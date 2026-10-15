"""VDL-ModernTCN: CBL-conditioned attention over VDL arrival windows + ModernTCN along depth.

Input x [B, L, C], channel layout [curves (n_curve) | VDL log-envelope bins (n_wave) | valid].

1. Waveform encoder, shared over depths: bins [B*L, 1, n_wave] -> Conv1d stack (two stride-2
   stages) -> time tokens [B*L, T, d_wave] + learned time-position embedding (arrival time is
   physically meaningful: casing arrivals come before formation arrivals).
2. Pooling of the time tokens into one vector per depth (`pool`):
   - "curve_attn" (proposed): the query is an MLP of the logging curves (+ valid flag) at the
     same depth; single-head scaled dot-product attention over the time tokens. It mirrors the
     CBL/VDL reading order: the amplitude curve first, then the VDL window that resolves what
     the curve leaves open (e.g. Moderate vs Good once CBL is at the tool floor).
   - "static_attn": one learned query shared by all depths (window selection without curves).
   - "mean": token average (no window selection).
3. Fusion: [curves, valid, pooled VDL vector] -> ModernTCN encoder exactly as in the ModernTCN
   baseline (Linear lift + official detection backbone) -> point-wise linear head.
4. `global_layer="reformer"`: one TSLib Reformer encoder layer (LSH self-attention along depth)
   on top of the ModernTCN output, residual; tests a convolution + LSH-attention hybrid.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..vendor.tslib.layers.SelfAttention_Family import ReformerLayer
from ..vendor.tslib.layers.Transformer_EncDec import EncoderLayer
from .baselines import ModernTCNEncoder

POOLS = ("curve_attn", "static_attn", "mean")
GLOBAL_LAYERS = ("none", "reformer")

DEFAULT_HP = {
    # ModernTCN backbone (same keys and meaning as the ModernTCN baseline)
    "d_lift": 16, "patch_size": 8, "d_model": 64, "ffn_ratio": 1, "num_blocks": 1, "large_size": 51,
    "small_size": 5, "dropout": 0.1,
    # waveform encoder
    "d_wave": 32, "wave_channels": 32,
    # optional Reformer layer
    "g_d_model": 64, "g_n_heads": 4, "g_bucket_size": 8, "g_n_hashes": 4,
}


class WaveEncoder(nn.Module):
    """[N, n_wave] -> time tokens [N, T, d_wave]."""

    def __init__(self, n_wave: int, d_wave: int, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels, 7, padding=3), nn.GELU(),
            nn.Conv1d(channels, channels, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(channels, d_wave, 5, stride=2, padding=2),
        )
        n_tok = ((n_wave + 1) // 2 + 1) // 2
        self.pos = nn.Parameter(torch.randn(1, n_tok, d_wave) * 0.02)
        self.norm = nn.LayerNorm(d_wave)

    def forward(self, w):
        tok = self.conv(w.unsqueeze(1)).transpose(1, 2)  # [N, T, d_wave]
        return self.norm(tok + self.pos)


class VdlModernTCN(nn.Module):
    def __init__(self, n_curve: int, n_wave: int, n_classes: int, seq_len: int, hp: dict,
                 pool: str = "curve_attn", global_layer: str = "none"):
        super().__init__()
        if pool not in POOLS or global_layer not in GLOBAL_LAYERS:
            raise ValueError(f"pool={pool!r} global_layer={global_layer!r}")
        unknown = set(hp) - set(DEFAULT_HP) - {"lr", "weight_decay", "batch_size"}
        if unknown:
            raise ValueError(f"VDL-ModernTCN: unknown hp keys {sorted(unknown)}")
        p = {**DEFAULT_HP, **{k: v for k, v in hp.items() if k in DEFAULT_HP}}
        self.n_curve, self.n_wave, self.pool = n_curve, n_wave, pool
        d_wave = p["d_wave"]
        self.wave = WaveEncoder(n_wave, d_wave, p["wave_channels"])
        if pool == "curve_attn":
            self.query = nn.Sequential(nn.Linear(n_curve + 1, d_wave), nn.GELU(), nn.Linear(d_wave, d_wave))
        elif pool == "static_attn":
            self.query_static = nn.Parameter(torch.randn(d_wave) * 0.02)
        self.encoder = ModernTCNEncoder(n_curve + 1 + d_wave, seq_len, p["d_lift"], patch_size=p["patch_size"],
                                        patch_stride=p["patch_size"] // 2, d_model=p["d_model"],
                                        ffn_ratio=p["ffn_ratio"], num_blocks=p["num_blocks"],
                                        large_size=p["large_size"], small_size=p["small_size"],
                                        dropout=p["dropout"])
        d_repr = p["d_lift"]
        self.global_layer = global_layer
        if global_layer == "reformer":
            g = p["g_d_model"]
            self.g_in = nn.Linear(d_repr, g)
            self.g_layer = EncoderLayer(ReformerLayer(None, g, p["g_n_heads"], bucket_size=p["g_bucket_size"],
                                                      n_hashes=p["g_n_hashes"]), g, 2 * g, dropout=p["dropout"],
                                        activation="gelu")
            self.g_out = nn.Linear(g, d_repr)
        self.head = nn.Linear(d_repr, n_classes)
        self.last_attention: torch.Tensor | None = None  # [B, L, T] of the last forward (pooling weights)

    def split_inputs(self, x):
        c, w = self.n_curve, self.n_wave
        if x.shape[-1] != c + w + 1:
            raise ValueError(f"expected {c}+{w}+1 channels, got {x.shape[-1]}")
        return x[..., :c], x[..., c:c + w], x[..., -1:]

    def pool_wave(self, curves_valid, wave):
        B, L, _ = wave.shape
        tok = self.wave(wave.reshape(B * L, -1))  # [BL, T, d]
        if self.pool == "mean":
            self.last_attention = None
            return tok.mean(1).reshape(B, L, -1)
        if self.pool == "curve_attn":
            q = self.query(curves_valid.reshape(B * L, -1))  # [BL, d]
        else:
            q = self.query_static.expand(B * L, -1)
        a = torch.softmax(torch.einsum("ntd,nd->nt", tok, q) / math.sqrt(tok.shape[-1]), dim=-1)
        self.last_attention = a.detach().reshape(B, L, -1)
        return torch.einsum("nt,ntd->nd", a, tok).reshape(B, L, -1)

    def forward(self, x):
        curves, wave, valid = self.split_inputs(x)
        cv = torch.cat([curves, valid], -1)
        h = self.encoder(torch.cat([cv, self.pool_wave(cv, wave)], -1))  # [B, L, d_lift]
        if self.global_layer == "reformer":
            g, _ = self.g_layer(self.g_in(h))
            h = h + self.g_out(g)
        return self.head(h), h
