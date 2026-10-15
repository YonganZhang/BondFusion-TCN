"""Baseline registry for point-wise cement-bond classification.

Every model maps x [B, L, F] -> (logits [B, L, C], h [B, L, D]); h is the
point-wise representation fed to the final `nn.Linear(D, C)` head.

No baseline uses per-window instance normalization (RevIN / Non-stationary
norm / TSLib `means, stdev` norm + de-norm): inputs are already globally scaled
and absolute amplitude carries physical meaning.

Per-model wiring (sources: vendored TSLib under ../vendor/tslib, ModernTCN in
moderntcn.py):
  LSTM / GRU   torch bidirectional multi-layer RNN, D = 2 * hidden.
  Transformer  TSLib DataEmbedding (value + position) + Encoder(FullAttention,
               mask_flag=False) as in TSLib Transformer imputation/anomaly path;
               D = d_model.
  Reformer     backbones.ReformerEncoder; D = d_model.
  TimesNet     TSLib DataEmbedding + TimesBlock stack + LayerNorm (anomaly path,
               Non-stationary norm/de-norm removed); D = d_model.
  Crossformer, TimeMixer, ModernTCN natively emit one value per input variable
               per time step. Wiring: Linear(F -> d_lift) channel lift, run the
               model's own dense output path on the d_lift "variables", then
               h = per-point, per-variable output [B, L, d_lift]; D = d_lift.
    Crossformer  DSW PatchEmbedding + TSA scale_block Encoder + FlattenHead
                 (TSLib imputation/anomaly path).
    TimeMixer    multi-scale avg-pool inputs, channel-independent
                 DataEmbedding_wo_pos + PastDecomposableMixing, Linear(d_model, 1)
                 (TSLib anomaly path with Normalize layers removed).
    ModernTCN    official detection backbone (see moderntcn.py).
"""
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
from einops import rearrange

from ..vendor.tslib.layers.Crossformer_EncDec import Encoder as CrossformerEncoderStack
from ..vendor.tslib.layers.Crossformer_EncDec import scale_block
from ..vendor.tslib.layers.Embed import DataEmbedding, DataEmbedding_wo_pos, PatchEmbedding
from ..vendor.tslib.layers.SelfAttention_Family import AttentionLayer, FullAttention
from ..vendor.tslib.layers.Transformer_EncDec import Encoder, EncoderLayer
from ..vendor.tslib.models.PatchTST import FlattenHead
from ..vendor.tslib.models.TimeMixer import PastDecomposableMixing
from ..vendor.tslib.models.TimesNet import TimesBlock
from .backbones import ReformerEncoder
from .moderntcn import ModernTCNPointwise

BASELINE_NAMES = ["LSTM", "GRU", "Transformer", "Reformer", "Crossformer", "TimesNet", "TimeMixer", "ModernTCN"]

# training-loop keys that may ride along in hp; ignored by model construction
TRAIN_HP_KEYS = {"lr", "weight_decay", "batch_size"}


class PointwiseClassifier(nn.Module):
    """encoder: x [B, L, F] -> h [B, L, D]; head: Linear(D, C)."""

    def __init__(self, encoder, d_repr, n_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(d_repr, n_classes)

    def forward(self, x):
        h = self.encoder(x)
        return self.head(h), h


class RNNEncoder(nn.Module):
    def __init__(self, n_in, hidden, num_layers, dropout, cell):
        super().__init__()
        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[cell]
        self.rnn = rnn_cls(n_in, hidden, num_layers=num_layers, batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.dropout(h)


class TransformerEncoder(nn.Module):
    def __init__(self, n_in, d_model, n_heads, e_layers, d_ff, dropout):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.embedding = DataEmbedding(n_in, d_model, dropout=dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(False, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation="gelu",
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

    def forward(self, x):
        h = self.embedding(x, None)
        h, _ = self.encoder(h, attn_mask=None)
        return h


class TimesNetEncoder(nn.Module):
    def __init__(self, n_in, seq_len, d_model, d_ff, e_layers, top_k, num_kernels, dropout):
        super().__init__()
        cfg = SimpleNamespace(seq_len=seq_len, pred_len=0, top_k=top_k, d_model=d_model, d_ff=d_ff,
                              num_kernels=num_kernels)
        self.seq_len = seq_len
        self.embedding = DataEmbedding(n_in, d_model, dropout=dropout)
        self.blocks = nn.ModuleList([TimesBlock(cfg) for _ in range(e_layers)])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        if x.shape[1] != self.seq_len:
            raise ValueError(f"TimesNet built for L={self.seq_len}, got L={x.shape[1]}")
        h = self.embedding(x, None)
        for block in self.blocks:
            h = self.layer_norm(block(h))
        return h


class CrossformerEncoder(nn.Module):
    """Lift F -> d_lift, Crossformer encoder, FlattenHead back to L points per variable."""

    def __init__(self, n_in, seq_len, d_lift, d_model, n_heads, e_layers, d_ff, seg_len, factor, dropout,
                 win_size=2):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        cfg = SimpleNamespace(factor=factor, dropout=dropout)
        self.seq_len = seq_len
        self.lift = nn.Linear(n_in, d_lift)
        pad_in_len = math.ceil(seq_len / seg_len) * seg_len
        in_seg_num = pad_in_len // seg_len
        out_seg_num = math.ceil(in_seg_num / (win_size ** (e_layers - 1)))
        self.value_embedding = PatchEmbedding(d_model, seg_len, seg_len, pad_in_len - seq_len, 0)
        self.pos_embedding = nn.Parameter(torch.randn(1, d_lift, in_seg_num, d_model))
        self.pre_norm = nn.LayerNorm(d_model)
        self.encoder = CrossformerEncoderStack(
            [
                scale_block(cfg, 1 if l == 0 else win_size, d_model, n_heads, d_ff, 1, dropout,
                            in_seg_num if l == 0 else math.ceil(in_seg_num / win_size ** l), factor)
                for l in range(e_layers)
            ]
        )
        self.head = FlattenHead(d_lift, d_model * out_seg_num, seq_len, head_dropout=dropout)

    def forward(self, x):
        if x.shape[1] != self.seq_len:
            raise ValueError(f"Crossformer built for L={self.seq_len}, got L={x.shape[1]}")
        z = self.lift(x)  # [B, L, M]
        z, n_vars = self.value_embedding(z.permute(0, 2, 1))
        z = rearrange(z, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        z = self.pre_norm(z + self.pos_embedding)
        enc_out, _ = self.encoder(z)
        y = self.head(enc_out[-1].permute(0, 1, 3, 2))  # [B, M, L]
        return y.permute(0, 2, 1)


class TimeMixerEncoder(nn.Module):
    """Lift F -> d_lift, channel-independent TimeMixer, Linear(d_model, 1) per variable."""

    def __init__(self, n_in, seq_len, d_lift, d_model, d_ff, e_layers, down_sampling_layers, moving_avg,
                 dropout, down_sampling_window=2, decomp_method="moving_avg", top_k=5):
        super().__init__()
        if down_sampling_layers < 1:
            raise ValueError("TimeMixer multi-scale mixing needs down_sampling_layers >= 1")
        if seq_len % (down_sampling_window ** down_sampling_layers) != 0:
            raise ValueError(f"seq_len={seq_len} not divisible by {down_sampling_window}**{down_sampling_layers}")
        cfg = SimpleNamespace(seq_len=seq_len, pred_len=0, down_sampling_window=down_sampling_window,
                              down_sampling_layers=down_sampling_layers, d_model=d_model, d_ff=d_ff,
                              dropout=dropout, channel_independence=1, decomp_method=decomp_method,
                              moving_avg=moving_avg, top_k=top_k)
        self.seq_len = seq_len
        self.down_sampling_layers = down_sampling_layers
        self.lift = nn.Linear(n_in, d_lift)
        self.down_pool = nn.AvgPool1d(down_sampling_window)
        self.embedding = DataEmbedding_wo_pos(1, d_model, dropout=dropout)
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(cfg) for _ in range(e_layers)])
        self.projection = nn.Linear(d_model, 1, bias=True)

    def forward(self, x):
        if x.shape[1] != self.seq_len:
            raise ValueError(f"TimeMixer built for L={self.seq_len}, got L={x.shape[1]}")
        z = self.lift(x)  # [B, L, M]
        B, _, M = z.shape
        # multi-scale inputs (down_sampling_method='avg')
        zc = z.permute(0, 2, 1)
        scales = [zc]
        for _ in range(self.down_sampling_layers):
            zc = self.down_pool(zc)
            scales.append(zc)
        # channel independence: [B, M, T] -> [B*M, T, 1]
        enc_out_list = [self.embedding(s.reshape(B * M, -1, 1), None) for s in scales]
        for block in self.pdm_blocks:
            enc_out_list = block(enc_out_list)
        y = self.projection(enc_out_list[0])  # [B*M, L, 1]
        return y.reshape(B, M, -1).permute(0, 2, 1)


class ModernTCNEncoder(nn.Module):
    """Lift F -> d_lift, ModernTCN detection backbone, one value per variable per point."""

    def __init__(self, n_in, seq_len, d_lift, **kw):
        super().__init__()
        self.seq_len = seq_len
        self.lift = nn.Linear(n_in, d_lift)
        self.backbone = ModernTCNPointwise(d_lift, seq_len, **kw)

    def forward(self, x):
        if x.shape[1] != self.seq_len:
            raise ValueError(f"ModernTCN built for L={self.seq_len}, got L={x.shape[1]}")
        return self.backbone(self.lift(x).permute(0, 2, 1)).permute(0, 2, 1)


_DEFAULT_HP = {
    "LSTM": {"hidden": 64, "num_layers": 2},
    "GRU": {"hidden": 64, "num_layers": 2},
    "Transformer": {"d_model": 64, "n_heads": 4, "e_layers": 2, "d_ff": 128},
    "Reformer": {"d_model": 64, "n_heads": 4, "e_layers": 2, "d_ff": 128, "bucket_size": 4, "n_hashes": 4},
    "Crossformer": {"d_lift": 16, "d_model": 64, "n_heads": 4, "e_layers": 2, "d_ff": 128, "seg_len": 12,
                    "factor": 1},
    "TimesNet": {"d_model": 32, "d_ff": 32, "e_layers": 2, "top_k": 3, "num_kernels": 6},
    "TimeMixer": {"d_lift": 16, "d_model": 16, "d_ff": 32, "e_layers": 2, "down_sampling_layers": 3,
                  "moving_avg": 25},
    "ModernTCN": {"d_lift": 16, "patch_size": 8, "d_model": 64, "ffn_ratio": 1, "num_blocks": 1,
                  "large_size": 51, "small_size": 5},
}


def default_hp(name):
    if name not in _DEFAULT_HP:
        raise KeyError(f"unknown baseline {name!r}; expected one of {BASELINE_NAMES}")
    return {**_DEFAULT_HP[name], "dropout": 0.1}


def suggest_hp(name, trial):
    """Model-specific search space (lr / weight_decay / dropout are searched by the caller)."""
    s = trial
    if name in ("LSTM", "GRU"):
        return {"hidden": s.suggest_categorical("hidden", [32, 64, 128]),
                "num_layers": s.suggest_int("num_layers", 1, 3)}
    if name in ("Transformer", "Reformer"):
        hp = {"d_model": s.suggest_categorical("d_model", [32, 64, 128]),
              "n_heads": s.suggest_categorical("n_heads", [2, 4, 8]),
              "e_layers": s.suggest_int("e_layers", 1, 3),
              "d_ff": s.suggest_categorical("d_ff", [64, 128, 256])}
        if name == "Reformer":
            hp["bucket_size"] = s.suggest_categorical("bucket_size", [4, 8, 16])
            hp["n_hashes"] = s.suggest_categorical("n_hashes", [2, 4, 8])
        return hp
    if name == "Crossformer":
        return {"d_lift": s.suggest_categorical("d_lift", [8, 16, 32]),
                "d_model": s.suggest_categorical("d_model", [32, 64, 128]),
                "n_heads": s.suggest_categorical("n_heads", [2, 4, 8]),
                "e_layers": s.suggest_int("e_layers", 1, 3),
                "d_ff": s.suggest_categorical("d_ff", [64, 128, 256]),
                "seg_len": s.suggest_categorical("seg_len", [8, 12, 16, 24, 32]),
                "factor": s.suggest_categorical("factor", [1, 5, 10])}
    if name == "TimesNet":
        return {"d_model": s.suggest_categorical("d_model", [16, 32, 64]),
                "d_ff": s.suggest_categorical("d_ff", [16, 32, 64]),
                "e_layers": s.suggest_int("e_layers", 1, 3),
                "top_k": s.suggest_int("top_k", 2, 5),
                "num_kernels": s.suggest_categorical("num_kernels", [4, 6])}
    if name == "TimeMixer":
        # channel independence runs B * d_lift sequences: kept small so that the largest
        # config stays < ~5 GB at batch 64, L=512
        return {"d_lift": s.suggest_categorical("d_lift", [8, 16]),
                "d_model": s.suggest_categorical("d_model", [16, 32]),
                "d_ff": s.suggest_categorical("d_ff", [32, 64]),
                "e_layers": s.suggest_int("e_layers", 1, 3),
                "down_sampling_layers": s.suggest_int("down_sampling_layers", 1, 3),
                "moving_avg": s.suggest_categorical("moving_avg", [13, 25, 49])}
    if name == "ModernTCN":
        return {"d_lift": s.suggest_categorical("d_lift", [8, 16, 32]),
                "patch_size": s.suggest_categorical("patch_size", [4, 8, 16]),
                "d_model": s.suggest_categorical("d_model", [32, 64]),
                "ffn_ratio": s.suggest_categorical("ffn_ratio", [1, 2]),
                "num_blocks": s.suggest_int("num_blocks", 1, 2),
                "large_size": s.suggest_categorical("large_size", [31, 51, 71]),
                "small_size": 5}
    raise KeyError(f"unknown baseline {name!r}; expected one of {BASELINE_NAMES}")


def build_baseline(name, n_features, n_classes, seq_len, hp):
    """hp: model keys (subset of default_hp(name)) plus optional TRAIN_HP_KEYS; missing keys use defaults."""
    base = default_hp(name)
    unknown = set(hp) - set(base) - TRAIN_HP_KEYS
    if unknown:
        raise ValueError(f"{name}: unknown hp keys {sorted(unknown)}")
    p = {**base, **{k: v for k, v in hp.items() if k in base}}
    dropout = p["dropout"]

    if name in ("LSTM", "GRU"):
        enc = RNNEncoder(n_features, p["hidden"], p["num_layers"], dropout, name)
        d_repr = 2 * p["hidden"]
    elif name == "Transformer":
        enc = TransformerEncoder(n_features, p["d_model"], p["n_heads"], p["e_layers"], p["d_ff"], dropout)
        d_repr = p["d_model"]
    elif name == "Reformer":
        enc = ReformerEncoder(n_features, d_model=p["d_model"], n_heads=p["n_heads"], e_layers=p["e_layers"],
                              d_ff=p["d_ff"], dropout=dropout, bucket_size=p["bucket_size"],
                              n_hashes=p["n_hashes"])
        d_repr = p["d_model"]
    elif name == "Crossformer":
        enc = CrossformerEncoder(n_features, seq_len, p["d_lift"], p["d_model"], p["n_heads"], p["e_layers"],
                                 p["d_ff"], p["seg_len"], p["factor"], dropout)
        d_repr = p["d_lift"]
    elif name == "TimesNet":
        enc = TimesNetEncoder(n_features, seq_len, p["d_model"], p["d_ff"], p["e_layers"], p["top_k"],
                              p["num_kernels"], dropout)
        d_repr = p["d_model"]
    elif name == "TimeMixer":
        enc = TimeMixerEncoder(n_features, seq_len, p["d_lift"], p["d_model"], p["d_ff"], p["e_layers"],
                               p["down_sampling_layers"], p["moving_avg"], dropout)
        d_repr = p["d_lift"]
    elif name == "ModernTCN":
        enc = ModernTCNEncoder(n_features, seq_len, p["d_lift"], patch_size=p["patch_size"],
                               patch_stride=p["patch_size"] // 2, d_model=p["d_model"], ffn_ratio=p["ffn_ratio"],
                               num_blocks=p["num_blocks"], large_size=p["large_size"],
                               small_size=p["small_size"], dropout=dropout)
        d_repr = p["d_lift"]
    else:
        raise KeyError(f"unknown baseline {name!r}; expected one of {BASELINE_NAMES}")
    return PointwiseClassifier(enc, d_repr, n_classes)
