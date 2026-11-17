"""First-principles BondFusion-TCN: ModernTCN multi-variable patched backbone upgraded with
decoupled curve prior, zero-phase symmetric multiscale depth filtering, 2D Hilbert-envelope
channel preservation with within-trace arrival contrast, and CBL-conditioned rational residual bridge.

Designed under the 4-slot Deep Learning AutoResearch invariants:
  - Law 1 (Shortcut Isolation): ``forward_prior`` reads only logging curves + valid mask
    (zero gradient / dependence on VDL waveform channels).
  - Law 2 (Symmetry & Zero-Phase): symmetric ``reflect`` multi-scale filters with odd
    symmetrized kernels eliminate phase shift; rational floor-distance gate
    guarantees ``boundary_gate(0) == 0``.
  - Law 6 (Zero-Degradation Bridge): VDL arrival-time cues preserve full bin resolution
    via channel-preserving linear lifting modulated by a CBL-conditioned residual gate.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moderntcn import Stage

DEFAULT_HP = {
    "d_lift": 16,
    "patch_size": 8,
    "d_model": 64,
    "ffn_ratio": 1,
    "num_blocks": 1,
    "large_size": 51,
    "small_size": 5,
    "dropout": 0.1,
}


class ZeroPhaseMultiScaleFilter(nn.Module):
    """Symmetric zero-phase 1D depth filter supporting both [B, 1, L] and [B, C, L]."""

    def __init__(self, small_size: int = 5, large_size: int = 31) -> None:
        super().__init__()
        if small_size % 2 == 0 or large_size % 2 == 0:
            raise ValueError("filter kernels must be odd for zero-phase symmetry")
        self.small_size = small_size
        self.large_size = large_size
        self.kernel_small = nn.Parameter(torch.randn(small_size) * 0.005)
        self.kernel_large = nn.Parameter(torch.randn(large_size) * 0.002)
        self.identity_weight = nn.Parameter(torch.tensor(1.0))

    def _sym_conv(self, x: torch.Tensor, raw_kernel: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        sym = 0.5 * (raw_kernel + raw_kernel.flip(0))
        half = sym.numel() // 2
        if length > 1 and half >= length:
            keep_half = length - 1
            trim = half - keep_half
            sym = sym[trim : sym.numel() - trim]
            half = keep_half
        elif length <= 1:
            sym = sym[half : half + 1]
            half = 0
        padded = F.pad(x, (half, half), mode="reflect") if half > 0 else x
        channels = x.shape[1]
        weight = sym.view(1, 1, -1).expand(channels, 1, -1).to(dtype=x.dtype, device=x.device)
        return F.conv1d(padded, weight, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply zero-phase smoothing along the last dimension of ``x`` [B, C, L]."""
        return self.identity_weight * x + self._sym_conv(x, self.kernel_small) + self._sym_conv(x, self.kernel_large)


class BondFusionTCN(nn.Module):
    """Decoupled curve-prior + arrival-time-preserving ModernTCN fusion model."""

    def __init__(
        self,
        n_curve: int,
        n_wave: int,
        n_classes: int,
        seq_len: int,
        hp: dict | None = None,
        use_vdl: bool = True,
        use_cbl_gate: bool = True,
        use_zero_phase: bool = True,
        use_trace_contrast: bool = True,
        use_group_norm: bool = False,
    ) -> None:
        super().__init__()
        del seq_len, use_group_norm
        p = {**DEFAULT_HP, **(hp or {})}
        d_lift = int(p["d_lift"])
        patch_size = int(p.get("patch_size", 8))
        patch_stride = max(1, patch_size // 2)
        d_model = int(p.get("d_model", 64))
        ffn_ratio = int(p.get("ffn_ratio", 1))
        num_blocks = int(p.get("num_blocks", 1))
        large_size = int(p.get("large_size", 51))
        small_size = int(p.get("small_size", 5))
        dropout = float(p.get("dropout", 0.1))

        self.n_curve = n_curve
        self.n_wave = n_wave
        self.n_classes = n_classes
        self.d_lift = d_lift
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.use_vdl = use_vdl and n_wave > 0
        self.use_cbl_gate = use_cbl_gate
        self.use_zero_phase = use_zero_phase
        self.use_trace_contrast = use_trace_contrast

        # Decoupled prior branch: reads ONLY curves + valid mask (never VDL channels).
        self.curve_lift = nn.Linear(n_curve + 1, d_lift)
        if self.use_zero_phase:
            self.multiscale_filter = ZeroPhaseMultiScaleFilter(small_size=small_size, large_size=min(large_size, 31))
        self.prior_head = nn.Linear(d_lift, n_classes)

        if self.use_vdl:
            # Explicit arrival-bin channel preservation (zero mean-collapse across arrival time).
            self.vdl_lift = nn.Linear(n_wave, d_lift, bias=False)
            if self.use_trace_contrast:
                self.vdl_contrast_lift = nn.Linear(n_wave, d_lift, bias=False)
                self.contrast_scale = nn.Parameter(torch.tensor(0.15))
            if self.use_cbl_gate:
                self.gate_proj = nn.Linear(n_curve + 1, d_lift)
                nn.init.constant_(self.gate_proj.bias, 1.0)
                self.gate_scale = nn.Parameter(torch.tensor(-0.5))
                self.prior_scale = nn.Parameter(torch.tensor(0.10))

        # Patched multi-variable ModernTCN backbone.
        self.stem = nn.Linear(patch_size, d_model)
        self.stage = Stage(
            ffn_ratio=ffn_ratio,
            num_blocks=num_blocks,
            large_size=large_size,
            small_size=small_size,
            dmodel=d_model,
            nvars=d_lift,
            drop=dropout,
        )
        self.patch_head = nn.Linear(d_model, patch_stride)
        self.head = nn.Linear(d_lift, n_classes)

    def _extract_tensor(self, batch_or_x: torch.Tensor | dict) -> torch.Tensor:
        return batch_or_x["x"] if isinstance(batch_or_x, dict) else batch_or_x

    def _split_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        curves_valid = torch.cat([x[..., : self.n_curve], x[..., -1:]], dim=-1)
        if not self.use_vdl:
            return curves_valid, None
        expected = self.n_curve + self.n_wave + 1
        if x.shape[-1] != expected:
            raise ValueError(f"expected {expected} channels, got {x.shape[-1]}")
        wave = x[..., self.n_curve : self.n_curve + self.n_wave]
        return curves_valid, wave

    def boundary_gate(self, dist: torch.Tensor) -> torch.Tensor:
        """Rational zero-boundary gate g(d) = d^2 / (d^2 + d0^2), exact g(0) == 0."""
        d0 = F.softplus(self.gate_scale) + 1e-2 if hasattr(self, "gate_scale") else dist.new_tensor(1.0)
        d2 = dist.float().pow(2)
        return (d2 / (d2 + d0.float().pow(2) + 1e-12)).to(dtype=dist.dtype)

    def forward_prior(self, batch_or_x: torch.Tensor | dict) -> torch.Tensor:
        """Pure curve-only prior logits without reading VDL channels (zero shortcut leakage)."""
        x = self._extract_tensor(batch_or_x)
        curves_valid = torch.cat([x[..., : self.n_curve], x[..., -1:]], dim=-1)
        z_curve = self.curve_lift(curves_valid)
        if self.use_zero_phase:
            z_curve = self.multiscale_filter(z_curve.transpose(1, 2)).transpose(1, 2)
        return self.prior_head(F.gelu(z_curve))

    def forward(self, batch_or_x: torch.Tensor | dict):
        """Return ``(logits, h)`` for Tensor input or dict contract for Lightning batch input."""
        is_dict = isinstance(batch_or_x, dict)
        x = self._extract_tensor(batch_or_x)
        B, L, _ = x.shape
        curves_valid, wave = self._split_inputs(x)

        z_curve = self.curve_lift(curves_valid)
        if self.use_zero_phase:
            z_curve = self.multiscale_filter(z_curve.transpose(1, 2)).transpose(1, 2)
        prior_logits = self.prior_head(F.gelu(z_curve))

        if self.use_vdl and wave is not None:
            vdl_feat = self.vdl_lift(wave)
            if self.use_trace_contrast:
                rel = (wave - wave.mean(dim=-1, keepdim=True)) / (wave.std(dim=-1, keepdim=True) + 1e-5)
                vdl_feat = vdl_feat + self.contrast_scale * self.vdl_contrast_lift(rel)
            if self.use_cbl_gate:
                gate = self.boundary_gate(self.gate_proj(curves_valid))
                z_fused = z_curve + (0.5 + 0.5 * gate) * vdl_feat
            else:
                z_fused = z_curve + vdl_feat
        else:
            z_fused = z_curve

        z = z_fused.permute(0, 2, 1)  # [B, d_lift, L]
        if self.patch_size != self.patch_stride:
            pad_len = self.patch_size - self.patch_stride
            if self.use_zero_phase:
                pad_left = pad_len // 2
                pad_right = pad_len - pad_left
                if z.shape[-1] > max(pad_left, pad_right):
                    z = F.pad(z, (pad_left, pad_right), mode="reflect")
                else:
                    z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_len)], dim=-1)
            else:
                z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_len)], dim=-1)

        z = z.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)  # [B, M, N, P]
        z = self.stem(z).permute(0, 1, 3, 2)  # [B, M, d_model, N]
        z = self.stage(z).permute(0, 1, 3, 2)  # [B, M, N, d_model]
        h = self.patch_head(z).reshape(B, self.d_lift, -1)[:, :, :L].permute(0, 2, 1)  # [B, L, d_lift]

        logits = self.head(h) + (
            self.prior_scale * prior_logits if (self.use_vdl and self.use_cbl_gate) else 0.0 * prior_logits
        )

        if is_dict:
            return {"pred": logits, "logits": logits, "prior_logits": prior_logits, "h": h}
        return logits, h
