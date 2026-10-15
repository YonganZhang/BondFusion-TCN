"""ModernTCN backbone with a point-wise (dense) output.

Source: official implementation, https://github.com/luodhhh/ModernTCN
  commit 56a9a2c018385cd5acef015378cae7f084d1b11c,
  file ModernTCN-detection/models/ModernTCN.py (anomaly-detection variant, the
  official dense / per-time-step output path). Paper: Luo & Wang, "ModernTCN: A
  Modern Pure Convolution Structure for General Time Series Analysis", ICLR 2024.

Modifications (all relative to the detection file above):
  1. RevIN removed (revin=False hard-wired): absolute amplitude is informative
     here, no per-window normalization / de-normalization.
  2. Single stage only (as in every official detection script, num_blocks=1 stage);
     the multi-stage downsampling branch and the unused Flatten_Head /
     use_multi_scale / individual / classification arguments are dropped.
  3. Detection head `head_dection1 = Linear(d_model, patch_size)` changed to
     `Linear(d_model, patch_stride)`: each token j covers points
     [j*stride, j*stride + patch_size); emitting `stride` points per token makes
     the N*stride outputs line up with the L input points. Upstream emits
     N*patch_size points and truncates to L, which misaligns time whenever
     patch_size != patch_stride (e.g. the official PSM/SMD scripts).
  4. Structural re-parameterization helpers (merge_kernel / PaddingTwoEdge1d /
     structural_reparam) dropped: inference-only speed-up, and upstream
     PaddingTwoEdge1d calls torch.cat(..., dims=-1), which raises. Training graph
     (large DW conv + parallel small DW conv, both with BN) is unchanged.
  5. Unused custom LayerNorm class (upstream references nn.Layernorm, a typo)
     dropped.
  The backbone input is [B, M, L]; the caller (baselines.py) lifts the F raw
  channels to M = d_lift channels with a Linear layer first.

Upstream license (MIT):
  Copyright (c) 2024 luodhhh
  Permission is hereby granted, free of charge, to any person obtaining a copy of
  this software and associated documentation files (the "Software"), to deal in
  the Software without restriction, including without limitation the rights to
  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
  the Software, and to permit persons to whom the Software is furnished to do so,
  subject to the following conditions: The above copyright notice and this
  permission notice shall be included in all copies or substantial portions of
  the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
  EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
  OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
"""
import torch
from torch import nn


def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm1d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result


class ReparamLargeKernelConv(nn.Module):
    """Large-kernel DW conv + parallel small-kernel DW conv (training form)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, small_kernel):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # the conv keeps the feature map size, so padding = k // 2
        padding = kernel_size // 2
        self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=1, groups=groups, bias=False)
        if small_kernel is not None:
            assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
            self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel,
                                      stride=stride, padding=small_kernel // 2, groups=groups, dilation=1, bias=False)

    def forward(self, inputs):
        out = self.lkb_origin(inputs)
        if hasattr(self, 'small_conv'):
            out += self.small_conv(inputs)
        return out


class Block(nn.Module):
    """DWConv (temporal) -> BN -> ConvFFN1 (per variable) -> ConvFFN2 (across variables)."""

    def __init__(self, large_size, small_size, dmodel, dff, nvars, drop=0.1):
        super(Block, self).__init__()
        self.dw = ReparamLargeKernelConv(in_channels=nvars * dmodel, out_channels=nvars * dmodel,
                                         kernel_size=large_size, stride=1, groups=nvars * dmodel,
                                         small_kernel=small_size)
        self.norm = nn.BatchNorm1d(dmodel)

        # convffn1
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        # convffn2
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

    def forward(self, x):
        input = x
        B, M, D, N = x.shape
        x = x.reshape(B, M * D, N)
        x = self.dw(x)
        x = x.reshape(B * M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M * D, N)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)

        return input + x


class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, nvars, drop=0.1):
        super(Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio
        self.blocks = nn.ModuleList([
            Block(large_size=large_size, small_size=small_size, dmodel=dmodel, dff=d_ffn, nvars=nvars, drop=drop)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class ModernTCNPointwise(nn.Module):
    """x [B, M, L] -> y [B, M, L]: one value per variable per time step (detection path)."""

    def __init__(self, nvars, seq_len, patch_size=8, patch_stride=4, d_model=64, ffn_ratio=1, num_blocks=1,
                 large_size=51, small_size=5, dropout=0.1):
        super().__init__()
        if patch_size < patch_stride:
            raise ValueError(f"patch_size={patch_size} must be >= patch_stride={patch_stride}")
        if seq_len % patch_stride != 0:
            raise ValueError(f"seq_len={seq_len} must be divisible by patch_stride={patch_stride}")
        if large_size % 2 == 0 or small_size % 2 == 0:
            raise ValueError("large_size and small_size must be odd to keep the length")
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        # stem: patchify each variable, Linear(patch_size -> d_model)
        self.stem = nn.Linear(patch_size, d_model)
        self.stage = Stage(ffn_ratio, num_blocks, large_size, small_size, dmodel=d_model, nvars=nvars, drop=dropout)
        # dense head: each token -> patch_stride points (modification 3)
        self.head = nn.Linear(d_model, patch_stride)

    def forward_feature(self, x):
        # x: [B, M, L]
        if self.patch_size != self.patch_stride:
            # stem layer padding (replicate last value), as upstream
            pad_len = self.patch_size - self.patch_stride
            x = torch.cat([x, x[:, :, -1:].repeat(1, 1, pad_len)], dim=-1)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)  # [B, M, N, P]
        x = self.stem(x).permute(0, 1, 3, 2)  # [B, M, D, N]
        return self.stage(x)

    def forward(self, x):
        B, M, L = x.shape
        z = self.forward_feature(x).permute(0, 1, 3, 2)  # [B, M, N, D]
        y = self.head(z).reshape(B, M, -1)  # [B, M, N * stride] == [B, M, L]
        return y[:, :, :self.seq_len]
