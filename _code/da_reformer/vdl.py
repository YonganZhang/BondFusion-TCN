"""VDL (variable-density log) waveform inputs: one 2000-sample acoustic trace per depth.

`_data/raw/<well>_vdl.npy` (int16 [N, 2000], same row order as `<well>.csv`) is
produced by step_01. Preprocessing, fitted on the training wells of the split only:

- keep samples [0, n_keep): the protocol keeps the first 1900 of 2000 samples,
  excluding terminal values at or near the int16 floor; the excluded tail is
  not uniformly constant;
- log-envelope proxy: log1p(mean(|v|)) in each non-overlapping bin. This
  summarises waveform magnitude, drops polarity/phase and compresses range;
- per-bin robust scaling (median, IQR/1.349 of the training wells), clipped to ±clip.

Two granularities are used: fine bins (e.g. 190 bins of 10 samples) feed the
waveform encoder of VDL-ModernTCN; coarse blocks (e.g. 19 blocks of 100 samples)
are appended as plain feature channels for the "+VDL" baselines.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_vdl(raw_dir: Path, name: str, n_keep: int) -> np.ndarray:
    v = np.load(Path(raw_dir) / f"{name}_vdl.npy")
    if v.ndim != 2 or v.shape[1] < n_keep:
        raise ValueError(f"well {name}: VDL shape {v.shape}, need >= {n_keep} samples")
    return v[:, :n_keep].astype(np.float32)


def build_log_envelope(v: np.ndarray, bin_size: int) -> np.ndarray:
    n, t = v.shape
    if t % bin_size:
        raise ValueError(f"n_keep={t} is not a multiple of bin={bin_size}")
    return np.log1p(np.abs(v).reshape(n, t // bin_size, bin_size).mean(2)).astype(np.float32)


class VdlScaler:
    def __init__(self, clip: float = 8.0):
        self.clip = clip
        self.median: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, envs: list[np.ndarray]) -> "VdlScaler":
        x = np.concatenate(envs)
        self.median = np.median(x, axis=0)
        q75, q25 = np.percentile(x, [75, 25], axis=0)
        iqr = q75 - q25
        self.scale = np.where(iqr > 1e-8, iqr / 1.349, 1.0)
        return self

    def transform(self, env: np.ndarray) -> np.ndarray:
        return np.clip((env - self.median) / self.scale, -self.clip, self.clip).astype(np.float32)


def build_vdl_channels(raw_dir: Path, names: list[str], train_names: list[str], n_keep: int, bin_size: int,
                       clip: float = 8.0) -> dict[str, np.ndarray]:
    """Scaled log-envelope [N, n_keep // bin_size] per well; scaler fitted on `train_names`."""
    envs = {n: build_log_envelope(load_vdl(raw_dir, n, n_keep), bin_size) for n in names}
    scaler = VdlScaler(clip).fit([envs[n] for n in train_names])
    return {n: scaler.transform(e) for n, e in envs.items()}
