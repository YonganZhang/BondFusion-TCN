"""Well loading, cleaning, train-only scaling and segment-aware windowing.

Conventions
- One row of `_data/raw/<well>.csv` = one depth sample; rows are kept as-is
  (duplicated depths included) so every raw sample gets exactly one prediction.
- A well is split into contiguous segments wherever the depth step exceeds
  `gap_m`; windows never cross a segment boundary. Segments shorter than the
  window are zero-padded and flagged invalid.
- Scaling statistics are fitted on the training wells of the current split only.
- Model input = scaled features (+ optional missing-indicator channels) + a
  final `valid` channel (1 = real sample, 0 = padding).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL_MAP = {"胶结差": 0, "胶结中等": 1, "胶结好": 2, "胶结良好": 2, "胶结优": 2}
CLASS_NAMES = ["Poor", "Moderate", "Good"]
MISSING_THRESHOLD = -999.0
IGNORE_INDEX = -100


@dataclass
class WellData:
    name: str
    depth: np.ndarray            # [N]
    x_raw: np.ndarray            # [N, F] float, NaN = missing
    y: np.ndarray                # [N] int64
    segments: list[tuple[int, int]] = field(default_factory=list)  # [start, end)


def load_well(raw_dir: Path, name: str, features: list[str], gap_m: float) -> WellData:
    df = pd.read_csv(Path(raw_dir) / f"{name}.csv")
    unknown = set(df["label_raw"]) - set(LABEL_MAP)
    if unknown:
        raise ValueError(f"well {name}: unknown labels {unknown}")
    x = df[features].to_numpy(dtype=np.float64, copy=True)
    x[x <= MISSING_THRESHOLD] = np.nan
    depth = df["DEPTH"].to_numpy(dtype=np.float64)
    if np.any(np.diff(depth) < 0):
        raise ValueError(f"well {name}: depth is not non-decreasing")
    cuts = np.where(np.diff(depth) > gap_m)[0] + 1
    bounds = np.r_[0, cuts, len(depth)]
    segments = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]
    return WellData(name, depth, x, df["label_raw"].map(LABEL_MAP).to_numpy(np.int64), segments)


def load_wells(raw_dir: Path, names: list[str], features: list[str], gap_m: float) -> dict[str, WellData]:
    return {n: load_well(raw_dir, n, features, gap_m) for n in names}


def compute_class_counts(wells: list[WellData], n_classes: int = 3) -> np.ndarray:
    y = np.concatenate([w.y for w in wells])
    return np.bincount(y, minlength=n_classes).astype(np.float64)


def compute_class_weights(counts: np.ndarray, power: float = 0.5) -> np.ndarray:
    """Inverse-frequency^power weights, normalised to a sample-weighted mean of 1."""
    freq = counts / counts.sum()
    w = (1.0 / np.maximum(freq, 1e-12)) ** power
    return w / (w * freq).sum()


class FeatureScaler:
    """Per-feature optional signed-log transform, then median/IQR scaling and clipping.

    Missing values become 0 after scaling (= training median); features listed in
    `missing_indicators` additionally get a 0/1 "missing" channel.
    """

    def __init__(self, features: list[str], transforms: dict[str, str] | None = None,
                 clip: float = 8.0, missing_indicators: list[str] | None = None):
        self.features = list(features)
        self.transforms = dict(transforms or {})
        self.clip = clip
        self.missing_indicators = list(missing_indicators or [])
        self.median: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def _apply_transforms(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        for j, f in enumerate(self.features):
            t = self.transforms.get(f)
            if t == "signed_log1p":
                x[:, j] = np.sign(x[:, j]) * np.log1p(np.abs(x[:, j]))
            elif t is not None:
                raise ValueError(f"unknown transform {t}")
        return x

    def fit(self, wells: list[WellData]) -> "FeatureScaler":
        x = self._apply_transforms(np.concatenate([w.x_raw for w in wells]))
        self.median = np.nanmedian(x, axis=0)
        q75, q25 = np.nanpercentile(x, [75, 25], axis=0)
        iqr = q75 - q25
        self.scale = np.where(iqr > 1e-8, iqr / 1.349, 1.0)
        return self

    @property
    def n_channels(self) -> int:
        return len(self.features) + len(self.missing_indicators)

    def transform(self, well: WellData) -> np.ndarray:
        x = self._apply_transforms(well.x_raw)
        missing = np.isnan(x)
        z = np.clip((x - self.median) / self.scale, -self.clip, self.clip)
        z[missing] = 0.0
        extra = [missing[:, self.features.index(f)][:, None].astype(np.float64) for f in self.missing_indicators]
        return np.concatenate([z] + extra, axis=1).astype(np.float32)


def build_windows(segments: list[tuple[int, int]], seq_len: int, stride: int, phase: int = 0) -> list[tuple[int, int]]:
    """Window starts/lengths inside segments. Returns (start, length<=seq_len)."""
    out = []
    for a, b in segments:
        n = b - a
        if n <= seq_len:
            out.append((a, n))
            continue
        starts = list(range(a + (phase % stride), b - seq_len + 1, stride))
        if not starts or starts[0] != a:
            starts.insert(0, a)
        if starts[-1] != b - seq_len:
            starts.append(b - seq_len)
        out.extend((s, seq_len) for s in starts)
    return out


@dataclass
class PreparedWell:
    name: str
    x: np.ndarray        # [N, C] scaled
    y: np.ndarray        # [N]
    segments: list[tuple[int, int]]
    domain: int          # index among training wells (-1 for non-training wells)


def build_prepared(wells: dict[str, WellData], scaler: FeatureScaler, train_names: list[str]) -> dict[str, PreparedWell]:
    return {n: PreparedWell(n, scaler.transform(w), w.y, w.segments,
                            train_names.index(n) if n in train_names else -1)
            for n, w in wells.items()}


def build_window_tensor(pw: PreparedWell, start: int, length: int, seq_len: int):
    c = pw.x.shape[1]
    x = np.zeros((seq_len, c + 1), np.float32)
    y = np.full(seq_len, IGNORE_INDEX, np.int64)
    x[:length, :c] = pw.x[start:start + length]
    x[:length, c] = 1.0
    y[:length] = pw.y[start:start + length]
    return x, y


class TrainWindows(Dataset):
    """Training windows over the training wells; call `reshuffle(epoch)` each epoch
    to re-draw the window phase (random-offset augmentation)."""

    def __init__(self, wells: list[PreparedWell], seq_len: int, stride: int, seed: int = 0):
        self.wells, self.seq_len, self.stride = wells, seq_len, stride
        self.rng = np.random.default_rng(seed)
        self.index: list[tuple[int, int, int]] = []
        self.reshuffle(0)

    def reshuffle(self, epoch: int) -> None:
        phase = int(self.rng.integers(0, self.stride)) if epoch > 0 else 0
        self.index = [(i, s, n) for i, w in enumerate(self.wells)
                      for s, n in build_windows(w.segments, self.seq_len, self.stride, phase)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int):
        i, s, n = self.index[k]
        w = self.wells[i]
        x, y = build_window_tensor(w, s, n, self.seq_len)
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(w.domain, dtype=torch.long)
