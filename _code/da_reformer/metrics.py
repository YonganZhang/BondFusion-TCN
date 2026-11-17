"""Classification metrics used across sample-level and run-level evaluations."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_fscore_support, roc_auc_score)

from .data import CLASS_NAMES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3, probs: np.ndarray | None = None) -> dict:
    labels = list(range(n_classes))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(np.mean(r[s > 0])) if np.any(s > 0) else None,
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    for k, name in enumerate(CLASS_NAMES[:n_classes]):
        out[f"precision_{name}"] = float(p[k])
        out[f"recall_{name}"] = float(r[k])
        out[f"f1_{name}"] = float(f[k])
        out[f"support_{name}"] = int(s[k])
    mask = np.isin(y_true, [1, 2])
    out["roc_auc_good_vs_mod"] = None
    if probs is not None and len(np.unique(y_true[mask])) == 2:
        score = probs[mask, 2] / np.maximum(probs[mask, 1] + probs[mask, 2], 1e-12)
        out["roc_auc_good_vs_mod"] = float(roc_auc_score(y_true[mask] == 2, score))
    return out


def constant_label_runs(y, segments=None):
    """Half-open constant-label runs, optionally split at every depth segment boundary."""
    y = np.asarray(y)
    runs = []
    for a, b in (segments if segments is not None else [(0, len(y))]):
        if b <= a:
            continue
        cuts = np.r_[a, np.flatnonzero(y[a + 1:b] != y[a:b - 1]) + a + 1, b]
        runs.extend((int(i), int(j)) for i, j in zip(cuts[:-1], cuts[1:]))
    return runs


def compute_run_metrics(y, probs, segments=None):
    """Equal run weights; majority ties resolve to smallest class ID."""
    runs = constant_label_runs(y, segments)
    truth = np.array([y[a] for a, b in runs])
    mean_probs = np.array([probs[a:b].mean(0) for a, b in runs])
    majority = np.array([np.bincount(probs[a:b].argmax(1), minlength=3).argmax() for a, b in runs])
    return {
        "majority": compute_metrics(truth, majority),
        "mean_probability": compute_metrics(truth, mean_probs.argmax(1), probs=mean_probs),
    }


def transition_mask(y, segments, radius=2):
    """Mark +-radius row indices around the first sample carrying a new label; no gap crossing."""
    mask = np.zeros(len(y), dtype=bool)
    for a, b in segments:
        for i in np.flatnonzero(y[a + 1:b] != y[a:b - 1]) + a + 1:
            mask[max(a, i - radius):min(b, i + radius + 1)] = True
    return mask
