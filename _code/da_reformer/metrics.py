"""Classification metrics used throughout the paper."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_fscore_support)

from .data import CLASS_NAMES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> dict:
    labels = list(range(n_classes))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
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
    return out
