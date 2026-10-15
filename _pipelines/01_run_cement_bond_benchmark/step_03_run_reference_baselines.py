"""Step 03: non-sequential reference baselines on the blind wells (sanity anchors).

- CBL two-threshold rule (thresholds fitted on training wells by macro-F1);
- point-wise gradient boosting on the main inputs (GR, CBL, CCL);
- the same with AC + AC-missing indicator, and the AC-missing indicator alone,
  to quantify the AC-missingness shortcut that motivated excluding AC;
- protocols with a `vdl` section: point-wise gradient boosting on GR, CBL, CCL + the coarse
  VDL log-envelope blocks (the tabular counterpart of the "+VDL" sequence baselines).
Deep sequence models should be judged against these anchors.

Usage: python step_03_run_reference_baselines.py [--protocol protocol_v03.yaml]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "_code"))

from da_reformer.data import LABEL_MAP  # noqa: E402
from da_reformer.experiment import load_protocol  # noqa: E402
from da_reformer.metrics import compute_metrics  # noqa: E402
from da_reformer.vdl import build_log_envelope, load_vdl  # noqa: E402


def load_frame(protocol: dict, wells: list[str]) -> pd.DataFrame:
    dfs = [pd.read_csv(ROOT / protocol["raw_dir"] / f"{w}.csv").assign(well=w) for w in wells]
    df = pd.concat(dfs, ignore_index=True)
    df["y"] = df["label_raw"].map(LABEL_MAP)
    for c in ["GR", "AC", "CBL", "CCL"]:
        df.loc[df[c] <= -999, c] = np.nan
    df["AC_missing"] = df["AC"].isna().astype(float)
    if "vdl" in protocol:
        v = protocol["vdl"]
        env = np.concatenate([build_log_envelope(load_vdl(ROOT / protocol["raw_dir"], w, v["n_keep"]), v["block"])
                              for w in wells])
        df = pd.concat([df, pd.DataFrame(env, columns=[f"VDL_block_{k}" for k in range(env.shape[1])])], axis=1)
    return df


def compute_macro_f1_fast(y: np.ndarray, p: np.ndarray, n_classes: int = 3) -> float:
    cm = np.bincount(y * n_classes + p, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def compute_threshold_rule(tr: pd.DataFrame) -> tuple[float, float]:
    """Poor if CBL >= t1, Moderate if t2 <= CBL < t1, Good otherwise; grid = CBL quantiles."""
    cbl, y = tr["CBL"].to_numpy(), tr["y"].to_numpy()
    grid = np.unique(np.quantile(cbl, np.linspace(0.01, 0.99, 197)))
    best = (-1.0, float("nan"), float("nan"))
    for t1 in grid:
        for t2 in grid[grid < t1]:
            f = compute_macro_f1_fast(y, np.where(cbl >= t1, 0, np.where(cbl >= t2, 1, 2)))
            if f > best[0]:
                best = (f, float(t1), float(t2))
    return best[1], best[2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", type=Path, default=HERE / "protocol_v03.yaml")
    args = ap.parse_args(argv)
    protocol = load_protocol(args.protocol)
    tr = load_frame(protocol, protocol["train_wells"])
    te = load_frame(protocol, protocol["test_wells"])
    res = {}
    t1, t2 = compute_threshold_rule(tr)
    pred = np.where(te["CBL"] >= t1, 0, np.where(te["CBL"] >= t2, 1, 2))
    res["CBL two-threshold rule"] = {"thresholds": [t1, t2], **compute_metrics(te["y"].to_numpy(), pred)}
    sets = {"HGB point-wise (GR,CBL,CCL)": ["GR", "CBL", "CCL"],
            "HGB point-wise (GR,AC,CBL,CCL + AC-missing)": ["GR", "AC", "CBL", "CCL", "AC_missing"],
            "HGB AC-missing indicator only": ["AC_missing"]}
    if "vdl" in protocol:
        sets["HGB point-wise (GR,CBL,CCL + VDL blocks)"] = ["GR", "CBL", "CCL"] + [c for c in tr if c.startswith("VDL_block_")]
    for name, cols in sets.items():
        m = HistGradientBoostingClassifier(max_iter=300, random_state=0, class_weight="balanced")
        m.fit(tr[cols].to_numpy(), tr["y"].to_numpy())
        res[name] = compute_metrics(te["y"].to_numpy(), m.predict(te[cols].to_numpy()))
    out = HERE / "outputs" / protocol["version"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "reference_baselines.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in res.items():
        print(f"{k:48s} acc={v['accuracy']:.3f} macroF1={v['macro_f1']:.3f} kappa={v['kappa']:.3f} "
              f"F1(P/M/G)={v['f1_Poor']:.3f}/{v['f1_Moderate']:.3f}/{v['f1_Good']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
