"""Step 02: data audit — counts, labels, missingness (incl. AC-missing vs label),
depth segments and train/test class distribution. Fails loudly on unexpected counts.

Usage: python step_02_validate_data.py [--protocol protocol_v03.yaml]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "_code"))

from da_reformer.data import CLASS_NAMES, LABEL_MAP, load_well  # noqa: E402
from da_reformer.experiment import load_protocol  # noqa: E402

EXPECTED_ROWS = {"A": 6031, "B": 4610, "C": 3994, "D": 6856, "E": 14573, "F": 19091}


def run_audit(protocol: dict) -> dict:
    raw = ROOT / protocol["raw_dir"]
    audit = {"wells": {}}
    for w, n_exp in EXPECTED_ROWS.items():
        df = pd.read_csv(raw / f"{w}.csv")
        if len(df) != n_exp:
            raise SystemExit(f"well {w}: {len(df)} rows, expected {n_exp}")
        y = df["label_raw"].map(LABEL_MAP)
        info = {"rows": len(df), "class_counts": {CLASS_NAMES[k]: int((y == k).sum()) for k in range(3)}}
        for c in ["GR", "AC", "CBL", "CCL"]:
            miss = df[c] <= -999
            info[f"missing_{c}"] = float(miss.mean())
        ac_miss = df["AC"] <= -999
        info["AC_missing_rate_by_class"] = {CLASS_NAMES[k]: float(ac_miss[y == k].mean()) if (y == k).any() else None
                                            for k in range(3)}
        wd = load_well(raw, w, protocol["features"], protocol["gap_m"])
        lens = np.array([b - a for a, b in wd.segments])
        info["segments"] = {"n": int(len(lens)), "min": int(lens.min()), "median": float(np.median(lens)),
                            "max": int(lens.max()), "share_points_in_segments_ge_seq_len":
                            float(lens[lens >= protocol["train"]["seq_len"]].sum() / lens.sum())}
        audit["wells"][w] = info
    for split in ("train_wells", "test_wells"):
        tot = np.sum([[audit["wells"][w]["class_counts"][c] for c in CLASS_NAMES] for w in protocol[split]], axis=0)
        audit[split] = {"rows": int(tot.sum()), **{c: {"count": int(v), "share": float(v / tot.sum())}
                                                   for c, v in zip(CLASS_NAMES, tot)}}
    return audit


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", type=Path, default=HERE / "protocol_v03.yaml")
    args = ap.parse_args(argv)
    protocol = load_protocol(args.protocol)
    audit = run_audit(protocol)
    out = HERE / "outputs" / protocol["version"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    for s in ("train_wells", "test_wells"):
        print(s, {c: f"{audit[s][c]['count']} ({audit[s][c]['share']:.1%})" for c in CLASS_NAMES}, "rows", audit[s]["rows"])
    for w, i in audit["wells"].items():
        print(w, "AC missing", f"{i['missing_AC']:.0%}", "by class", {k: (None if v is None else round(v, 2)) for k, v in i["AC_missing_rate_by_class"].items()},
              "segments", i["segments"]["n"])
    print("wrote", out / "data_audit.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
