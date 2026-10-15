"""Step 09 (protocol v03+, run before step_04): design screen on training-well CV only.

Every candidate in protocol `screen.candidates` is trained with the same fixed
hyper-parameters (`screen.hp_from`) under leave-one-well-out CV on the training wells,
once per screen seed. Test wells are never loaded. A candidate may override `transforms`.

Outputs: outputs/<version>/screen/<id>_seed<k>.json (resumable: existing files are skipped)
and, with --summarize, screen_summary.csv plus the decision-rule quantities printed.

Usage:
    python step_09_run_design_screen.py --protocol protocol_v03.yaml [--candidate S4 ...] [--device cuda:0]
    python step_09_run_design_screen.py --protocol protocol_v03.yaml --summarize
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "_code"))

from da_reformer.experiment import load_protocol, run_cv  # noqa: E402


def run_candidates(protocol: dict, ids: list[str] | None, seeds: list[int], device, out: Path) -> None:
    hp = json.loads((HERE / protocol["screen"]["hp_from"]).read_text(encoding="utf-8"))["hp"]
    for cand in protocol["screen"]["candidates"]:
        if ids and cand["id"] not in ids:
            continue
        p = copy.deepcopy(protocol)
        if "transforms" in cand:
            p["transforms"] = cand["transforms"]
        for seed in seeds:
            path = out / f"{cand['id']}_seed{seed}.json"
            if path.exists():
                print(f"{cand['id']} seed {seed}: exists, skipped")
                continue
            cv = run_cv(cand["model"], hp, p, device, seed=seed)
            path.write_text(json.dumps({**cand, "seed": seed, "hp": hp, **cv}, ensure_ascii=False, indent=2),
                            encoding="utf-8")
            print(f"{cand['id']:10s} {cand['model']:32s} seed {seed}: CV macro-F1 {cv['mean_macro_f1']:.4f} "
                  + " ".join(f"{f['val_well']}:{f['macro_f1']:.3f}/G{f['f1_Good']:.2f}" for f in cv["folds"]), flush=True)


def run_summary(protocol: dict, out: Path) -> pd.DataFrame:
    rows = []
    for cand in protocol["screen"]["candidates"]:
        runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob(f"{cand['id']}_seed*.json"))]
        if not runs:
            continue
        scores = [r["mean_macro_f1"] for r in runs]
        row = {"id": cand["id"], "model": cand["model"], "n_seeds": len(runs),
               "cv_macro_f1_mean": float(np.mean(scores)), "cv_macro_f1_std": float(np.std(scores, ddof=1)) if len(runs) > 1 else np.nan}
        for c in ("Poor", "Moderate", "Good"):
            row[f"cv_f1_{c}_mean"] = float(np.mean([np.mean([f[f"f1_{c}"] for f in r["folds"]]) for r in runs]))
        for w in protocol["train_wells"]:
            row[f"fold_{w}_macro_f1"] = float(np.mean([f["macro_f1"] for r in runs for f in r["folds"] if f["val_well"] == w]))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "screen_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.round(4).to_string(index=False))
    if len(df):
        noise = float(np.sqrt(np.nanmean(df["cv_macro_f1_std"] ** 2)))
        print(f"pooled noise std (over seeds) = {noise:.4f}")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", type=Path, default=HERE / "protocol_v03.yaml")
    ap.add_argument("--candidate", action="append", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args(argv)
    protocol = load_protocol(args.protocol)
    out = HERE / "outputs" / protocol["version"] / "screen"
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        run_summary(protocol, out)
        return 0
    torch.set_num_threads(4)
    seeds = args.seeds if args.seeds is not None else protocol["screen"]["seeds"]
    run_candidates(protocol, args.candidate, seeds, torch.device(args.device), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
