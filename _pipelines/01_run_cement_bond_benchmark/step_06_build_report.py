"""Step 06: aggregate final multi-seed results into paper tables and plotting inputs.

Writes to outputs/<version>/report/:
  summary_merged.csv     mean/std over seeds on the merged blind set (E+F), plus ens_* =
                         metrics of the seed ensemble (probabilities averaged over seeds)
  summary_per_well.csv   same per blind well
  hpo_summary.csv        best CV macro-F1, trial counts and tuned hyper-parameters
  predictions/<safe model name> <well>.csv   point-wise predictions of the seed whose merged
                         macro-F1 is closest to the model's mean (input for figures)
  _artifact_manifest.json  sha256/size of untracked artefacts (probs_*.npy, predictions/*.csv)

Models listed under protocol `reuse` are read from the final runs of `reuse.from_version`.

Usage: python step_06_build_report.py [--protocol protocol_v03.yaml]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "_code"))

from da_reformer.data import CLASS_NAMES, load_well  # noqa: E402
from da_reformer.experiment import build_safe_name, load_protocol, resolve_final_dir  # noqa: E402
from da_reformer.metrics import compute_metrics  # noqa: E402

KEYS = ["accuracy", "macro_f1", "kappa", "mcc"] + [f"{m}_{c}" for c in CLASS_NAMES for m in ("f1", "precision", "recall")]


def load_runs(model_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(model_dir.glob("seed*/metrics.json"))]


def build_summary(runs: list[dict], getter) -> dict:
    vals = {k: [getter(r)[k] for r in runs] for k in KEYS}
    out = {"n_seeds": len(runs)}
    for k, v in vals.items():
        out[f"{k}_mean"], out[f"{k}_std"] = float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", type=Path, default=HERE / "protocol_v03.yaml")
    args = ap.parse_args(argv)
    protocol = load_protocol(args.protocol)
    base = HERE / "outputs" / protocol["version"]
    final_dir, out = base / "final", base / "report"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    m = protocol["models"]
    models = ([m["proposed"]] + m["ablations"] + m.get("plugins", []) + m.get("vdl_baselines", [])
              + m["baselines"])
    wells = {w: load_well(ROOT / protocol["raw_dir"], w, ["GR", "CBL", "CCL"], protocol["gap_m"]) for w in protocol["test_wells"]}
    merged_rows, well_rows = [], []
    for name in models:
        model_dir = resolve_final_dir(HERE / "outputs", protocol, name)
        runs = load_runs(model_dir)
        if not runs:
            print(f"{name}: no final runs yet")
            continue
        row = {"model": name, **build_summary(runs, lambda r: r["merged"])}
        # seed ensemble: average the probabilities of all seeds, then evaluate once
        seed_dirs = [model_dir / f"seed{r['seed']}" for r in runs]
        ens_pred = np.concatenate([np.mean([np.load(d / f"probs_{w}.npy") for d in seed_dirs], 0).argmax(1) for w in wells])
        ens = compute_metrics(np.concatenate([wells[w].y for w in wells]), ens_pred)
        row.update({f"ens_{k}": ens[k] for k in KEYS})
        merged_rows.append(row)
        for w in protocol["test_wells"]:
            well_rows.append({"model": name, "well": w, **build_summary(runs, lambda r, w=w: r["per_well"][w])})
        mean_f1 = np.mean([r["merged"]["macro_f1"] for r in runs])
        rep = min(runs, key=lambda r: abs(r["merged"]["macro_f1"] - mean_f1))
        rep_dir = model_dir / f"seed{rep['seed']}"
        for w, wd in wells.items():
            prob = np.load(rep_dir / f"probs_{w}.npy")
            pd.DataFrame({"DEPTH": wd.depth, "true_class": [CLASS_NAMES[k] for k in wd.y],
                          "pred_class": [CLASS_NAMES[k] for k in prob.argmax(1)],
                          **{f"prob_{c}": prob[:, k] for k, c in enumerate(CLASS_NAMES)},
                          "input_GR": wd.x_raw[:, 0], "input_CBL": wd.x_raw[:, 1], "input_CCL": wd.x_raw[:, 2],
                          "seed": rep["seed"], "protocol_version": rep["protocol_version"]}).to_csv(out / "predictions" / f"{build_safe_name(name)} {w}.csv", index=False)
    pd.DataFrame(merged_rows).to_csv(out / "summary_merged.csv", index=False)
    pd.DataFrame(well_rows).to_csv(out / "summary_per_well.csv", index=False)
    hpo_rows = []
    for p in sorted((base / "hpo").glob("*.best.json")):
        b = json.loads(p.read_text(encoding="utf-8"))
        hpo_rows.append({"model": b["model"], "cv_macro_f1": b["value"], "best_trial": b["trial"], "n_trials": b["n_trials"],
                         "stop_epoch": b["stop_epoch"], "hp": json.dumps(b["hp"], ensure_ascii=False)})
    pd.DataFrame(hpo_rows).to_csv(out / "hpo_summary.csv", index=False)
    manifest = {str(p.relative_to(base)): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                for p in sorted([*final_dir.glob("*/seed*/probs_*.npy"), *(out / "predictions").glob("*.csv")])}
    (out / "_artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    cols = ["accuracy", "macro_f1", "kappa", "mcc", "f1_Poor", "f1_Moderate", "f1_Good"]
    print(f"{'model':32s} " + " ".join(f"{c:>15s}" for c in cols) + "   ens_macro_f1 ens_kappa")
    for r in merged_rows:
        print(f"{r['model']:32s} " + " ".join(f"{r[c + '_mean']:.3f}±{r[c + '_std']:.3f}".rjust(15) for c in cols)
              + f"   {r['ens_macro_f1']:.3f}        {r['ens_kappa']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
