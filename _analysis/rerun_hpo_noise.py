"""Is the HPO winner distinguishable from the runners-up? (decision space of an HPO "agent")

Re-evaluates the frozen top-k completed Optuna configurations (protocol v03, A-D leave-one-well-out
CV only, no test wells) with several fresh training seeds and compares the spread between configs
with the spread between seeds. If the top configs are indistinguishable, any smarter
hyper-parameter selector (Bayesian, LLM agent, ...) has little room to improve the outcome.

Usage: python _analysis/rerun_hpo_noise.py --model "VDL-ModernTCN (mean pool)" [--top 5] [--seeds 1 2 3] [--device cuda]
Output: _outputs/hpo_noise/<safe model>.json (one record per config x seed), summary printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PIPE = ROOT / "_pipelines" / "01_run_cement_bond_benchmark"
sys.path.insert(0, str(ROOT / "_code"))

from da_reformer.experiment import build_safe_name, load_protocol, run_cv  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-root", type=Path, default=ROOT,
                    help="directory containing authorized raw data")
    ap.add_argument("--configs", type=Path, default=ROOT / "results/support/hpo_noise_configs.json")
    ap.add_argument("--out", type=Path, default=ROOT / "_outputs/hpo_noise")
    args = ap.parse_args(argv)
    protocol = load_protocol(PIPE / "protocol_v03.yaml")
    safe = build_safe_name(args.model)
    if args.model != "VDL-ModernTCN (mean pool)":
        ap.error("released top-five configs belong to VDL-ModernTCN (mean pool)")
    protocol["_root"] = args.data_root.resolve()
    configs = json.loads(args.configs.read_text())
    done = [SimpleNamespace(number=r["trial"], value=r["hpo_value"], user_attrs={"hp": r["hp"]})
            for r in sorted(configs, key=lambda r: -r["hpo_value"])]
    out_path = args.out / f"{safe}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(out_path.read_text()) if out_path.exists() else []
    seen = {(r["trial"], r["seed"]) for r in records}
    torch.set_num_threads(4)
    for t in done[:args.top]:
        for seed in args.seeds:
            if (t.number, seed) in seen:
                continue
            cv = run_cv(args.model, t.user_attrs["hp"], protocol, torch.device(args.device), seed=seed)
            records.append({"trial": t.number, "hpo_value": t.value, "seed": seed, "cv_macro_f1": cv["mean_macro_f1"],
                            "folds": cv["folds"]})
            out_path.write_text(json.dumps(records, indent=1))
            print(f"trial {t.number} (HPO {t.value:.4f}) seed {seed}: {cv['mean_macro_f1']:.4f}", flush=True)
    by = {}
    for r in records:
        by.setdefault(r["trial"], []).append(r["cv_macro_f1"])
    means = {k: float(np.mean(v)) for k, v in by.items()}
    seed_sd = float(np.sqrt(np.mean([np.var(v, ddof=1) for v in by.values() if len(v) > 1])))
    print("trial  HPO-value  re-eval mean ± sd")
    for t in done[:args.top]:
        v = by.get(t.number, [])
        print(f"{t.number:5d}  {t.value:.4f}     {np.mean(v):.4f} ± {np.std(v, ddof=1) if len(v) > 1 else float('nan'):.4f}")
    print(f"between-config sd of re-eval means = {np.std(list(means.values()), ddof=1):.4f}; pooled seed sd = {seed_sd:.4f}")
    print(f"winner's curse (HPO best value - its re-eval mean) = {done[0].value - means[done[0].number]:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
