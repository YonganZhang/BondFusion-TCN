"""Step 05: train on all training wells with the tuned hyper-parameters (several seeds)
and evaluate once on the blind wells. Ablations reuse another model's tuned
hyper-parameters via --hp-from (only modules are switched off).

Outputs per seed: outputs/<version>/final/<model>/seed<k>/{metrics.json, probs_<well>.npy, model.pt}.

Usage:
    python step_05_run_final.py --model Transformer [--seeds 0 1 2 3 4]
    python step_05_run_final.py --model "VDL-ModernTCN w/o VDL" --hp-from "VDL-ModernTCN (mean pool)"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "_code"))

from da_reformer.experiment import build_safe_name, load_protocol, run_fit_eval  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", type=Path, default=HERE / "protocol_v03.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--hp-from", default=None, help="take tuned hyper-parameters from this model's study")
    ap.add_argument("--hpo-dir", type=Path, default=None, help="defaults to outputs/<version>/hpo of --protocol")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    protocol = load_protocol(args.protocol)
    base = HERE / "outputs" / protocol["version"]
    hpo_dir = args.hpo_dir or base / "hpo"
    src = args.hp_from or args.model
    best = json.loads((hpo_dir / f"{build_safe_name(src)}.best.json").read_text(encoding="utf-8"))
    seeds = args.seeds if args.seeds is not None else protocol["final"]["seeds"]
    torch.set_num_threads(4)
    for seed in seeds:
        out = base / "final" / build_safe_name(args.model) / f"seed{seed}"
        if (out / "metrics.json").exists():
            print(f"{args.model} seed {seed}: exists, skipped")
            continue
        r = run_fit_eval(args.model, best["hp"], protocol, list(protocol["train_wells"]), list(protocol["test_wells"]),
                         torch.device(args.device), seed, validate=False, stop_epoch=best["stop_epoch"])
        out.mkdir(parents=True, exist_ok=True)
        for w, p in r["probs"].items():
            np.save(out / f"probs_{w}.npy", p.astype(np.float32))
        torch.save(r["model"].state_dict(), out / "model.pt")
        (out / "metrics.json").write_text(json.dumps({
            "model": args.model, "hp_from": src, "hp": best["hp"], "stop_epoch": best["stop_epoch"], "seed": seed,
            "protocol_version": protocol["version"], "n_params": r["n_params"], "seconds": r["fit"]["seconds"],
            "merged": r["metrics"], "per_well": r["per_well"], "history": r["fit"]["history"]},
            ensure_ascii=False, indent=2), encoding="utf-8")
        m = r["metrics"]
        print(f"{args.model} seed {seed}: acc {m['accuracy']:.3f} macroF1 {m['macro_f1']:.3f} kappa {m['kappa']:.3f} "
              f"F1 P/M/G {m['f1_Poor']:.3f}/{m['f1_Moderate']:.3f}/{m['f1_Good']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
