"""Step 04: Optuna search (equal budget) for one or more models on leave-one-well-out CV.

Resumable: the study is stored in outputs/<version>/hpo/<model>.db; rerunning the
same command continues until n_trials trials exist. Writes <model>.best.json.

Usage:
    python step_04_run_hpo.py --model Transformer [--model LSTM ...] [--n-trials 30] [--device cuda]
    python step_04_run_hpo.py --model all          # every tuned model of the protocol: baselines not
                                                   # reused from an earlier version, "+VDL" baselines, proposed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "_code"))

from da_reformer.experiment import load_protocol  # noqa: E402
from da_reformer.hpo import run_study  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", type=Path, default=HERE / "protocol_v03.yaml")
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    protocol = load_protocol(args.protocol)
    models = args.model
    if models == ["all"]:
        reused = set((protocol.get("reuse") or {}).get("models", []))
        m = protocol["models"]
        models = [n for n in m["baselines"] if n not in reused] + m.get("vdl_baselines", []) + [m["proposed"]]
    out = HERE / "outputs" / protocol["version"] / "hpo"
    torch.set_num_threads(4)
    for name in models:
        study = run_study(name, protocol, out, torch.device(args.device), args.n_trials)
        states = {}
        for t in study.trials:
            states[t.state.name] = states.get(t.state.name, 0) + 1
        print(f"{name}: best CV macro-F1 {study.best_value:.4f} (trial {study.best_trial.number}); trials {states}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
