"""Compute missing Well D 5-inner-fold validation-checkpointed baseline predictions
into _outputs/p04_v04_lightning/baseline_cache/ without touching the active worktree.

Usage:
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=_code python \
        _pipelines/01_run_cement_bond_benchmark/cache_missing_d_baselines_v04.py --study TimeMixer_plusVDL
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "_code"))

from da_reformer.experiment import (  # noqa: E402
    build_split,
    build_train_config,
    load_protocol,
    resolve_vdl_mode,
    run_fit_eval,
)
from da_reformer.metrics import compute_metrics  # noqa: E402
from da_reformer.train import run_predict_well  # noqa: E402

V04_OUT = Path(os.environ.get("BONDFUSION_V04_OUT", str(ROOT / "_pipelines/01_run_cement_bond_benchmark/outputs/v04")))
V03_HPO = ROOT / "_pipelines/01_run_cement_bond_benchmark/outputs/v03/hpo"
CACHE_DIR = ROOT / "_outputs/p04_v04_lightning/baseline_cache"

STUDY_TO_MODEL = {
    "TimeMixer_plusVDL": "TimeMixer +VDL",
    "TimesNet": "TimesNet",
    "TimesNet_plusVDL": "TimesNet +VDL",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", nargs="+", default=list(STUDY_TO_MODEL.keys()), choices=list(STUDY_TO_MODEL.keys()))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    protocol = load_protocol(HERE / "protocol_v03.yaml")
    protocol["_root"] = ROOT
    protocol["train_wells"] = ["A", "B", "C", "E", "F"]
    protocol["test_wells"] = ["D"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    for study in args.study:
        out_sub = CACHE_DIR / study
        out_sub.mkdir(parents=True, exist_ok=True)
        out_prob = out_sub / "probs_D_cv.npy"
        wt_prob = V04_OUT / "deep" / "D" / study / "probs_D_cv.npy"
        if wt_prob.exists():
            pr = np.load(wt_prob)
            np.save(out_prob, pr)
            print(f"[{study}] reused worktree {wt_prob}", flush=True)
            continue
        if out_prob.exists():
            print(f"[{study}] already cached at {out_prob}", flush=True)
            continue

        model_name = STUDY_TO_MODEL[study]
        hp = json.loads((V03_HPO / f"{study}.best.json").read_text())["hp"]
        vdl_mode = resolve_vdl_mode(model_name)
        train_wells = ["A", "B", "C", "E", "F"]
        cv_probs = []
        best_epochs = []
        for val_w in train_wells:
            inner_tr = [w for w in train_wells if w != val_w]
            r = run_fit_eval(
                model_name,
                hp,
                protocol,
                inner_tr,
                [val_w],
                device,
                seed=0,
                validate=True,
            )
            prepared, *_ = build_split(protocol, inner_tr, ["D"], vdl_mode)
            cfg = build_train_config(protocol, hp, 0, None)
            pr_d = run_predict_well(r["model"], prepared["D"], cfg.seq_len, cfg.infer_stride, device)
            cv_probs.append(pr_d)
            best_epochs.append(int(r["fit"]["best_epoch"]))
            del r

        pr = np.mean(cv_probs, axis=0).astype(np.float32)
        np.save(out_prob, pr)
        m = compute_metrics(prepared["D"].y, pr.argmax(1), probs=pr)
        (out_sub / "result_D.json").write_text(json.dumps({
            "study": study,
            "model": model_name,
            "best_epochs": best_epochs,
            "hp": hp,
            "macro_f1": m["macro_f1"],
            "f1_Good": m["f1_Good"],
            "roc_auc_good_vs_mod": m["roc_auc_good_vs_mod"],
        }, indent=2))
        print(
            f"[{study}] Well D computed (ep={best_epochs}): "
            f"MacroF1={m['macro_f1']:.4f} F1_Good={m['f1_Good']:.4f} AUC_GM={m['roc_auc_good_vs_mod']:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
