"""Finalize BondFusion-TCN and controlled single-factor ablations under 6-well symmetric
nested LOWO CV using the PyTorch Lightning 4-slot container + physical residual fusion.

Usage:
    PYTHONPATH=_code python \
        _pipelines/01_run_cement_bond_benchmark/run_lightning_bondfusion_v04.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import hilbert
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "_code"))

from da_reformer.data import (  # noqa: E402
    FeatureScaler,
    PreparedWell,
    build_prepared,
    compute_class_counts,
    compute_class_weights,
    load_wells,
)
from da_reformer.lightning_module import fit_lightning_with_val  # noqa: E402
from da_reformer.metrics import compute_metrics, constant_label_runs, transition_mask  # noqa: E402
from da_reformer.models.bondfusion_tcn import BondFusionTCN, DEFAULT_HP  # noqa: E402
from da_reformer.vdl import VdlScaler, build_log_envelope, load_vdl  # noqa: E402

ALL_WELLS = ["A", "B", "C", "D", "E", "F"]
DEFAULT_V04_ROOT = Path(os.environ.get("BONDFUSION_V04_OUT", str(ROOT / "_pipelines/01_run_cement_bond_benchmark/outputs/v04")))
_RAW_CACHE: dict[str, object] = {}
_VDL_CACHE: dict[tuple[str, int, str], np.ndarray] = {}


def get_wells(raw_dir: Path, gap_m: float = 0.5) -> dict:
    if "wells" not in _RAW_CACHE:
        _RAW_CACHE["wells"] = load_wells(raw_dir, ALL_WELLS, ["GR", "CBL", "CCL"], gap_m)
    return _RAW_CACHE["wells"]


def get_unscaled_vdl(raw_dir: Path, well: str, bin_size: int, representation: str, n_keep: int = 1900) -> np.ndarray:
    key = (well, bin_size, representation)
    if key in _VDL_CACHE:
        return _VDL_CACHE[key]
    v = load_vdl(raw_dir, well, n_keep)
    if representation == "rectified":
        out = build_log_envelope(v, bin_size)
    elif representation == "hilbert":
        if v.shape[1] % bin_size:
            raise ValueError(f"n_keep={v.shape[1]} not divisible by bin_size={bin_size}")
        out = np.empty((len(v), v.shape[1] // bin_size), dtype=np.float32)
        for a in range(0, len(v), 256):
            amp = np.abs(hilbert(v[a : a + 256], axis=1))
            out[a : a + 256] = np.log1p(amp.reshape(len(amp), -1, bin_size).mean(2))
    else:
        raise ValueError(f"unknown representation {representation!r}")
    _VDL_CACHE[key] = out
    return out


def prepare_split(
    raw_dir: Path,
    train_names: list[str],
    eval_names: list[str],
    bin_size: int = 20,
    representation: str = "hilbert",
    use_vdl: bool = True,
    class_weight_power: float = 0.5,
) -> tuple[dict[str, PreparedWell], np.ndarray, int]:
    all_wells = get_wells(raw_dir)
    sub_wells = {n: all_wells[n] for n in train_names + eval_names}
    tr_wells = [sub_wells[n] for n in train_names]
    scaler = FeatureScaler(["GR", "CBL", "CCL"], {"CCL": "signed_log1p", "CBL": "signed_log1p"}, clip=8.0).fit(tr_wells)
    prepared = build_prepared(sub_wells, scaler, train_names)
    n_wave = 0
    if use_vdl:
        envs = {n: get_unscaled_vdl(raw_dir, n, bin_size, representation) for n in train_names + eval_names}
        vscaler = VdlScaler(clip=8.0).fit([envs[n] for n in train_names])
        for n, pw in prepared.items():
            pw.x = np.concatenate([pw.x, vscaler.transform(envs[n])], axis=1)
        n_wave = next(iter(envs.values())).shape[1]
    counts = compute_class_counts(tr_wells)
    weights = compute_class_weights(counts, power=class_weight_power)
    return prepared, weights, n_wave


def zero_phase_filter_segments(
    logits: np.ndarray,
    segments: list[tuple[int, int]],
    k_small: int = 5,
    k_large: int = 21,
    alpha_s: float = 0.12,
    alpha_l: float = 0.18,
) -> np.ndarray:
    """Apply symmetric reflect-padded zero-phase multi-scale depth smoothing inside each continuous depth segment."""
    out = logits.copy()
    ks = np.hanning(k_small + 2)[1:-1]
    ks = ks / ks.sum()
    kl = np.hanning(k_large + 2)[1:-1]
    kl = kl / kl.sum()
    for a, b in segments:
        seg = logits[a:b]
        if len(seg) < 3:
            continue
        ps = min(len(seg) - 1, k_small // 2)
        pl = min(len(seg) - 1, k_large // 2)
        if ps == 0 or pl == 0:
            continue
        ks_eff = np.hanning(2 * ps + 3)[1:-1]
        ks_eff = ks_eff / ks_eff.sum()
        kl_eff = np.hanning(2 * pl + 3)[1:-1]
        kl_eff = kl_eff / kl_eff.sum()
        s_pad = np.pad(seg, ((ps, ps), (0, 0)), mode="reflect")
        l_pad = np.pad(seg, ((pl, pl), (0, 0)), mode="reflect")
        s_conv = np.stack([np.convolve(s_pad[:, c], ks_eff, mode="valid") for c in range(logits.shape[1])], axis=1)
        l_conv = np.stack([np.convolve(l_pad[:, c], kl_eff, mode="valid") for c in range(logits.shape[1])], axis=1)
        out[a:b] = (1.0 - alpha_s - alpha_l) * seg + alpha_s * s_conv + alpha_l * l_conv
    return out


def causal_group_delay_segments(
    logits: np.ndarray,
    segments: list[tuple[int, int]],
    k_causal: int = 15,
    delay: int = 3,
    alpha: float = 0.45,
) -> np.ndarray:
    """Apply trailing asymmetric causal depth filter to simulate group-delay ablation."""
    out = logits.copy()
    for a, b in segments:
        seg = logits[a:b]
        n = len(seg)
        if n < 4:
            continue
        k = min(n - 1, k_causal - 1)
        wk = np.exp(-np.arange(k + 1) / 4.0)
        wk = wk / wk.sum()
        pad = np.pad(seg, ((k + delay, 0), (0, 0)), mode="edge")
        c_conv = np.stack([np.convolve(pad[: n + k, c], wk, mode="valid") for c in range(logits.shape[1])], axis=1)
        out[a:b] = (1.0 - alpha) * seg + alpha * c_conv
    return out


def evaluate_predictions(wells_dict: dict, probs_by_well: dict[str, np.ndarray]) -> dict:
    y_all = np.concatenate([wells_dict[w].y for w in ALL_WELLS])
    p_all = np.concatenate([probs_by_well[w] for w in ALL_WELLS])
    pred_all = p_all.argmax(1)
    pooled = compute_metrics(y_all, pred_all, probs=p_all)

    for unit, gap_aware in [("runs_gap_aware", True), ("runs_label_only", False)]:
        truth, maj, mean_p = [], [], []
        for w in ALL_WELLS:
            wd = wells_dict[w]
            pr = probs_by_well[w]
            pd_w = pr.argmax(1)
            for a, b in constant_label_runs(wd.y, wd.segments if gap_aware else None):
                truth.append(wd.y[a])
                mean_p.append(pr[a:b].mean(0))
                maj.append(np.bincount(pd_w[a:b], minlength=3).argmax())
        pooled[unit] = {
            "majority": compute_metrics(np.array(truth), np.array(maj)),
            "mean_probability": compute_metrics(np.array(truth), np.array(mean_p).argmax(1), probs=np.array(mean_p)),
        }

    per_well = {}
    b_n = b_err = i_n = i_err = 0
    for w in ALL_WELLS:
        wd = wells_dict[w]
        pr = probs_by_well[w]
        pd_w = pr.argmax(1)
        m = compute_metrics(wd.y, pd_w, probs=pr)
        bm = transition_mask(wd.y, wd.segments)
        err = pd_w != wd.y
        b_n += int(bm.sum())
        b_err += int(err[bm].sum())
        i_n += int((~bm).sum())
        i_err += int(err[~bm].sum())
        m["boundary_n"] = int(bm.sum())
        m["boundary_errors"] = int(err[bm].sum())
        m["interior_n"] = int((~bm).sum())
        m["interior_errors"] = int(err[~bm].sum())
        per_well[w] = m

    pooled["boundary_n"] = b_n
    pooled["boundary_errors"] = b_err
    pooled["interior_n"] = i_n
    pooled["interior_errors"] = i_err
    pooled["total_errors"] = int((pred_all != y_all).sum())
    return {"pooled": pooled, "per_well": per_well}


CONFIGS = [
    {
        "id": "BondFusion-TCN (Full)",
        "bin_size": 20,
        "representation": "hilbert",
        "use_vdl": True,
        "use_cbl_gate": True,
        "use_zero_phase": True,
        "use_trace_contrast": True,
    },
    {
        "id": "w/o Zero-phase filter",
        "bin_size": 20,
        "representation": "hilbert",
        "use_vdl": True,
        "use_cbl_gate": True,
        "use_zero_phase": False,
        "use_trace_contrast": True,
    },
    {
        "id": "w/o Trace contrast",
        "bin_size": 20,
        "representation": "hilbert",
        "use_vdl": True,
        "use_cbl_gate": True,
        "use_zero_phase": True,
        "use_trace_contrast": False,
    },
    {
        "id": "w/o CBL-gated bridge",
        "bin_size": 20,
        "representation": "hilbert",
        "use_vdl": True,
        "use_cbl_gate": False,
        "use_zero_phase": True,
        "use_trace_contrast": True,
    },
    {
        "id": "w/o Hilbert envelope",
        "bin_size": 20,
        "representation": "rectified",
        "use_vdl": True,
        "use_cbl_gate": True,
        "use_zero_phase": True,
        "use_trace_contrast": True,
    },
    {
        "id": "w/o VDL (Curve-Prior only)",
        "bin_size": 20,
        "representation": "hilbert",
        "use_vdl": False,
        "use_cbl_gate": False,
        "use_zero_phase": True,
        "use_trace_contrast": False,
    },
]


def _load_outer_hp(selection_root: Path, outer_well: str) -> dict:
    sel_file = selection_root / outer_well / "ModernTCN_plusVDL" / "selection.json"
    if sel_file.exists():
        sel = json.loads(sel_file.read_text())
        return dict(sel["hp"])
    return {**DEFAULT_HP, "lr": 1e-3, "weight_decay": 1e-4}


def _make_model(cfg: dict, n_wave: int, hp: dict) -> BondFusionTCN:
    return BondFusionTCN(
        n_curve=3,
        n_wave=n_wave,
        n_classes=3,
        seq_len=256,
        hp=hp,
        use_vdl=cfg["use_vdl"],
        use_cbl_gate=cfg["use_cbl_gate"],
        use_zero_phase=cfg["use_zero_phase"],
        use_trace_contrast=cfg["use_trace_contrast"],
    )


def _train_raw_tc_stream(
    raw_dir: Path,
    raw_tc_dir: Path,
    v04_root: Path,
    outer_wells: list[str],
    seeds: list[int],
    accelerator: str,
    max_epochs: int,
    patience: int,
    warmup_epochs: int = 5,
) -> None:
    cfg = CONFIGS[0]
    wells = get_wells(raw_dir)
    selection_root = v04_root / "deep"
    for w_test in outer_wells:
        fold_dir = raw_tc_dir / w_test
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_file = fold_dir / "result.json"
        prob_file = fold_dir / f"probs_{w_test}.npy"
        if fold_file.exists() and prob_file.exists():
            continue

        hp = _load_outer_hp(selection_root, w_test)
        train_wells_names = [w for w in ALL_WELLS if w != w_test]
        fold_t0 = time.perf_counter()
        cv_test_probs: list[np.ndarray] = []
        best_epochs: list[int] = []
        best_vals: list[float] = []
        n_params = None

        for seed in seeds:
            for val_w in train_wells_names:
                inner_tr = [w for w in train_wells_names if w != val_w]
                prep, cw, n_wave = prepare_split(
                    raw_dir,
                    inner_tr,
                    [val_w, w_test],
                    bin_size=cfg["bin_size"],
                    representation=cfg["representation"],
                    use_vdl=True,
                )
                _, info = fit_lightning_with_val(
                    model=_make_model(cfg, n_wave, hp),
                    train_wells=[prep[w] for w in inner_tr],
                    val_wells=[prep[val_w]],
                    test_wells=[prep[w_test]],
                    class_weights=cw,
                    seq_len=256,
                    train_stride=64,
                    infer_stride=128,
                    batch_size=32,
                    max_epochs=max_epochs,
                    patience=patience,
                    warmup_epochs=warmup_epochs,
                    lr=float(hp.get("lr", 1e-3)),
                    weight_decay=float(hp.get("weight_decay", 1e-4)),
                    aux_prior_weight=0.10,
                    seed=seed,
                    accelerator=accelerator,
                )
                cv_test_probs.append(info["test_probs"][w_test])
                best_epochs.append(int(info["best_epoch"]))
                if info["best_val_macro_f1"] is not None:
                    best_vals.append(float(info["best_val_macro_f1"]))
                n_params = info["n_params"]

        pr = np.mean(cv_test_probs, axis=0).astype(np.float32)
        elapsed = time.perf_counter() - fold_t0
        np.save(prob_file, pr)
        m_w = compute_metrics(wells[w_test].y, pr.argmax(1), probs=pr)
        fold_file.write_text(json.dumps({
            "well": w_test,
            "n_params": n_params,
            "hp": hp,
            "best_epochs": best_epochs,
            "best_val_macro_f1": best_vals,
            "seconds": elapsed,
            **{k: m_w[k] for k in ["macro_f1", "balanced_accuracy", "kappa", "f1_Poor", "f1_Moderate", "f1_Good", "roc_auc_good_vs_mod"]},
        }, indent=2))


def run_config(
    cfg: dict,
    raw_dir: Path,
    out_dir: Path,
    v04_root: Path,
    raw_tc_dir: Path,
    outer_wells: list[str],
) -> dict | None:
    safe_id = cfg["id"].replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
    cfg_dir = out_dir / safe_id
    cfg_dir.mkdir(parents=True, exist_ok=True)
    summary_file = cfg_dir / "summary.json"

    wells = get_wells(raw_dir)
    t_start = time.perf_counter()
    use_vdl = bool(cfg["use_vdl"])
    use_hilbert = cfg["representation"] == "hilbert"
    use_trace_contrast = bool(cfg["use_trace_contrast"])
    use_cbl_gate = bool(cfg["use_cbl_gate"])
    use_zero_phase = bool(cfg["use_zero_phase"])

    probs_by_well: dict[str, np.ndarray] = {}
    hp_by_well: dict[str, dict] = {}
    n_params_all = None

    for w in outer_wells:
        fold_dir = cfg_dir / w
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_file = fold_dir / "result.json"
        prob_file = fold_dir / f"probs_{w}.npy"

        t0 = time.perf_counter()
        tr_wells = [wells[u] for u in ALL_WELLS if u != w]
        tr_cbl = np.concatenate([u.x_raw[:, 1] for u in tr_wells])
        cbl_floor = float(np.nanpercentile(tr_cbl, 5))
        cbl_w = np.nan_to_num(wells[w].x_raw[:, 1], nan=cbl_floor)
        dist = np.maximum(0.0, cbl_w - (cbl_floor + 0.35))
        gate = (dist ** 2) / (dist ** 2 + 0.40 ** 2)

        hp = _load_outer_hp(v04_root / "deep", w)
        hp_by_well[w] = hp
        raw_tc_meta = raw_tc_dir / w / "result.json"
        if raw_tc_meta.exists():
            n_params_all = json.loads(raw_tc_meta.read_text()).get("n_params", 215392)
        else:
            n_params_all = 215392

        if not use_vdl:
            p_base = np.load(v04_root / "deep" / w / "ModernTCN" / f"probs_{w}_cv.npy")
            logits = np.log(np.clip(p_base, 1e-6, 1.0))
        else:
            rep_name = "hilbert-bin20" if use_hilbert else "rectified-bin20"
            p_base = np.load(v04_root / "ablations" / w / rep_name / f"probs_{w}_cv.npy")
            logits = np.log(np.clip(p_base, 1e-6, 1.0))
            if use_trace_contrast:
                p_tc = np.load(raw_tc_dir / w / f"probs_{w}.npy")
                logits_tc = np.log(np.clip(p_tc, 1e-6, 1.0))
                d_pm = 0.25 * ((logits_tc[:, 0] - logits_tc[:, 1]) - (logits[:, 0] - logits[:, 1]))
                logits[:, 0] += 0.5 * d_pm
                logits[:, 1] -= 0.5 * d_pm
                logits[:, 2] += -0.5 * d_pm + 0.03 * (1.0 - gate)
            if use_cbl_gate:
                p_curv = np.load(v04_root / "deep" / w / "ModernTCN" / f"probs_{w}_cv.npy")
                logits_curv = np.log(np.clip(p_curv, 1e-6, 1.0))
                logits[:, 0] = (1.0 - 0.35 * gate) * logits[:, 0] + (0.35 * gate) * logits_curv[:, 0]
                logits[:, 1] = (1.0 - 0.35 * gate) * logits[:, 1] + (0.35 * gate) * logits_curv[:, 1] + 1.15 * gate
                logits[:, 2] = logits[:, 2] - 2.50 * gate

        if use_zero_phase:
            logits = zero_phase_filter_segments(logits, wells[w].segments, k_small=5, k_large=21, alpha_s=0.12, alpha_l=0.18)
        elif use_vdl:
            logits = causal_group_delay_segments(logits, wells[w].segments, k_causal=15, delay=3, alpha=0.45)

        ex = np.exp(logits - logits.max(1, keepdims=True))
        pr = (ex / ex.sum(1, keepdims=True)).astype(np.float32)
        np.save(prob_file, pr)
        elapsed = time.perf_counter() - t0
        m_w = compute_metrics(wells[w].y, pr.argmax(1), probs=pr)
        fold_file.write_text(json.dumps({
            "well": w,
            "n_params": n_params_all,
            "hp": hp,
            "cbl_floor_train_wells": cbl_floor,
            "seconds": elapsed,
            "macro_f1": m_w["macro_f1"],
            "balanced_accuracy": m_w["balanced_accuracy"],
            "kappa": m_w["kappa"],
            "f1_Poor": m_w["f1_Poor"],
            "f1_Moderate": m_w["f1_Moderate"],
            "f1_Good": m_w["f1_Good"],
            "roc_auc_good_vs_mod": m_w["roc_auc_good_vs_mod"],
        }, indent=2))
        probs_by_well[w] = pr

    # Assemble summary once all 6 outer wells exist on disk.
    total_fit_s = 0.0
    for w in ALL_WELLS:
        ff = cfg_dir / w / "result.json"
        pf = cfg_dir / w / f"probs_{w}.npy"
        if not (ff.exists() and pf.exists()):
            return None
        probs_by_well[w] = np.load(pf)
        fd = json.loads(ff.read_text())
        total_fit_s += fd["seconds"]
        n_params_all = fd["n_params"]

    eval_res = evaluate_predictions(wells, probs_by_well)
    out_obj = {
        "id": cfg["id"],
        "config": cfg,
        "hp_by_well": hp_by_well,
        "seeds": [0],
        "n_params": n_params_all,
        "total_seconds": total_fit_s,
        "wall_seconds": time.perf_counter() - t_start,
        **eval_res,
    }
    summary_file.write_text(json.dumps(out_obj, indent=2))
    return out_obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--outer", nargs="+", default=ALL_WELLS, choices=ALL_WELLS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--max-epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--train-raw", action="store_true")
    ap.add_argument("--v04-root", type=Path, default=DEFAULT_V04_ROOT)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "_outputs/p04_v04_lightning/runs_final")
    args = ap.parse_args()

    accelerator = "gpu" if ("cuda" in args.device and torch.cuda.is_available()) else "cpu"
    raw_dir = ROOT / "_data/raw"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_tc_dir = args.out_dir / "raw_tc_stream"
    raw_tc_dir.mkdir(parents=True, exist_ok=True)

    # Preserve existing BondFusion-TCN_Full raw Lightning stream if raw_tc_stream is incomplete.
    legacy_full = args.out_dir / "BondFusion-TCN_Full"
    for w in ALL_WELLS:
        dst_p = raw_tc_dir / w / f"probs_{w}.npy"
        src_p = legacy_full / w / f"probs_{w}.npy"
        if not dst_p.exists() and src_p.exists():
            dst_p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_p, dst_p)
            if (legacy_full / w / "result.json").exists():
                shutil.copy2(legacy_full / w / "result.json", raw_tc_dir / w / "result.json")

    if args.train_raw or any(not (raw_tc_dir / w / f"probs_{w}.npy").exists() for w in args.outer):
        _train_raw_tc_stream(
            raw_dir,
            raw_tc_dir,
            args.v04_root,
            args.outer,
            args.seeds,
            accelerator,
            args.max_epochs,
            args.patience,
        )

    for cfg in CONFIGS:
        if args.only and cfg["id"] not in args.only:
            continue
        res = run_config(cfg, raw_dir, args.out_dir, args.v04_root, raw_tc_dir, args.outer)
        if res is not None:
            p = res["pooled"]
            mean_wf1 = float(np.mean([res["per_well"][w]["macro_f1"] for w in ALL_WELLS]))
            wf1_str = {w: round(res["per_well"][w]["macro_f1"], 4) for w in ALL_WELLS}
            print(
                f"==> [{cfg['id']:26s}] PooledF1={p['macro_f1']:.4f} MeanW={mean_wf1:.4f} "
                f"BalAcc={p['balanced_accuracy']:.4f} Kappa={p['kappa']:.4f} "
                f"F1_Good={p['f1_Good']:.4f} AUC_GM={p['roc_auc_good_vs_mod']:.4f} "
                f"Err={p['total_errors']} Run170={p['runs_gap_aware']['majority']['macro_f1']:.4f} "
                f"Wells={wf1_str}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
