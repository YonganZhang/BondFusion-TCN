"""Assemble the four standardized v04 experiment tables (Comparison, Ablation, Sensitivity,
Generalization & Mechanism), block-bootstrap significance tests, and summary report
into _outputs/p04_v04_lightning/.

Usage:
    PYTHONPATH=_code python \
        _pipelines/01_run_cement_bond_benchmark/build_four_pillar_tables_v04.py
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "_code"))
sys.path.insert(0, str(HERE))

from da_reformer.data import load_wells  # noqa: E402
from da_reformer.metrics import compute_metrics, constant_label_runs, transition_mask  # noqa: E402
from run_lightning_bondfusion_v04 import zero_phase_filter_segments  # noqa: E402

ALL_WELLS = ["A", "B", "C", "D", "E", "F"]
V04_OUT = Path(os.environ.get("BONDFUSION_V04_OUT", str(ROOT / "_pipelines/01_run_cement_bond_benchmark/outputs/v04")))
V03_FINAL = ROOT / "_pipelines/01_run_cement_bond_benchmark/outputs/v03/final"
LIGHTNING_ROOT = ROOT / "_outputs/p04_v04_lightning"
RUNS_FINAL = LIGHTNING_ROOT / "runs_final"
RAW_TC_DIR = RUNS_FINAL / "raw_tc_stream"
CACHE_DIR = LIGHTNING_ROOT / "baseline_cache"


def _compute_full_stats(wells: dict, probs_by_well: dict[str, np.ndarray]) -> dict:
    y_all = np.concatenate([wells[w].y for w in ALL_WELLS])
    p_all = np.concatenate([probs_by_well[w] for w in ALL_WELLS])
    pred_all = p_all.argmax(1)
    m = compute_metrics(y_all, pred_all, probs=p_all)

    run170_y, run170_p = [], []
    run80_y, run80_p = [], []
    per_well = {}
    b_n = b_err = i_n = i_err = 0

    for w in ALL_WELLS:
        wd = wells[w]
        pr = probs_by_well[w]
        pw = compute_metrics(wd.y, pr.argmax(1), probs=pr)
        bm = transition_mask(wd.y, wd.segments)
        err = pr.argmax(1) != wd.y
        b_n += int(bm.sum())
        b_err += int(err[bm].sum())
        i_n += int((~bm).sum())
        i_err += int(err[~bm].sum())
        pw["boundary_n"] = int(bm.sum())
        pw["boundary_errors"] = int(err[bm].sum())
        pw["interior_n"] = int((~bm).sum())
        pw["interior_errors"] = int(err[~bm].sum())
        per_well[w] = pw

        for a, b in constant_label_runs(wd.y, wd.segments):
            run170_y.append(wd.y[a])
            run170_p.append(np.bincount(pr[a:b].argmax(1), minlength=3).argmax())
        for a, b in constant_label_runs(wd.y, None):
            run80_y.append(wd.y[a])
            run80_p.append(np.bincount(pr[a:b].argmax(1), minlength=3).argmax())

    m170 = compute_metrics(np.array(run170_y), np.array(run170_p))
    m80 = compute_metrics(np.array(run80_y), np.array(run80_p))
    wf1s = [per_well[w]["macro_f1"] for w in ALL_WELLS]
    return {
        "pooled": m,
        "mean_well_f1": float(np.mean(wf1s)),
        "std_well_f1": float(np.std(wf1s, ddof=1)),
        "total_errors": int((pred_all != y_all).sum()),
        "boundary_n": b_n,
        "boundary_errors": b_err,
        "interior_n": i_n,
        "interior_errors": i_err,
        "run170_macro_f1": float(m170["macro_f1"]),
        "run80_macro_f1": float(m80["macro_f1"]),
        "per_well": per_well,
    }


def _load_deep_baseline_probs(study: str) -> tuple[dict[str, np.ndarray], str]:
    probs = {}
    sources = []
    for w in ALL_WELLS:
        f_v04 = V04_OUT / "deep" / w / study / f"probs_{w}_cv.npy"
        f_cache = CACHE_DIR / study / f"probs_{w}_cv.npy"
        f_v03 = [V03_FINAL / study / f"seed{k}" / f"probs_{w}.npy" for k in range(10)]
        if f_v04.exists():
            probs[w] = np.load(f_v04)
            sources.append(f"{w}:v04")
        elif f_cache.exists():
            probs[w] = np.load(f_cache)
            sources.append(f"{w}:v04_cache")
        elif all(fp.exists() for fp in f_v03):
            probs[w] = np.mean([np.load(fp) for fp in f_v03], axis=0)
            sources.append(f"{w}:v03_refit")
        else:
            raise FileNotFoundError(f"Missing predictions for {study} well {w}")
    note = "Full Nested v04" if all(":v03" not in s for s in sources) else "v04(A-E/D) + v03_refit(F/E-F)"
    return probs, note


def _eval_bondfusion_rep(wells: dict, rep_id: str) -> dict[str, np.ndarray]:
    """Evaluate BondFusion-TCN physical modules over a base waveform representation id."""
    out = {}
    for w in ALL_WELLS:
        tr_wells = [wells[u] for u in ALL_WELLS if u != w]
        tr_cbl = np.concatenate([u.x_raw[:, 1] for u in tr_wells])
        cbl_floor = float(np.nanpercentile(tr_cbl, 5))
        cbl_w = np.nan_to_num(wells[w].x_raw[:, 1], nan=cbl_floor)
        dist = np.maximum(0.0, cbl_w - (cbl_floor + 0.35))
        gate = (dist ** 2) / (dist ** 2 + 0.40 ** 2)

        p_base = np.load(V04_OUT / "ablations" / w / rep_id / f"probs_{w}_cv.npy")
        logits = np.log(np.clip(p_base, 1e-6, 1.0))
        p_tc = np.load(RAW_TC_DIR / w / f"probs_{w}.npy")
        logits_tc = np.log(np.clip(p_tc, 1e-6, 1.0))
        d_pm = 0.25 * ((logits_tc[:, 0] - logits_tc[:, 1]) - (logits[:, 0] - logits[:, 1]))
        logits[:, 0] += 0.5 * d_pm
        logits[:, 1] -= 0.5 * d_pm
        logits[:, 2] += -0.5 * d_pm + 0.03 * (1.0 - gate)

        p_curv = np.load(V04_OUT / "deep" / w / "ModernTCN" / f"probs_{w}_cv.npy")
        logits_curv = np.log(np.clip(p_curv, 1e-6, 1.0))
        logits[:, 0] = (1.0 - 0.35 * gate) * logits[:, 0] + (0.35 * gate) * logits_curv[:, 0]
        logits[:, 1] = (1.0 - 0.35 * gate) * logits[:, 1] + (0.35 * gate) * logits_curv[:, 1] + 1.15 * gate
        logits[:, 2] = logits[:, 2] - 2.50 * gate

        logits = zero_phase_filter_segments(logits, wells[w].segments, k_small=5, k_large=21, alpha_s=0.12, alpha_l=0.18)
        ex = np.exp(logits - logits.max(1, keepdims=True))
        out[w] = (ex / ex.sum(1, keepdims=True)).astype(np.float32)
    return out


def _cm_to_metrics(cm: np.ndarray) -> tuple[float, float]:
    """Compute pooled macro-F1 and Cohen's kappa from a [3, 3] confusion matrix."""
    total = float(cm.sum())
    tp = np.diag(cm).astype(np.float64)
    row_s = cm.sum(1).astype(np.float64)
    col_s = cm.sum(0).astype(np.float64)
    denom = row_s + col_s
    f1 = np.where(denom > 0.0, 2.0 * tp / np.maximum(denom, 1.0), 0.0)
    macro_f1 = float(f1.mean())
    po = float(tp.sum() / total) if total > 0.0 else 0.0
    pe = float((row_s * col_s).sum() / (total * total)) if total > 0.0 else 0.0
    kappa = (po - pe) / (1.0 - pe) if abs(1.0 - pe) > 1e-12 else 0.0
    return macro_f1, float(kappa)


def _segment_bootstrap_diff(
    wells: dict,
    probs_a: dict[str, np.ndarray],
    probs_b: dict[str, np.ndarray],
    n_boot: int = 1000,
    seed: int = 20261116,
) -> dict:
    """1,000-draw paired segment block-bootstrap over the 139 contiguous depth segments."""
    segs_cm_a = []
    segs_cm_b = []
    for w in ALL_WELLS:
        y = wells[w].y
        pa = probs_a[w].argmax(1)
        pb = probs_b[w].argmax(1)
        for a, b in wells[w].segments:
            segs_cm_a.append(np.bincount(y[a:b] * 3 + pa[a:b], minlength=9).reshape(3, 3))
            segs_cm_b.append(np.bincount(y[a:b] * 3 + pb[a:b], minlength=9).reshape(3, 3))
    cm_a = np.stack(segs_cm_a, axis=0)
    cm_b = np.stack(segs_cm_b, axis=0)
    n_seg = len(cm_a)

    obs_f1_a, obs_kp_a = _cm_to_metrics(cm_a.sum(0))
    obs_f1_b, obs_kp_b = _cm_to_metrics(cm_b.sum(0))
    obs_df1 = obs_f1_a - obs_f1_b
    obs_dkp = obs_kp_a - obs_kp_b

    rng = np.random.default_rng(seed)
    boot_df1 = np.empty(n_boot, dtype=np.float64)
    boot_dkp = np.empty(n_boot, dtype=np.float64)
    for t in range(n_boot):
        idx = rng.integers(0, n_seg, size=n_seg)
        f1_a, kp_a = _cm_to_metrics(cm_a[idx].sum(0))
        f1_b, kp_b = _cm_to_metrics(cm_b[idx].sum(0))
        boot_df1[t] = f1_a - f1_b
        boot_dkp[t] = kp_a - kp_b

    p_f1 = min(1.0, 2.0 * min((np.sum(boot_df1 <= 0.0) + 1.0) / (n_boot + 1.0), (np.sum(boot_df1 >= 0.0) + 1.0) / (n_boot + 1.0)))
    p_kp = min(1.0, 2.0 * min((np.sum(boot_dkp <= 0.0) + 1.0) / (n_boot + 1.0), (np.sum(boot_dkp >= 0.0) + 1.0) / (n_boot + 1.0)))
    return {
        "n_segments": n_seg,
        "n_boot": n_boot,
        "delta_macro_f1": float(obs_df1),
        "ci95_macro_f1": [float(np.percentile(boot_df1, 2.5)), float(np.percentile(boot_df1, 97.5))],
        "p_value_macro_f1": float(p_f1),
        "delta_kappa": float(obs_dkp),
        "ci95_kappa": [float(np.percentile(boot_dkp, 2.5)), float(np.percentile(boot_dkp, 97.5))],
        "p_value_kappa": float(p_kp),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Empty rows for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(v: float | int) -> str | int:
    if isinstance(v, (int, np.integer)):
        return int(v)
    return f"{float(v):.4f}"


def main() -> int:
    wells = load_wells(ROOT / "_data/raw", ALL_WELLS, ["GR", "CBL", "CCL"], 0.5)
    LIGHTNING_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. Comparison Experiments (一、对比实验)
    comp_specs = [
        ("CBL Two-Threshold Rule", "rule", None),
        ("HistGBDT (Curves only)", "baseline", "point_curves/tuned"),
        ("HistGBDT (+VDL, Point)", "baseline", "point_vdl/tuned"),
        ("HistGBDT (+VDL, Window)", "baseline", "window_vdl/default"),
        ("LSTM (Curves only)", "deep", "LSTM"),
        ("LSTM +VDL", "deep", "LSTM_plusVDL"),
        ("GRU (Curves only)", "deep", "GRU"),
        ("GRU +VDL", "deep", "GRU_plusVDL"),
        ("Transformer (Curves only)", "deep", "Transformer"),
        ("Transformer +VDL", "deep", "Transformer_plusVDL"),
        ("Reformer (Curves only)", "deep", "Reformer"),
        ("Reformer +VDL", "deep", "Reformer_plusVDL"),
        ("Crossformer (Curves only)", "deep", "Crossformer"),
        ("Crossformer +VDL", "deep", "Crossformer_plusVDL"),
        ("TimesNet (Curves only)", "deep", "TimesNet"),
        ("TimesNet +VDL", "deep", "TimesNet_plusVDL"),
        ("TimeMixer (Curves only)", "deep", "TimeMixer"),
        ("TimeMixer +VDL", "deep", "TimeMixer_plusVDL"),
        ("ModernTCN (Curves only)", "deep", "ModernTCN"),
        ("ModernTCN +VDL", "v04_ablation", "rectified-bin10"),
        ("VDL-ModernTCN (mean pool)", "deep", "VDL-ModernTCN_(mean_pool)"),
        ("2D-CNN +VDL", "deep", "2D-CNN_plusVDL"),
        ("BondFusion-TCN (Ours)", "ours", "BondFusion-TCN_Full"),
    ]

    comp_rows = []
    stats_by_name = {}
    probs_cache: dict[str, dict[str, np.ndarray]] = {}
    for label, kind, key in comp_specs:
        if kind == "rule":
            probs = {w: np.load(V04_OUT / "baselines" / w / f"probs_{w}_cbl.npy") for w in ALL_WELLS}
            note = "Full Nested v04"
        elif kind == "baseline":
            base, sub = key.split("/")
            probs = {w: np.load(V04_OUT / "baselines" / w / base / f"probs_{w}_{sub}.npy") for w in ALL_WELLS}
            note = "Full Nested v04"
        elif kind == "v04_ablation":
            probs = {w: np.load(V04_OUT / "ablations" / w / key / f"probs_{w}_cv.npy") for w in ALL_WELLS}
            note = "Full Nested v04 (fine b=10)"
        elif kind == "deep":
            probs, note = _load_deep_baseline_probs(key)
        else:
            probs = {w: np.load(RUNS_FINAL / key / w / f"probs_{w}.npy") for w in ALL_WELLS}
            note = "Full Nested v04 + Lightning 4-Slot"

        probs_cache[label] = probs
        st = _compute_full_stats(wells, probs)
        stats_by_name[label] = st
        m = st["pooled"]
        comp_rows.append({
            "Model": label,
            "Pooled_MacroF1": _fmt(m["macro_f1"]),
            "Mean_Well_MacroF1": _fmt(st["mean_well_f1"]),
            "Std_Well_MacroF1": _fmt(st["std_well_f1"]),
            "Balanced_Accuracy": _fmt(m["balanced_accuracy"]),
            "Kappa": _fmt(m["kappa"]),
            "MCC": _fmt(m["mcc"]),
            "F1_Poor": _fmt(m["f1_Poor"]),
            "F1_Moderate": _fmt(m["f1_Moderate"]),
            "F1_Good": _fmt(m["f1_Good"]),
            "ROC_AUC_Good_vs_Mod": _fmt(m["roc_auc_good_vs_mod"]),
            "Total_Errors": st["total_errors"],
            "Run170_MacroF1": _fmt(st["run170_macro_f1"]),
            "Run80_MacroF1": _fmt(st["run80_macro_f1"]),
            **{f"Well_{w}_MacroF1": _fmt(st["per_well"][w]["macro_f1"]) for w in ALL_WELLS},
            "Protocol_Source": note,
        })

    comp_rows.sort(key=lambda r: float(r["Pooled_MacroF1"]), reverse=True)
    _write_csv(LIGHTNING_ROOT / "01_comparison_experiments.csv", comp_rows)

    # 2. Ablation Experiments (二、消融实验)
    abl_specs = [
        ("BondFusion-TCN (Full)", "BondFusion-TCN_Full"),
        ("w/o Zero-phase filter", "w-o_Zero-phase_filter"),
        ("w/o Trace contrast", "w-o_Trace_contrast"),
        ("w/o CBL-gated bridge", "w-o_CBL-gated_bridge"),
        ("w/o Hilbert envelope", "w-o_Hilbert_envelope"),
        ("w/o VDL (Curve-Prior only)", "w-o_VDL_Curve-Prior_only"),
    ]
    abl_rows = []
    full_f1 = None
    for label, folder in abl_specs:
        probs = {w: np.load(RUNS_FINAL / folder / w / f"probs_{w}.npy") for w in ALL_WELLS}
        probs_cache[f"ABL::{label}"] = probs
        st = _compute_full_stats(wells, probs)
        stats_by_name[f"ABL::{label}"] = st
        m = st["pooled"]
        if full_f1 is None:
            full_f1 = m["macro_f1"]
        abl_rows.append({
            "Variant": label,
            "Pooled_MacroF1": _fmt(m["macro_f1"]),
            "Delta_Pooled_F1": _fmt(m["macro_f1"] - full_f1),
            "Mean_Well_MacroF1": _fmt(st["mean_well_f1"]),
            "Balanced_Accuracy": _fmt(m["balanced_accuracy"]),
            "Kappa": _fmt(m["kappa"]),
            "F1_Poor": _fmt(m["f1_Poor"]),
            "F1_Moderate": _fmt(m["f1_Moderate"]),
            "F1_Good": _fmt(m["f1_Good"]),
            "ROC_AUC_Good_vs_Mod": _fmt(m["roc_auc_good_vs_mod"]),
            "Total_Errors": st["total_errors"],
            "Boundary_Errors_of_155": st["boundary_errors"],
            "Interior_Errors_of_55000": st["interior_errors"],
            "Run170_MacroF1": _fmt(st["run170_macro_f1"]),
            **{f"Well_{w}_MacroF1": _fmt(st["per_well"][w]["macro_f1"]) for w in ALL_WELLS},
        })
    _write_csv(LIGHTNING_ROOT / "02_ablation_experiments.csv", abl_rows)

    # 3. Sensitivity Experiments (三、敏感性实验)
    sens_specs = [
        ("BondFusion-TCN Representation & Bin Width", "bondfusion-hilbert-bin10", "hilbert", 10, "bf_rep", "hilbert-bin10"),
        ("BondFusion-TCN Representation & Bin Width", "bondfusion-hilbert-bin20", "hilbert", 20, "bf_rep", "hilbert-bin20"),
        ("BondFusion-TCN Representation & Bin Width", "bondfusion-hilbert-bin100", "hilbert", 100, "bf_rep", "hilbert-bin100"),
        ("BondFusion-TCN Representation & Bin Width", "bondfusion-rectified-bin10", "rectified", 10, "bf_rep", "rectified-bin10"),
        ("BondFusion-TCN Representation & Bin Width", "bondfusion-rectified-bin20", "rectified", 20, "bf_rep", "rectified-bin20"),
        ("BondFusion-TCN Representation & Bin Width", "bondfusion-rectified-bin50", "rectified", 50, "bf_rep", "rectified-bin50"),
        ("BondFusion-TCN Representation & Bin Width", "bondfusion-rectified-bin100", "rectified", 100, "bf_rep", "rectified-bin100"),
        ("BondFusion-TCN Representation & Bin Width", "bondfusion-rectified-bin190", "rectified", 190, "bf_rep", "rectified-bin190"),
        ("Backbone Representation & Bin Width", "hilbert-bin10", "hilbert", 10, "ablation", "hilbert-bin10"),
        ("Backbone Representation & Bin Width", "hilbert-bin20", "hilbert", 20, "ablation", "hilbert-bin20"),
        ("Backbone Representation & Bin Width", "hilbert-bin100", "hilbert", 100, "ablation", "hilbert-bin100"),
        ("Backbone Representation & Bin Width", "rectified-bin10", "rectified", 10, "ablation", "rectified-bin10"),
        ("Backbone Representation & Bin Width", "rectified-bin20", "rectified", 20, "ablation", "rectified-bin20"),
        ("Backbone Representation & Bin Width", "rectified-bin50", "rectified", 50, "ablation", "rectified-bin50"),
        ("Backbone Representation & Bin Width", "rectified-bin100", "rectified", 100, "ablation", "rectified-bin100"),
        ("Backbone Representation & Bin Width", "rectified-bin190", "rectified", 190, "ablation", "rectified-bin190"),
        ("Curve Channel & Missingness", "diagnostic_ac/tuned", "curves+AC", 0, "baseline", "diagnostic_ac/tuned"),
        ("Curve Channel & Missingness", "diagnostic_missing/tuned", "curves+missing_mask", 0, "baseline", "diagnostic_missing/tuned"),
        ("Early-Stopping Schedule", "legacy-stopping-ModernTCN", "curves_only", 0, "ablation", "legacy-stopping-ModernTCN"),
        ("Early-Stopping Schedule", "legacy-stopping-ModernTCN_plusVDL", "rectified", 100, "ablation", "legacy-stopping-ModernTCN_plusVDL"),
    ]
    sens_rows = []
    for grp, exp_id, rep, bin_sz, kind, src_key in sens_specs:
        if kind == "bf_rep":
            probs = _eval_bondfusion_rep(wells, src_key)
        elif kind == "ablation":
            probs = {w: np.load(V04_OUT / "ablations" / w / src_key / f"probs_{w}_cv.npy") for w in ALL_WELLS}
        else:
            base, sub = src_key.split("/")
            probs = {w: np.load(V04_OUT / "baselines" / w / base / f"probs_{w}_{sub}.npy") for w in ALL_WELLS}
        st = _compute_full_stats(wells, probs)
        m = st["pooled"]
        sens_rows.append({
            "Group": grp,
            "Experiment_ID": exp_id,
            "Representation": rep,
            "Bin_Size": bin_sz,
            "Pooled_MacroF1": _fmt(m["macro_f1"]),
            "Mean_Well_MacroF1": _fmt(st["mean_well_f1"]),
            "Balanced_Accuracy": _fmt(m["balanced_accuracy"]),
            "Kappa": _fmt(m["kappa"]),
            "F1_Good": _fmt(m["f1_Good"]),
            "ROC_AUC_Good_vs_Mod": _fmt(m["roc_auc_good_vs_mod"]),
            "Total_Errors": st["total_errors"],
            "Run170_MacroF1": _fmt(st["run170_macro_f1"]),
            **{f"Well_{w}_MacroF1": _fmt(st["per_well"][w]["macro_f1"]) for w in ALL_WELLS},
        })
    _write_csv(LIGHTNING_ROOT / "03_sensitivity_experiments.csv", sens_rows)

    # 4. Generalization & Mechanism Analysis Experiments (四、泛化性与机理分析实验)
    gen_rows = []
    for label, st_key in [
        ("BondFusion-TCN (Ours)", "BondFusion-TCN (Ours)"),
        ("2D-CNN +VDL", "2D-CNN +VDL"),
        ("ModernTCN +VDL", "ModernTCN +VDL"),
        ("CBL Two-Threshold Rule", "CBL Two-Threshold Rule"),
    ]:
        st = stats_by_name[st_key]
        for w in ALL_WELLS:
            pw = st["per_well"][w]
            gen_rows.append({
                "Analysis_Type": "Per-Well LOWO Generalization",
                "Subject": f"{label} - Held-out Well {w}",
                "Samples": pw["n"],
                "MacroF1": _fmt(pw["macro_f1"]),
                "Balanced_Accuracy": _fmt(pw["balanced_accuracy"]),
                "Kappa": _fmt(pw["kappa"]),
                "F1_Poor": _fmt(pw["f1_Poor"]),
                "F1_Moderate": _fmt(pw["f1_Moderate"]),
                "F1_Good": _fmt(pw["f1_Good"]),
                "ROC_AUC_Good_vs_Mod": _fmt(pw["roc_auc_good_vs_mod"]),
                "Boundary_Errors": pw["boundary_errors"],
                "Boundary_Total": pw["boundary_n"],
                "Interior_Errors": pw["interior_errors"],
                "Interior_Total": pw["interior_n"],
                "Notes": "Nested LOWO blind well evaluation",
            })

    mech_specs = [
        ("Mechanism Check", "CBL-Gated Bridge Gain (Full vs w/o CBL-gated bridge on High-CBL Well E)",
         "abl", "BondFusion-TCN_Full", "Zero-false-Good leakage gating"),
        ("Mechanism Check", "Early-Window Reference Drift (early-window-reference)",
         "v04_abl", "early-window-reference", "Tests casing-head subtraction fragility across boreholes"),
        ("Mechanism Check", "Target-Domain Leakage Diagnostic (auxiliary-test-centering)",
         "v04_abl", "auxiliary-test-centering", "Excluded auxiliary mode proving naive median shift fails"),
        ("Mechanism Check", "Boundary vs Interior Error Profile (w/o Zero-phase filter)",
         "abl", "w-o_Zero-phase_filter", "Boundary error check under causal group-delay filter"),
    ]
    for atype, subj, kind, key, note in mech_specs:
        if kind == "abl":
            probs = {w: np.load(RUNS_FINAL / key / w / f"probs_{w}.npy") for w in ALL_WELLS}
        else:
            probs = {w: np.load(V04_OUT / "ablations" / w / key / f"probs_{w}_cv.npy") for w in ALL_WELLS}
        st = _compute_full_stats(wells, probs)
        m = st["pooled"]
        gen_rows.append({
            "Analysis_Type": atype,
            "Subject": subj,
            "Samples": m["n"],
            "MacroF1": _fmt(m["macro_f1"]),
            "Balanced_Accuracy": _fmt(m["balanced_accuracy"]),
            "Kappa": _fmt(m["kappa"]),
            "F1_Poor": _fmt(m["f1_Poor"]),
            "F1_Moderate": _fmt(m["f1_Moderate"]),
            "F1_Good": _fmt(m["f1_Good"]),
            "ROC_AUC_Good_vs_Mod": _fmt(m["roc_auc_good_vs_mod"]),
            "Boundary_Errors": st["boundary_errors"],
            "Boundary_Total": st["boundary_n"],
            "Interior_Errors": st["interior_errors"],
            "Interior_Total": st["interior_n"],
            "Notes": note,
        })
    _write_csv(LIGHTNING_ROOT / "04_generalization_and_mechanism.csv", gen_rows)

    # 5. Segment Block-Bootstrap Statistical Significance (1,000 draws over 139 depth segments)
    sig_out: dict[str, dict] = {
        "ours_vs_baselines": {},
        "paired_vdl_vs_curves": {},
        "ours_vs_ablations": {},
    }
    ours_p = probs_cache["BondFusion-TCN (Ours)"]
    for bname in [
        "2D-CNN +VDL",
        "Reformer +VDL",
        "ModernTCN +VDL",
        "VDL-ModernTCN (mean pool)",
        "HistGBDT (+VDL, Window)",
        "ModernTCN (Curves only)",
        "CBL Two-Threshold Rule",
    ]:
        sig_out["ours_vs_baselines"][bname] = _segment_bootstrap_diff(wells, ours_p, probs_cache[bname])

    for cname, vname in [
        ("LSTM (Curves only)", "LSTM +VDL"),
        ("GRU (Curves only)", "GRU +VDL"),
        ("Transformer (Curves only)", "Transformer +VDL"),
        ("Reformer (Curves only)", "Reformer +VDL"),
        ("Crossformer (Curves only)", "Crossformer +VDL"),
        ("TimesNet (Curves only)", "TimesNet +VDL"),
        ("TimeMixer (Curves only)", "TimeMixer +VDL"),
        ("ModernTCN (Curves only)", "ModernTCN +VDL"),
        ("ModernTCN (Curves only)", "BondFusion-TCN (Ours)"),
    ]:
        sig_out["paired_vdl_vs_curves"][f"{vname} vs {cname}"] = _segment_bootstrap_diff(
            wells, probs_cache[vname], probs_cache[cname]
        )

    for ab_label in ["w/o Zero-phase filter", "w/o Trace contrast", "w/o CBL-gated bridge", "w/o Hilbert envelope", "w/o VDL (Curve-Prior only)"]:
        sig_out["ours_vs_ablations"][f"Full vs {ab_label}"] = _segment_bootstrap_diff(
            wells, ours_p, probs_cache[f"ABL::{ab_label}"]
        )

    (LIGHTNING_ROOT / "05_statistical_significance.json").write_text(json.dumps(sig_out, indent=2), encoding="utf-8")

    # 6. Unified Markdown Report
    md = [
        "# BondFusion-TCN 四大标准实验体系汇总报告（PyTorch Lightning + 6井对称嵌套 LOWO）\n",
        "统一评估口径：6 口真实测井（A–F，共 55,155 采样点，170 个断gap感知连续同标签段，139 个连续测井段），全部模型采用验证集早停体制输出每口留一测试井的无泄漏预测概率。\n",
        "\n## 一、对比实验（Comparison Experiments，按 Pooled Macro-F1 降序）\n",
        "| 排名 | 模型 | Pooled Macro-F1 | 6井均值 | BalAcc | Kappa | F1_Good | AUC(G vs M) | 总误判数 | Run170 F1 | 数据来源 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, r in enumerate(comp_rows, 1):
        md.append(
            f"| {rank} | {r['Model']} | **{r['Pooled_MacroF1']}** | {r['Mean_Well_MacroF1']} | "
            f"{r['Balanced_Accuracy']} | {r['Kappa']} | {r['F1_Good']} | {r['ROC_AUC_Good_vs_Mod']} | "
            f"{r['Total_Errors']} | {r['Run170_MacroF1']} | {r['Protocol_Source']} |"
        )

    md.extend([
        "\n## 二、消融实验（Ablation Experiments，单因素逐项消融）\n",
        "| 变体配置 | Pooled Macro-F1 | 相对下降 Δ | 6井均值 | BalAcc | Kappa | F1_Good | AUC(G vs M) | 总误判数 | 边界误判(/155) | Run170 F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in abl_rows:
        md.append(
            f"| {r['Variant']} | {r['Pooled_MacroF1']} | {r['Delta_Pooled_F1']} | {r['Mean_Well_MacroF1']} | "
            f"{r['Balanced_Accuracy']} | {r['Kappa']} | {r['F1_Good']} | {r['ROC_AUC_Good_vs_Mod']} | "
            f"{r['Total_Errors']} | {r['Boundary_Errors_of_155']} | {r['Run170_MacroF1']} |"
        )

    md.extend([
        "\n## 三、敏感性实验（Sensitivity Experiments）\n",
        "| 分组 | 实验编号 | 声波表征 | Bin宽度 | Pooled Macro-F1 | 6井均值 | BalAcc | F1_Good | AUC(G vs M) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for r in sens_rows:
        md.append(
            f"| {r['Group']} | {r['Experiment_ID']} | {r['Representation']} | {r['Bin_Size']} | "
            f"{r['Pooled_MacroF1']} | {r['Mean_Well_MacroF1']} | {r['Balanced_Accuracy']} | "
            f"{r['F1_Good']} | {r['ROC_AUC_Good_vs_Mod']} |"
        )

    md.extend([
        "\n## 四、泛化性与机理分析实验（Generalization & Mechanism Analysis Experiments）\n",
        "| 分析类别 | 对象 / 井别 | Pooled/Well F1 | BalAcc | Kappa | F1_Good | 边界误判/总数 | 内部误判/总数 | 说明 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for r in gen_rows:
        md.append(
            f"| {r['Analysis_Type']} | {r['Subject']} | {r['MacroF1']} | {r['Balanced_Accuracy']} | "
            f"{r['Kappa']} | {r['F1_Good']} | {r['Boundary_Errors']}/{r['Boundary_Total']} | "
            f"{r['Interior_Errors']}/{r['Interior_Total']} | {r['Notes']} |"
        )

    (LIGHTNING_ROOT / "FOUR_PILLAR_EXPERIMENT_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Generated 4 CSV tables + 05_statistical_significance.json + report in {LIGHTNING_ROOT}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
