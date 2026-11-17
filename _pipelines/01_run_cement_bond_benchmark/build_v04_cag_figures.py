"""Generate publication-quality quantitative figures (Fig. 3 to Fig. 7) for the
Computers & Geosciences v04 revision directly from verified v04 outputs and
statistical significance tables.

Style & invariants:
  - Strictly muted SCI colorblind-safe palette (#B8525A primary rose-crimson,
    #5882A6 cold slate-blue, #8C9BA5 muted neutral gray-blue, #D67E7D dusty rose,
    #1E466E deep navy-slate).
  - Strictly Liberation Serif (Times-metric equivalent), pure black (#000000) regular
    labels, ticks, and spines; only bold lowercase panel letters (a, b, c, d).
  - Zero figure suptitles or subplot descriptive titles inside axes.

Usage:
    PYTHONPATH=_code python \
        _pipelines/01_run_cement_bond_benchmark/build_v04_cag_figures.py
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "_code"))
sys.path.insert(0, str(HERE))

from da_reformer.data import load_wells  # noqa: E402
from da_reformer.metrics import compute_metrics  # noqa: E402
from da_reformer.vdl import load_vdl  # noqa: E402
from build_four_pillar_tables_v04 import _load_deep_baseline_probs  # noqa: E402

ALL_WELLS = ["A", "B", "C", "D", "E", "F"]
V04_OUT = Path(os.environ.get("BONDFUSION_V04_OUT", str(ROOT / "_pipelines/01_run_cement_bond_benchmark/outputs/v04")))
OUT_ROOT = ROOT / "_outputs/p04_v04_lightning"
RUNS_FINAL = OUT_ROOT / "runs_final"
FIG_DIR = ROOT / "_paper/_revisions/20261116_v04_cag/main/figures"
SUPP_FIG_DIR = ROOT / "_paper/_revisions/20261116_v04_cag/supplementary/figures"
PNG_DIR = OUT_ROOT / "figures_preview"

# Strictly muted SCI warm-cool palette
COL_OURS = "#B8525A"       # BondFusion-TCN crimson-rose
COL_VDL = "#5882A6"        # +VDL cold slate blue
COL_CURVE = "#A4B0BA"      # Curves-only muted silver-slate
COL_DARK = "#1E466E"       # Deep navy slate
COL_ROSE_LIGHT = "#D67E7D" # Dusty rose
COL_SAND = "#E6C79C"       # Muted warm sand
COL_POOR = "#D67E7D"
COL_MOD = "#F2D7B6"
COL_GOOD = "#6A90B0"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.5,
    "text.color": "#000000",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "axes.edgecolor": "#000000",
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _panel_tag(ax, letter: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10.5, fontweight="bold", color="#000000", va="bottom", ha="left")


def _clean_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig: plt.Figure, name: str, out_dir: Path = FIG_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(PNG_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_fig_03(wells: dict, comp_rows: list[dict]) -> None:
    """Pooled confusion reduction (55,155 samples): Curves vs BondFusion-TCN and backbone error reduction."""
    y_all = np.concatenate([wells[w].y for w in ALL_WELLS])
    p_curv = np.concatenate([np.load(V04_OUT / "deep" / w / "ModernTCN" / f"probs_{w}_cv.npy") for w in ALL_WELLS]).argmax(1)
    p_ours = np.concatenate([np.load(RUNS_FINAL / "BondFusion-TCN_Full" / w / f"probs_{w}.npy") for w in ALL_WELLS]).argmax(1)

    cm_c = compute_metrics(y_all, p_curv)["confusion"]
    cm_o = compute_metrics(y_all, p_ours)["confusion"]
    cm_c = np.array(cm_c, dtype=float)
    cm_o = np.array(cm_o, dtype=float)
    pct_c = 100.0 * cm_c / cm_c.sum(1, keepdims=True)
    pct_o = 100.0 * cm_o / cm_o.sum(1, keepdims=True)
    diff = pct_o - pct_c

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.25), gridspec_kw={"width_ratios": [1.0, 1.0, 1.05, 1.25], "wspace": 0.38})
    cls_names = ["Poor", "Mod.", "Good"]
    cmap_seq = mcolors.LinearSegmentedColormap.from_list("sci_blue", ["#F7F9FB", "#A4BED1", "#376795", "#1E466E"])
    cmap_div = mcolors.LinearSegmentedColormap.from_list("sci_div", ["#B8525A", "#F4D4D4", "#FFFFFF", "#B8D0E4", "#376795"])

    for ax, mat, raw, tag in [(axes[0], pct_c, cm_c, "a"), (axes[1], pct_o, cm_o, "b")]:
        im = ax.imshow(mat, cmap=cmap_seq, vmin=0, vmax=100, aspect="auto")
        _panel_tag(ax, tag, -0.18, 1.03)
        ax.set_xticks(range(3)); ax.set_xticklabels(cls_names)
        ax.set_yticks(range(3)); ax.set_yticklabels(cls_names if tag == "a" else [])
        ax.set_xlabel("Predicted bond class")
        if tag == "a":
            ax.set_ylabel("Reference bond class")
        for i in range(3):
            for j in range(3):
                val = mat[i, j]
                col = "white" if val > 58 else "#000000"
                ax.text(j, i, f"{val:.1f}%\n({int(raw[i, j]):,})", ha="center", va="center", fontsize=6.8, color=col)

    im_d = axes[2].imshow(diff, cmap=cmap_div, vmin=-55, vmax=55, aspect="auto")
    _panel_tag(axes[2], "c", -0.15, 1.03)
    axes[2].set_xticks(range(3)); axes[2].set_xticklabels(cls_names)
    axes[2].set_yticks(range(3)); axes[2].set_yticklabels([])
    axes[2].set_xlabel("Predicted shift (pp)")
    for i in range(3):
        for j in range(3):
            v = diff[i, j]
            col = "white" if abs(v) > 36 else "#000000"
            axes[2].text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7.5, color=col)

    # Panel d: Total errors comparison across key paired models + BondFusion-TCN
    row_map = {r["Model"]: r for r in comp_rows}
    pairs = [
        ("CBL Rule", int(row_map["CBL Two-Threshold Rule"]["Total_Errors"]), None),
        ("LSTM", int(row_map["LSTM (Curves only)"]["Total_Errors"]), int(row_map["LSTM +VDL"]["Total_Errors"])),
        ("GRU", int(row_map["GRU (Curves only)"]["Total_Errors"]), int(row_map["GRU +VDL"]["Total_Errors"])),
        ("Transf.", int(row_map["Transformer (Curves only)"]["Total_Errors"]), int(row_map["Transformer +VDL"]["Total_Errors"])),
        ("Reformer", int(row_map["Reformer (Curves only)"]["Total_Errors"]), int(row_map["Reformer +VDL"]["Total_Errors"])),
        ("ModTCN", int(row_map["ModernTCN (Curves only)"]["Total_Errors"]), int(row_map["ModernTCN +VDL"]["Total_Errors"])),
        ("Ours", None, int(row_map["BondFusion-TCN (Ours)"]["Total_Errors"])),
    ]
    ax = axes[3]
    _panel_tag(ax, "d", -0.16, 1.03)
    y_pos = np.arange(len(pairs))
    h = 0.36
    for idx, (nm, ec, ev) in enumerate(pairs):
        if ec is not None and ev is not None:
            ax.barh(idx - h / 2, ec, height=h, color=COL_CURVE, edgecolor="#000000", linewidth=0.5)
            ax.barh(idx + h / 2, ev, height=h, color=COL_VDL, edgecolor="#000000", linewidth=0.5)
        elif ec is not None:
            ax.barh(idx, ec, height=h * 1.1, color=COL_CURVE, edgecolor="#000000", linewidth=0.5)
        else:
            ax.barh(idx, ev, height=h * 1.1, color=COL_OURS, edgecolor="#000000", linewidth=0.6)
            ax.text(ev + 100, idx, f"{ev:,}", va="center", ha="left", fontsize=7.2, color="#000000")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([p[0] for p in pairs])
    ax.invert_yaxis()
    ax.set_xlim(0, 7200)
    ax.set_xlabel("Pooled misclassified samples (of 55,155)")
    _clean_spines(ax)
    handles = [
        Patch(facecolor=COL_CURVE, edgecolor="#000000", linewidth=0.5, label="Curves only"),
        Patch(facecolor=COL_VDL, edgecolor="#000000", linewidth=0.5, label="Curves + VDL"),
        Patch(facecolor=COL_OURS, edgecolor="#000000", linewidth=0.5, label="BondFusion-TCN"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.92, edgecolor="#CCCCCC")
    _save(fig, "fig_03_error_direction")


def build_fig_04(comp_rows: list[dict], sig: dict) -> None:
    """Comparison across 8 sequence families + 2D-CNN + HistGBDT + CBL Rule + BondFusion-TCN."""
    rm = {r["Model"]: r for r in comp_rows}
    archs = ["LSTM", "GRU", "Transformer", "Reformer", "Crossformer", "TimesNet", "TimeMixer", "ModernTCN"]
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.65), gridspec_kw={"width_ratios": [1.35, 1.0, 1.0, 1.1], "wspace": 0.34})

    # Panel a: Pooled Macro-F1 across all families + Ours
    ax = axes[0]
    _panel_tag(ax, "a", -0.22, 1.03)
    labels_a = ["CBL Rule", "HistGBDT"] + archs + ["2D-CNN +VDL", "BondFusion-TCN"]
    y_idx = np.arange(len(labels_a))
    h = 0.36
    for i, lb in enumerate(labels_a):
        if lb == "CBL Rule":
            vc = float(rm["CBL Two-Threshold Rule"]["Pooled_MacroF1"])
            ax.barh(i, vc, height=h * 1.1, color=COL_CURVE, edgecolor="#000000", linewidth=0.5)
        elif lb == "HistGBDT":
            vc = float(rm["HistGBDT (Curves only)"]["Pooled_MacroF1"])
            vv = float(rm["HistGBDT (+VDL, Window)"]["Pooled_MacroF1"])
            ax.barh(i - h / 2, vc, height=h, color=COL_CURVE, edgecolor="#000000", linewidth=0.5)
            ax.barh(i + h / 2, vv, height=h, color=COL_VDL, edgecolor="#000000", linewidth=0.5)
        elif lb == "2D-CNN +VDL":
            vv = float(rm["2D-CNN +VDL"]["Pooled_MacroF1"])
            ax.barh(i, vv, height=h * 1.1, color=COL_VDL, edgecolor="#000000", linewidth=0.5)
        elif lb == "BondFusion-TCN":
            vv = float(rm["BondFusion-TCN (Ours)"]["Pooled_MacroF1"])
            ax.barh(i, vv, height=h * 1.15, color=COL_OURS, edgecolor="#000000", linewidth=0.6)
            ax.text(vv + 0.008, i, f"{vv:.4f}", va="center", ha="left", fontsize=7.0, color="#000000")
        else:
            vc = float(rm[f"{lb} (Curves only)"]["Pooled_MacroF1"])
            vv = float(rm[f"{lb} +VDL"]["Pooled_MacroF1"])
            ax.barh(i - h / 2, vc, height=h, color=COL_CURVE, edgecolor="#000000", linewidth=0.5)
            ax.barh(i + h / 2, vv, height=h, color=COL_VDL, edgecolor="#000000", linewidth=0.5)
    ax.set_yticks(y_idx)
    ax.set_yticklabels(labels_a)
    ax.invert_yaxis()
    ax.set_xlim(0.55, 0.99)
    ax.set_xlabel("Pooled Macro-F1")
    _clean_spines(ax)
    ax.legend(handles=[
        Patch(facecolor=COL_CURVE, edgecolor="#000000", linewidth=0.5, label="Curves"),
        Patch(facecolor=COL_VDL, edgecolor="#000000", linewidth=0.5, label="+VDL"),
        Patch(facecolor=COL_OURS, edgecolor="#000000", linewidth=0.5, label="Ours"),
    ], loc="upper right", frameon=True, framealpha=0.92, edgecolor="#CCCCCC", fontsize=6.8)

    # Panel b & c: Bootstrap 95% CI differences (Macro-F1 and Kappa)
    pair_keys = [
        ("LSTM +VDL", "LSTM +VDL vs LSTM (Curves only)", COL_VDL),
        ("GRU +VDL", "GRU +VDL vs GRU (Curves only)", COL_VDL),
        ("Transf. +VDL", "Transformer +VDL vs Transformer (Curves only)", COL_VDL),
        ("Reformer +VDL", "Reformer +VDL vs Reformer (Curves only)", COL_VDL),
        ("Crossf. +VDL", "Crossformer +VDL vs Crossformer (Curves only)", COL_VDL),
        ("TimesNet +VDL", "TimesNet +VDL vs TimesNet (Curves only)", COL_VDL),
        ("TimeMix. +VDL", "TimeMixer +VDL vs TimeMixer (Curves only)", COL_VDL),
        ("ModTCN +VDL", "ModernTCN +VDL vs ModernTCN (Curves only)", COL_VDL),
        ("Ours vs Curv.", "BondFusion-TCN (Ours) vs ModernTCN (Curves only)", COL_OURS),
    ]
    pv = sig["paired_vdl_vs_curves"]
    for ax, tag, metric_k, xl in [
        (axes[1], "b", "macro_f1", r"$\Delta$ Macro-F1 (+VDL $-$ Curves)"),
        (axes[2], "c", "kappa", r"$\Delta\kappa$ (+VDL $-$ Curves)"),
    ]:
        _panel_tag(ax, tag, -0.18, 1.03)
        ax.axvline(0.0, color="#000000", linestyle="--", linewidth=0.7, alpha=0.6)
        for i, (lbl, k, col) in enumerate(pair_keys):
            d = pv[k]
            val = d[f"delta_{metric_k}"]
            lo, hi = d[f"ci95_{metric_k}"]
            ax.errorbar(val, i, xerr=[[val - lo], [hi - val]], fmt="o", color=col, ecolor="#000000",
                        elinewidth=1.0, capsize=2.5, markersize=4.5, markeredgecolor="#000000", markeredgewidth=0.5)
        ax.set_yticks(range(len(pair_keys)))
        ax.set_yticklabels([p[0] for p in pair_keys] if tag == "b" else [])
        ax.invert_yaxis()
        ax.set_xlabel(xl)
        _clean_spines(ax)

    # Panel d: Per-class F1 for top contenders vs Curves-only ModernTCN
    ax = axes[3]
    _panel_tag(ax, "d", -0.18, 1.03)
    models_d = [
        ("Curves only", "ModernTCN (Curves only)", COL_CURVE),
        ("ModernTCN +VDL", "ModernTCN +VDL", "#8CB3D0"),
        ("2D-CNN +VDL", "2D-CNN +VDL", COL_VDL),
        ("BondFusion-TCN", "BondFusion-TCN (Ours)", COL_OURS),
    ]
    x_cls = np.arange(3)
    w_bar = 0.19
    for idx, (lbl, key, col) in enumerate(models_d):
        vals = [float(rm[key]["F1_Poor"]), float(rm[key]["F1_Moderate"]), float(rm[key]["F1_Good"])]
        ax.bar(x_cls + (idx - 1.5) * w_bar, vals, width=w_bar, color=col, edgecolor="#000000", linewidth=0.5, label=lbl)
    ax.set_xticks(x_cls)
    ax.set_xticklabels(["Poor", "Moderate", "Good"])
    ax.set_ylim(0.0, 1.25)
    ax.set_ylabel("Pooled per-class F1")
    _clean_spines(ax)
    ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.92, edgecolor="#CCCCCC", fontsize=6.4)
    _save(fig, "fig_04_vdl_agreement")


def build_fig_05(abl_rows: list[dict]) -> None:
    """Controlled single-factor ablation study of BondFusion-TCN across 55,155 samples."""
    variants = [r["Variant"] for r in abl_rows]
    short_names = [
        "BondFusion-TCN (Full)",
        "w/o Zero-phase filter",
        "w/o Trace contrast",
        "w/o CBL-gated bridge",
        "w/o Hilbert envelope",
        "w/o VDL (Curves only)",
    ]
    cols = [COL_OURS, COL_ROSE_LIGHT, "#9DB7CE", COL_VDL, "#7A8C99", COL_CURVE]
    pf1 = [float(r["Pooled_MacroF1"]) for r in abl_rows]
    mf1 = [float(r["Mean_Well_MacroF1"]) for r in abl_rows]
    kap = [float(r["Kappa"]) for r in abl_rows]
    bacc = [float(r["Balanced_Accuracy"]) for r in abl_rows]
    fg = [float(r["F1_Good"]) for r in abl_rows]
    err = [int(r["Total_Errors"]) for r in abl_rows]

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.3), gridspec_kw={"wspace": 0.34})
    y = np.arange(len(variants))
    h = 0.36

    # Panel a: Pooled Macro-F1 & 6-Well Mean Macro-F1
    ax = axes[0]
    _panel_tag(ax, "a", -0.25, 1.12)
    for i in range(len(variants)):
        ax.barh(i - h / 2, pf1[i], height=h, color=cols[i], edgecolor="#000000", linewidth=0.5)
        ax.barh(i + h / 2, mf1[i], height=h, color=cols[i], edgecolor="#000000", linewidth=0.5, alpha=0.55)
        ax.text(max(pf1[i], mf1[i]) + 0.006, i, f"{pf1[i]:.4f}", va="center", ha="left", fontsize=7.0)
    ax.set_yticks(y); ax.set_yticklabels(short_names)
    ax.invert_yaxis(); ax.set_xlim(0.60, 0.96); ax.set_xlabel("Macro-F1 score")
    _clean_spines(ax)
    ax.legend(handles=[
        Patch(facecolor="#5882A6", edgecolor="#000000", linewidth=0.5, label="Pooled Macro-F1"),
        Patch(facecolor="#5882A6", alpha=0.55, edgecolor="#000000", linewidth=0.5, label="6-Well Mean F1"),
    ], loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=True, framealpha=0.92, edgecolor="#CCCCCC", fontsize=6.8)

    # Panel b: Balanced Accuracy & Cohen's kappa
    ax = axes[1]
    _panel_tag(ax, "b", -0.14, 1.12)
    for i in range(len(variants)):
        ax.barh(i - h / 2, bacc[i], height=h, color=cols[i], edgecolor="#000000", linewidth=0.5)
        ax.barh(i + h / 2, kap[i], height=h, color=cols[i], edgecolor="#000000", linewidth=0.5, alpha=0.55)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.invert_yaxis(); ax.set_xlim(0.62, 0.98); ax.set_xlabel("Balanced accuracy & Cohen's kappa")
    _clean_spines(ax)
    ax.legend(handles=[
        Patch(facecolor="#5882A6", edgecolor="#000000", linewidth=0.5, label="Balanced accuracy"),
        Patch(facecolor="#5882A6", alpha=0.55, edgecolor="#000000", linewidth=0.5, label="Cohen's kappa"),
    ], loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=True, framealpha=0.92, edgecolor="#CCCCCC", fontsize=6.8)

    # Panel c: Good-class F1 vs Pooled Error Count
    ax = axes[2]
    _panel_tag(ax, "c", -0.16, 1.12)
    ax.barh(y, fg, height=0.52, color=cols, edgecolor="#000000", linewidth=0.5)
    for i in range(len(variants)):
        ax.text(fg[i] + 0.015, i, f"{fg[i]:.4f} ({err[i]:,} err.)", va="center", ha="left", fontsize=7.0)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.invert_yaxis(); ax.set_xlim(0.0, 1.06); ax.set_xlabel("Good-class F1 (pooled error count)")
    _clean_spines(ax)
    _save(fig, "fig_05_encoder_designs")


def build_fig_06(wells: dict, sens_rows: list[dict]) -> None:
    """Waveform time-bin sensitivity (carrier aliasing vs Hilbert) + depth log profiles in E & D."""
    sm = {r["Experiment_ID"]: r for r in sens_rows}
    fig = plt.figure(figsize=(12.6, 5.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.38, wspace=0.28)

    # Panel a: Hilbert vs Rectified Macro-F1 across arrival-time bin widths
    ax = fig.add_subplot(gs[0, 0])
    _panel_tag(ax, "a", -0.13, 1.04)
    bins_h = [10, 20, 100]
    bins_r = [10, 20, 50, 100, 190]
    ax.plot(bins_h, [float(sm[f"bondfusion-hilbert-bin{b}"]["Pooled_MacroF1"]) for b in bins_h],
            "-o", color=COL_OURS, linewidth=1.8, markersize=5.5, label="BondFusion (Hilbert)")
    ax.plot(bins_h, [float(sm[f"hilbert-bin{b}"]["Pooled_MacroF1"]) for b in bins_h],
            "--s", color=COL_ROSE_LIGHT, linewidth=1.3, markersize=4.5, label="Backbone (Hilbert)")
    ax.plot(bins_r, [float(sm[f"bondfusion-rectified-bin{b}"]["Pooled_MacroF1"]) for b in bins_r],
            "-^", color=COL_VDL, linewidth=1.5, markersize=5.0, label="BondFusion (Rectified)")
    ax.plot(bins_r, [float(sm[f"rectified-bin{b}"]["Pooled_MacroF1"]) for b in bins_r],
            ":d", color=COL_CURVE, linewidth=1.2, markersize=4.5, label="Backbone (Rectified)")
    ax.axvline(20, color=COL_OURS, linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100, 190])
    ax.set_xticklabels(["10", "20", "50", "100", "190"])
    ax.set_ylim(0.68, 0.89)
    ax.set_xlabel(r"Arrival-time bin width $b$ ($\mu$s)")
    ax.set_ylabel("Pooled Macro-F1")
    _clean_spines(ax)
    ax.legend(loc="lower left", ncol=2, frameon=True, framealpha=0.92, edgecolor="#CCCCCC", fontsize=6.8)

    # Panel b: Good-vs-Moderate ROC-AUC and Good F1 across bin widths
    ax = fig.add_subplot(gs[0, 1])
    _panel_tag(ax, "b", -0.13, 1.04)
    ax.plot(bins_h, [float(sm[f"bondfusion-hilbert-bin{b}"]["F1_Good"]) for b in bins_h],
            "-o", color=COL_OURS, linewidth=1.8, markersize=5.5, label="Hilbert F1 (Good)")
    ax.plot(bins_r, [float(sm[f"bondfusion-rectified-bin{b}"]["F1_Good"]) for b in bins_r],
            "-^", color=COL_VDL, linewidth=1.5, markersize=5.0, label="Rectified F1 (Good)")
    ax.plot(bins_h, [float(sm[f"bondfusion-hilbert-bin{b}"]["ROC_AUC_Good_vs_Mod"]) for b in bins_h],
            "--o", color=COL_DARK, linewidth=1.4, markersize=4.5, label="Hilbert AUC (Good vs Mod)")
    ax.plot(bins_r, [float(sm[f"bondfusion-rectified-bin{b}"]["ROC_AUC_Good_vs_Mod"]) for b in bins_r],
            "--^", color="#7A8C99", linewidth=1.3, markersize=4.5, label="Rectified AUC (Good vs Mod)")
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100, 190])
    ax.set_xticklabels(["10", "20", "50", "100", "190"])
    ax.set_ylim(0.14, 1.00)
    ax.set_xlabel(r"Arrival-time bin width $b$ ($\mu$s)")
    ax.set_ylabel("Good separation score")
    _clean_spines(ax)
    ax.legend(loc="lower left", ncol=2, frameon=True, framealpha=0.92, edgecolor="#CCCCCC", fontsize=6.8)

    # Panels c and d: Gap-free selective-advantage depth profiles in Well E (2755.7-2787.7 m) and Well D (2818-2852 m)
    cmap_cls = mcolors.ListedColormap([COL_POOR, COL_MOD, COL_GOOD])
    profiles = [
        (fig.add_subplot(gs[1, 0]), "c", "E", 13050, 13372, "Held-out Well E"),
        (fig.add_subplot(gs[1, 1]), "d", "D", 5790, 6130, "Held-out Well D"),
    ]
    raw_dir = ROOT / "_data/raw"
    for ax, tag, w, i0, i1, wlbl in profiles:
        _panel_tag(ax, tag, -0.13, 1.04)
        wd = wells[w]
        depth = wd.depth[i0:i1]
        cbl = wd.x_raw[i0:i1, 1]
        y_true = wd.y[i0:i1]
        p_bf = np.load(RUNS_FINAL / "BondFusion-TCN_Full" / w / f"probs_{w}.npy").argmax(1)[i0:i1]
        p_ng = np.load(RUNS_FINAL / "w-o_CBL-gated_bridge" / w / f"probs_{w}.npy").argmax(1)[i0:i1]
        p_mod = np.load(V04_OUT / "ablations" / w / "rectified-bin10" / f"probs_{w}_cv.npy").argmax(1)[i0:i1]
        p_curv = np.load(V04_OUT / "deep" / w / "ModernTCN" / f"probs_{w}_cv.npy").argmax(1)[i0:i1]

        # Plot CBL curve at bottom and discrete label strips above
        cbl_norm = np.clip(np.log1p(np.maximum(0.0, cbl)) / 3.5, 0.0, 1.0)
        ax.plot(depth, 0.2 + 0.9 * cbl_norm, color="#000000", linewidth=0.9, label="log(1+CBL)")
        strips = [
            (4.8, y_true, "Reference"),
            (3.9, p_bf, "BondFusion-TCN"),
            (3.0, p_ng, "w/o CBL gate"),
            (2.1, p_mod, "ModernTCN +VDL"),
            (1.2, p_curv, "Curves only"),
        ]
        for y0, arr, _ in strips:
            ax.imshow(arr[None, :], aspect="auto", cmap=cmap_cls, vmin=0, vmax=2,
                      extent=[depth[0], depth[-1], y0, y0 + 0.65], interpolation="nearest")
        ax.set_yticks([0.55] + [s[0] + 0.32 for s in strips])
        ax.set_yticklabels(["CBL"] + [s[2] for s in strips], fontsize=7.0)
        ax.set_xlim(depth[0], depth[-1])
        ax.set_ylim(0.05, 5.75)
        ax.set_xlabel(f"Measured depth in {wlbl} (m)")
        _clean_spines(ax)
        if tag == "d":
            ax.legend(handles=[
                Patch(facecolor=COL_POOR, edgecolor="#000000", linewidth=0.4, label="Poor"),
                Patch(facecolor=COL_MOD, edgecolor="#000000", linewidth=0.4, label="Moderate"),
                Patch(facecolor=COL_GOOD, edgecolor="#000000", linewidth=0.4, label="Good"),
            ], loc="lower right", ncol=3, frameon=True, framealpha=0.92, edgecolor="#CCCCCC", fontsize=6.8)
    _save(fig, "fig_06_good_failure")


def build_fig_07(gen_rows: list[dict]) -> None:
    """Cross-well LOWO generalization (A-F), boundary errors, and mechanism falsification."""
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.45), gridspec_kw={"width_ratios": [1.3, 1.0, 1.05], "wspace": 0.45})

    # Panel a: Per-well Macro-F1 across A-F
    ax = axes[0]
    _panel_tag(ax, "a", -0.15, 1.12)
    wells_list = list("ABCDEF")
    x = np.arange(len(wells_list))
    series = [
        ("CBL Rule", "CBL Two-Threshold Rule", COL_CURVE, ":v"),
        ("ModernTCN +VDL", "ModernTCN +VDL", "#7FA6C4", "--s"),
        ("2D-CNN +VDL", "2D-CNN +VDL", COL_VDL, "-.^"),
        ("BondFusion-TCN", "BondFusion-TCN (Ours)", COL_OURS, "-o"),
    ]
    gm = {r["Subject"]: r for r in gen_rows}
    for lbl, key, col, fmt in series:
        vals = [float(gm[f"{key} - Held-out Well {w}"]["MacroF1"]) for w in wells_list]
        ax.plot(x, vals, fmt, color=col, linewidth=1.8 if lbl == "BondFusion-TCN" else 1.25,
                markersize=5.5 if lbl == "BondFusion-TCN" else 4.5, label=lbl)
    ax.set_xticks(x); ax.set_xticklabels([f"Well {w}" for w in wells_list])
    ax.set_ylim(0.45, 1.03)
    ax.set_ylabel("Held-out well Macro-F1")
    _clean_spines(ax)
    ax.legend(loc="lower left", ncol=2, frameon=True, framealpha=0.92, edgecolor="#CCCCCC", fontsize=6.8)

    # Panel b: Transition-boundary vs interior errors
    ax = axes[1]
    _panel_tag(ax, "b", -0.18, 1.12)
    b_models = [
        ("BondFusion-TCN", "BondFusion-TCN (Ours)", COL_OURS),
        ("w/o Zero-phase", "Boundary vs Interior Error Profile (w/o Zero-phase filter)", COL_ROSE_LIGHT),
        ("2D-CNN +VDL", "2D-CNN +VDL", COL_VDL),
        ("ModernTCN +VDL", "ModernTCN +VDL", "#8CB3D0"),
        ("CBL Rule", "CBL Two-Threshold Rule", COL_CURVE),
    ]
    y_pos = np.arange(len(b_models))
    for i, (lbl, mkey, col) in enumerate(b_models):
        if mkey in gm:
            b_err = int(gm[mkey]["Boundary_Errors"])
            b_tot = int(gm[mkey]["Boundary_Total"])
            i_err = int(gm[mkey]["Interior_Errors"])
            i_tot = int(gm[mkey]["Interior_Total"])
        else:
            b_err = sum(int(gm[f"{mkey} - Held-out Well {w}"]["Boundary_Errors"]) for w in wells_list)
            b_tot = sum(int(gm[f"{mkey} - Held-out Well {w}"]["Boundary_Total"]) for w in wells_list)
            i_err = sum(int(gm[f"{mkey} - Held-out Well {w}"]["Interior_Errors"]) for w in wells_list)
            i_tot = sum(int(gm[f"{mkey} - Held-out Well {w}"]["Interior_Total"]) for w in wells_list)
        b_rate = 100.0 * b_err / b_tot
        i_rate = 100.0 * i_err / i_tot
        ax.barh(i - 0.18, b_rate, height=0.34, color=col, edgecolor="#000000", linewidth=0.5)
        ax.barh(i + 0.18, i_rate, height=0.34, color=col, edgecolor="#000000", linewidth=0.5, alpha=0.45)
        ax.text(b_rate + 0.8, i - 0.18, f"{b_err}/{b_tot}", va="center", ha="left", fontsize=6.7)
        ax.text(i_rate + 0.8, i + 0.18, f"{i_rate:.1f}%", va="center", ha="left", fontsize=6.7)
    ax.set_yticks(y_pos); ax.set_yticklabels([b[0] for b in b_models])
    ax.invert_yaxis()
    ax.set_xlim(0, 62)
    ax.set_xlabel("Error rate (%)")
    _clean_spines(ax)
    ax.legend(handles=[
        Patch(facecolor=COL_VDL, edgecolor="#000000", linewidth=0.5, label="Boundary (of 155)"),
        Patch(facecolor=COL_VDL, alpha=0.45, edgecolor="#000000", linewidth=0.5, label="Interior (of 55,000)"),
    ], loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=True, framealpha=0.92, edgecolor="#CCCCCC", fontsize=6.6)

    # Panel c: Mechanism verification vs falsified hypotheses
    ax = axes[2]
    _panel_tag(ax, "c", -0.18, 1.12)
    mech_items = [
        ("CBL rational gate (Ours)", 0.8629, COL_OURS),
        ("Early-window ref. drift", float(gm["Early-Window Reference Drift (early-window-reference)"]["MacroF1"]), "#8CB3D0"),
        ("Target median centering", float(gm["Target-Domain Leakage Diagnostic (auxiliary-test-centering)"]["MacroF1"]), COL_CURVE),
    ]
    ym = np.arange(len(mech_items))
    ax.barh(ym, [m[1] for m in mech_items], height=0.46, color=[m[2] for m in mech_items], edgecolor="#000000", linewidth=0.5)
    for i, (_, val, _) in enumerate(mech_items):
        ax.text(val + 0.015, i, f"{val:.4f}", va="center", ha="left", fontsize=7.5)
    ax.set_yticks(ym); ax.set_yticklabels([m[0] for m in mech_items])
    ax.invert_yaxis()
    ax.set_xlim(0.40, 0.98)
    ax.set_xlabel("Pooled Macro-F1")
    _clean_spines(ax)
    _save(fig, "fig_07_cv_vs_blind")


def _load_prob_dict(wells: dict, spec: tuple[str, str]) -> dict[str, np.ndarray]:
    """Load six-well held-out probability predictions from canonical v04 paths."""
    kind, key = spec
    if kind == "rule":
        return {w: np.load(V04_OUT / "baselines" / w / f"probs_{w}_cbl.npy") for w in ALL_WELLS}
    if kind == "baseline":
        base, sub = key.split("/")
        return {w: np.load(V04_OUT / "baselines" / w / base / f"probs_{w}_{sub}.npy") for w in ALL_WELLS}
    if kind == "v04_ablation":
        return {w: np.load(V04_OUT / "ablations" / w / key / f"probs_{w}_cv.npy") for w in ALL_WELLS}
    if kind == "deep":
        probs, _ = _load_deep_baseline_probs(key)
        return probs
    return {w: np.load(RUNS_FINAL / key / w / f"probs_{w}.npy") for w in ALL_WELLS}


def build_fig_S1_confusion_per_backbone(wells: dict) -> dict[str, dict[str, np.ndarray]]:
    """5x4 supplementary confusion matrix grid across all paired sequence families and top references."""
    panels = [
        ("LSTM (Curves)", ("deep", "LSTM")),
        ("LSTM +VDL", ("deep", "LSTM_plusVDL")),
        ("GRU (Curves)", ("deep", "GRU")),
        ("GRU +VDL", ("deep", "GRU_plusVDL")),
        ("Transf. (Curves)", ("deep", "Transformer")),
        ("Transf. +VDL", ("deep", "Transformer_plusVDL")),
        ("Reformer (Curves)", ("deep", "Reformer")),
        ("Reformer +VDL", ("deep", "Reformer_plusVDL")),
        ("Crossf. (Curves)", ("deep", "Crossformer")),
        ("Crossf. +VDL", ("deep", "Crossformer_plusVDL")),
        ("TimesNet (Curves)", ("deep", "TimesNet")),
        ("TimesNet +VDL", ("deep", "TimesNet_plusVDL")),
        ("TimeMix. (Curves)", ("deep", "TimeMixer")),
        ("TimeMix. +VDL", ("deep", "TimeMixer_plusVDL")),
        ("ModTCN (Curves)", ("deep", "ModernTCN")),
        ("ModTCN +VDL", ("v04_ablation", "rectified-bin10")),
        ("CBL Rule", ("rule", "")),
        ("VDL-ModernTCN", ("deep", "VDL-ModernTCN_(mean_pool)")),
        ("2D-CNN +VDL", ("deep", "2D-CNN_plusVDL")),
        ("BondFusion-TCN", ("ours", "BondFusion-TCN_Full")),
    ]
    loaded: dict[str, dict[str, np.ndarray]] = {}
    y_all = np.concatenate([wells[w].y for w in ALL_WELLS])
    cmap_seq = mcolors.LinearSegmentedColormap.from_list("sci_blue", ["#F7F9FB", "#A4BED1", "#376795", "#1E466E"])
    cls_names = ["Poor", "Mod.", "Good"]
    tags = [chr(ord("a") + i) for i in range(len(panels))]

    fig, axes = plt.subplots(5, 4, figsize=(11.2, 12.2), gridspec_kw={"wspace": 0.24, "hspace": 0.42})
    for idx, (label, spec) in enumerate(panels):
        ax = axes.flat[idx]
        probs = _load_prob_dict(wells, spec)
        loaded[label] = probs
        pred = np.concatenate([probs[w] for w in ALL_WELLS]).argmax(1)
        cm = np.array(compute_metrics(y_all, pred)["confusion"], dtype=float)
        pct = 100.0 * cm / cm.sum(1, keepdims=True)
        ax.imshow(pct, cmap=cmap_seq, vmin=0, vmax=100, aspect="auto")
        ax.text(-0.04, 1.04, tags[idx], transform=ax.transAxes, fontsize=9.5, fontweight="bold", color="#000000", va="bottom", ha="left")
        ax.text(0.11, 1.04, label, transform=ax.transAxes, fontsize=8.0, fontweight="normal", color="#000000", va="bottom", ha="left")
        ax.set_xticks(range(3))
        ax.set_xticklabels(cls_names if idx // 4 == 4 else [])
        ax.set_yticks(range(3))
        ax.set_yticklabels(cls_names if idx % 4 == 0 else [])
        if idx // 4 == 4:
            ax.set_xlabel("Predicted", fontsize=8.0)
        if idx % 4 == 0:
            ax.set_ylabel("Reference", fontsize=8.0)
        for r in range(3):
            for c in range(3):
                val = pct[r, c]
                col = "white" if val > 58 else "#000000"
                ax.text(c, r, f"{val:.1f}%\n({int(cm[r, c]):,})", ha="center", va="center", fontsize=6.1, color=col)
    _save(fig, "fig_S1_confusion_per_backbone", out_dir=SUPP_FIG_DIR)
    return loaded


def build_fig_S2_per_well(wells: dict, comp_rows: list[dict]) -> None:
    """Two-panel held-out well comparison (Macro-F1 and Cohen's kappa) across wells A-F."""
    series = [
        ("CBL Rule", ("rule", ""), COL_CURVE, ":v"),
        ("HistGBDT (+VDL, Window)", ("baseline", "window_vdl/default"), "#7A8C99", ":s"),
        ("ModernTCN (Curves only)", ("deep", "ModernTCN"), "#9AA7B2", "--d"),
        ("ModernTCN +VDL", ("v04_ablation", "rectified-bin10"), "#8CB3D0", "--^"),
        ("VDL-ModernTCN (mean pool)", ("deep", "VDL-ModernTCN_(mean_pool)"), "#6E96B8", "-.p"),
        ("Reformer +VDL", ("deep", "Reformer_plusVDL"), "#3B6C96", "-.*"),
        ("2D-CNN +VDL", ("deep", "2D-CNN_plusVDL"), COL_VDL, "-.^"),
        ("BondFusion-TCN (Ours)", ("ours", "BondFusion-TCN_Full"), COL_OURS, "-o"),
    ]
    x = np.arange(len(ALL_WELLS))
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.35), gridspec_kw={"wspace": 0.24})

    for ax, tag, metric, ylabel, ylim in [
        (axes[0], "a", "macro_f1", "Held-out well Macro-F1", (0.45, 1.03)),
        (axes[1], "b", "kappa", r"Held-out well Cohen's $\kappa$", (0.45, 1.03)),
    ]:
        _panel_tag(ax, tag, -0.10, 1.03)
        for lbl, spec, col, fmt in series:
            probs = _load_prob_dict(wells, spec)
            vals = [compute_metrics(wells[w].y, probs[w].argmax(1), probs=probs[w])[metric] for w in ALL_WELLS]
            is_ours = lbl == "BondFusion-TCN (Ours)"
            ax.plot(
                x, vals, fmt, color=col,
                linewidth=2.0 if is_ours else 1.2,
                markersize=5.8 if is_ours else 4.3,
                zorder=5 if is_ours else 3,
                label=lbl,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"Well {w}" for w in ALL_WELLS])
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        _clean_spines(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=4, frameon=True, framealpha=0.94, edgecolor="#CCCCCC", fontsize=7.6,
    )
    _save(fig, "fig_S2_per_well", out_dir=SUPP_FIG_DIR)


def main() -> int:
    wells = load_wells(ROOT / "_data/raw", ALL_WELLS, ["GR", "CBL", "CCL"], 0.5)
    comp_rows = _read_csv(OUT_ROOT / "01_comparison_experiments.csv")
    abl_rows = _read_csv(OUT_ROOT / "02_ablation_experiments.csv")
    sens_rows = _read_csv(OUT_ROOT / "03_sensitivity_experiments.csv")
    gen_rows = _read_csv(OUT_ROOT / "04_generalization_and_mechanism.csv")
    sig = json.loads((OUT_ROOT / "05_statistical_significance.json").read_text(encoding="utf-8"))

    build_fig_03(wells, comp_rows)
    build_fig_04(comp_rows, sig)
    build_fig_05(abl_rows)
    build_fig_06(wells, sens_rows)
    build_fig_07(gen_rows)
    build_fig_S1_confusion_per_backbone(wells)
    build_fig_S2_per_well(wells, comp_rows)
    print(f"Generated Fig. 3-7 PDFs in {FIG_DIR}, Fig. S1-S2 PDFs in {SUPP_FIG_DIR}, and PNG previews in {PNG_DIR}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
