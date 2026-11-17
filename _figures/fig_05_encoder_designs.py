"""Fig. 5 — Controlled mechanism comparison of BondFusion-TCN variants (outer CV, wells A–F).

a  Macro-F1 across 6 held-out wells (dots), 6-well mean (bar), pooled 55,155-sample score (diamond)
b  same for Cohen's kappa
c  same for Good-class F1
BondFusion-TCN (row 1) is set in bold; dashed line separates single-component removals (rows 2–5) from
alternative waveform representations and the Curve-Prior-only baseline (rows 6–8).
"""
from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import _paper_data as P  # noqa: E402

VARIANTS = [
    ("BondFusion-TCN", "BondFusion-TCN"),
    ("w/o Zero-phase filter", "w/o Zero-phase filter"),
    ("w/o Trace contrast", "w/o Trace contrast"),
    ("w/o CBL-gated bridge", "w/o CBL-gated bridge"),
    ("w/o Hilbert envelope", "w/o Hilbert envelope"),
    ("ModernTCN +VDL (raw)", "Raw VDL + ModernTCN (b = 10)"),
    ("VDL-ModernTCN (mean pool)", "Mean pooling (b = 100)"),
    ("w/o VDL (Curve-Prior only)", "Curve-Prior only (w/o VDL)"),
]
LABEL_Y = -0.14


def wrap(label: str) -> str:
    return textwrap.fill(label, 28, break_on_hyphens=False)


def strip(ax, rows, values, ens, colors, bold=0, jitter=0.12):
    rng = np.random.default_rng(0)
    for i, (v, c) in enumerate(zip(values, colors)):
        v = np.asarray(v, dtype=float)
        hl = bold is not None and i == bold
        ax.scatter(
            v,
            i + rng.uniform(-jitter, jitter, len(v)),
            s=10 if hl else 8,
            color=c,
            alpha=0.82,
            zorder=3,
            lw=0.4 if hl else 0,
            edgecolor="k" if hl else "none",
        )
        ax.plot([v.mean(), v.mean()], [i - 0.3, i + 0.3], color=c, lw=2.0 if hl else 1.4, zorder=4)
        ax.plot(ens[i], i, marker="D", ms=4.5, mfc=c, mec="k", mew=0.5, ls="none", zorder=5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.grid(axis="x", lw=0.3, alpha=0.4)


def main():
    P.setup_style()
    keys = [k for k, _ in VARIANTS]
    labels = [wrap(lbl) for _, lbl in VARIANTS]
    cols = [P.INPUT_COLORS[P.input_set(k)] for k in keys]

    for k in keys:
        em = P.ensemble_metrics(k)
        sv = P.seed_values(k, "macro_f1")
        print(
            f"{k:28s} pooled F1={em['macro_f1']:.4f} well_mean={sv.mean():.4f}±{sv.std(ddof=1):.4f} "
            f"kappa={em['kappa']:.4f} f1_Good={em['f1_Good']:.4f}"
        )

    fig = plt.figure(figsize=(7.0, 2.75))
    gs = GridSpec(1, 3, figure=fig, wspace=0.12, left=0.255, right=0.985, top=0.84, bottom=0.17)
    axes = []
    specs = [
        ("macro_f1", "Macro-F1", "a", [0.65, 0.75, 0.85, 0.95]),
        ("kappa", "Kappa", "b", [0.75, 0.85, 0.95]),
        ("f1_Good", "Good-class F1", "c", [0.20, 0.40, 0.60, 0.80]),
    ]
    for j, (metric, lab, letter, ticks) in enumerate(specs):
        ax = fig.add_subplot(gs[j], sharey=axes[0] if axes else None)
        vals = [P.seed_values(k, metric) for k in keys]
        ens = [P.ensemble_metrics(k)[metric] for k in keys]
        strip(ax, keys, vals, ens, cols, bold=0)
        for idx in range(len(keys) - 1):
            if idx != 4:
                ax.axhline(idx + 0.5, color="0.88", lw=0.35, zorder=0)
        ax.axhline(4.5, color="0.45", lw=0.65, ls=":")
        ax.set_xlabel(lab)
        ax.xaxis.set_label_coords(0.5, LABEL_Y)
        if j == 0:
            ax.set_yticks(range(len(labels)), labels)
            ax.get_yticklabels()[0].set_fontweight("bold")
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xticks(ticks)
        P.panel_label(ax, letter, x_pt=-126 if j == 0 else -8, y_pt=6)
        axes.append(ax)

    handles = [
        Line2D([], [], marker="o", ls="none", color=P.INPUT_COLORS["Curves"], ms=3.5, label="Curves only"),
        Line2D([], [], marker="o", ls="none", color=P.INPUT_COLORS["Curves + VDL"], ms=3.5, label="Curves + VDL"),
        Line2D([], [], color="0.3", lw=1.4, label="Well mean"),
        Line2D([], [], marker="D", ls="none", mfc="0.55", mec="k", mew=0.5, ms=4.5, label="Pooled"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.60, 0.995),
        ncol=4,
        handletextpad=0.35,
        columnspacing=1.6,
    )
    P.save(fig, "fig_05_encoder_designs")


if __name__ == "__main__":
    main()
