"""Fig. S2 — per-well outer-CV scores across wells A–F, Curves vs Curves + VDL.

a  Macro-F1   b  Kappa   (*ModernTCN under Curves + VDL = BondFusion-TCN)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import _paper_data as P  # noqa: E402

WELL_MARKER = {"A": "o", "B": "s", "C": "^", "D": "v", "E": "D", "F": "P"}
OFF = {"Curves": -0.17, "Curves + VDL": 0.17}


def main():
    P.setup_style()
    spw = P.summary_per_well().set_index(["model", "well"])
    fig = plt.figure(figsize=P.MODES["B_wide"]["figsize"])
    gs = GridSpec(1, 2, figure=fig, wspace=0.08, left=0.125, right=0.985, top=0.84, bottom=0.13)
    B = P.BACKBONES
    axes = []
    for j, (key, lab) in enumerate([("macro_f1", "Macro-F1"), ("kappa", "Kappa")]):
        ax = fig.add_subplot(gs[j], sharey=axes[0] if axes else None)
        for i, b in enumerate(B):
            for name in (b, P.vdl(b)):
                iset = P.input_set(name)
                col = P.INPUT_COLORS[iset]
                y0 = i + OFF[iset]
                vals = []
                for w in P.ALL_WELLS:
                    v = float(spw.loc[(name, w), key])
                    vals.append(v)
                    ax.plot(
                        v,
                        y0,
                        marker=WELL_MARKER[w],
                        ms=3.4,
                        color=col,
                        mfc="white" if iset == "Curves" else col,
                        mew=0.75,
                        ls="none",
                        alpha=0.85,
                        zorder=3,
                    )
                mu = float(np.mean(vals))
                ax.plot([mu, mu], [y0 - 0.12, y0 + 0.12], color=col, lw=1.4, zorder=4)
        ax.set_ylim(len(B) - 0.5, -0.5)
        ax.set_xlabel(f"{lab}, per held-out well (A–F)")
        ax.grid(axis="x", lw=0.3, alpha=0.4)
        for i in range(len(B) - 1):
            ax.axhline(i + 0.5, color="0.9", lw=0.4)
        axes.append(ax)
    axes[0].set_yticks(range(len(B)), [f"{b}*" if b == "ModernTCN" else b for b in B])
    axes[1].tick_params(axis="y", labelleft=False)
    P.panel_label(axes[0], "a", x_pt=-64, y_pt=6)
    P.panel_label(axes[1], "b", x_pt=-10, y_pt=6)

    h = [
        Line2D([], [], marker=WELL_MARKER[w], ls="none", color="0.3", mfc="0.3", ms=3.4, label=f"Well {w}")
        for w in P.ALL_WELLS
    ]
    h += [
        Line2D([], [], marker="o", ls="none", color=P.INPUT_COLORS["Curves"], mfc="white", mew=0.8, ms=3.6, label="Curves"),
        Line2D([], [], marker="o", ls="none", color=P.INPUT_COLORS["Curves + VDL"], ms=3.6, label="Curves + VDL"),
        Line2D([], [], color="0.3", lw=1.4, label="6-well mean"),
    ]
    fig.legend(
        handles=h,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.995),
        ncol=5,
        columnspacing=1.2,
        handletextpad=0.3,
    )
    P.save(fig, "fig_S2_per_well")


if __name__ == "__main__":
    main()
