"""Fig. 7 — Inner-CV selection alignment and Hilbert bin-width sensitivity (outer CV, wells A–F).

a  Inner-CV macro-F1 (mean across 6 outer folds) vs held-out 6-well mean macro-F1 across all 18 tuned models;
   arrows connect each architecture's Curves to Curves + VDL run; star = BondFusion-TCN
b  Sensitivity to temporal bin width b (10–190 samples) comparing Hilbert log-envelope against rectified waveform,
   for BondFusion-TCN (solid) and the bare ModernTCN backbone (dashed)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import _paper_data as P  # noqa: E402

NAME_OFFSET = {
    "LSTM": (5, 0, "left"),
    "GRU": (0, 6, "center"),
    "Transformer": (-6, 2, "right"),
    "Reformer": (5, 0, "left"),
    "Crossformer": (5, 0, "left"),
    "TimesNet": (-6, 2, "right"),
    "TimeMixer": (5, -3, "left"),
    "ModernTCN": (-7, 0, "right"),
}


def main():
    P.setup_style()
    df = P.cv_vs_blind()
    rho_w, p_w = spearmanr(df["cv_macro_f1"], df["blind_macro_f1_mean"])
    rho_p, p_p = spearmanr(df["cv_macro_f1"], df["blind_macro_f1_pooled"])
    print(f"Spearman well-mean: rho={rho_w:.3f} p={p_w:.2e} (n={len(df)})")
    print(f"Spearman pooled:    rho={rho_p:.3f} p={p_p:.2e} (n={len(df)})")

    fig = plt.figure(figsize=(7.2, 3.15))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1], wspace=0.28, left=0.08, right=0.985, top=0.96, bottom=0.15)
    ax = fig.add_subplot(gs[0])
    d = df.set_index("model")
    for b in P.BACKBONES:
        rc, rv = d.loc[b], d.loc[P.vdl(b)]
        ax.annotate(
            "",
            xy=(rv["cv_macro_f1"], rv["blind_macro_f1_mean"]),
            xytext=(rc["cv_macro_f1"], rc["blind_macro_f1_mean"]),
            arrowprops=dict(arrowstyle="-|>", color="0.65", lw=0.6, mutation_scale=6, shrinkA=3, shrinkB=3.5),
            zorder=1,
        )
        dx, dy, ha = NAME_OFFSET[b]
        lbl = "BondFusion" if b == "ModernTCN" else b
        t = ax.annotate(
            lbl,
            (rv["cv_macro_f1"], rv["blind_macro_f1_mean"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            color="0.25",
        )
        t.set_gid("small_text")
    for extra, lbl, off in [("2D-CNN +VDL", "2D-CNN", (-6, 0)), (P.MEAN_POOL, "Mean pool", (0, -8))]:
        r = d.loc[extra]
        t = ax.annotate(
            lbl,
            (r["cv_macro_f1"], r["blind_macro_f1_mean"]),
            xytext=off,
            textcoords="offset points",
            ha="left" if off[0] > 0 else ("center" if off[0] == 0 else "right"),
            va="top" if off[1] < 0 else "center",
            color="0.25",
        )
        t.set_gid("small_text")
    for _, r in df.iterrows():
        col = P.INPUT_COLORS[r["input_set"]]
        if r["model"] == P.REF_VDL:
            ax.scatter(
                r["cv_macro_f1"],
                r["blind_macro_f1_mean"],
                marker="*",
                s=85,
                color=col,
                edgecolor="k",
                lw=0.4,
                zorder=4,
            )
        else:
            ax.scatter(
                r["cv_macro_f1"],
                r["blind_macro_f1_mean"],
                marker="o",
                s=22,
                color=col,
                edgecolor="white",
                lw=0.4,
                zorder=3,
            )
    ax.set_xlabel("Inner-CV Macro-F1 (6 outer folds)")
    ax.set_ylabel("Held-out Macro-F1, well mean (A–F)")
    ax.grid(lw=0.3, alpha=0.4)
    ax.text(
        0.97,
        0.04,
        f"Spearman \u03c1 = {P.fmt_signed(rho_w).lstrip('+')}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )
    h = [
        Line2D([], [], marker="o", ls="none", color=P.INPUT_COLORS["Curves"], ms=4, label="Curves"),
        Line2D([], [], marker="o", ls="none", color=P.INPUT_COLORS["Curves + VDL"], ms=4, label="Curves + VDL"),
        Line2D(
            [],
            [],
            marker="*",
            ls="none",
            color=P.INPUT_COLORS["Curves + VDL"],
            mec="k",
            mew=0.4,
            ms=7.5,
            label="BondFusion-TCN",
        ),
    ]
    ax.legend(handles=h, loc="upper left", handletextpad=0.25, borderaxespad=0.3)
    P.panel_label(ax, "a", x_pt=-36, y_pt=2)

    # b: bin-width sensitivity (Hilbert vs Rectified)
    rows = P.read_csv_table("03_sensitivity_experiments.csv")
    axb = fig.add_subplot(gs[1])
    series = [
        ("BondFusion-TCN Representation & Bin Width", "hilbert", "BondFusion-TCN, Hilbert", P.INPUT_COLORS["Curves + VDL"], "-", "o", 1.5, 4.5),
        ("BondFusion-TCN Representation & Bin Width", "rectified", "BondFusion-TCN, Rectified", P.PAL4[3], "-", "s", 1.3, 3.8),
        ("Backbone Representation & Bin Width", "hilbert", "ModernTCN, Hilbert", P.PAL4[1], "--", "^", 1.1, 3.8),
        ("Backbone Representation & Bin Width", "rectified", "ModernTCN, Rectified", P.INPUT_COLORS["Curves"], "--", "v", 1.0, 3.6),
    ]
    for grp, rep, lab, col, ls, mk, lw, ms in series:
        sub = sorted(
            [r for r in rows if r["Group"] == grp and r["Representation"] == rep],
            key=lambda r: int(r["Bin_Size"]),
        )
        xs = [int(r["Bin_Size"]) for r in sub]
        ys = [float(r["Pooled_MacroF1"]) for r in sub]
        axb.plot(xs, ys, color=col, ls=ls, lw=lw, marker=mk, ms=ms, mec="white", mew=0.3, label=lab, zorder=3 if ls == "-" else 2)
        if grp.startswith("BondFusion") and rep == "hilbert":
            b20 = next(y for x, y in zip(xs, ys) if x == 20)
            axb.scatter([20], [b20], marker="*", s=95, color=col, edgecolor="k", lw=0.4, zorder=5)
            t = axb.annotate(
                f"b = 20 ({b20:.4f})",
                (20, b20),
                xytext=(6, 4),
                textcoords="offset points",
                ha="left",
                va="bottom",
                color="0.2",
            )
            t.set_gid("small_text")
        if grp.startswith("BondFusion") and rep == "rectified":
            r20 = next(y for x, y in zip(xs, ys) if x == 20)
            t = axb.annotate(
                f"Alias dip ({r20:.4f})",
                (20, r20),
                xytext=(14, -3),
                textcoords="offset points",
                ha="left",
                va="center",
                color=P.PAL4[3],
            )
            t.set_gid("small_text")
    axb.set_xscale("log")
    axb.set_xticks([10, 20, 50, 100, 190])
    axb.set_xticklabels(["10", "20", "50", "100", "190"])
    axb.minorticks_off()
    axb.set_xlim(8.5, 225)
    axb.set_ylim(0.685, 0.895)
    axb.set_xlabel("VDL temporal bin width b (time samples)")
    axb.set_ylabel("Pooled Macro-F1 (wells A–F)")
    axb.grid(lw=0.3, alpha=0.4)
    axb.legend(loc="lower left", handlelength=1.6, borderaxespad=0.3)
    P.panel_label(axb, "b", x_pt=-38, y_pt=2)
    P.save(fig, "fig_07_cv_vs_blind")


if __name__ == "__main__":
    main()
