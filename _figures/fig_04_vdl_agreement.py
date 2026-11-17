"""Fig. 4 — effect of the VDL input on performance across eight architectures (outer CV, wells A–F).

a  Macro-F1 (%), cross-well mean ± sd (A–F), Curves vs Curves + VDL; filled diamonds = pooled (55,155 samples)
b  pooled Δkappa, c  pooled Δmacro-F1 (Curves + VDL − Curves) with 95% 139-segment block-bootstrap intervals;
   filled black square = interval excludes 0, open = includes 0
d  pooled Δ per-class F1, Curves + VDL − Curves (*ModernTCN Curves + VDL = BondFusion-TCN)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import _paper_data as P  # noqa: E402


def main():
    P.setup_style()
    B = P.BACKBONES
    yy = np.arange(len(B))
    sig = P.sig_vdl_pairs()

    fig = plt.figure(figsize=P.MODES["B_wide"]["figsize"])
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1.35, 0.8, 0.8, 1.0], wspace=0.12,
                  left=0.11, right=0.92, top=0.86, bottom=0.14)
    ax = fig.add_subplot(gs[0])
    off = {"Curves": -0.15, "Curves + VDL": 0.15}
    for b, y in zip(B, yy):
        m = {}
        for name in (b, P.vdl(b)):
            k = P.seed_values(name, "macro_f1") * 100
            iset = P.input_set(name)
            m[iset] = k.mean()
            ax.errorbar(k.mean(), y + off[iset], xerr=k.std(ddof=1), fmt="o", ms=3.5, color=P.INPUT_COLORS[iset],
                        elinewidth=0.8, capsize=1.5, zorder=3)
            ax.plot(P.ensemble_metrics(name)["macro_f1"] * 100, y + off[iset], marker="D", ms=5.5, mfc=P.INPUT_COLORS[iset],
                    mec="k", mew=0.6, ls="none", zorder=4)
        ax.plot([m["Curves"], m["Curves + VDL"]], [y + off["Curves"], y + off["Curves + VDL"]], color="0.75",
                lw=0.6, zorder=1)
    ax.set_yticks(yy, [f"{b}*" if b == "ModernTCN" else b for b in B])
    ax.set_ylim(len(B) - 0.5, -0.5)
    ax.set_xlabel("Macro-F1 (%)")
    ax.grid(axis="x", lw=0.3, alpha=0.4)
    handles = [Line2D([], [], marker="o", ls="none", color=P.INPUT_COLORS["Curves"], ms=3.5, label="Curves"),
               Line2D([], [], marker="o", ls="none", color=P.INPUT_COLORS["Curves + VDL"], ms=3.5,
                      label="Curves + VDL"),
               Line2D([], [], marker="D", ls="none", mfc="0.55", mec="k", mew=0.6, ms=5.5, label="Pooled")]
    handles += [Line2D([], [], marker="s", color="k", ms=3.8, label="Δ, 95% CI excludes 0"),
                Line2D([], [], marker="s", color="k", mfc="white", ms=3.8, label="Δ, 95% CI includes 0")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5, handletextpad=0.3,
               columnspacing=1.4, borderaxespad=0.1)
    P.panel_label(ax, "a", x_pt=-62, y_pt=6)

    for j, (key, ci, lab, letter) in enumerate([("ensemble_kappa_diff", "boot_kappa_diff_ci95", "Δ Kappa", "b"),
                                                ("ensemble_macro_f1_diff", "boot_macro_f1_diff_ci95", "Δ Macro-F1", "c")]):
        axb = fig.add_subplot(gs[1 + j], sharey=ax)
        for b, y in zip(B, yy):
            r = sig[f"{P.vdl(b)} vs {b}"]
            lo, hi = r[ci]
            excl = lo > 0 or hi < 0
            axb.plot([lo, hi], [y, y], color="k", lw=1.0)
            axb.plot(r[key], y, marker="s", ms=3.8, color="k", mfc="k" if excl else "white", mew=0.9, zorder=3)
        axb.axvline(0, color="k", lw=0.6, ls="--")
        axb.set_xlabel(lab)
        axb.tick_params(axis="y", labelleft=False)
        axb.grid(axis="x", lw=0.3, alpha=0.4)
        P.panel_label(axb, letter, x_pt=-8, y_pt=6)

    axc = fig.add_subplot(gs[3], sharey=ax)
    D = np.array([[P.ensemble_metrics(P.vdl(b))[f"f1_{c}"] - P.ensemble_metrics(b)[f"f1_{c}"]
                   for c in P.CLASSES] for b in B])
    lim = float(np.ceil(np.abs(D).max() * 20) / 20)
    im = axc.imshow(D, cmap=P.diverging_cmap(), vmin=-lim, vmax=lim, aspect="auto")
    P.annotate_cells(axc, im, D, [[P.fmt_signed(v) for v in row] for row in D])
    axc.set_xticks(range(3), P.CLASSES)
    axc.set_xlabel("Δ F1 by class")
    axc.tick_params(axis="y", labelleft=False, left=False)
    for s in axc.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.55)
        s.set_edgecolor("k")
    cb = fig.colorbar(im, ax=axc, fraction=0.08, pad=0.04)
    cb.set_label("Δ F1 (Curves + VDL − Curves)")
    cb.outline.set_linewidth(0.4)
    P.panel_label(axc, "d", x_pt=-8, y_pt=6)
    print("delta per-class F1 (rows architectures, cols Poor/Moderate/Good):\n", np.round(D, 3))
    P.save(fig, "fig_04_vdl_agreement")


if __name__ == "__main__":
    main()
