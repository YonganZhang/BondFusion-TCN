"""Fig. 3 — direction of the errors and where the agreement gain comes from (blind wells E+F merged, 10-seed ensembles).

a  confusion matrix pooled over the eight Curves architectures: colour and large text = row percentage
   (share of the true class), small text = mean count per architecture (pooled count / 8, rounded)
b  same for the eight Curves + VDL architectures
c  difference of the row percentages (Curves + VDL − Curves), percentage points
d  errors that overstate bond quality (predicted class above the interpreted class, Poor < Moderate < Good) per
   architecture, as a share of the held-out samples, Curves vs Curves + VDL (10-seed ensembles)
e  errors that understate bond quality, same layout
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

import _paper_data as P  # noqa: E402
import _post_hoc as H  # noqa: E402


def cell_text(ax, i, j, big, small, facecolor):
    col = P.fs.heatmap_text_color(facecolor)
    for txt, dy, size in ((big, -0.1, 8), (small, 0.24, 6)):
        if txt is None:
            continue
        t = ax.text(j, i + dy, txt, ha="center", va="center", fontsize=size, color=col)
        t.set_gid(P.fs.HEATMAP_CELL_GID)


def main():
    P.setup_style()
    n = len(P.BACKBONES)
    cm_c = P.pooled_confusion(P.BACKBONES)
    cm_v = P.pooled_confusion([P.vdl(b) for b in P.BACKBONES])
    pct_c = 100 * cm_c / cm_c.sum(1, keepdims=True)
    pct_v = 100 * cm_v / cm_v.sum(1, keepdims=True)
    diff = pct_v - pct_c
    print("Curves (pooled counts):\n", cm_c, "\nCurves + VDL (pooled counts):\n", cm_v)
    print("row % Curves:\n", np.round(pct_c, 1), "\nrow % Curves + VDL:\n", np.round(pct_v, 1))
    print("mean count per architecture Curves:\n", np.rint(cm_c / n).astype(int),
          "\nmean count per architecture Curves + VDL:\n", np.rint(cm_v / n).astype(int))
    print("delta row % (pp):\n", np.round(diff, 1))

    fig = plt.figure(figsize=(P.MODES["E_panorama"]["figsize"][0], 4.3))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 0.95], hspace=0.5, left=0.1, right=0.93, top=0.95,
                     bottom=0.09)
    gs = outer[0].subgridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.42)
    seq = P.sequential_cmap()
    axes = []
    for k, (cm, pct, letter) in enumerate([(cm_c, pct_c, "a"), (cm_v, pct_v, "b")]):
        ax = fig.add_subplot(gs[k])
        im = ax.imshow(pct, cmap=seq, vmin=0, vmax=100)
        for (i, j), v in np.ndenumerate(pct):
            cell_text(ax, i, j, f"{v:.1f}%", f"{int(np.rint(cm[i, j] / n)):,}", im.cmap(im.norm(v)))
        axes.append(ax)
        P.panel_label(ax, letter, x_pt=-40 if letter == "a" else -16, y_pt=2)
    cax_a = make_axes_locatable(axes[0]).append_axes("right", size="5%", pad=0.06)
    cax_a.set_visible(False)  # keeps panel a the same size as b and c
    cb = fig.colorbar(im, cax=make_axes_locatable(axes[1]).append_axes("right", size="5%", pad=0.06))
    cb.set_label("Share of true class (%)")
    cb.outline.set_linewidth(0.4)
    ax = fig.add_subplot(gs[2])
    lim = float(np.abs(diff).max())
    im = ax.imshow(diff, cmap=P.diverging_cmap(), vmin=-lim, vmax=lim)
    for (i, j), v in np.ndenumerate(diff):
        cell_text(ax, i, j, P.fmt_signed(v, 1), None, im.cmap(im.norm(v)))
    axes.append(ax)
    P.panel_label(ax, "c", x_pt=-12, y_pt=2)
    cb = fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.06))
    cb.set_label("Δ percentage points of true class")
    cb.outline.set_linewidth(0.4)
    for k, ax in enumerate(axes):
        ax.set_xticks(range(3), P.CLASSES)
        ax.set_yticks(range(3), P.CLASSES if k == 0 else [""] * 3)
        ax.set_xlabel("Predicted class")
        ax.tick_params(length=0)
        for pos in (0.5, 1.5):
            ax.axhline(pos, color="0.75", lw=0.45, zorder=2)
            ax.axvline(pos, color="0.75", lw=0.45, zorder=2)
        for s_ in ax.spines.values():
            s_.set_visible(True)
            s_.set_linewidth(0.65)
            s_.set_edgecolor("k")
    axes[0].set_ylabel("True class")

    # d, e: direction of the errors per architecture
    ed = H.error_direction()
    n_s = ed["n_samples"]
    gs2 = outer[1].subgridspec(1, 2, wspace=0.12)
    names = P.BACKBONES
    ypos = np.arange(len(names))[::-1]
    share = {(key, tag): np.array([100 * ed["per_architecture"][b][tag]["pooled"][key] / n_s for b in names])
             for key in ("overstate", "understate") for tag in ("curves", "vdl")}
    xmax = 1.08 * max(v.max() for v in share.values())  # shared scale so that d and e can be compared directly
    for k, (key, letter, lab) in enumerate([("overstate", "d", "Errors overstating bond quality (% of samples)"),
                                             ("understate", "e", "Errors understating bond quality (% of samples)")]):
        ax = fig.add_subplot(gs2[k])
        xc, xv = share[(key, "curves")], share[(key, "vdl")]
        for y0, a, b_ in zip(ypos, xc, xv):
            ax.annotate("", xy=(b_, y0), xytext=(a, y0),
                        arrowprops=dict(arrowstyle="-|>", color="0.6", lw=0.7, shrinkA=2.5, shrinkB=2.5,
                                        mutation_scale=6))
        ax.scatter(xc, ypos, s=18, color=P.INPUT_COLORS["Curves"], zorder=3, label="Curves")
        ax.scatter(xv, ypos, s=18, color=P.INPUT_COLORS["Curves + VDL"], zorder=3, label="Curves + VDL")
        disp = [f"{b}*" if b == "ModernTCN" else b for b in names]
        ax.set_yticks(ypos, disp if k == 0 else [""] * len(names))
        ax.set_xlabel(lab)
        ax.set_xlim(0, xmax)
        ax.grid(axis="x", lw=0.3, color="0.85")
        ax.tick_params(axis="y", length=0)
        P.panel_label(ax, letter, x_pt=-58 if letter == "d" else -10, y_pt=2)
        if k == 1:
            ax.legend(loc="lower right", handletextpad=0.2, borderaxespad=0.2)
    P.save(fig, "fig_03_error_direction")


if __name__ == "__main__":
    main()
