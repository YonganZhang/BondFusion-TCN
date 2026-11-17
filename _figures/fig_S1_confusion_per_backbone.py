"""Fig. S1 — held-out outer-CV confusion matrices (wells A–F pooled, 55,155 samples) per architecture, Curves vs Curves + VDL.
Colour and large text = row percentage (share of the true class); small text = counts."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

import _paper_data as P  # noqa: E402

def cell_text(ax, i, j, big, small, facecolor):
    col = P.fs.heatmap_text_color(facecolor)
    for txt, dy, size in ((big, -0.13, 7), (small, 0.25, 6)):
        t = ax.text(j, i + dy, txt, ha="center", va="center", fontsize=size, color=col)
        t.set_gid(P.fs.HEATMAP_CELL_GID)


def main():
    P.setup_style()
    fig = plt.figure(figsize=(7.2, 6.9))
    outer = GridSpec(4, 2, figure=fig, wspace=0.16, hspace=0.2, left=0.085, right=0.9, top=0.965, bottom=0.06)
    seq = P.sequential_cmap()
    first = None
    for k, b in enumerate(P.BACKBONES):
        inner = outer[k // 2, k % 2].subgridspec(1, 2, wspace=0.08)
        for j, name in enumerate((b, P.vdl(b))):
            ax = fig.add_subplot(inner[j])
            cm = np.array(P.ensemble_metrics(name)["confusion"])
            pct = 100 * cm / cm.sum(1, keepdims=True)
            im = ax.imshow(pct, cmap=seq, vmin=0, vmax=100)
            for (i, jj), v in np.ndenumerate(pct):
                cell_text(ax, i, jj, f"{v:.1f}%", f"{int(cm[i, jj]):,}", im.cmap(im.norm(v)))
            ax.set_xticks(range(3), P.CLASSES if k >= 6 else [""] * 3)
            ax.set_yticks(range(3), P.CLASSES if (j == 0 and k % 2 == 0) else [""] * 3)
            ax.tick_params(length=0)
            for pos in (0.5, 1.5):
                ax.axhline(pos, color="0.75", lw=0.4, zorder=2)
                ax.axvline(pos, color="0.75", lw=0.4, zorder=2)
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_linewidth(0.55)
                s.set_edgecolor("k")
            ax.set_xlabel(P.REF_DISPLAY[name] if name == P.REF_VDL else f"{b}, {P.input_set(name)}", labelpad=2)
            ax.xaxis.set_label_position("top")
            if first is None:
                first = ax
    cax = fig.add_axes([0.92, 0.3, 0.015, 0.4])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Share of true class (%)")
    cb.outline.set_linewidth(0.4)
    fig.text(0.5, 0.016, "Predicted class", ha="center", va="bottom", fontsize=8)
    fig.text(0.005, 0.5, "True class", ha="left", va="center", rotation=90, fontsize=8)
    P.save(fig, "fig_S1_confusion_per_backbone")


if __name__ == "__main__":
    main()
