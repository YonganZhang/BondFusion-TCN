"""Fig. 2 — data and the VDL class signature.

a  two intervals of one training well: GR, CBL, CCL (recorded units), raw VDL (variable density), reference labels;
   two red dashed lines in the VDL track = start and end of the window of panel b
b  class-wise median Hilbert log-envelope (bin 20) vs time sample, pooled over A–F, IQR shading;
   shaded band = contiguous samples where the Moderate median is > 1 log unit below Poor and Good
(class proportions per well are in Table 1, not repeated here)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

import _paper_data as P  # noqa: E402

HALF_M = 30.0  # half-width of each interval around a class transition


def pick_intervals() -> tuple[str, list[tuple[int, int]]]:
    """No contiguous A–D segment contains all three classes (labels are long blocks), so panel a shows two
    intervals of ONE training well: around a Poor<->Moderate and a Moderate<->Good transition, each inside one
    segment (uniform 0.1 m step), clipped to +-HALF_M. Picks the well with the most balanced class counts."""
    best = None
    for w in P.TRAIN_WELLS:
        d = P.well(w)
        picks = {}
        for a, b in d.segments:
            if not np.allclose(np.diff(d.depth[a:b]), 0.1, atol=1e-3):
                continue
            for t in np.flatnonzero(np.diff(d.y[a:b]) != 0) + a + 1:
                pair = frozenset((int(d.y[t - 1]), int(d.y[t])))
                lo = max(a, int(np.searchsorted(d.depth, d.depth[t] - HALF_M)))
                hi = min(b, int(np.searchsorted(d.depth, d.depth[t] + HALF_M)))
                cnt = np.bincount(d.y[lo:hi], minlength=3)
                score = min(c for c in cnt if c > 0)
                if pair not in picks or score > picks[pair][0]:
                    picks[pair] = (score, lo, hi)
        need = [frozenset((0, 1)), frozenset((1, 2))]
        if all(k in picks for k in need):
            score = min(picks[k][0] for k in need)
            if best is None or score > best[0]:
                best = (score, w, sorted([picks[k][1:] for k in need]))
    if best is None:
        raise RuntimeError("no training well has both transitions inside segments")
    return best[1], best[2]


def draw_tracks(axs, w, s, e, win, first: bool):
    d = P.well(w)
    df = P.curves(w).iloc[s:e]
    depth = d.depth[s:e]
    tracks = [("GR", P.PAL10[1]), ("CBL", P.PAL10[8]), ("CCL", P.PAL10[0])]
    for ax, (c, col) in zip(axs[:3], tracks):
        ax.plot(df[c].to_numpy(), depth, color=col, lw=0.6)
    v = P.vdl_raw(w)[s:e]
    clip = float(np.percentile(np.abs(v), 99))
    axs[3].imshow(v, aspect="auto", cmap="gray", vmin=-clip, vmax=clip, interpolation="antialiased",
                  extent=[0, P.N_KEEP, depth[-1] + 0.05, depth[0] - 0.05], rasterized=True)
    for xx in (win["sample_start"], win["sample_end"]):
        axs[3].axvline(xx, color=P.PAL4[3], lw=0.7, ls="--")
    axs[4].imshow(d.y[s:e, None], aspect="auto", cmap=P.class_cmap(), vmin=-0.5, vmax=2.5,
                  interpolation="nearest", extent=[0, 1, depth[-1] + 0.05, depth[0] - 0.05])
    axs[0].set_ylim(depth[-1] + 0.05, depth[0] - 0.05)  # depth increases downward
    for ax in axs:
        ax.spines["right"].set_visible(True)
        ax.spines["top"].set_visible(True)
        if ax is not axs[0]:
            ax.tick_params(axis="y", labelleft=False)
        if not first:
            ax.tick_params(axis="x", labeltop=False, labelbottom=False, top=True, bottom=False)
        else:
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
    axs[4].set_xticks([])
    axs[3].set_xticks([0, 500, 1000, 1500])
    if first:
        for ax, lab in zip(axs[:3], ["GR", "CBL", "CCL"]):
            ax.set_xlabel(lab)
        axs[3].set_xlabel("VDL time sample")
        axs[4].set_xlabel("Interpreted\nlabel", linespacing=0.95)
    print(f"interval: well {w} {depth[0]:.2f}-{depth[-1]:.2f} m, n={e - s}, classes={np.bincount(d.y[s:e], minlength=3)}")
    return depth[0], depth[-1]


def main():
    P.setup_style()
    w, ivs = pick_intervals()
    win = P.vdl_window()

    fig = plt.figure(figsize=(7.2, 4.9))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 0.92], wspace=0.24,
                  left=0.085, right=0.985, top=0.885, bottom=0.085)
    lens = [P.well(w).depth[e - 1] - P.well(w).depth[s] for s, e in ivs]
    ga = gs[0, 0].subgridspec(2, 5, width_ratios=[1, 1, 1, 2.2, 0.55], height_ratios=lens, wspace=0.12, hspace=0.06)
    rows = []
    for r, (s, e) in enumerate(ivs):
        axs = [fig.add_subplot(ga[r, i]) for i in range(5)]
        for i in range(1, 5):
            axs[i].sharey(axs[0])
        if r:
            for i in range(5):
                axs[i].sharex(rows[0][i])
        draw_tracks(axs, w, s, e, win, first=(r == 0))
        rows.append(axs)
    for i in range(2):
        rows[0][i].locator_params(axis="x", nbins=3)
    rows[0][2].locator_params(axis="x", nbins=2)
    # depth break marks between the two intervals
    for ax_top, ax_bot in zip(rows[0], rows[1]):
        kw = dict(transform=ax_top.transAxes, color="k", clip_on=False, lw=0.6)
        ax_top.plot([-0.06, 0.06], [-0.012, 0.012], **kw)
        ax_top.plot([0.94, 1.06], [-0.012, 0.012], **kw)
    fig.supylabel(f"Depth of well {w} (m)", x=0.012, fontsize=8)
    P.panel_label(rows[0][0], "a", x_pt=-30, y_pt=22)
    # b: class-wise envelope
    axb = fig.add_subplot(gs[0, 1])
    st = P.class_envelope_stats(P.TRAIN_WELLS)
    x = P.bin_centers()
    axb.axvspan(win["sample_start"], win["sample_end"], color="0.88", lw=0, zorder=0)
    for c in P.CLASSES:
        col = P.CLASS_COLORS[c] if c != "Moderate" else "#E0A800"  # darker yellow line for contrast
        axb.fill_between(x, st[c]["q25"], st[c]["q75"], color=P.CLASS_COLORS[c], alpha=0.28, lw=0)
        axb.plot(x, st[c]["median"], color=col, lw=1.0, label=f"{c} (n = {st[c]['n']:,})")
    axb.set_xlim(0, P.N_KEEP)
    axb.set_xlabel("VDL time sample")
    axb.set_ylabel("Log envelope (wells A–F)")
    axb.legend(loc="lower right", handlelength=1.4, borderaxespad=0.2)
    P.panel_label(axb, "b", x_pt=-28, y_pt=22)
    print(f"VDL window (red dashed lines in a, grey band in b): samples {win['sample_start']}-{win['sample_end']}")

    P.save(fig, "fig_02_data_vdl")


if __name__ == "__main__":
    main()
