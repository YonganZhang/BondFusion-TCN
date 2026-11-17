"""Fig. 6 — the minority Good class on the blind wells.

a  well F, the Good zone ± 10 m: CBL, raw VDL, reference label and 10-seed-ensemble predictions of four models
   (columns 1-4: ModernTCN Curves, ModernTCN Curves + VDL, Mean pooling, TimeMixer Curves; key in the second
   legend row); black dotted line = top of the labelled Good zone
b  same for well E
c  Good-class F1 per seed (dots), seed mean (bar) and 10-seed ensemble (diamond) for the 16 architecture models
   (Curves and Curves + VDL of one architecture adjacent, order of Fig. 3) and the encoder designs
d  Good-class precision vs recall, one point per seed, 16 architecture models, coloured by input set
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import _paper_data as P  # noqa: E402

MARGIN_M = 10.0
STRIP_MODELS = ["ModernTCN", "ModernTCN +VDL", "VDL-ModernTCN (mean pool)", "TimeMixer"]
STRIP_HEADERS = ["Interpreted\nlabel", "1", "2", "3", "4"]  # model columns numbered, key in the figure legend
STRIP_KEY = ["ModernTCN, Curves", "BondFusion-TCN", "Mean pooling", "TimeMixer, Curves"]


def window(w):
    z = P.good_zones(w)
    if len(z) != 1:
        raise ValueError(f"{w}: expected one Good zone, found {len(z)}")
    z = z[0]
    d = P.well(w)
    top, base = z["top_m"] - MARGIN_M, z["base_m"] + MARGIN_M
    s, e = int(np.searchsorted(d.depth, top)), int(np.searchsorted(d.depth, base, side="right"))
    return z, s, e, top, base


def pieces(w, s, e):
    """Contiguous (segment-internal) index ranges inside [s, e)."""
    return [(max(a, s), min(b, e)) for a, b in P.well(w).segments if a < e and b > s]


def draw_well(fig, sub, w, letter):
    z, s, e, top, base = window(w)
    d = P.well(w)
    preds = [d.y] + [P.ensemble_pred(m)[w] for m in STRIP_MODELS]
    axs = [fig.add_subplot(sub[0, i]) for i in range(2 + len(preds))]
    for ax in axs[1:]:
        ax.sharey(axs[0])
    cbl = P.curves(w)["CBL"].to_numpy()
    v = P.vdl_raw(w)
    clip = float(np.percentile(np.abs(v[s:e]), 99))
    for a, b in pieces(w, s, e):
        dep = d.depth[a:b]
        ext = [dep[-1] + 0.05, dep[0] - 0.05]
        axs[0].plot(cbl[a:b], dep, color=P.PAL10[8], lw=0.7)
        axs[1].imshow(v[a:b], aspect="auto", cmap="gray", vmin=-clip, vmax=clip, interpolation="antialiased",
                      extent=[0, P.N_KEEP, *ext], rasterized=True)
        for ax, p in zip(axs[2:], preds):
            ax.imshow(p[a:b, None], aspect="auto", cmap=P.class_cmap(), vmin=-0.5, vmax=2.5,
                      interpolation="nearest", extent=[0, 1, *ext])
    axs[0].set_ylim(base, top)  # depth increases downward
    axs[0].set_xlim(left=0)
    for ax in axs[1:]:
        ax.set_xlim(*(ax.get_xlim() if ax is axs[1] else (0, 1)))
    axs[1].set_xlim(0, P.N_KEEP)
    axs[0].set_ylabel(f"Depth of well {w} (m)")
    axs[0].set_xlabel("CBL")
    axs[0].locator_params(axis="x", nbins=3)
    axs[1].set_xlabel("VDL time sample")
    axs[1].set_xticks([500, 1500])  # no tick at 0: keeps clear of the last CBL tick label
    for ax in axs:
        ax.spines["right"].set_visible(True)
        ax.spines["top"].set_visible(True)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        if ax is not axs[0]:
            ax.tick_params(axis="y", labelleft=False)
    for ax, lab in zip(axs[2:], STRIP_HEADERS):
        ax.set_xticks([0.5], [lab], linespacing=0.95)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.axhline(z["top_m"], color="k", lw=0.5, ls=":")
    pcs = pieces(w, s, e)
    nodata = [(top, d.depth[pcs[0][0]])] + [(d.depth[b - 1], d.depth[a2]) for (_, b), (a2, _) in zip(pcs, pcs[1:])]
    nodata += [(d.depth[pcs[-1][1] - 1], base)]
    for g0, g1 in nodata:
        if g1 - g0 > 0.2:
            for ax in axs:
                ax.axhspan(g0, g1, color="0.9", lw=0, zorder=0)
            axs[1].text(P.N_KEEP / 2, (g0 + g1) / 2, "No data", ha="center", va="center", color="0.35")
    P.panel_label(axs[0], letter, x_pt=-34, y_pt=26)
    return z


def main():
    P.setup_style()
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = GridSpec(1, 2, figure=fig, wspace=0.26, left=0.075, right=0.985, top=0.855, bottom=0.535)
    gs2 = GridSpec(1, 1, figure=fig, left=0.255, right=0.975, top=0.45, bottom=0.065)
    wr = [0.95, 1.25, 0.62] + [0.36] * 4
    zF = draw_well(fig, gs[0, 0].subgridspec(1, 7, width_ratios=wr, wspace=0.12), "F", "a")
    zE = draw_well(fig, gs[0, 1].subgridspec(1, 7, width_ratios=wr, wspace=0.12), "E", "b")
    print("Good zones:", zF, zE)

    # c: Good F1 per held-out well & pooled
    extra = [
        "w/o CBL-gated bridge",
        "w/o Hilbert envelope",
        "VDL-ModernTCN (mean pool)",
        "ModernTCN +VDL (raw)",
        "2D-CNN +VDL",
        "w/o VDL (Curve-Prior only)",
    ]
    order = [m for b in P.BACKBONES for m in (b, P.vdl(b))] + extra
    disp_map = {
        "VDL-ModernTCN (mean pool)": "Mean pooling (b = 100)",
        "ModernTCN +VDL (raw)": "Raw VDL + ModernTCN",
        "w/o VDL (Curve-Prior only)": "Curve-Prior only (w/o VDL)",
    }
    vals = {m: P.seed_values(m, "f1_Good") for m in order}
    sub = gs2[0].subgridspec(1, 2, width_ratios=[1.25, 1], wspace=0.25)
    axc = fig.add_subplot(sub[0])
    rng = np.random.default_rng(0)
    for i, m in enumerate(order):
        c = P.INPUT_COLORS[P.input_set(m)]
        axc.scatter(vals[m], i + rng.uniform(-0.15, 0.15, len(vals[m])), s=6, color=c, alpha=0.75, lw=0, zorder=3)
        axc.plot([vals[m].mean()] * 2, [i - 0.32, i + 0.32], color=c, lw=1.2, zorder=4)
        axc.plot(P.ensemble_metrics(m)["f1_Good"], i, marker="D", ms=4.0, mfc=c, mec="k", mew=0.5,
                 ls="none", zorder=5)
    for i in range(1, 2 * len(P.BACKBONES) + 1, 2):  # separators between architecture pairs and variants
        axc.axhline(i + 0.5, color="0.85", lw=0.4, zorder=0)
    axc.set_yticks(range(len(order)), [disp_map.get(m, P.display_name(m)) for m in order])
    axc.tick_params(axis="y", labelsize=6)
    axc.set_ylim(len(order) - 0.5, -0.5)
    axc.set_xlim(-0.02, 1.03)
    axc.set_xlabel("Good-class F1 (wells A–F)")
    axc.grid(axis="x", lw=0.3, alpha=0.4)
    P.panel_label(axc, "c", x_pt=-128, y_pt=4)

    # d: precision vs recall (well dots + pooled diamonds)
    axd = fig.add_subplot(sub[1])
    for m in P.BACKBONES + [P.vdl(b) for b in P.BACKBONES]:
        iset = P.input_set(m)
        axd.scatter(P.seed_values(m, "recall_Good"), P.seed_values(m, "precision_Good"), s=7,
                    color=P.INPUT_COLORS[iset], alpha=0.35, lw=0, zorder=2)
        em = P.ensemble_metrics(m)
        axd.plot(em["recall_Good"], em["precision_Good"], marker="D", ms=4.2, mfc=P.INPUT_COLORS[iset],
                 mec="k", mew=0.5, ls="none", zorder=4)
    axd.set_xlabel("Good-class recall (wells A–F)")
    axd.set_ylabel("Good-class precision (wells A–F)")
    axd.set_xlim(-0.03, 1.03)
    axd.set_ylim(-0.03, 1.03)
    axd.grid(lw=0.3, alpha=0.4)
    P.panel_label(axd, "d", x_pt=-34, y_pt=4)

    handles = [Patch(color=P.CLASS_COLORS[c], label=c) for c in P.CLASSES]
    handles += [Line2D([], [], marker="o", ls="none", color=P.INPUT_COLORS[k], ms=3.5, label=k)
                for k in ("Curves", "Curves + VDL")]
    handles += [Line2D([], [], color="0.3", lw=1.2, label="Well mean"),
                Line2D([], [], marker="D", ls="none", mfc="0.55", mec="k", mew=0.5, ms=4.0, label="Pooled")]
    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.998), ncol=7, handletextpad=0.3,
                     columnspacing=1.1, handlelength=1.2)
    fig.add_artist(leg)
    key = [Line2D([], [], marker=f"${k}$", ls="none", color="k", ms=5, label=lab)
           for k, lab in zip(STRIP_HEADERS[1:], STRIP_KEY)]
    fig.legend(handles=key, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4, handletextpad=0.2,
               columnspacing=1.6, handlelength=1.0)
    P.save(fig, "fig_06_good_failure")


if __name__ == "__main__":
    main()
