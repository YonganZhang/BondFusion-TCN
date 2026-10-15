"""Numerical interval-selection helper extracted unchanged from the Fig. 2 script."""
from __future__ import annotations

import numpy as np
import _paper_data as P

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
