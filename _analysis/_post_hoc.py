"""Post hoc analyses of the frozen v03 blind-well predictions, added after the second P5.4 review (no training).

They ask what the pre-specified comparison cannot: whether the kappa gain from VDL reflects better discrimination
or a shift in how often Good is predicted, how much of the "eight of eight" survives multiplicity, shared test
segments and a per-well view, and how the encoder designs compare directly with block features. Everything here is
labelled post hoc in the paper.

Bootstrap: the frozen significance files (outputs/v03/report/significance_*.json) were written by step_08 with
--n-boot 1000 and rng seed 0. `boot_indices()` regenerates exactly those draws, and `verify_frozen()` checks that
the recomputed VDL-pair intervals equal the frozen ones before anything new is derived from the same draws.

Entry point: compute() -> dict, stored under "post_hoc" in paper_numbers.json by build_paper_numbers.py.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

import _paper_data as P

N_BOOT = 1000
ALPHA = 0.05
HALF_WINDOW = 128  # samples (12.8 m): half of the 256-sample model window
ENCODERS = ["VDL-ModernTCN (mean pool)", "VDL-ModernTCN (static query)", "VDL-ModernTCN", "VDL-ModernTCN + Reformer"]
BLOCKS = "VDL-ModernTCN w/o waveform encoder"


def _probs(m: str) -> np.ndarray:
    return np.concatenate([P.ensemble_probs(m)[w] for w in P.TEST_WELLS])


@functools.lru_cache(maxsize=None)
def boot_indices() -> list[np.ndarray]:
    """step_08's draws: segments resampled with replacement inside each blind well (E then F)."""
    rng = np.random.default_rng(0)
    off = np.cumsum([0] + [len(P.well(w).y) for w in P.TEST_WELLS])
    draws = []
    for _ in range(N_BOOT):
        parts = []
        for k, w in enumerate(P.TEST_WELLS):
            segs = P.well(w).segments
            for i in rng.integers(0, len(segs), len(segs)):
                a, b = segs[i]
                parts.append(np.arange(off[k] + a, off[k] + b))
        draws.append(np.concatenate(parts))
    return draws


def _boot_diff(pa: np.ndarray, pb: np.ndarray) -> dict:
    """Ensemble difference a - b in macro-F1 and kappa with percentile intervals over the frozen draws."""
    y = P.y_true()
    fa, ka = P.compute_f1_kappa(y, pa)
    fb, kb = P.compute_f1_kappa(y, pb)
    d = np.array([[x - z for x, z in zip(P.compute_f1_kappa(y[i], pa[i]), P.compute_f1_kappa(y[i], pb[i]))]
                  for i in boot_indices()])
    return {"d_macro_f1": fa - fb, "ci_macro_f1": np.percentile(d[:, 0], [2.5, 97.5]).tolist(),
            "d_kappa": ka - kb, "ci_kappa": np.percentile(d[:, 1], [2.5, 97.5]).tolist(),
            "p_le_0_kappa": float(np.mean(d[:, 1] <= 0)), "boot_kappa": d[:, 1]}


def verify_frozen() -> None:
    sig = P.sig_vdl_pairs()
    for b in P.BACKBONES:
        r = _boot_diff(_probs(P.vdl(b)).argmax(1), _probs(b).argmax(1))
        ref = sig[f"{b} +VDL vs {b}"]
        for got, want in [(r["ci_kappa"], ref["boot_kappa_diff_ci95"]), (r["ci_macro_f1"], ref["boot_macro_f1_diff_ci95"])]:
            if not np.allclose(got, want, atol=1e-12):
                raise ValueError(f"{b}: bootstrap draws do not reproduce the frozen file ({got} vs {want})")


def holm(pvals: dict[str, float]) -> list[str]:
    """Names rejected by Holm's step-down procedure at ALPHA."""
    order = sorted(pvals, key=lambda k: pvals[k])
    rejected = []
    for i, name in enumerate(order):
        if pvals[name] > ALPHA / (len(order) - i):
            break
        rejected.append(name)
    return rejected


def decision_shift() -> dict:
    """Kappa of the ensembles when Good may not be predicted, threshold-free Good-vs-Moderate ranking, Poor-vs-bonded."""
    y = P.y_true()
    bonded = y > 0
    rows = {}
    for b in P.BACKBONES:
        rec = {}
        for tag, m in (("curves", b), ("vdl", P.vdl(b))):
            pr = _probs(m)
            score = pr[bonded, 2] / (pr[bonded, 1] + pr[bonded, 2])
            rec[tag] = {"kappa": cohen_kappa_score(y, pr.argmax(1)),
                        "kappa_no_good": cohen_kappa_score(y, pr[:, :2].argmax(1)),
                        "auc_good_vs_moderate": roc_auc_score(y[bonded] == 2, score),
                        "kappa_poor_vs_bonded": cohen_kappa_score(y > 0, pr.argmax(1) > 0),
                        "share_predicted_good": float(np.mean(pr.argmax(1) == 2))}
        rows[b] = rec
    cur_nog = [r["curves"]["kappa_no_good"] for r in rows.values()]
    vdl_k = [r["vdl"]["kappa"] for r in rows.values()]
    return {"per_architecture": rows,
            "curves_no_good_kappa_range": [min(cur_nog), max(cur_nog)], "vdl_kappa_range": [min(vdl_k), max(vdl_k)],
            "n_vdl_kappa_above_curves_no_good": int(sum(r["vdl"]["kappa"] > r["curves"]["kappa_no_good"]
                                                        for r in rows.values())),
            "n_auc_up": int(sum(r["vdl"]["auc_good_vs_moderate"] > r["curves"]["auc_good_vs_moderate"]
                                for r in rows.values())),
            "auc_down": [b for b, r in rows.items()
                         if r["vdl"]["auc_good_vs_moderate"] < r["curves"]["auc_good_vs_moderate"]],
            "n_poor_vs_bonded_up": int(sum(r["vdl"]["kappa_poor_vs_bonded"] > r["curves"]["kappa_poor_vs_bonded"]
                                           for r in rows.values())),
            "poor_vs_bonded_down": [b for b, r in rows.items()
                                    if r["vdl"]["kappa_poor_vs_bonded"] < r["curves"]["kappa_poor_vs_bonded"]]}


def multiplicity_and_pooling() -> dict:
    sig = P.sig_vdl_pairs()
    p_two = {b: min(1.0, 2 * min(r["boot_kappa_p_le_0"], 1 - r["boot_kappa_p_le_0"]))
             for b in P.BACKBONES for r in [sig[f"{b} +VDL vs {b}"]]}
    excl = [b for b in P.BACKBONES if sig[f"{b} +VDL vs {b}"]["boot_kappa_diff_ci95"][0] > 0]
    boots = np.array([_boot_diff(_probs(P.vdl(b)).argmax(1), _probs(b).argmax(1))["boot_kappa"] for b in P.BACKBONES])
    mean_d = float(np.mean([sig[f"{b} +VDL vs {b}"]["ensemble_kappa_diff"] for b in P.BACKBONES]))
    return {"ci_excludes_zero": excl, "p_two_sided_kappa": p_two, "holm_rejected": holm(p_two),
            "mean_d_kappa_over_architectures": mean_d,
            "mean_d_kappa_ci": np.percentile(boots.mean(0), [2.5, 97.5]).tolist(),
            "n_test_segments": {w: len(P.well(w).segments) for w in P.TEST_WELLS},
            "n_test_segments_total": sum(len(P.well(w).segments) for w in P.TEST_WELLS)}


def per_well() -> dict:
    out = {}
    for w in P.TEST_WELLS:
        ens_up = [b for b in P.BACKBONES
                  if P.ensemble_metrics(P.vdl(b), w)["kappa"] > P.ensemble_metrics(b, w)["kappa"]]
        seed_up = [b for b in P.BACKBONES
                   if P.seed_values(P.vdl(b), "kappa", w).mean() > P.seed_values(b, "kappa", w).mean()]
        out[w] = {"ensemble_kappa_up": ens_up, "n_ensemble_kappa_up": len(ens_up),
                  "seed_mean_kappa_up": seed_up, "n_seed_mean_kappa_up": len(seed_up)}
    return out


def bootstrap_good_coverage() -> dict:
    """How many draws contain no Good sample, or Good samples from one blind well only."""
    y = P.y_true()
    n_e = len(P.well(P.TEST_WELLS[0]).y)
    none = one = 0
    for i in boot_indices():
        g = i[y[i] == 2]
        none += len(g) == 0
        one += len(g) > 0 and (bool(np.all(g < n_e)) or bool(np.all(g >= n_e)))
    return {"share_draws_without_good": none / N_BOOT, "share_draws_good_from_one_well_only": one / N_BOOT}


def direct_design_comparisons() -> dict:
    """Each encoder design minus block features (same hyper-parameters) and minus ModernTCN +VDL (tuned)."""
    out = {}
    for m in ENCODERS:
        for ref in (BLOCKS, P.REF_VDL):
            r = _boot_diff(_probs(m).argmax(1), _probs(ref).argmax(1))
            r.pop("boot_kappa")
            out[f"{P.display_name(m)} vs {P.display_name(ref)}"] = r
    return out


def label_run_agreement() -> dict:
    """Majority vote of the ensemble over each constant-label run of the blind wells (the interpreters' units)."""
    runs = []
    off = 0
    for w in P.TEST_WELLS:
        y = P.well(w).y
        cut = np.flatnonzero(np.diff(y)) + 1
        runs += [(off + a, off + b) for a, b in zip(np.r_[0, cut], np.r_[cut, len(y)])]
        off += len(y)
    yt = P.y_true()
    res = {}
    for b in P.BACKBONES:
        res[b] = {}
        for tag, m in (("curves", b), ("vdl", P.vdl(b))):
            pred = _probs(m).argmax(1)
            res[b][tag] = int(sum(np.bincount(pred[a:c], minlength=3).argmax() == yt[a] for a, c in runs))
    return {"n_runs": len(runs), "n_runs_by_class": dict(zip(P.CLASSES, map(int, np.bincount([yt[a] for a, _ in runs],
                                                                                              minlength=3)))),
            "runs_correct": res,
            "n_vdl_more_runs_correct": int(sum(r["vdl"] > r["curves"] for r in res.values())),
            "n_vdl_fewer_runs_correct": int(sum(r["vdl"] < r["curves"] for r in res.values()))}


def good_context() -> dict:
    """Where Good samples sit relative to segment ends, per well, and which raw label names form Good."""
    out = {}
    for w in P.ALL_WELLS:
        d = P.well(w)
        near = total = 0
        for a, b in d.segments:
            idx = np.flatnonzero(d.y[a:b] == 2)
            total += len(idx)
            near += int(np.sum(np.minimum(idx, b - a - 1 - idx) < HALF_WINDOW))
        names = P.curves(w)["label_raw"].value_counts().to_dict()
        out[w] = {"n_good": total, "share_good_within_half_window_of_segment_end": near / total if total else None,
                  "good_label_names": {k: int(v) for k, v in names.items() if k in ("胶结好", "胶结良好", "胶结优")}}
    return out


def confusion_extras() -> dict:
    """Pooled (summed over architectures) Poor -> Moderate shares, curves vs +VDL."""
    y = P.y_true()
    res = {}
    for tag, models in (("curves", P.BACKBONES), ("vdl", [P.vdl(b) for b in P.BACKBONES])):
        cm = sum(pd.crosstab(pd.Categorical(y, [0, 1, 2]), pd.Categorical(_probs(m).argmax(1), [0, 1, 2]),
                             dropna=False).to_numpy() for m in models)
        res[tag] = {"poor_to_moderate": cm[0, 1] / cm[0].sum(), "moderate_to_poor": cm[1, 0] / cm[1].sum()}
    return res


def error_direction() -> dict:
    """Errors that overstate bond quality (predicted class above the label in the order Poor < Moderate < Good) and
    errors that understate it, per architecture and input, pooled and per held-out well; also Poor called bonded."""
    y = P.y_true()
    n_e = len(P.well(P.TEST_WELLS[0]).y)
    parts = {"pooled": slice(None), P.TEST_WELLS[0]: slice(0, n_e), P.TEST_WELLS[1]: slice(n_e, None)}
    rows = {}
    for b in P.BACKBONES:
        rows[b] = {}
        for tag, m in (("curves", b), ("vdl", P.vdl(b))):
            pred = _probs(m).argmax(1)
            rows[b][tag] = {k: {"overstate": int(np.sum(pred[s_] > y[s_])), "understate": int(np.sum(pred[s_] < y[s_])),
                                "poor_called_bonded": int(np.sum((y[s_] == 0) & (pred[s_] > 0)))}
                            for k, s_ in parts.items()}
    tot = {tag: {k: sum(r[tag]["pooled"][k] for r in rows.values()) for k in ("overstate", "understate",
                                                                            "poor_called_bonded")}
           for tag in ("curves", "vdl")}
    nog = {b: int(np.sum(_probs(b)[:, :2].argmax(1) > y)) for b in P.BACKBONES}
    ci = {}  # share of samples with an overstating error, VDL minus curves, over the frozen bootstrap draws
    for b in P.BACKBONES:
        oc = (_probs(b).argmax(1) > y).astype(float)
        ov = (_probs(P.vdl(b)).argmax(1) > y).astype(float)
        d = np.array([ov[i].mean() - oc[i].mean() for i in boot_indices()])
        ci[b] = {"d_share": float(ov.mean() - oc.mean()), "ci": np.percentile(d, [2.5, 97.5]).tolist()}
    count = lambda key, part: [b for b, r in rows.items() if r["vdl"][part][key] < r["curves"][part][key]]
    return {"n_samples": int(len(y)), "per_architecture": rows, "totals": tot,
            "overstate_change": tot["vdl"]["overstate"] / tot["curves"]["overstate"] - 1,
            "understate_ratio": tot["vdl"]["understate"] / tot["curves"]["understate"],
            "total_error_change": (tot["vdl"]["overstate"] + tot["vdl"]["understate"])
                                  / (tot["curves"]["overstate"] + tot["curves"]["understate"]) - 1,
            "poor_called_bonded_change": tot["vdl"]["poor_called_bonded"] / tot["curves"]["poor_called_bonded"] - 1,
            "overstate_change_range": [min(r["vdl"]["pooled"]["overstate"] / r["curves"]["pooled"]["overstate"] - 1
                                           for r in rows.values()),
                                       max(r["vdl"]["pooled"]["overstate"] / r["curves"]["pooled"]["overstate"] - 1
                                           for r in rows.values())],
            "overstate_down": {part: count("overstate", part) for part in parts},
            "understate_up": {part: [b for b, r in rows.items()
                                     if r["vdl"][part]["understate"] > r["curves"][part]["understate"]] for part in parts},
            "poor_called_bonded_down": {part: count("poor_called_bonded", part) for part in parts},
            "curves_no_good_overstate": nog, "overstate_share_diff_ci": ci,
            "n_overstate_ci_below_zero": int(sum(v["ci"][1] < 0 for v in ci.values())),
            "n_vdl_overstate_below_no_good": int(sum(rows[b]["vdl"]["pooled"]["overstate"] < nog[b]
                                                     for b in P.BACKBONES))}


def boundaries_and_costs() -> dict:
    """Errors at the two decision boundaries, both directions, per architecture (pooled wells):
    remediation boundary Poor | bonded (missed poor bond = Poor called bonded; false alarm = bonded called Poor) and
    grading boundary Moderate | Good. Break-even cost ratio = the cost of a missed poor sample relative to a false
    alarm above which VDL lowers the total cost at the remediation boundary (None when one input is better at every
    ratio). Also the curves-only no-Good baseline and a class-prior sensitivity (error rates re-weighted to the
    training-well class shares)."""
    y = P.y_true()
    n_tr = np.bincount(np.concatenate([P.well(w).y for w in P.TRAIN_WELLS]), minlength=3)
    pi_tr = n_tr / n_tr.sum()
    rows, tot = {}, {"curves": np.zeros(4), "vdl": np.zeros(4), "curves_no_good": np.zeros(4)}
    rew = {"curves": [], "vdl": []}
    for b in P.BACKBONES:
        rec = {}
        for tag, pred in (("curves", _probs(b).argmax(1)), ("vdl", _probs(P.vdl(b)).argmax(1)),
                          ("curves_no_good", _probs(b)[:, :2].argmax(1))):
            v = np.array([np.sum((y == 0) & (pred > 0)), np.sum((y > 0) & (pred == 0)),
                          np.sum((y == 1) & (pred == 2)), np.sum((y == 2) & (pred == 1))])
            rec[tag] = dict(zip(["missed_poor", "false_alarm_poor", "moderate_called_good", "good_called_moderate"],
                                map(int, v)))
            tot[tag] += v
            if tag in rew:  # error rate per true class, re-weighted to the training class shares
                err_c = np.array([np.mean(pred[y == c] != c) for c in range(3)])
                over_c = np.array([np.mean(pred[y == c] > c) for c in range(3)])
                under_c = np.array([np.mean(pred[y == c] < c) for c in range(3)])
                rew[tag].append([float(pi_tr @ err_c), float(pi_tr @ over_c), float(pi_tr @ under_c)])
        dm = rec["vdl"]["missed_poor"] - rec["curves"]["missed_poor"]
        df = rec["vdl"]["false_alarm_poor"] - rec["curves"]["false_alarm_poor"]
        rec["remediation_errors_change"] = (dm + df) / (rec["curves"]["missed_poor"] + rec["curves"]["false_alarm_poor"])
        rec["break_even_ratio"] = (df / -dm) if (dm < 0 < df) else None
        rec["vdl_better_at_any_ratio"] = bool(dm <= 0 and df <= 0)
        rec["vdl_worse_at_any_ratio"] = bool(dm >= 0 and df >= 0)
        # threshold-free and matched operating point at the remediation boundary (curves-only threshold on p(Poor)
        # re-chosen with the held-out labels so that its false alarms equal those of the VDL ensemble: an oracle that
        # favours the curves-only model)
        pc, pv = _probs(b)[:, 0], _probs(P.vdl(b))[:, 0]
        rec["auc_poor_vs_bonded"] = {"curves": roc_auc_score(y == 0, pc), "vdl": roc_auc_score(y == 0, pv)}
        fa_v = rec["vdl"]["false_alarm_poor"]
        t = np.sort(pc[y > 0])[::-1][fa_v - 1] if fa_v > 0 else np.inf
        rec["curves_missed_poor_at_vdl_false_alarms"] = int(np.sum((y == 0) & (pc < t)))
        rows[b] = rec
    keys = ["missed_poor", "false_alarm_poor", "moderate_called_good", "good_called_moderate"]
    totals = {t: dict(zip(keys, map(int, v))) for t, v in tot.items()}
    dm = totals["vdl"]["missed_poor"] - totals["curves"]["missed_poor"]
    df = totals["vdl"]["false_alarm_poor"] - totals["curves"]["false_alarm_poor"]
    be = {b: r["break_even_ratio"] for b, r in rows.items() if r["break_even_ratio"] is not None}
    rw = {t: np.mean(v, 0) for t, v in rew.items()}
    ds = decision_shift()["per_architecture"]
    total_err = lambda t: sum(int(np.sum(_probs(b if t != "vdl" else P.vdl(b)).argmax(1) != y)) for b in P.BACKBONES)
    nog_err = sum(int(np.sum(_probs(b)[:, :2].argmax(1) != y)) for b in P.BACKBONES)
    return {"per_architecture": rows, "totals": totals,
            "remediation_errors_change_pooled": (dm + df) / (totals["curves"]["missed_poor"]
                                                             + totals["curves"]["false_alarm_poor"]),
            "n_remediation_errors_down": int(sum(r["remediation_errors_change"] < 0 for r in rows.values())),
            "break_even_ratio_pooled": df / -dm,
            "break_even_ratio_range": [min(be.values()), max(be.values())] if be else None,
            "vdl_better_at_any_ratio": [b for b, r in rows.items() if r["vdl_better_at_any_ratio"]],
            "vdl_worse_at_any_ratio": [b for b, r in rows.items() if r["vdl_worse_at_any_ratio"]],
            "moderate_good_break_even_pooled": (totals["vdl"]["good_called_moderate"]
                                                - totals["curves"]["good_called_moderate"])
                                               / (totals["curves"]["moderate_called_good"]
                                                  - totals["vdl"]["moderate_called_good"]),
            "no_good_mean_kappa": float(np.mean([r["curves"]["kappa_no_good"] for r in ds.values()])),
            "vdl_mean_kappa": float(np.mean([r["vdl"]["kappa"] for r in ds.values()])),
            "curves_mean_kappa": float(np.mean([r["curves"]["kappa"] for r in ds.values()])),
            "mean_auc_curves": float(np.mean([r["curves"]["auc_good_vs_moderate"] for r in ds.values()])),
            "mean_auc_vdl": float(np.mean([r["vdl"]["auc_good_vs_moderate"] for r in ds.values()])),
            "total_errors": {"curves": total_err("curves"), "vdl": total_err("vdl"), "curves_no_good": nog_err},
            "n_vdl_fewer_missed_poor_than_rethresholded_curves": int(sum(
                r["vdl"]["missed_poor"] < r["curves_missed_poor_at_vdl_false_alarms"] for r in rows.values())),
            "auc_poor_vs_bonded_range": [min(min(r["auc_poor_vs_bonded"].values()) for r in rows.values()),
                                         max(max(r["auc_poor_vs_bonded"].values()) for r in rows.values())],
            "training_prior": pi_tr.tolist(),
            "reweighted_to_training_prior": {t: {"error_rate": float(v[0]), "overstate_rate": float(v[1]),
                                                 "understate_rate": float(v[2])} for t, v in rw.items()},
            "reweighted_error_change": float(rw["vdl"][0] / rw["curves"][0] - 1),
            "reweighted_overstate_change": float(rw["vdl"][1] / rw["curves"][1] - 1),
            "reweighted_understate_ratio": float(rw["vdl"][2] / rw["curves"][2])}


def final_epochs_vs_warmup() -> dict:
    warm = int(P.PROTOCOL["train"]["warmup_epochs"])
    ep = {m: int(P.hpo_best(m)["stop_epoch"]) for m in P.TUNED_MODELS}
    return {"warmup_epochs": warm, "final_epochs": ep}


def compute() -> dict:
    verify_frozen()
    return {"n_boot": N_BOOT, "decision_shift": decision_shift(), "multiplicity": multiplicity_and_pooling(),
            "per_well": per_well(), "bootstrap_good_coverage": bootstrap_good_coverage(),
            "direct_design_comparisons": direct_design_comparisons(), "label_runs": label_run_agreement(),
            "good_context": good_context(), "confusion_extras": confusion_extras(), "error_direction": error_direction(),
            "boundaries_and_costs": boundaries_and_costs(),
            "final_epochs": final_epochs_vs_warmup()}


if __name__ == "__main__":
    import json
    print(json.dumps(compute(), indent=1, ensure_ascii=False, default=float))
