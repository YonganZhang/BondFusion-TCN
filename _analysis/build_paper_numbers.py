"""Build the P04 (v03) paper tables and the single numbers file, after verifying every recomputed number
against the existing v03 summaries.

Outputs (R/_outputs/p04_v03_paper/):
  table_01_wells.{csv,tex}  table_02_main.{csv,tex} (tex = slim main-text table)
  table_02_paired_accuracy.csv (unrounded mean-accuracy comparisons)  table_03_designs.csv
  table_S1_hyperparameters.csv  table_S1_search_space.md  table_S2_per_well.csv  table_S3_significance.csv
  paper_numbers.json  verification.json
Verification failures raise (exit != 0); nothing is silently replaced.

Usage: python scripts/rebuild_paper_analysis.py numbers
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _paper_data as P
import _post_hoc

TOL = 1e-6
METRICS = ["accuracy", "macro_f1", "kappa", "mcc"]
CLS_KEYS = [f"{m}_{c}" for c in P.CLASSES for m in ("f1", "precision", "recall")]
CHECKS: list[dict] = []


def check(name: str, got: float, ref: float, tol: float = TOL):
    d = abs(float(got) - float(ref))
    CHECKS.append({"check": name, "got": float(got), "ref": float(ref), "abs_diff": d, "ok": d < tol})


def r(x, nd=6):
    return None if x is None else round(float(x), nd)


def pm(m, s, nd=3):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


# ------------------------------------------------------------------------------------------------
# Verification
# ------------------------------------------------------------------------------------------------
def verify():
    S = P.summary_merged()
    spw = P.summary_per_well().set_index(["model", "well"])
    yt = P.y_true()
    for m in P.ALL_MODELS:
        e = P.ensemble_metrics(m)
        pred = np.concatenate([P.ensemble_pred(m)[w] for w in P.TEST_WELLS])
        f1, k = P.compute_f1_kappa(yt, pred)
        check(f"{m}: ens macro_f1 (step_08 formula) vs summary", f1, S.loc[m, "ens_macro_f1"])
        check(f"{m}: ens kappa (step_08 formula) vs summary", k, S.loc[m, "ens_kappa"])
        for key in METRICS + CLS_KEYS:
            check(f"{m}: ens {key} vs summary", e[key], S.loc[m, f"ens_{key}"])
        for key in METRICS + CLS_KEYS:
            v = P.seed_values(m, key)
            check(f"{m}: seed mean {key} vs summary", v.mean(), S.loc[m, f"{key}_mean"])
            check(f"{m}: seed sd {key} vs summary", v.std(ddof=1), S.loc[m, f"{key}_std"])
        for w in P.TEST_WELLS:
            for key in METRICS + [f"f1_{c}" for c in P.CLASSES]:
                v = P.seed_values(m, key, w)
                check(f"{m}/{w}: seed mean {key} vs summary_per_well", v.mean(), spw.loc[(m, w), f"{key}_mean"])
        # per-seed metrics.json vs recomputation from that seed's probs
        for i, pr in enumerate(P.seed_preds(m)):
            check(f"{m}: seed{i} accuracy probs vs metrics.json", np.mean(yt == pr), P.seed_metrics(m)[i]["merged"]["accuracy"])
            f1s, ks = P.compute_f1_kappa(yt, pr)
            check(f"{m}: seed{i} macro_f1 probs vs metrics.json", f1s, P.seed_metrics(m)[i]["merged"]["macro_f1"])
            check(f"{m}: seed{i} kappa probs vs metrics.json", ks, P.seed_metrics(m)[i]["merged"]["kappa"])
    for key, sig in [("vdl_pairs", P.sig_vdl_pairs()), ("vs_mean_pool", P.sig_vs_mean_pool())]:
        for name, rec in sig.items():
            a, b = rec["model"], rec["other"]
            ea, eb = P.ensemble_metrics(a), P.ensemble_metrics(b)
            check(f"sig {name}: ens dF1", ea["macro_f1"] - eb["macro_f1"], rec["ensemble_macro_f1_diff"])
            check(f"sig {name}: ens dkappa", ea["kappa"] - eb["kappa"], rec["ensemble_kappa_diff"])
            check(f"sig {name}: seed dF1", P.seed_values(a, "macro_f1").mean() - P.seed_values(b, "macro_f1").mean(),
                  rec["seed_mean_macro_f1_diff"])
            check(f"sig {name}: seed dkappa", P.seed_values(a, "kappa").mean() - P.seed_values(b, "kappa").mean(),
                  rec["seed_mean_kappa_diff"])
    hs = pd.read_csv(P.REPORT / "hpo_summary.csv").set_index("model")
    for m in P.TUNED_MODELS:
        check(f"{m}: hpo best value vs hpo_summary", P.hpo_best(m)["value"], hs.loc[m, "cv_macro_f1"])
    sc = P.screen_table()
    for sid in sc.index:
        v = np.array(sc.loc[sid, "seed_values"])
        check(f"screen {sid}: mean", v.mean(), sc.loc[sid, "cv_macro_f1_mean"])
        check(f"screen {sid}: sd (ddof=1)", v.std(ddof=1), sc.loc[sid, "cv_macro_f1_std"])
    hn = P.hpo_noise()
    check("hpo noise between-config sd vs run.log (4 dp)", round(hn["between_config_sd"], 4), 0.0119, 1e-9)
    check("hpo noise pooled seed sd vs run.log (4 dp)", round(hn["pooled_seed_sd"], 4), 0.0126, 1e-9)
    bad = [c for c in CHECKS if not c["ok"]]
    print(f"verification: {len(CHECKS)} checks, {len(bad)} failed, max |diff| = "
          f"{max(c['abs_diff'] for c in CHECKS):.2e}")
    return bad


# ------------------------------------------------------------------------------------------------
# LaTeX
# ------------------------------------------------------------------------------------------------
def tex_escape(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#").replace("±", r"$\pm$").replace("Δ", r"$\Delta$")
            .replace("–", "--").replace("—", "---"))


def write_tex(df: pd.DataFrame, path, caption: str, label: str, midrules: list[int] | None = None):
    const = [c for c in df.columns if df[c].nunique(dropna=False) == 1 and len(df) > 1]
    d = df.drop(columns=const)
    cols = list(d.columns)
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", f"\\caption{{{tex_escape(caption)}}}",
             f"\\label{{{label}}}", *([r"\resizebox{\textwidth}{!}{%"] if len(cols) > 8 else []),
             r"\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(tex_escape(c) for c in cols) + r" \\", r"\midrule"]
    for i, (_, row) in enumerate(d.iterrows()):
        if midrules and i in midrules:
            lines.append(r"\midrule")
        lines.append(" & ".join(tex_escape("" if pd.isna(v) else v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"] + (["}"] if len(cols) > 8 else []) + [r"\end{table*}", ""]
    if len(cols) > 8:
        lines.insert(0, r"% wide table scaled to \textwidth: requires \usepackage{graphicx}")
    if const:
        lines.insert(0, f"% columns constant across rows dropped: {', '.join(const)} = "
                        f"{', '.join(str(df[c].iloc[0]) for c in const)}")
    path.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------------------------------------
# Tables
# ------------------------------------------------------------------------------------------------
def well_facts(w):
    d = P.well(w)
    cnt = np.bincount(d.y, minlength=3)
    dd = np.diff(d.depth)
    return {"role": "training" if w in P.TRAIN_WELLS else "blind", "n": int(len(d.y)),
            "depth_top_m": float(d.depth[0]), "depth_base_m": float(d.depth[-1]),
            "median_step_m": float(np.round(np.median(dd), 4)), "n_duplicate_depths": int((dd == 0).sum()),
            "n_segments": len(d.segments), "logged_length_m": float(np.round(len(d.y) * np.median(dd), 1)),
            "n_label_runs": int((np.diff(d.y) != 0).sum() + 1),
            "segments_with_all_three_classes": int(sum(len(set(d.y[a:b])) == 3 for a, b in d.segments)),
            "counts": {c: int(cnt[k]) for k, c in enumerate(P.CLASSES)},
            "shares": {c: float(cnt[k] / cnt.sum()) for k, c in enumerate(P.CLASSES)}}


def table_wells(out):
    rows = []
    for w in P.ALL_WELLS:
        f = well_facts(w)
        rows.append({"Well": w, "Role": "Training" if f["role"] == "training" else "Held-out", "Samples": f"{f['n']:,}",
                     "Depth range (m)": f"{f['depth_top_m']:.1f}–{f['depth_base_m']:.1f}",
                     "Median step (m)": f"{f['median_step_m']:.1f}",
                     **{c: f"{f['counts'][c]:,} ({100 * f['shares'][c]:.1f}%)" for c in P.CLASSES},
                     "Segments": f["n_segments"]})
    df = pd.DataFrame(rows)
    df.to_csv(out / "table_01_wells.csv", index=False)
    write_tex(df, out / "table_01_wells.tex", "Wells, roles and class composition. Segments are split at depth "
              "gaps > 0.5 m; all wells have a median depth step of 0.1 m.", "tab:wells", midrules=[4])
    return df


def model_row(m):
    S = P.summary_merged()
    row = {"Model": m.replace(" +VDL", ""), "Input": P.input_set(m)}
    for k, lab in zip(METRICS, ["Accuracy", "Macro-F1", "Kappa", "MCC"]):
        v = P.seed_values(m, k)
        row[lab] = pm(v.mean(), v.std(ddof=1))
    for c in P.CLASSES:
        row[f"F1 {c}"] = f"{P.seed_values(m, f'f1_{c}').mean():.3f}"
    e = P.ensemble_metrics(m)
    row["Ens. accuracy"] = f"{e['accuracy']:.3f}"
    row["Ens. macro-F1"] = f"{e['macro_f1']:.3f}"
    row["Ens. kappa"] = f"{e['kappa']:.3f}"
    assert abs(e["kappa"] - S.loc[m, "ens_kappa"]) < TOL
    return row


def table_main(out):
    rows = [model_row(m) for b in P.BACKBONES for m in (b, P.vdl(b))]
    ref = P.reference_baselines()
    for k, disp in P.REFERENCE_DISPLAY.items():
        x = ref[k]
        name, inp = disp.split(", ") if ", " in disp else (disp, "Curves (CBL)")
        rows.append({"Model": name, "Input": inp, "Accuracy": f"{x['accuracy']:.3f}", "Macro-F1": f"{x['macro_f1']:.3f}",
                     "Kappa": f"{x['kappa']:.3f}", "MCC": f"{x['mcc']:.3f}",
                     **{f"F1 {c}": f"{x[f'f1_{c}']:.3f}" for c in P.CLASSES}, "Ens. accuracy": "—", "Ens. macro-F1": "—", "Ens. kappa": "—"})
    df = pd.DataFrame(rows)
    df.to_csv(out / "table_02_main.csv", index=False)  # full table: supplementary Table (tab:s-full)
    write_main_slim(out)
    return df


def paired_accuracy(b):
    """Same-architecture comparison, derived before rounding from individual-run means."""
    ac = float(P.seed_values(b, "accuracy").mean())
    av = float(P.seed_values(P.vdl(b), "accuracy").mean())
    return {"accuracy_curves_mean": ac, "accuracy_vdl_mean": av,
            "accuracy_gain_percentage_points": 100 * (av - ac),
            "relative_error_reduction_percent": 100 * (av - ac) / (1 - ac)}


def write_main_slim(out):
    """Main Table 2: mean accuracy and same-model error reduction, with agreement metrics."""
    f3 = lambda x: f"{x:.3f}"
    rows = []
    pairs = {b: paired_accuracy(b) for b in P.BACKBONES}
    pd.DataFrame.from_dict(pairs, orient="index").rename_axis("Architecture").to_csv(
        out / "table_02_paired_accuracy.csv")
    for b in P.BACKBONES:
        v = P.vdl(b)
        cells = [b]
        for m in (b, v):
            x = P.seed_values(m, "accuracy")
            cells.append(f"${100*x.mean():.1f}\\pm{100*x.std(ddof=1):.1f}$")
        cells.append(f"{pairs[b]['relative_error_reduction_percent']:.1f}".replace("-", "$-$"))
        for k in ("kappa", "macro_f1"):
            cells.extend(f3(P.seed_values(m, k).mean()) for m in (b, v))
        rows.append(" & ".join(cells) + r" \\")
    ref = P.reference_baselines()
    hc, hv = ref["HGB point-wise (GR,CBL,CCL)"], ref["HGB point-wise (GR,CBL,CCL + VDL blocks)"]
    cbl = ref["CBL two-threshold rule"]
    refs = []
    for name, c, v in [("Gradient boosting$^{a}$", hc, hv), ("CBL threshold rule$^{a}$", cbl, None)]:
        er = 100 * (v["accuracy"]-c["accuracy"])/(1-c["accuracy"]) if v else None
        cells = [name, f"{100*c['accuracy']:.1f}", f"{100*v['accuracy']:.1f}" if v else "--",
                 f"{er:.1f}" if v else "--"]
        for k in ("kappa", "macro_f1"):
            cells += [f3(c[k]), f3(v[k]) if v else "--"]
        refs.append(" & ".join(cells) + r" \\")
    tex = [r"\begin{table*}[!t]", r"\centering", r"\small",
           r"\caption{Paired input results on held-out wells E and F (33,664 samples). "
           r"Accuracy: mean $\pm$ standard deviation over ten training seeds, in percent. "
           r"Relative error reduction uses unrounded mean accuracies and the same curves-only architecture; "
           r"negative values indicate increased error. Kappa and macro-F1 are seed means. "
           r"The ModernTCN +VDL entry is BondFusion-TCN; its Curves entry is the backbone reference. "
           r"Full seed statistics, class-wise F1, MCC and ensemble results are in Table~\ref{S-tab:s-full}. "
           r"$^{a}$Point-wise references: gradient boosting uses one untuned fit; CBL thresholds are fitted on the training wells.}",
           r"\label{tab:main}", r"\begin{tabular}{lccccccc}", r"\toprule",
           r" & \multicolumn{2}{c}{Accuracy (\%)} & Error reduction (\%) & \multicolumn{2}{c}{Kappa} & \multicolumn{2}{c}{Macro-F1} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
           r"Architecture & Curves & +VDL & Relative to curves & Curves & +VDL & Curves & +VDL \\",
           r"\midrule", *rows, r"\midrule", *refs, r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (out / "table_02_main.tex").write_text("\n".join(tex), encoding="utf-8")


DESIGN_ROWS = [P.REF_CURVES, P.REF_VDL, "VDL-ModernTCN w/o waveform encoder", "VDL-ModernTCN (mean pool)",
               "VDL-ModernTCN (static query)", "VDL-ModernTCN", "VDL-ModernTCN + Reformer", "VDL-ModernTCN w/o VDL"]
SCREEN_OF = {"VDL-ModernTCN w/o VDL": ("S0", "S0-logCBL"), "VDL-ModernTCN w/o waveform encoder": ("S1", "S1-logCBL"),
             "VDL-ModernTCN (mean pool)": ("S2", "S2-logCBL"), "VDL-ModernTCN (static query)": ("S3", None),
             "VDL-ModernTCN": ("S4", None), "VDL-ModernTCN + Reformer": ("S5", None)}


def table_designs(out):
    sc = P.screen_table()
    rows = []
    for m in DESIGN_ROWS:
        lin, log = SCREEN_OF.get(m, (None, None))
        row = {"Design": P.REF_DISPLAY.get(m) or P.DESIGN_DISPLAY[m],
               "Hyper-parameters": "tuned separately" if m in P.REF_DISPLAY else "tuned for mean pooling"}
        for col, sid in [("Screen CV macro-F1, linear CBL", lin), ("Screen CV macro-F1, log CBL", log)]:
            row[col] = pm(sc.loc[sid, "cv_macro_f1_mean"], sc.loc[sid, "cv_macro_f1_std"]) if sid else "—"
        for k, lab in [("macro_f1", "Blind macro-F1"), ("kappa", "Blind kappa"), ("f1_Good", "Good F1")]:
            v = P.seed_values(m, k)
            row[lab] = pm(v.mean(), v.std(ddof=1))
        e = P.ensemble_metrics(m)
        row["Ens. macro-F1"] = f"{e['macro_f1']:.3f}"
        row["Ens. kappa"] = f"{e['kappa']:.3f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "table_03_designs.csv", index=False)  # supplementary Table (tab:s-designs)
    return df


class _Recorder:
    """Stands in for an optuna.Trial and records every suggest_* call (search space read from the code)."""

    def __init__(self):
        self.space = []

    def suggest_float(self, name, low, high, log=False):
        self.space.append((name, f"float [{low:g}, {high:g}]" + (" log" if log else "")))
        return low

    def suggest_int(self, name, low, high):
        self.space.append((name, f"int [{low}, {high}]"))
        return low

    def suggest_categorical(self, name, choices):
        self.space.append((name, "{" + ", ".join(map(str, choices)) + "}"))
        return choices[0]


def table_hyperparameters(out):
    from da_reformer.hpo import suggest_all
    rows = []
    for m in P.TUNED_MODELS:
        h = P.hpo_best(m)
        rows.append({"model": m, "display": P.display_name(m), "cv_best_macro_f1": h["value"], "best_trial": h["trial"],
                     "n_trials_in_study": h["n_trials"], "stop_epoch": h["stop_epoch"], **h["hp"]})
    pd.DataFrame(rows).convert_dtypes().to_csv(out / "table_S1_hyperparameters.csv", index=False)
    md = ["# Table S1b — hyper-parameter search space (read from `_code/da_reformer/hpo.py` and "
          "`models/baselines.py::suggest_hp` by recording the suggest calls)", "",
          "30 Optuna trials per model (TPE sampler seed 2026, median pruner after 8 startup trials, fold-level), "
          "objective = mean A–D leave-one-well-out validation macro-F1 (protocol_v03.yaml `hpo`). "
          "'+VDL' models search the space of their base model; Mean pooling searches the ModernTCN space plus the "
          "waveform encoder. ModernTCN small_size is fixed at 5.", "",
          "| Model | Parameter | Space |", "|---|---|---|"]
    for m in [*P.BACKBONES, P.MEAN_POOL]:
        rec = _Recorder()
        suggest_all(m, rec)
        for name, space in rec.space:
            md.append(f"| {P.display_name(m) if m == P.MEAN_POOL else m} | {name} | {space} |")
    (out / "table_S1_search_space.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def table_per_well(out):
    df = P.summary_per_well().copy()
    df.insert(1, "display", df["model"].map(P.display_name))
    df.insert(2, "input_set", df["model"].map(P.input_set))
    df.to_csv(out / "table_S2_per_well.csv", index=False)


def table_significance(out):
    rows = []
    for comp_set, sig in [("VDL vs Curves per backbone", P.sig_vdl_pairs()),
                          ("Mean pooling vs other models", P.sig_vs_mean_pool())]:
        for rec in sig.values():
            rows.append({"comparison_set": comp_set, "model": rec["model"], "other": rec["other"],
                         "model_display": P.display_name(rec["model"]), "other_display": P.display_name(rec["other"]),
                         "ens_d_macro_f1": rec["ensemble_macro_f1_diff"],
                         "ci95_lo_macro_f1": rec["boot_macro_f1_diff_ci95"][0],
                         "ci95_hi_macro_f1": rec["boot_macro_f1_diff_ci95"][1],
                         "p_le_0_macro_f1": rec["boot_macro_f1_p_le_0"], "ens_d_kappa": rec["ensemble_kappa_diff"],
                         "ci95_lo_kappa": rec["boot_kappa_diff_ci95"][0], "ci95_hi_kappa": rec["boot_kappa_diff_ci95"][1],
                         "p_le_0_kappa": rec["boot_kappa_p_le_0"],
                         "seed_mean_d_macro_f1": rec["seed_mean_macro_f1_diff"],
                         "seed_mean_d_kappa": rec["seed_mean_kappa_diff"]})
    pd.DataFrame(rows).to_csv(out / "table_S3_significance.csv", index=False)


# ------------------------------------------------------------------------------------------------
# paper_numbers.json
# ------------------------------------------------------------------------------------------------
def model_numbers(m):
    out = {"display": P.display_name(m), "input_set": P.input_set(m), "seed_mean": {}, "seed_sd": {}, "ensemble": {}}
    for k in METRICS + CLS_KEYS:
        v = P.seed_values(m, k)
        out["seed_mean"][k], out["seed_sd"][k] = r(v.mean()), r(v.std(ddof=1))
    e = P.ensemble_metrics(m)
    out["ensemble"] = {k: r(e[k]) for k in METRICS + CLS_KEYS}
    out["ensemble"]["confusion"] = e["confusion"]
    g = P.seed_values(m, "f1_Good")
    out["good_f1_per_seed"] = [r(x) for x in g]
    out["good_f1_seed_median"], out["good_f1_seed_min"], out["good_f1_seed_max"] = r(np.median(g)), r(g.min()), r(g.max())
    out["per_well"] = {}
    for w in P.TEST_WELLS:
        ew = P.ensemble_metrics(m, w)
        out["per_well"][w] = {
            "seed_mean": {k: r(P.seed_values(m, k, w).mean()) for k in METRICS + [f"f1_{c}" for c in P.CLASSES]
                          + ["recall_Good", "precision_Good"]},
            "seed_sd": {k: r(P.seed_values(m, k, w).std(ddof=1)) for k in ["macro_f1", "kappa", "f1_Good"]},
            "ensemble": {k: r(ew[k]) for k in METRICS + ["f1_Good", "recall_Good", "precision_Good"]}}
    if m in P.TUNED_MODELS:
        out["hpo_cv_best_macro_f1"] = r(P.hpo_best(m)["value"])
    return out


def curve_and_ac_facts() -> dict:
    """CBL amplitude by class (training vs blind) and AC availability by class, from the raw well files."""
    raw = P.ROOT / "_data" / "raw"
    label_map = {"胶结差": "Poor", "胶结中等": "Moderate", "胶结好": "Good", "胶结良好": "Good", "胶结优": "Good"}
    out = {"cbl_quantiles_by_class": {}, "ac_missing_share_by_well": {}, "ac_missing_share_by_class_training": {}}
    frames = {w: pd.read_csv(raw / f"{w}.csv", usecols=["CBL", "AC", "label_raw"]) for w in P.ALL_WELLS}
    for w, df in frames.items():
        df["cls"] = df["label_raw"].map(label_map)
        df["ac_missing"] = df["AC"].isna() | (df["AC"] <= -999)
        out["ac_missing_share_by_well"][w] = round(float(df["ac_missing"].mean()), 4)
        if w in P.TRAIN_WELLS:
            out["ac_missing_share_by_class_training"][w] = {c: round(float(g["ac_missing"].mean()), 4)
                                                            for c, g in df.groupby("cls")}
    for role, ws in [("training", P.TRAIN_WELLS), ("blind", P.TEST_WELLS)]:
        df = pd.concat([frames[w] for w in ws])
        out["cbl_quantiles_by_class"][role] = {c: {f"q{int(q * 100)}": round(float(g["CBL"].quantile(q)), 3)
                                                   for q in (0.1, 0.5, 0.9)} for c, g in df.groupby("cls")}
    out["note"] = "AC counted missing when NaN or <= -999 (the file uses -9999 as the null value); CBL in the units of the source file"
    return out


def build_numbers(fig2_interval):
    nums = {"_about": {
        "description": "Every number cited in the P04 v03 paper (VDL empirical study). Computed by "
                       "_analysis/build_paper_numbers.py from protocol v03 outputs; nothing trained.",
        "sources": {"final_runs": "_pipelines/01_run_cement_bond_benchmark/outputs/v03/final/<model>/seed*/",
                    "report": "outputs/v03/report/", "hpo": "outputs/v03/hpo/*.best.json",
                    "screen": "outputs/v03/screen/", "reference_baselines": "outputs/v03/reference_baselines.json",
                    "hpo_noise": "_sandbox/2026-10-10-hpo-selection-noise/out/VDL-ModernTCN_(mean_pool).json"},
        "conventions": {"sd": "sample sd (ddof=1) over the 10 seeds", "ensemble": "mean of 10 seeds' probabilities, "
                        "argmax", "blind": "wells E+F merged, 33,664 samples, each predicted once",
                        "labels": "0 Poor (胶结差), 1 Moderate (胶结中等), 2 Good (胶结好/胶结良好/胶结优)",
                        "bootstrap": "block bootstrap over depth segments within each well (step_08; numbers taken "
                                     "from report/significance_*.json)"}}}
    # data
    wells = {w: well_facts(w) for w in P.ALL_WELLS}
    agg = {}
    for role, ws in [("training", P.TRAIN_WELLS), ("blind", P.TEST_WELLS)]:
        cnt = np.sum([[wells[w]["counts"][c] for c in P.CLASSES] for w in ws], 0)
        agg[role] = {"wells": ws, "n": int(cnt.sum()), "counts": dict(zip(P.CLASSES, map(int, cnt))),
                     "shares": {c: float(x / cnt.sum()) for c, x in zip(P.CLASSES, cnt)}}
    nums["data"] = {"wells": wells, "totals": agg,
                    "n_total": int(sum(wells[w]["n"] for w in P.ALL_WELLS)),
                    "vdl": {"samples_per_trace_raw": 2000, "n_keep": P.N_KEEP, "envelope_bin": P.BIN,
                            "n_envelope_bins": P.N_KEEP // P.BIN, "block_features": P.N_KEEP // P.PROTOCOL["vdl"]["block"]},
                    "note_labels_are_blocks": "labels are long interval annotations: no contiguous training segment "
                                              "contains all three classes; see n_label_runs per well"}
    nums["data"]["curve_and_ac_facts"] = curve_and_ac_facts()
    # VDL window
    win = P.vdl_window()
    nums["vdl_window"] = {
        "rule": "contiguous log-envelope bins (bin 10) where the A–D Moderate median is below both the Poor and "
                "Good medians by more than 1 log unit (A–D labels only)",
        "sample_range": [win["sample_start"], win["sample_end"]], "sample_range_note": "half-open [start, end)",
        "last_sample_inclusive": win["sample_end"] - 1,
        "all_qualifying_runs": win["all_runs_samples"],
        "class_medians_AD": P.window_class_medians(P.TRAIN_WELLS, tuple(win["bins"])),
        "post_hoc_E": P.window_class_medians(["E"], tuple(win["bins"])),
        "post_hoc_F": P.window_class_medians(["F"], tuple(win["bins"])),
        "post_hoc_note": "E/F medians use blind-well labels; diagnostic only, computed after all models were frozen"}
    nums["figure_2a_intervals"] = fig2_interval
    # models
    nums["models"] = {m: model_numbers(m) for m in P.ALL_MODELS}
    nums["reference_methods"] = {disp: {k: r(P.reference_baselines()[k0][k]) for k in METRICS + CLS_KEYS}
                                 | {"confusion": P.reference_baselines()[k0]["confusion"]}
                                 for k0, disp in P.REFERENCE_DISPLAY.items()}
    # VDL effect
    eff = {}
    for b in P.BACKBONES:
        rec = P.sig_vdl_pairs()[f"{P.vdl(b)} vs {b}"]
        eff[b] = {"ens_d_kappa": r(rec["ensemble_kappa_diff"]), "ci95_kappa": [r(x) for x in rec["boot_kappa_diff_ci95"]],
                  "p_le_0_kappa": rec["boot_kappa_p_le_0"],
                  "ens_d_macro_f1": r(rec["ensemble_macro_f1_diff"]),
                  "ci95_macro_f1": [r(x) for x in rec["boot_macro_f1_diff_ci95"]],
                  "p_le_0_macro_f1": rec["boot_macro_f1_p_le_0"],
                  "seed_mean_d_kappa": r(rec["seed_mean_kappa_diff"]),
                  "seed_mean_d_macro_f1": r(rec["seed_mean_macro_f1_diff"]),
                  "seed_mean_d_f1_by_class": {c: r(P.seed_values(P.vdl(b), f"f1_{c}").mean()
                                                   - P.seed_values(b, f"f1_{c}").mean()) for c in P.CLASSES}}
    dk = [eff[b]["ens_d_kappa"] for b in P.BACKBONES]
    nums["vdl_effect"] = {
        "per_backbone": eff,
        "n_backbones_ens_d_kappa_positive": int(sum(x > 0 for x in dk)),
        "ens_d_kappa_range": [min(dk), max(dk)],
        "backbones_kappa_ci_excludes_0": [b for b in P.BACKBONES if eff[b]["ci95_kappa"][0] > 0],
        "backbones_macro_f1_ci_excludes_0": [b for b in P.BACKBONES if eff[b]["ci95_macro_f1"][0] > 0
                                             or eff[b]["ci95_macro_f1"][1] < 0],
        "n_backbones_seed_d_kappa_positive": int(sum(eff[b]["seed_mean_d_kappa"] > 0 for b in P.BACKBONES)),
        "n_backbones_ens_d_macro_f1_positive": int(sum(eff[b]["ens_d_macro_f1"] > 0 for b in P.BACKBONES)),
        "mean_over_backbones_seed_d_f1_by_class": {c: r(np.mean([eff[b]["seed_mean_d_f1_by_class"][c]
                                                                 for b in P.BACKBONES])) for c in P.CLASSES},
        "n_backbones_seed_d_f1_up": {c: int(sum(eff[b]["seed_mean_d_f1_by_class"][c] > 0 for b in P.BACKBONES))
                                     for c in P.CLASSES},
        "n_backbones_ens_d_f1_up": {c: int(sum(P.ensemble_metrics(P.vdl(b))[f"f1_{c}"] > P.ensemble_metrics(b)[f"f1_{c}"]
                                               for b in P.BACKBONES)) for c in P.CLASSES},
        "n_backbones_seed_d_macro_f1_positive": int(sum(eff[b]["seed_mean_d_macro_f1"] > 0 for b in P.BACKBONES)),
        "mean_seed_f1_by_class": {inp: {c: r(np.mean([P.seed_values(m, f"f1_{c}").mean() for m in ms]))
                                        for c in P.CLASSES}
                                  for inp, ms in [("curves", list(P.BACKBONES)), ("curves_vdl", [P.vdl(b) for b in P.BACKBONES])]}}
    # Headline comparison: full precision, same baseline and mean-run convention.
    nums["paired_input_accuracy"] = {
        "convention": "Ten individual-run accuracies averaged on E+F; not probability ensembles",
        "relative_error_reduction_formula": "100*(A_VDL-A_curves)/(1-A_curves)",
        "per_backbone": {b: paired_accuracy(b) for b in P.BACKBONES},
        "n_mean_accuracy_increased": int(sum(paired_accuracy(b)["accuracy_vdl_mean"] >
                                            paired_accuracy(b)["accuracy_curves_mean"] for b in P.BACKBONES))}
    # pooled confusion
    cc = P.pooled_confusion(P.BACKBONES)
    cv = P.pooled_confusion([P.vdl(b) for b in P.BACKBONES])
    nums["pooled_confusion_8_backbones"] = {
        "rows_true_cols_pred": P.CLASSES, "curves": cc.tolist(), "curves_vdl": cv.tolist(),
        "diff_per_backbone": ((cv - cc) / 8).tolist(),
        "row_share_curves": np.round(cc / cc.sum(1, keepdims=True), 6).tolist(),
        "row_share_curves_vdl": np.round(cv / cv.sum(1, keepdims=True), 6).tolist(),
        "errors_curves_vdl": {"total": int(cv.sum() - np.trace(cv)),
                              "poor_moderate": int(cv[0, 1] + cv[1, 0]),
                              "involving_good": int(cv[0, 2] + cv[2, 0] + cv[1, 2] + cv[2, 1])},
        "errors_curves": {"total": int(cc.sum() - np.trace(cc)),
                          "poor_moderate": int(cc[0, 1] + cc[1, 0]),
                          "involving_good": int(cc[0, 2] + cc[2, 0] + cc[1, 2] + cc[2, 1])},
        "note": "sum over the 8 backbones of the 10-seed-ensemble confusion matrices (E+F)"}
    nums["pooled_confusion_8_backbones"]["errors_curves_vdl"]["share_involving_good"] = r(
        nums["pooled_confusion_8_backbones"]["errors_curves_vdl"]["involving_good"]
        / nums["pooled_confusion_8_backbones"]["errors_curves_vdl"]["total"])
    # label-free VDL amplitude level per well: median over depth samples of the mean log envelope of samples 0-300
    nums["vdl_early_time_level"] = {
        "rule": "median over depth samples of the mean raw log envelope over time samples 0-300 (bins 0-29); no labels",
        "per_well": {w: r(float(np.median(P.log_envelope(w)[:, :30].mean(1)))) for w in P.ALL_WELLS}}
    nums["final_training_epochs"] = {m: int(P.hpo_best(m)["stop_epoch"]) for m in P.TUNED_MODELS}
    # designs
    nums["designs"] = {"rows": {m: (P.REF_DISPLAY.get(m) or P.DESIGN_DISPLAY[m]) for m in DESIGN_ROWS},
                       "significance_vs_mean_pooling": {
                           P.display_name(rec["other"]): {
                               "ens_d_macro_f1": r(rec["ensemble_macro_f1_diff"]),
                               "ci95_macro_f1": [r(x) for x in rec["boot_macro_f1_diff_ci95"]],
                               "ens_d_kappa": r(rec["ensemble_kappa_diff"]),
                               "ci95_kappa": [r(x) for x in rec["boot_kappa_diff_ci95"]],
                               "p_le_0_macro_f1": rec["boot_macro_f1_p_le_0"], "p_le_0_kappa": rec["boot_kappa_p_le_0"]}
                           for rec in P.sig_vs_mean_pool().values()}}
    # screen
    sc = P.screen_table()
    nums["screen"] = {
        "note": "A–D leave-one-well-out CV, fixed screening hyper-parameters (ModernTCN tuned on curves in the earlier benchmark), 3 seeds; S0–S5 linear CBL, "
                "*-logCBL with signed-log CBL",
        "candidates": {sid: {"display": P.SCREEN_DISPLAY[sid], "model": sc.loc[sid, "model"],
                             "mean": r(sc.loc[sid, "cv_macro_f1_mean"]), "sd": r(sc.loc[sid, "cv_macro_f1_std"]),
                             "seeds": [r(x) for x in sc.loc[sid, "seed_values"]],
                             "folds_mean": {w: r(sc.loc[sid, f"fold_{w}_macro_f1"]) for w in P.TRAIN_WELLS}}
                       for sid in P.SCREEN_ORDER},
        "pooled_noise_sd_rms_round1": r(np.sqrt(np.mean(sc.loc[["S0", "S0-logCBL", "S1", "S2", "S3", "S4", "S5"],
                                                               "cv_macro_f1_std"] ** 2)))}
    # Good class
    nums["good_class"] = {
        "zones": {w: P.good_zones(w) for w in P.TEST_WELLS},
        "zones_training": {w: P.good_zones(w) for w in P.TRAIN_WELLS},
        "per_model_seed_sd_good_f1": {P.display_name(m): r(P.seed_values(m, "f1_Good").std(ddof=1)) for m in P.ALL_MODELS},
        "median_over_models_seed_sd_good_f1": r(np.median([P.seed_values(m, "f1_Good").std(ddof=1)
                                                           for m in P.ALL_MODELS])),
        "n_seeds_good_f1_zero": {P.display_name(m): int((P.seed_values(m, "f1_Good") == 0).sum()) for m in P.ALL_MODELS}}
    # CV vs blind
    df = P.cv_vs_blind()
    rho, p = spearmanr(df["cv_macro_f1"], df["blind_macro_f1_mean"])
    rk, pk = spearmanr(df["cv_macro_f1"], df["blind_kappa_mean"])
    nums["cv_vs_blind"] = {"n_models": len(df), "spearman_rho_macro_f1": r(rho), "spearman_p_macro_f1": r(p),
                           "spearman_rho_vs_blind_kappa": r(rk), "spearman_p_vs_blind_kappa": r(pk),
                           "points": df.drop(columns=["display"]).to_dict(orient="records")}
    hn = P.hpo_noise()
    nums["hpo_selection_noise"] = {"model": P.MEAN_POOL, **hn,
                                   "note": "top-5 completed Optuna trials re-run with fresh seeds 1-3, A–D CV only"}
    nums["hpo_notes"] = {
        "reformer_curves_and_vdl_same_best_hp": P.hpo_best("Reformer")["hp"] == P.hpo_best("Reformer +VDL")["hp"],
        "studies_with_31_trials": [m for m in P.TUNED_MODELS if P.hpo_best(m)["n_trials"] != 30]}
    ref = P.reference_baselines()
    nums["reference_extras"] = {
        "cbl_rule_thresholds": ref["CBL two-threshold rule"]["thresholds"],
        "hgb_ac_missing_only_kappa": r(ref["HGB AC-missing indicator only"]["kappa"]),
        "hgb_with_ac_kappa": r(ref["HGB point-wise (GR,AC,CBL,CCL + AC-missing)"]["kappa"])}
    nums["n_params"] = {m: int(P.seed_metrics(m)[0]["n_params"]) for m in P.ALL_MODELS
                        if "n_params" in P.seed_metrics(m)[0]}
    nums["post_hoc"] = _post_hoc.compute()  # labelled post hoc in the paper (second P5.4 review)
    return nums


def main():
    out = P.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    bad = verify()
    (out / "verification.json").write_text(json.dumps(
        {"n_checks": len(CHECKS), "n_failed": len(bad), "tolerance": TOL,
         "max_abs_diff": max(c["abs_diff"] for c in CHECKS), "failed": bad}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    if bad:
        for c in bad[:30]:
            print("MISMATCH", c)
        sys.exit(1)
    import _intervals as fig_02_data_vdl
    w, ivs = fig_02_data_vdl.pick_intervals()
    d = P.well(w)
    fig2 = {"well": w, "intervals_m": [[float(d.depth[s]), float(d.depth[e - 1])] for s, e in ivs],
            "class_counts": [dict(zip(P.CLASSES, map(int, np.bincount(d.y[s:e], minlength=3)))) for s, e in ivs]}
    table_wells(out)
    table_main(out)
    table_designs(out)
    table_hyperparameters(out)
    table_per_well(out)
    table_significance(out)
    nums = build_numbers(fig2)
    (out / "paper_numbers.json").write_text(json.dumps(nums, indent=1, ensure_ascii=False, default=float),
                                            encoding="utf-8")
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
