"""Write the LaTeX supplementary tables from the CSVs built by build_paper_numbers.py.

Inputs : _outputs/p04_v03_paper/table_02_main.csv, table_03_designs.csv, table_S1_hyperparameters.csv,
         table_S1_search_space.md, table_S2_per_well.csv, table_S3_significance.csv
Outputs: _outputs/p04_v03_paper/table_S_full_results.tex, table_S_designs.tex, table_S_selected_hp.tex,
         table_S_search_space.tex, table_S_per_well.tex, table_S_significance.tex   (longtable, booktabs)
File names and labels (tab:s-*) are semantic; the printed S-numbers follow the order in supplementary.tex and the
main text refers to them through xr (\\ref{S-tab:s-...}).

Usage: python scripts/rebuild_paper_analysis.py tables
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

import _paper_data as P

OUT = P.OUT_DIR
ARCH_KEYS = ["hidden", "num_layers", "d_model", "n_heads", "e_layers", "d_ff", "bucket_size", "n_hashes", "d_lift",
             "seg_len", "factor", "top_k", "num_kernels", "down_sampling_layers", "moving_avg", "patch_size",
             "ffn_ratio", "num_blocks", "large_size", "d_wave", "wave_channels"]
SHARED_PARAMS = ["lr", "weight_decay", "dropout"]
RAGGED = r">{\raggedright\arraybackslash}"
# display order shared with the main text: eight architectures (curves, +VDL), then the encoder designs
ORDER = [n for b in P.BACKBONES for n in (P.display_name(b), P.display_name(P.vdl(b)))] + \
        [P.DESIGN_DISPLAY[m] for m in ("VDL-ModernTCN (mean pool)", "VDL-ModernTCN (static query)", "VDL-ModernTCN",
                                       "VDL-ModernTCN + Reformer", "VDL-ModernTCN w/o waveform encoder",
                                       "VDL-ModernTCN w/o VDL")]


def tex(s) -> str:
    s = str(s).replace("{", r"\{").replace("}", r"\}")
    s = s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("±", r"$\pm$")
    return s.replace("—", "--").replace("–", "--")


def signed(x: float) -> str:
    return f"{x:+.3f}".replace("-", "$-$")


def sci(x: float) -> str:
    e = math.floor(math.log10(abs(x)))
    return f"${x / 10 ** e:.2f}\\times10^{{{e}}}$"


def longtable(path: Path, cols: str, header: list[str], rows: list, caption: str, label: str,
              size: str = r"\small") -> None:
    """rows: list of cell lists; a string row is written verbatim (e.g. a \\midrule or a group heading)."""
    head = " & ".join(header) + r" \\"
    body = "\n".join(r if isinstance(r, str) else " & ".join(r) + r" \\" for r in rows)
    path.write_text(
        f"{{{size}\n\\begin{{longtable}}{{{cols}}}\n\\caption{{{caption}}}\\label{{{label}}}\\\\\n\\toprule\n{head}\n"
        f"\\midrule\n\\endfirsthead\n\\toprule\n{head}\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n{body}\n"
        f"\\end{{longtable}}}}\n", encoding="utf-8")


def build_full_results() -> None:
    df = pd.read_csv(OUT / "table_02_main.csv")
    rows, ensemble_rows = [], []
    for i, r in df.iterrows():
        if i == 2 * len(P.BACKBONES):
            rows.append(r"\midrule")
        rows.append([tex(r["Model"]), tex(r["Input"])] + [f"{float(r[c]):.3f}" if c.startswith("F1 ") else tex(r[c]) for c in
                    ("Accuracy", "Macro-F1", "Kappa", "MCC", "F1 Poor", "F1 Moderate", "F1 Good")])
        if i < 2 * len(P.BACKBONES):
            ensemble_rows.append([tex(r["Model"]), tex(r["Input"])] +
                                 [tex(r[c]) for c in ("Ens. accuracy", "Ens. macro-F1", "Ens. kappa")])
    path = OUT / "table_S_full_results.tex"
    longtable(path, RAGGED + "p{2.6cm}" + RAGGED + "p{1.6cm}" + "c" * 7,
              ["Model", "Input", "Accuracy", "Macro-F1", "Kappa", "MCC", "F1 P", "F1 M", "F1 G"], rows,
              "Held-out-well results of the sixteen architecture--input pairs and reference methods (wells E and F, "
              "33,664 samples). (a) Individual-run results. Accuracy, macro-F1, kappa and MCC: mean $\\pm$ sd "
              "over 10 seeds; per-class F1 (P = Poor, M = Moderate, G = Good): seed mean. "
              "(b) Probability-ensemble results. Reference methods are single deterministic runs and have no ensemble.",
              "tab:s-full", size=r"\footnotesize")
    # Keep the existing S1 number and expose ensemble accuracy without an over-wide table.
    part_b = [r"\par\noindent\begin{minipage}{\textwidth}", r"\footnotesize",
              r"\textit{Table~\ref{tab:s-full} (continued). (b) Ten-seed probability ensembles.}",
              r"\begin{center}", r"\begin{tabular}{llccc}", r"\toprule",
              r"Model & Input & Accuracy & Macro-F1 & Kappa \\", r"\midrule"]
    part_b += [" & ".join(row) + r" \\" for row in ensemble_rows]
    part_b += [r"\bottomrule", r"\end{tabular}", r"\end{center}", r"\end{minipage}", ""]
    path.write_text(path.read_text() + "\n".join(part_b), encoding="utf-8")


def build_designs() -> None:
    df = pd.read_csv(OUT / "table_03_designs.csv").rename(columns={"Held-out macro-F1": "Blind macro-F1", "Held-out kappa": "Blind kappa"})
    rows = []
    for group, title in [("tuned separately", "Separately tuned ModernTCN references"),
                         ("tuned for mean pooling", "Encoder designs (hyper-parameters tuned for mean pooling)")]:
        rows.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{title}}}}} \\")
        for _, r in df[df["Hyper-parameters"] == group].iterrows():
            rows.append([tex(r["Design"])] + [tex(r[c]) for c in
                        ("Screen CV macro-F1, linear CBL", "Screen CV macro-F1, log CBL", "Blind macro-F1",
                         "Blind kappa", "Good F1", "Ens. macro-F1", "Ens. kappa")])
    longtable(OUT / "table_S_designs.tex", RAGGED + "p{3.6cm}" + "c" * 7,
              ["Design", "CV, linear CBL", "CV, log CBL", "Macro-F1", "Kappa", "Good F1", "Ens. F1",
               "Ens. $\\kappa$"], rows,
              "Waveform-encoder designs on the ModernTCN backbone and the two separately tuned ModernTCN references. "
              "CV: macro-F1 of the design screen (leave-one-well-out cross-validation on wells A--D with one fixed set "
              "of screening hyper-parameters, mean $\\pm$ sd over 3 seeds); in the No VDL and Block features rows "
              "these columns give the curves-only and block-feature inputs. The remaining columns are held-out-well "
              "results (wells E and F, mean $\\pm$ sd over 10 seeds; Ens.: 10-seed ensemble).", "tab:s-designs",
              size=r"\footnotesize")


def build_selected_hp() -> None:
    df = pd.read_csv(OUT / "table_S1_hyperparameters.csv").set_index("display")
    n_params = json.loads((OUT / "paper_numbers.json").read_text(encoding="utf-8"))["n_params"]
    rows = []
    for name in [n for n in ORDER if n in df.index]:
        r = df.loc[name]
        arch = "; ".join(f"\\texttt{{{tex(k)}}}\\,=\\,{int(r[k]) if float(r[k]).is_integer() else r[k]}"
                         for k in ARCH_KEYS if pd.notna(r[k]))
        rows.append([tex(name), f"{r['cv_best_macro_f1']:.3f}", str(int(r["n_trials_in_study"])),
                     str(int(r["stop_epoch"])), sci(r["lr"]), sci(r["weight_decay"]), f"{r['dropout']:.2f}",
                     f"{n_params[r['model']]:,}", arch])
    longtable(OUT / "table_S_selected_hp.tex", RAGGED + "p{2.6cm}ccccccc" + RAGGED + "p{4.9cm}",
              ["Model", "CV F1", "Trials", "Epochs", "Learning rate", "Weight decay", "Dropout", "Parameters",
               "Architecture"],
              rows,
              "Selected hyper-parameters of the seventeen tuned models. CV F1 is the best-epoch cross-validation "
              "macro-F1 of the selected configuration (optimistic, see the main text, Section~\\ref{M-sec:protocol}); "
              "Epochs is the number of epochs used to train the final models on wells A--D (the learning-rate warm-up lasts five "
              "epochs); Parameters is the number of trainable parameters. Two searches ran a 31st "
              "configuration after one configuration failed. Architecture parameters use the names of the "
              "Time-Series-Library implementation.", "tab:s-hp", size=r"\footnotesize")


def build_search_space() -> None:
    md = (OUT / "table_S1_search_space.md").read_text(encoding="utf-8").splitlines()
    recs = [[c.strip() for c in line.strip("|").split("|")] for line in md if line.startswith("|")][2:]
    shared = {p: {s for m, q, s in recs if q == p} for p in SHARED_PARAMS}
    assert all(len(v) == 1 for v in shared.values()), shared  # identical for every model, so listed once
    rows = [rf"\multicolumn{{3}}{{l}}{{\textit{{All models}}}} \\"]
    rows += [["", f"\\texttt{{{tex(p)}}}", tex(next(iter(shared[p])))] for p in SHARED_PARAMS]
    rows.append(rf"\multicolumn{{3}}{{l}}{{\textit{{Model-specific}}}} \\")
    last = None
    for m, p, s in recs:
        if p in SHARED_PARAMS:
            continue
        rows.append([tex(m) if m != last else "", f"\\texttt{{{tex(p)}}}", tex(s)])
        last = m
    longtable(OUT / "table_S_search_space.tex", "p{3.2cm}p{4.2cm}p{5cm}", ["Model", "Parameter", "Search space"], rows,
              "Hyper-parameter search spaces (30 configurations of the tree-structured Parzen estimator per model). "
              "Models with VDL block features search the space of their curves-only counterpart; the mean-pooling "
              "design searches the ModernTCN space plus the encoder width. ModernTCN \\texttt{small\\_size} is fixed "
              "at 5.", "tab:s-space")


def build_per_well() -> None:
    df = pd.read_csv(OUT / "table_S2_per_well.csv")
    rows = []
    for name in ORDER:
        sub = df[df["display"] == name].sort_values("well")
        for j, (_, r) in enumerate(sub.iterrows()):
            cells = [tex(name) if j == 0 else "", r["well"]] + \
                    [tex(f"{r[f'{k}_mean']:.3f} ± {r[f'{k}_std']:.3f}") for k in ("macro_f1", "kappa")] + \
                    [f"{r[f'f1_{c}_mean']:.3f}" for c in ("Poor", "Moderate", "Good")]
            # keep the two wells of one model on the same page
            rows.append(" & ".join(cells) + (r" \\*" if j == 0 else r" \\"))
    longtable(OUT / "table_S_per_well.tex", RAGGED + "p{4.6cm}cccccc",
              ["Model", "Well", "Macro-F1", "Kappa", "F1 Poor", "F1 Moderate", "F1 Good"], rows,
              "Held-out-well results per well (mean $\\pm$ sd over 10 seeds; per-class F1 = seed mean).", "tab:s-perwell")


def build_significance() -> None:
    df = pd.read_csv(OUT / "table_S3_significance.csv")
    rows = []
    for k, (cs, sub) in enumerate(df.groupby("comparison_set", sort=False)):
        title = ("Curves plus VDL minus curves, per architecture" if cs.startswith("VDL")
                 else "Mean pooling minus each other model")
        if k:
            rows.append(r"\midrule")
        rows.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{title}}}}} \\")
        for _, r in sub.iterrows():
            name = r["model"].replace(" +VDL", "") if cs.startswith("VDL") else r["other_display"]
            rows.append([tex(name),
                         f"{signed(r['ens_d_kappa'])} [{signed(r['ci95_lo_kappa'])}, {signed(r['ci95_hi_kappa'])}]",
                         f"{signed(r['ens_d_macro_f1'])} [{signed(r['ci95_lo_macro_f1'])}, "
                         f"{signed(r['ci95_hi_macro_f1'])}]",
                         signed(r["seed_mean_d_kappa"]), signed(r["seed_mean_d_macro_f1"])])
    direct = json.loads((OUT / "paper_numbers.json").read_text(encoding="utf-8"))["post_hoc"]["direct_design_comparisons"]
    for reference, title in [
        ("Block features (no encoder)", "Encoder minus block features (shared hyper-parameters)"),
        ("ModernTCN, Curves + VDL", "Encoder minus BondFusion-TCN (separately tuned reference)"),
    ]:
        rows.append(r"\midrule")
        rows.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{title}}}}} \\")
        for design in ["Mean pooling", "Static attention", "Curve-conditioned attention",
                       "Curve-conditioned attention + LSH layer"]:
            result = direct[f"{design} vs {reference}"]
            rows.append([tex(design),
                         f"{signed(result['d_kappa'])} [{signed(result['ci_kappa'][0])}, {signed(result['ci_kappa'][1])}]",
                         f"{signed(result['d_macro_f1'])} [{signed(result['ci_macro_f1'][0])}, {signed(result['ci_macro_f1'][1])}]",
                         "--", "--"])
    longtable(OUT / "table_S_significance.tex", RAGGED + "p{4.2cm}cccc",
              ["Model", "Ens. $\\Delta$kappa [95\\% CI]", "Ens. $\\Delta$macro-F1 [95\\% CI]", "Seed $\\Delta$kappa",
               "Seed $\\Delta$macro-F1"], rows,
              "Paired differences between seed ensembles in the held-out wells with 95\\% segment-bootstrap intervals "
              "(1000 draws), and differences between seed means. The bootstrap keeps the trained ensembles fixed and "
              "does not include seed variation. The final two groups report the direct encoder comparisons; dashes denote "
              "seed-mean differences not tabulated for these additional comparisons.", "tab:s-sig", size=r"\footnotesize")


def build_post_hoc() -> None:
    ph = json.loads((OUT / "paper_numbers.json").read_text(encoding="utf-8"))["post_hoc"]
    ds, runs = ph["decision_shift"]["per_architecture"], ph["label_runs"]
    bc = ph["boundaries_and_costs"]["per_architecture"]
    rows = []
    for b in P.BACKBONES:
        c, v, r = ds[b]["curves"], ds[b]["vdl"], bc[b]
        be = r["break_even_ratio"]
        be_txt = f"{be:.2f}" if be is not None else ("any" if r["vdl_better_at_any_ratio"] else "none")
        rows.append([b] + [f"{x:.3f}" for x in (c["kappa"], c["kappa_no_good"], v["kappa"], c["auc_good_vs_moderate"],
                                                 v["auc_good_vs_moderate"])]
                    + [f"{r[t][k]:,}" for k in ("missed_poor", "false_alarm_poor") for t in ("curves", "vdl")]
                    + [be_txt, f"{runs['runs_correct'][b]['curves']}", f"{runs['runs_correct'][b]['vdl']}"])
    longtable(OUT / "table_S_post_hoc.tex", "lccccccccccccc",
              ["", "$\\kappa$ C", "$\\kappa$ C$_{\\neg G}$", "$\\kappa$ V", "AUC C", "AUC V", "Miss C", "Miss V",
               "FA C", "FA V", "B/E", "Runs C", "Runs V"], rows,
              "Post hoc analyses of the seed ensembles in the held-out wells (C = curves, V = curves plus VDL). "
              "$\\kappa$ C$_{\\neg G}$: kappa of the curves-only ensemble when Good may not be predicted (argmax "
              "over Poor and Moderate only). AUC: area under the ROC curve for separating Good from Moderate among the "
              "bonded samples, using $p_{G}/(p_{G}+p_{M})$. Miss: Poor samples called bonded; FA (false alarm): bonded "
              "samples called Poor. B/E: break-even ratio of the cost of a missed poor sample to that of a false alarm "
              "above which VDL lowers the total cost at this boundary (any: VDL lowers both errors; none: VDL raises "
              f"both). Runs: number of the {runs['n_runs']} constant-label runs of wells E and F "
              f"({runs['n_runs_by_class']['Poor']} Poor, {runs['n_runs_by_class']['Moderate']} Moderate, "
              f"{runs['n_runs_by_class']['Good']} Good) whose majority prediction equals the label.",
              "tab:s-posthoc", size=r"\scriptsize")


def main() -> int:
    build_full_results()
    build_designs()
    build_selected_hp()
    build_search_space()
    build_per_well()
    build_significance()
    build_post_hoc()
    print("wrote", sorted(p.name for p in OUT.glob("table_S_*.tex")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
