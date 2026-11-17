# BondFusion-TCN

Open data and research code for point-wise cased-hole cement-bond quality
classification into **Poor**, **Moderate**, and **Good** using conventional
wireline logging curves (`GR`, `CBL`, `CCL`) and variable-density log (`VDL`)
waveforms.

In the **v04** study (`BondFusion-TCN` manuscript), the proposed architecture
(`_code/da_reformer/models/bondfusion_tcn.py` inside a PyTorch Lightning
container `_code/da_reformer/lightning_module.py`) bridges:
1. Sub-period analytic **Hilbert log-envelopes** (`20 us` resolution, 95 channels)
   and within-trace arrival contrast across the first 1,900 VDL time samples;
2. A **decoupled curve prior** combined with a **training-anchored CBL rational
   residual gate**; and
3. **Zero-phase symmetric multi-scale depth filtering** inside a
   non-normalizing multi-variable ModernTCN depth backbone.

## Open six-well dataset (`v0.4.0`)

**Both the complete anonymised six-well dataset and the full method codebase are
publicly released here under `_data/raw/` (as well as in the `v0.4.0` GitHub
Release archive).** Wells **A–F** provide `55,155` depth samples across `139`
contiguous logging segments:
- `_data/raw/{A..F}.csv`: UTF-8 curves (`DEPTH`, `GR`, `CBL`, `CCL`, optional
  `AC`, and `label_raw`)
- `_data/raw/{A..F}_vdl.npy`: aligned `int16` VDL waveforms `[N, 2000]`
- `_data/raw/manifest.json`: SHA-256 checksums and per-well shapes/counts

All six boreholes participate symmetrically under the unified **six-well nested
leave-one-well-out (LOWO) cross-validation protocol** (`run_lightning_bondfusion_v04.py`).
The earlier `v03` protocol (`protocol_v03.yaml`, `ModernTCN +VDL`) trained on A–D
and evaluated E/F as held-out wells and is preserved for historical provenance.

## Model identifiers

- **BondFusion-TCN (`v04`, manuscript model):** `BondFusionTCN`
  (`_code/da_reformer/models/bondfusion_tcn.py`), run under symmetric 6-well
  nested LOWO CV (`_pipelines/01_run_cement_bond_benchmark/run_lightning_bondfusion_v04.py`).
- **Paired backbones (`<Backbone>` / `<Backbone> +VDL`):** LSTM, GRU,
  Transformer, Reformer, Crossformer, TimesNet, TimeMixer, and ModernTCN under
  matched Optuna HPO budgets (30 trials per fold).
- **`v03` historical reference (`ModernTCN +VDL` / `VDL-ModernTCN (mean pool)`):**
  preserved in `protocol_v03.yaml`, `step_01`–`step_09`, and `results/` without
  modification.

## Install and verify

Python 3.11 is the reference interpreter. Install PyTorch with the CPU or CUDA
build appropriate to your machine, then install the remaining dependencies.
The direct dependency versions used for validation are in `requirements.txt`.
Broader package version ranges in `pyproject.toml` are convenience metadata,
not a claim that every combination has been tested.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python scripts/smoke_test.py
```

The CPU smoke test creates temporary synthetic wells, checks training-only
scaling and alignment, trains the fusion model for one epoch, and checks
waveform-encoder gradients. Its synthetic data and metrics are solely software
checks and do not represent experimental evidence or reproduce paper numbers.

## Data availability and input contract

The complete six-well anonymised dataset (Wells **A–F**, `55,155` samples) is
directly included in `_data/raw/` (`{A..F}.csv`, `{A..F}_vdl.npy`, and
`manifest.json`) as well as attached as a compressed archive to GitHub Release
`v0.4.0`. Schema details and integrity hashes are documented in
[`_data/raw/README.md`](_data/raw/README.md). To test another borehole dataset,
place compatible CSV + VDL arrays following the same schema.

## Reproduce the v04 BondFusion-TCN benchmark

Run mathematical invariant and PyTorch Lightning unit tests (CPU, ~5 seconds):

```bash
PYTHONPATH=_code pytest _code/da_reformer/tests/test_invariants.py _code/da_reformer/tests/test_models.py -q
```

Run the symmetric six-well nested LOWO CV pipeline and rebuild the four-pillar
summary tables (`01_comparison_experiments.csv`, `02_ablation_experiments.csv`,
`03_sensitivity_experiments.csv`, `04_generalization_and_mechanism.csv`, and
`05_statistical_significance.json`, archived under `results/v04/`):

```bash
PYTHONPATH=_code python _pipelines/01_run_cement_bond_benchmark/run_lightning_bondfusion_v04.py
PYTHONPATH=_code python _pipelines/01_run_cement_bond_benchmark/build_four_pillar_tables_v04.py
PYTHONPATH=_code python _pipelines/01_run_cement_bond_benchmark/build_v04_cag_figures.py
```

Publication-ready figure scripts and number checkers are in `_figures/`
(`_figures/make_all.sh`).

## Reproduce the v03 reference pipeline

The executable steps live in `_pipelines/01_run_cement_bond_benchmark/`.
All commands below run from the repository root; their default protocol is
`protocol_v03.yaml`. Every step provides `--help`. `--device cpu` is supported
for verification; full searches and final multi-seed runs are substantial jobs.

```bash
python _pipelines/01_run_cement_bond_benchmark/step_02_validate_data.py
python _pipelines/01_run_cement_bond_benchmark/step_09_run_design_screen.py --device cuda
python _pipelines/01_run_cement_bond_benchmark/step_09_run_design_screen.py --summarize
python _pipelines/01_run_cement_bond_benchmark/step_04_run_hpo.py --model all --device cuda
python _pipelines/01_run_cement_bond_benchmark/step_05_run_final.py --model "ModernTCN +VDL" --device cuda
python _pipelines/01_run_cement_bond_benchmark/step_06_build_report.py
python _pipelines/01_run_cement_bond_benchmark/step_08_compute_significance.py --reference "ModernTCN +VDL"
python _pipelines/01_run_cement_bond_benchmark/step_08_compute_significance.py --vdl-pairs
```

Selected v03 hyperparameters are included in the pipeline's
`outputs/v03/hpo/*.best.json`; the v02 ModernTCN configuration is included only
because the v03 design screen used it as its fixed starting configuration.
To use the frozen settings directly, skip HPO and run step 05 for each desired
model and seed. For full paired tables, run all entries in `baselines` and
`vdl_baselines`. Run the learned waveform model separately with
`--model "VDL-ModernTCN (mean pool)"`; the `ablations` and `plugins` entries
reuse its settings with `--hp-from "VDL-ModernTCN (mean pool)"`.
Step 06 aggregates only final runs present locally and reports missing ones.
Step 08 defaults to the historical `models.proposed`, so pass the explicit
reference above when comparing with BondFusion-TCN.

The archived screen table records the actual historical screen. The final v03
protocol already has log-CBL enabled, whereas the first screen round preceded
that decision. Consequently, rerunning the screen with this final protocol does
not reconstruct the original linear-CBL round unless its candidate transform
overrides are restored explicitly. The released protocol is preserved verbatim
rather than silently rewritten to imply an exact historical replay.

The included best JSON files are frozen selections, not the complete Optuna
study databases. Starting HPO in this release creates fresh studies; exact trial
histories are not reconstructed from those JSON files. GPU arithmetic and
library/backend changes can affect numerical reproducibility.

## Numerical paper analysis

The frozen `results/paper_numbers.json` provides the complete anonymized
numerical record, including post hoc decision-rule diagnostics, error costs,
Good-class/bootstrap coverage and direct encoder comparisons. It is copied
unchanged from the frozen artifact. Wells are identified only as A–F. Small
Good-zone and figure-interval summaries retain interval endpoints and counts;
there are no per-depth logs, waveforms, labels or prediction arrays. These
results are **aggregate evidence, not a release of the underlying dataset**.
The file's historical `_figures/` source labels refer to the private study's
original layout; equivalent public numerical modules are in `_analysis/`.

The original analysis algorithms and interval-selection rule are preserved.
Only local input/output resolution, aggregate fallbacks and plotting/font/shared
skill dependencies were adapted. No figure assets or shared skill files are
required. The existing runtime requirements cover this analysis; Matplotlib,
PowerPoint and font installation are unnecessary for the numerical commands.

```bash
python scripts/rebuild_paper_analysis.py --help
python scripts/rebuild_paper_analysis.py inspect
python scripts/rebuild_paper_analysis.py tables --from-release --output _outputs/reformatted_tables
python scripts/rebuild_paper_analysis.py numbers --data-root AUTHORIZED_INPUT_ROOT --output _outputs/recomputed_numbers
python scripts/rebuild_paper_analysis.py post-hoc --data-root AUTHORIZED_INPUT_ROOT --output _outputs/recomputed_post_hoc
```

`inspect` imports the analysis and lists the released sections. `tables
--from-release` formats the released aggregate CSV/JSON files into LaTeX table
fragments; it does not independently validate the underlying measurements.
`numbers` and `post-hoc` require the authorized raw six-well inputs and all
required frozen per-seed metrics/probabilities under `AUTHORIZED_INPUT_ROOT`,
with the same `_data/` and `_pipelines/` layout. For the complete number builder,
this means all 22 registered architecture/input/design runs with 10 seeds each.
Training is performed separately by the benchmark steps. The public analysis
commands themselves do not train and write only aggregate reports/tables and
ignored numerical caches. Recomputing the real paper numbers from released
aggregates alone is not possible.

`results/support/` supplies additional aggregate reference/significance,
screen-fold and HPO-noise records. The HPO-noise top-five settings are included
so their training-well CV reevaluation does not require the private study DB:

```bash
python _analysis/rerun_hpo_noise.py --help
python _analysis/rerun_hpo_noise.py --model "VDL-ModernTCN (mean pool)" --data-root AUTHORIZED_INPUT_ROOT --device cuda
```

This last optional command **does train** on A–D cross-validation; E/F are not
used. It is separate from the training-free paper analysis and has not been
rerun for this release. No new experiment or original manuscript number is
introduced by the analysis supplement.

## Repository contents

- `_data/raw/`: complete anonymised six-well curve CSVs (`A.csv`–`F.csv`),
  aligned VDL wavetrain arrays (`A_vdl.npy`–`F_vdl.npy`), and SHA-256
  `manifest.json` (`55,155` depth samples across 139 contiguous logging segments).
- `_code/da_reformer/`: preprocessing, sequence backbones, `BondFusionTCN`
  (`models/bondfusion_tcn.py`), `BondFusionLightningModule`
  (`lightning_module.py`), evaluation metrics, HPO, and CPU invariant tests
  (`tests/`).
- `_pipelines/01_run_cement_bond_benchmark/`: symmetric six-well nested LOWO CV
  scripts (`run_lightning_bondfusion_v04.py`, `build_four_pillar_tables_v04.py`,
  `build_v04_cag_figures.py`, `cache_missing_d_baselines_v04.py`) alongside the
  historical `v03` protocol and frozen HPO selections.
- `_figures/`: publication figure scripts (`fig_02`–`fig_07`, `fig_S1`–`fig_S2`),
  number builders, and plot styling (`_reproduction/`).
- `results/v04/` & `results/`: `v04` four-pillar summary tables, 139-segment
  block-bootstrap significance records, and `v03` reference tables.
- `_analysis/`: training-free statistical analysis and table formatters.
- `scripts/`: synthetic CPU verification and publication-boundary audit.

Model weights, sample-level checkpoints, unpublished manuscripts, consultation
histories, and personal or platform configuration are excluded.
Changes and release versions are tracked by Git.

## License

Original project code: MIT, copyright (c) 2026 Yongan Zhang et al.
ModernTCN and the vendored TSLib subset retain their original MIT licenses.
See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
