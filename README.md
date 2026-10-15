# BondFusion-TCN

Research code for point-wise cement-bond quality classification into **Poor**,
**Moderate**, and **Good** using logging curves and variable-density log (VDL)
waveforms. BondFusion-TCN combines GR, transformed CBL, and transformed CCL with
19 log-envelope VDL blocks, followed by a ModernTCN backbone and a point-wise
classification head. It is a task-specific fusion design using an existing
backbone; it does not introduce a new convolution operator.

## Model names and study scope

**BondFusion-TCN is registered as `ModernTCN +VDL` in the frozen v03 protocol.**
The code package name `da_reformer` is retained for import compatibility.
The protocol key `models.proposed` records the earlier training-well screen's
selection, `VDL-ModernTCN (mean pool)`, a learned waveform encoder design. It is
preserved as a design comparison and is not the manuscript's BondFusion-TCN.
Use `--model "ModernTCN +VDL"` to reproduce the manuscript model.

The benchmark pairs eight sequence backbones (LSTM, GRU, Transformer, Reformer,
Crossformer, TimesNet, TimeMixer, and ModernTCN) with curves-only and curves-plus-VDL
inputs under equal hyperparameter-search budgets. Additional waveform encoder
designs share the ModernTCN backbone. Preprocessing statistics are fitted only
on the training wells of each split. The protocol specifies training-well
leave-one-well-out cross-validation for selection and 10 final training seeds.

Wells A–D were used for training and selection; E/F were held out for the v03
final evaluation. E/F had already been evaluated in previous benchmark versions,
and the choice of ModernTCN was informed by those earlier results. They should
not be described as a newly acquired or untouched external blind test.

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

**This release does not redistribute the underlying well logs or VDL waveforms.**
No public data download or permission to redistribute the source dataset is
provided here. Obtain data from its rights holder under appropriate permission,
or prepare your own authorized data using the schema in `_data/raw/README.md`.
The code license does not grant rights to any dataset. Released aggregate tables
can be inspected without raw data; independent recomputation requires the data.

For the original authorized archive, `step_01_load_raw_wells.py` extracts the
curve CSVs and aligned waveform arrays. This extractor requires the external
`unar` command and the original archive plus its intact well-A CSV. It performs
an archive-specific size and prefix check; it is not a universal CSV converter.
For other datasets, prepare the documented arrays directly and edit a copy of
the protocol's well identifiers and paths. The fixed-count audit in step 02
applies to the original six-well benchmark.

## Reproduce the v03 pipeline

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

## Repository contents

- `_code/da_reformer/`: preprocessing, current sequence models, training,
  evaluation, and HPO. Abandoned dual-adversarial model implementations are
  excluded; retained generic training hooks are inactive for released models.
- `_pipelines/01_run_cement_bond_benchmark/`: protocol, executable steps, and
  frozen hyperparameters. Runtime outputs remain local.
- `results/`: aggregate tables from real v03 runs, without sample-level logs
  or predictions. `results/README.md` describes the tables.
- `scripts/`: a synthetic CPU smoke test and a publication-boundary audit.
- `RELEASE_PROVENANCE.json`: source commit and file hashes for this distribution.

Raw logs, model weights, sample-level predictions, unpublished manuscripts,
consultation histories, and personal or platform configuration are excluded.
Changes and release versions are tracked by Git.

## License

Original project code: MIT, copyright (c) 2026 Yongan Zhang et al.
ModernTCN and the vendored TSLib subset retain their original MIT licenses.
See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
