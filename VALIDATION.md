# Release validation

Validated on 2026-10-16. Runtime checks used Python 3.11 and the direct package
versions in `requirements.txt` (PyTorch 2.11.0+cu128 installed, CPU selected).

- `python scripts/smoke_test.py`: passed on CPU. Temporary synthetic wells only;
  envelope calculation, training-only curve/VDL scaling, segment-aware one-epoch
  training, per-depth probability coverage/normalization, and learned waveform
  encoder gradients were exercised. No real-data model was retrained.
- Eight pipeline `step_*.py --help` commands: passed, no stderr output.
- All 36 Python source files: parsed successfully.
- Wheel packaging: built successfully with no dependency installation; inspected
  33 wheel entries. The ModernTCN implementation and TSLib license were included,
  and the abandoned dual-adversarial implementation was absent. The build host
  used Python 3.10 with `--ignore-requires-python` solely to construct metadata;
  Python 3.11 was used for all runtime checks. No claim of Python 3.10 runtime
  support is made. Temporary build products were removed.
- `python scripts/audit_release.py`: passed, zero findings for recognized
  credentials, personal paths, email addresses, private configuration, raw
  arrays/weights/databases, unexpected binaries, large files and symlinks.

The audit is a defined pattern/file-boundary check, not proof against every
possible secret encoding. Original private Git history is not part of this
release. For the initial v0.1.0 release, full original-data recomputation was not
performed. The v0.1.1 supplement below adds that validation. Fresh-environment
dependency installation, GPU training, and exact historical HPO/screen replay
remain untested. Data access and historical-replay limits are in README.md.

## v0.1.1 numerical analysis supplement

- The public `rebuild_paper_analysis.py numbers` entry was executed against the
  authorized original local inputs and frozen per-seed outputs, on CPU with
  thread limits of 8, without training. Temporary cache/output files were
  removed after validation and are excluded from the release.
- All **2,023** checks against the frozen source summaries passed (maximum
  source-check difference 2.22e-16). All **3,773** numeric fields in the complete
  recomputed paper JSON matched the frozen artifact exactly, including the
  post hoc decision, cost, Good-class/bootstrap and direct design comparisons.
  See `ANALYSIS_VALIDATION.json` for the complete machine-readable evidence.
- Two non-statistical strings differed as expected: the public numerical module
  path and the adopted BondFusion-TCN display name in the design row. The
  distributed frozen `results/paper_numbers.json` remains byte-identical to
  its original source, with every original value preserved.
- `_post_hoc.py` is byte-identical to the original. The interval-selection
  function has the same Python AST. Only plotting/environment dependencies,
  input/output resolution and table column-label compatibility were adapted.
- New analysis entry and HPO-noise reevaluation `--help` checks passed; `inspect`
  imported all numerical modules without shared plotting/font files. The
  optional HPO-noise training entry was not executed.
- Formatting the public aggregates with `tables --from-release` successfully
  recreated seven supplementary table fragments in a temporary directory.
- All 43 Python files parsed successfully. Publication-boundary audit passed
  with zero findings. No raw or sample-level files are included.
