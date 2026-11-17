# Anonymised six-well cased-hole logging dataset

This directory distributes the complete anonymised six-well dataset (Wells **A–F**, `55,155` depth samples across `139` contiguous logging segments) used in the BondFusion-TCN (`v04`) and `v03` benchmarks. For each well `<W>` in `{A, B, C, D, E, F}`:

- `<W>.csv`: UTF-8 CSV, one row per depth, columns `DEPTH`, `GR`, `CBL`, `CCL`,
  `AC`, `label_raw`. `AC` is only needed for the benchmark audit and optional
  reference diagnostics; it is excluded from the main model. Values <= -999
  are treated as missing by the curve loader. Depth must be nondecreasing.
- `<W>_vdl.npy`: numeric array `[N, 2000]` (`int16`), exactly the
  same rows/order as the curve CSV. The protocol uses the first 1,900 samples.
- `manifest.json`: SHA-256 checksums, row counts, and array shapes for all six wells.
- `label_raw`: `胶结差` (Poor), `胶结中等` (Moderate), or one of `胶结好`,
  `胶结良好`, `胶结优` (Good). Add an explicit mapping for other label systems.

All wells A–F participate symmetrically under the v04 six-well nested leave-one-well-out (LOWO) cross-validation protocol (`run_lightning_bondfusion_v04.py`). The historical v03 configuration (`protocol_v03.yaml`) used A–D for training/selection and E/F as held-out evaluation wells.
