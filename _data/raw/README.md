# Authorized local data

Place authorized data here; raw files are excluded from version control.
For each well `<W>`, provide:

- `<W>.csv`: UTF-8 CSV, one row per depth, columns `DEPTH`, `GR`, `CBL`, `CCL`,
  `AC`, `label_raw`. `AC` is only needed for the benchmark audit and optional
  reference diagnostics; it is excluded from the main model. Values <= -999
  are treated as missing by the curve loader. Depth must be nondecreasing.
- `<W>_vdl.npy`: numeric array `[N, 2000]` (reference data: int16), exactly the
  same rows/order as the curve CSV. The protocol uses the first 1900 samples.
- `label_raw`: `胶结差` (Poor), `胶结中等` (Moderate), or one of `胶结好`,
  `胶结良好`, `胶结优` (Good). Add an explicit mapping for other label systems.

The v03 protocol expects A–F with A–D training and E/F evaluation; change the
protocol for other data. No waveforms or CSV data are distributed here.
