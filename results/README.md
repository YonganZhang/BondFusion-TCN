# Aggregate results

These small tables come from real v03 benchmark runs and the paper table
builder, not from the synthetic smoke test.

- `summary_merged.csv` and `summary_per_well.csv`: numerical seed means,
  sample standard deviations and available ensemble metrics, with original
  protocol model identifiers.
- `hpo_summary.csv`: selected CV scores, trial counts and hyperparameters.
- `screen_summary.csv`: training-well CV design-screen results.
- `significance_vdl_pairs.json`: depth-segment bootstrap for paired VDL inputs.
- `table_02_main.csv`, `table_02_paired_accuracy.csv`, `table_03_designs.csv`,
  `table_S3_significance.csv`: manuscript-facing aggregate tables (display
  rounding). `ModernTCN +VDL` denotes BondFusion-TCN.

Only table column wording `Blind` was changed to `Held-out` in the release;
numerical values are unchanged. E/F were used in earlier benchmark versions.
Raw inputs, per-depth predictions and study databases are not included.
Recomputation requires authorized data and the complete relevant final runs.
