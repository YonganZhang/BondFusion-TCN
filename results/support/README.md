# Aggregate analysis inputs

These are anonymized aggregate outputs needed by the numerical analysis:
reference baselines, ensemble significance versus mean pooling, per-screen-seed
CV summaries, and HPO selection-noise CV summaries plus the original top-five
configuration settings. They contain model/fold metrics and settings, with wells
identified only as A–F; there are no raw measurements or per-depth predictions.

The selected HPO settings were read from the original frozen study in read-only
mode. `hpo_noise_configs.json` permits the CV reevaluation to be rerun without
redistributing the study database. That optional reevaluation performs training;
the paper-number and post-hoc analysis commands do not train models.
