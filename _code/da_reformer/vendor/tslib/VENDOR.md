# Vendored: Time-Series-Library (TSLib)

- Upstream: https://github.com/thuml/Time-Series-Library (MIT, see `LICENSE` in this directory)
- Upstream commit: `4e938a1767106324dd753b2a44832bf870a0252e` (2026-04-18, "Update README.md"); the local clone had no uncommitted changes when copied (`git status --short` empty).
- Copied on 2026-10-10. Import as `da_reformer.vendor.tslib.<subpkg>.<module>` with `PYTHONPATH=_code`.
- Extra runtime deps used by these files: `einops`, `reformer_pytorch` (1.4.4).

## Files

| File | Status |
|---|---|
| `LICENSE` | verbatim |
| `layers/Embed.py` | verbatim |
| `layers/Transformer_EncDec.py` | verbatim |
| `layers/Conv_Blocks.py` | verbatim |
| `layers/Autoformer_EncDec.py` | verbatim |
| `layers/SelfAttention_Family.py` | import rewritten (1) |
| `layers/Crossformer_EncDec.py` | import rewritten (1) |
| `utils/masking.py` | verbatim |
| `models/TimesNet.py` | imports rewritten (1); `class Model` removed (2) |
| `models/TimeMixer.py` | imports rewritten (1); `StandardNorm` import and `class Model` removed (2) |
| `models/PatchTST.py` | only `Transpose` / `FlattenHead` kept; `class Model` and its imports removed (2) |
| `__init__.py`, `layers/__init__.py`, `models/__init__.py`, `utils/__init__.py` | new, empty (package markers) |

## Modifications

1. Absolute imports (`from layers.X`, `from utils.masking`) changed to package-relative imports (`from .X`, `from ..layers.X`, `from ..utils.masking`). No other line in the kept code changed.
2. The upstream `Model` classes of TimesNet / TimeMixer / PatchTST were deleted. They contain per-window instance normalization (Non-stationary `means/stdev` norm and de-norm, RevIN-style `Normalize`), which this project forbids because the absolute log amplitude is physically meaningful. Only the building blocks are kept (`FFT_for_Period`, `TimesBlock`; `DFT_series_decomp`, `MultiScaleSeasonMixing`, `MultiScaleTrendMixing`, `PastDecomposableMixing`; `Transpose`, `FlattenHead`). The point-wise model wrappers live in `../../models/baselines.py` and `../../models/backbones.py`; they mirror the upstream `imputation` / `anomaly_detection` forward paths with the normalization removed.

Upstream `models/Transformer.py`, `models/Reformer.py` and `models/Crossformer.py` were not copied: their encoders are rebuilt from the vendored `layers/` in `backbones.py` / `baselines.py`.

To check for drift: `diff -r <upstream>/layers <this dir>/layers` should show only the import lines above.
