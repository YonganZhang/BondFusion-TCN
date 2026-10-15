# Third-party software

The root MIT license applies to original project code. It does not replace the
licenses or notices for the following components.

- **ModernTCN**, copyright (c) 2024 luodhhh, MIT. Official source:
  https://github.com/luodhhh/ModernTCN, commit
  `56a9a2c018385cd5acef015378cae7f084d1b11c`,
  `ModernTCN-detection/models/ModernTCN.py`. Adapted in
  `_code/da_reformer/models/moderntcn.py`; its header lists the changes and
  retains the upstream copyright and license. The official license is also
  preserved verbatim in `LICENSES/ModernTCN-MIT.txt`.
- **Time-Series-Library**, MIT. Official source:
  https://github.com/thuml/Time-Series-Library, commit
  `4e938a1767106324dd753b2a44832bf870a0252e`. The selected source subset and
  verbatim license are in `_code/da_reformer/vendor/tslib/`; `VENDOR.md`
  describes import changes and removal of per-window normalization.

Other libraries are installed as dependencies rather than copied into this
repository; their licenses remain applicable. No raw data is licensed or
redistributed by this code release.
