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
release. Full original-data recomputation, fresh-environment dependency
installation, GPU training, and exact historical HPO/screen replay were not
performed. Data access and historical-replay limits are described in README.md.
