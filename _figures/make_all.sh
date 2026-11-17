#!/usr/bin/env bash
# Regenerate every P04 (v03) paper figure, table and paper_numbers.json from existing outputs (CPU only,
# no training). Usage: bash _figures/make_all.sh   (override the interpreter with PY=/path/to/python)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" CUDA_VISIBLE_DEVICES=""
"$PY" build_paper_numbers.py   # verification first: exits non-zero on any mismatch
"$PY" build_supplement_tables.py
for f in fig_02_data_vdl fig_03_error_direction fig_04_vdl_agreement fig_05_encoder_designs fig_06_good_failure \
         fig_07_cv_vs_blind fig_S1_confusion_per_backbone fig_S2_per_well; do
  echo "== $f"
  "$PY" "$f.py"
done
# Fig. 1: restore the pinned editable-PPT export; no network or image-generation calls.
"$PY" build_bondfusion_figure.py
"$PY" check_paper_numbers.py
if grep -nE 'set_title|plt\.title|suptitle\(' fig_*.py; then echo "title call found" >&2; exit 1; fi
# figure scripts must not put "proposed/ours/ablation" into figures (code names live in _paper_data.py)
if grep -nE '[Pp]roposed|\bours\b|[Aa]blation' fig_*.py; then echo "forbidden word in figure code" >&2; exit 1; fi
echo "done"
