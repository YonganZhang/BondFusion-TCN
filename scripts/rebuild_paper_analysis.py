"""Rebuild numerical paper analysis from authorized local inputs; never train models."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["numbers", "post-hoc", "tables", "inspect"])
    ap.add_argument("--data-root", type=Path, default=ROOT,
                    help="directory containing authorized _data and frozen pipeline outputs")
    ap.add_argument("--output", type=Path, default=ROOT / "_outputs/p04_v03_paper",
                    help="directory for aggregate outputs (no per-depth predictions are written)")
    ap.add_argument("--from-release", action="store_true",
                    help="tables only: format released aggregates, without recomputing from raw inputs")
    args = ap.parse_args()
    if args.from_release and args.mode != "tables":
        ap.error("--from-release is available only for tables")
    os.environ["BONDFUSION_DATA_ROOT"] = str(args.data_root.resolve())
    # Keep all derived caches under the explicitly chosen output directory.
    os.environ["BONDFUSION_CACHE_DIR"] = str(args.output.resolve() / ".cache")
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = str(min(8, int(os.environ.get(key, "8"))))
    sys.path.insert(0, str(ROOT / "_analysis"))
    import _paper_data as P
    P.OUT_DIR = args.output.resolve()
    if args.mode == "inspect":
        import _post_hoc
        import build_paper_numbers
        import build_supplement_tables
        nums = json.loads((ROOT / "results/paper_numbers.json").read_text())
        print(json.dumps({"analysis_imports": "passed", "frozen_sections": list(nums),
                          "post_hoc_sections": list(nums["post_hoc"]),
                          "statistics_definition": "original frozen analysis; plotting dependencies removed"}))
        return 0
    if args.mode in {"numbers", "post-hoc"}:
        raw = args.data_root / "_data/raw"
        final = args.data_root / "_pipelines/01_run_cement_bond_benchmark/outputs/v03/final"
        if not raw.is_dir() or not final.is_dir():
            ap.error("authorized raw inputs and frozen per-seed final outputs are required; released aggregates are insufficient")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "numbers":
        import build_paper_numbers
        build_paper_numbers.main()
    elif args.mode == "post-hoc":
        import _post_hoc
        result = _post_hoc.compute()
        (args.output / "post_hoc.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float) + "\n")
        print("wrote post_hoc.json")
    elif args.mode == "tables":
        if args.from_release:
            names = ["paper_numbers.json", "table_02_main.csv", "table_03_designs.csv",
                     "table_S1_hyperparameters.csv", "table_S1_search_space.md",
                     "table_S2_per_well.csv", "table_S3_significance.csv"]
            for name in names:
                shutil.copy2(ROOT / "results" / name, args.output / name)
        import build_supplement_tables
        build_supplement_tables.OUT = args.output.resolve()
        build_supplement_tables.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
