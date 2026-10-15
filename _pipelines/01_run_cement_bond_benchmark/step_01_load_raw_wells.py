"""Step 01: extract the four logging curves + label of wells A-F into _data/raw/.

Source of truth is the original archive `固井数据集.rar` (Gaia 1.5 project). Its
A.csv member is truncated (unar: "Attempted to read more data than was
available"), so well A is read from an intact copy whose first bytes must match
the truncated member byte-for-byte and whose size must equal the size recorded
in the archive listing.

Outputs per well: `<W>.csv` (DEPTH/GR/AC/CBL/CCL/label_raw, original tokens) and
`<W>_vdl.npy` (VDL_1..VDL_2000 waveform, int16 [N, 2000], same row order; the raw
values are integers inside the int16 range, -32768 is kept as recorded). The .npy
files are not tracked by Git; their sha256 is written to the manifest.

Usage:
    python step_01_load_raw_wells.py --rar <固井数据集.rar> --well-a <intact A.csv> [--out _data/raw]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEEP = ["DEPTH", "GR", "AC", "CBL", "CCL", "类型"]
OUT_COLS = ["DEPTH", "GR", "AC", "CBL", "CCL", "label_raw"]
WELLS = ["A", "B", "C", "D", "E", "F"]
A_SIZE_IN_ARCHIVE = 73467548
VDL_COLS = [f"VDL_{i}" for i in range(1, 2001)]


def load_member(rar: Path, member: str) -> bytes:
    proc = subprocess.run(["unar", "-q", "-o", "-", str(rar), member],
                          capture_output=True, stdin=subprocess.DEVNULL)
    return proc.stdout


def build_subset(raw: bytes) -> tuple[list[list[str]], int]:
    csv.field_size_limit(10**8)
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig")))
    header = next(reader)
    idx = [header.index(c) for c in KEEP]
    rows, skipped = [], 0
    for row in reader:
        if len(row) != len(header):
            skipped += 1
            continue
        rows.append([row[i] for i in idx])
    return rows, skipped


def build_vdl(raw: bytes, depth: list[str]) -> np.ndarray:
    df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", usecols=["DEPTH"] + VDL_COLS)
    if len(df) != len(depth) or not np.array_equal(df["DEPTH"].to_numpy(np.float64), np.array(depth, np.float64)):
        raise SystemExit("VDL rows do not line up with the curve rows")
    v = df[VDL_COLS].to_numpy(np.float64)
    if not np.array_equal(v, np.round(v)) or v.min() < -32768 or v.max() > 32767:
        raise SystemExit("VDL values are not int16 integers")
    return v.astype(np.int16)


def run_extract(rar: Path, well_a: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"rar": str(rar), "rar_sha256": hashlib.sha256(rar.read_bytes()).hexdigest(), "wells": {}}
    for w in WELLS:
        member = load_member(rar, f"results/{w}.csv")
        if w == "A":
            full = well_a.read_bytes()
            if len(full) != A_SIZE_IN_ARCHIVE or full[: len(member)] != member:
                raise SystemExit("intact A.csv does not match the archive member prefix/size")
            source, raw = str(well_a), full
        else:
            source, raw = f"{rar}::results/{w}.csv", member
        rows, skipped = build_subset(raw)
        if skipped:
            raise SystemExit(f"well {w}: {skipped} malformed rows")
        path = out / f"{w}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(OUT_COLS)
            wr.writerows(rows)
        vdl = build_vdl(raw, [r[0] for r in rows])
        vdl_path = out / f"{w}_vdl.npy"
        np.save(vdl_path, vdl)
        manifest["wells"][w] = {
            "source": source,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "rows": len(rows),
            "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "vdl_shape": list(vdl.shape),
            "vdl_sha256": hashlib.sha256(vdl_path.read_bytes()).hexdigest(),
        }
        print(f"{w}: {len(rows)} rows -> {path}")
    (out / "_extract_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rar", type=Path, required=True)
    ap.add_argument("--well-a", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "_data" / "raw")
    args = ap.parse_args(argv)
    run_extract(args.rar, args.well_a, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
