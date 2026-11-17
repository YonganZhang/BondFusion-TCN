"""Check that every number written in the manuscript text can be traced to paper_numbers.json.

Scans _paper/main.tex (abstract, highlights) and _paper/sections/*.tex (prose only; generated tables are skipped because they are built
from the same JSON), extracts decimal numbers and comma-grouped integers, and looks each one up
among the values of _outputs/p04_v03_paper/paper_numbers.json under the usual roundings
(3/2/1 decimals, percentages with 1 decimal, integers). Unmatched numbers are printed with
their context so that a person can confirm or correct them; exit code 1 if any remain.

Usage: python _figures/check_paper_numbers.py [--allow 2000 1900 ...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMBERS = ROOT / "_outputs" / "p04_v03_paper" / "paper_numbers.json"
SECTIONS = ROOT / "_paper" / "sections"
TOKEN = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+)(?![\w])")
# protocol constants that are design settings, not results
DEFAULT_ALLOW = {"2000", "1900", "190", "19", "10", "100", "256", "64", "128", "150", "25", "30", "32", "48", "7", "5",
                 "0.5", "0.1", "25.6", "1.349", "1.0", "2.5", "97.5", "97", "95", "8", "3", "2", "4", "1", "0", "17", "16", "6"}

# numbers quoted from the literature (source: _paper/bib/_verification.md)
LITERATURE = {"86.1", "95.7", "93.2", "91.5"}


def flatten(obj, out: set[str]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            flatten(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            flatten(v, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        x = float(obj)
        for d in (3, 2, 1):
            out.add(f"{x:.{d}f}")
            out.add(f"{abs(x):.{d}f}")
            out.add(f"{100 * x:.{d}f}")
            out.add(f"{abs(100 * x):.{d}f}")
        out.add(f"{100 * x:.0f}")   # whole percentages
        if abs(x - round(x)) < 1e-9:
            out.add(str(int(round(x))))
            out.add(str(abs(int(round(x)))))


def strip_tex(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)                                   # comments
    text = re.sub(r"\\(ref|label|cite[pt]?|citep|citet|papertable|paperfig|input|includegraphics|ead|url)\{[^}]*\}", "", text)
    text = re.sub(r"\\authornote\{[^}]*\}", "", text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow", nargs="*", default=[], help="extra literal tokens to accept")
    args = ap.parse_args(argv)
    known: set[str] = set()
    flatten(json.loads(NUMBERS.read_text(encoding="utf-8")), known)
    allow = DEFAULT_ALLOW | LITERATURE | set(args.allow)
    misses = 0
    for tex in [SECTIONS.parent / "main.tex", *sorted(SECTIONS.glob("*.tex"))]:
        for ln, line in enumerate(strip_tex(tex.read_text(encoding="utf-8")).splitlines(), 1):
            for m in TOKEN.finditer(line):
                tok = m.group(1)
                plain = tok.replace(",", "")
                if tok in allow or plain in allow or plain in known:
                    continue
                misses += 1
                ctx = line[max(0, m.start() - 50): m.end() + 30].strip()
                print(f"{tex.name}:{ln}: {tok}  …{ctx}…")
    print(f"unmatched numbers: {misses}")
    return 1 if misses else 0


if __name__ == "__main__":
    sys.exit(main())
