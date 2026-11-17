"""Check public release boundaries without printing suspected secret values."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WELLS = ("A", "B", "C", "D", "E", "F")
PUBLIC_RAW_FILES = {
    "_data/raw/README.md",
    "_data/raw/manifest.json",
    *{f"_data/raw/{w}.csv" for w in WELLS},
    *{f"_data/raw/{w}_vdl.npy" for w in WELLS},
}
PUBLIC_RAW_VDL_ARRAYS = {f"_data/raw/{w}_vdl.npy" for w in WELLS}
MAX_PUBLIC_RAW_BYTES = 80 * 1024 * 1024

SECRET_PATTERNS = {
    "github_token": re.compile(r"gh[pousr]_" + r"[A-Za-z0-9]{20,}|github_pat_" + r"[A-Za-z0-9_]{20,}"),
    "openai_key": re.compile(r"sk-" + r"(?:proj-)?[A-Za-z0-9_-]{30,}"),
    "aws_key": re.compile(r"AKIA" + r"[A-Z0-9]{16}"),
    "private_key": re.compile(r"-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential_url": re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
    "personal_path": re.compile(r"/(?:mnt/data|home|Users)/[^\s\"']+"),
    "email_address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    findings = []
    count = 0
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        rel_posix = rel.as_posix()
        if ".git" in rel.parts or "__pycache__" in rel.parts or p.suffix == ".pyc":
            continue
        count += 1
        if p.is_symlink():
            findings.append({"path": str(rel), "check": "symlink"})
        if p.suffix in {".npz", ".pt", ".pth", ".db", ".pdf", ".rar", ".zip"}:
            findings.append({"path": str(rel), "check": "excluded_binary_or_data"})
        if p.suffix == ".npy" and rel_posix not in PUBLIC_RAW_VDL_ARRAYS:
            findings.append({"path": str(rel), "check": "excluded_binary_or_data"})
        if p.name in {"AGENTS.md", "CLAUDE.md", ".env"}:
            findings.append({"path": str(rel), "check": "private_configuration"})
        if "_data" in rel.parts and rel_posix not in PUBLIC_RAW_FILES:
            findings.append({"path": str(rel), "check": "raw_data"})
        size_limit = MAX_PUBLIC_RAW_BYTES if rel_posix in PUBLIC_RAW_VDL_ARRAYS else 10 * 1024 * 1024
        if p.stat().st_size > size_limit:
            findings.append({"path": str(rel), "check": "large_file"})
        if rel_posix in PUBLIC_RAW_VDL_ARRAYS:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append({"path": str(rel), "check": "unexpected_nontext"})
            continue
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append({"path": str(rel), "check": name})
    print(json.dumps({"files_scanned": count, "findings": findings, "status": "passed" if not findings else "failed"}))
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
