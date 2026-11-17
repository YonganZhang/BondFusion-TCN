"""Restore Fig. 1 from the pinned editable-PowerPoint export (no network or generation)."""
from pathlib import Path
import hashlib
import json
import shutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "_figures/figure_01_source.json"


def main():
    info = json.loads(META.read_text())
    for key, digest in [("source_pptx", "sha256_pptx"), ("source_pdf", "sha256_pdf"), ("source_png", "sha256")]:
        source = ROOT / info[key]
        if hashlib.sha256(source.read_bytes()).hexdigest() != info[digest]:
            raise ValueError(f"Pinned Figure 1 source mismatch: {key}")
    with Image.open(ROOT / info["source_png"]) as artwork:
        if artwork.format != "PNG" or artwork.size != (4800, 2700):
            raise ValueError("Figure 1 requires the pinned 4800 x 2700 PPT export")
    # Preserve vector labels and embedded Times New Roman from the actual PPT export.
    shutil.copyfile(ROOT / info["source_pdf"], ROOT / "_figures/fig_01_study_design.pdf")
    shutil.copyfile(ROOT / info["source_png"], ROOT / "_figures/fig_01_study_design.png")
    print("Figure 1: editable PPT and its PDF/PNG export hashes verified")


if __name__ == "__main__":
    main()
