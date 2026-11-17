"""Shared data access, derived quantities and plot style for the P04 (v04) paper figures.

Every number comes from verified v04 outputs and the unified PyTorch Lightning 4-slot benchmark:
- labels / curves / VDL: `_data/raw/` via `da_reformer.data.load_well` and `da_reformer.vdl`;
- 6-well held-out CV probabilities for baselines: team-1384 `outputs/v04/deep/<W>/<study>/probs_<W>_cv.npy`
  and `outputs/v04/{baselines,ablations}/`;
- BondFusion-TCN & single-factor ablations: `_outputs/p04_v04_lightning/runs_final/<variant>/<W>/probs_<W>.npy`;
- summaries & 139-segment block-bootstrap significance: `_outputs/p04_v04_lightning/`.
"""
from __future__ import annotations

import csv
import functools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[1]
PIPE = ROOT / "_pipelines" / "01_run_cement_bond_benchmark"
V04_OUT = Path(os.environ.get("BONDFUSION_V04_OUT", str(ROOT / "_pipelines/01_run_cement_bond_benchmark/outputs/v04")))
V04_LIGHTNING = ROOT / "_outputs" / "p04_v04_lightning"
RUNS_FINAL = V04_LIGHTNING / "runs_final"
FIG_DIR = ROOT / "_figures"
OUT_DIR = V04_LIGHTNING
CACHE = ROOT / "_tmp" / "p04_v04_paper_cache"
RAW = ROOT / "_data" / "raw"

sys.path.insert(0, str(ROOT / "_code"))
from da_reformer.data import CLASS_NAMES, load_well  # noqa: E402
from da_reformer.metrics import compute_metrics  # noqa: E402
from da_reformer.vdl import load_vdl  # noqa: E402

ALL_WELLS = ["A", "B", "C", "D", "E", "F"]
TRAIN_WELLS = ["A", "B", "C", "D", "E", "F"]
TEST_WELLS = ALL_WELLS
GAP_M = 0.5
N_KEEP = 1900
BIN = 20
CLASSES = list(CLASS_NAMES)  # 0 Poor, 1 Moderate, 2 Good

BACKBONES = ["LSTM", "GRU", "Transformer", "Reformer", "Crossformer", "TimesNet", "TimeMixer", "ModernTCN"]


def vdl(b: str) -> str:
    return f"{b} +VDL"


MEAN_POOL = "VDL-ModernTCN (mean pool)"
REF_CURVES = "ModernTCN"
REF_VDL = "ModernTCN +VDL"
REF_DISPLAY = {REF_CURVES: "ModernTCN, Curves", REF_VDL: "BondFusion-TCN"}

ABLATION_FOLDERS = {
    "BondFusion-TCN": "BondFusion-TCN_Full",
    "BondFusion-TCN (Full)": "BondFusion-TCN_Full",
    "w/o Zero-phase filter": "w-o_Zero-phase_filter",
    "w/o Trace contrast": "w-o_Trace_contrast",
    "w/o CBL-gated bridge": "w-o_CBL-gated_bridge",
    "w/o Hilbert envelope": "w-o_Hilbert_envelope",
    "w/o VDL (Curve-Prior only)": "w-o_VDL_Curve-Prior_only",
}
ABLATION_ORDER = [
    "BondFusion-TCN",
    "w/o Zero-phase filter",
    "w/o Trace contrast",
    "w/o CBL-gated bridge",
    "w/o Hilbert envelope",
    "ModernTCN +VDL (raw)",
    "VDL-ModernTCN (mean pool)",
    "w/o VDL (Curve-Prior only)",
]

STUDY_DIR = {
    "LSTM": "LSTM",
    "LSTM +VDL": "LSTM_plusVDL",
    "GRU": "GRU",
    "GRU +VDL": "GRU_plusVDL",
    "Transformer": "Transformer",
    "Transformer +VDL": "Transformer_plusVDL",
    "Reformer": "Reformer",
    "Reformer +VDL": "Reformer_plusVDL",
    "Crossformer": "Crossformer",
    "Crossformer +VDL": "Crossformer_plusVDL",
    "TimesNet": "TimesNet",
    "TimesNet +VDL": "TimesNet_plusVDL",
    "TimeMixer": "TimeMixer",
    "TimeMixer +VDL": "TimeMixer_plusVDL",
    "ModernTCN": "ModernTCN",
    "VDL-ModernTCN (mean pool)": "VDL-ModernTCN_(mean_pool)",
    "2D-CNN +VDL": "2D-CNN_plusVDL",
}


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in name)


def display_name(name: str) -> str:
    if name in REF_DISPLAY:
        return REF_DISPLAY[name]
    if name in ABLATION_FOLDERS:
        return name
    if name.endswith(" +VDL"):
        return f"{name[:-5]}, Curves + VDL"
    if name in BACKBONES:
        return f"{name}, Curves"
    return name


def input_set(name: str) -> str:
    """'Curves' or 'Curves + VDL'."""
    if name in BACKBONES or "w/o VDL" in name or "Curves" in name or "CBL Two-Threshold" in name:
        return "Curves"
    return "Curves + VDL"


# ----------------------------------------------------------------------------------------------
# Raw data
# ----------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def well(w: str):
    """WellData with .depth, .y, .segments."""
    return load_well(RAW, w, ["CBL"], GAP_M)


@functools.lru_cache(maxsize=None)
def curves(w: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / f"{w}.csv")
    for c in ("GR", "AC", "CBL", "CCL"):
        df.loc[df[c] <= -999.0, c] = np.nan
    if not np.array_equal(df["DEPTH"].to_numpy(), well(w).depth):
        raise ValueError(f"{w}: csv depth differs from load_well")
    return df


def vdl_raw(w: str) -> np.ndarray:
    return load_vdl(RAW, w, N_KEEP)


@functools.lru_cache(maxsize=None)
def log_envelope(w: str) -> np.ndarray:
    """[N, 95] analytic Hilbert log-envelope (bin=20 us) matching Eq. (1)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    src = RAW / f"{w}_vdl.npy"
    f = CACHE / f"hilbert_logenv_bin{BIN}_{w}.npz"
    sig = f"{src.stat().st_size}:{src.stat().st_mtime_ns}"
    if f.exists():
        z = np.load(f)
        if str(z["sig"]) == sig:
            return z["env"]
    raw = vdl_raw(w).astype(np.float64)
    amp = np.abs(hilbert(raw, axis=1))
    n_b = N_KEEP // BIN
    env = np.log1p(amp[:, : n_b * BIN].reshape(len(amp), n_b, BIN).mean(axis=2)).astype(np.float32)
    np.savez(f, env=env, sig=sig)
    return env


def y_true(wells: list[str] | None = None) -> np.ndarray:
    return np.concatenate([well(w).y for w in (wells or TEST_WELLS)])


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(int(a), int(b)) for a, b in zip(e[::2], e[1::2])]


# ----------------------------------------------------------------------------------------------
# v04 Model Outputs (Full 6-Well Nested CV)
# ----------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def ensemble_probs(name: str) -> dict[str, np.ndarray]:
    """Load 6-well held-out probabilities (A-F). ModernTCN +VDL maps to BondFusion-TCN."""
    if name in (REF_VDL, "BondFusion-TCN (Ours)", "BondFusion-TCN"):
        folder = "BondFusion-TCN_Full"
        out = {w: np.load(RUNS_FINAL / folder / w / f"probs_{w}.npy") for w in ALL_WELLS}
    elif name in ABLATION_FOLDERS:
        folder = ABLATION_FOLDERS[name]
        out = {w: np.load(RUNS_FINAL / folder / w / f"probs_{w}.npy") for w in ALL_WELLS}
    elif name in ("ModernTCN +VDL (raw)", "ModernTCN +VDL (b=10)", "rectified-bin10"):
        out = {w: np.load(V04_OUT / "ablations" / w / "rectified-bin10" / f"probs_{w}_cv.npy") for w in ALL_WELLS}
    elif name == "CBL Two-Threshold Rule":
        out = {w: np.load(V04_OUT / "baselines" / w / f"probs_{w}_cbl.npy") for w in ALL_WELLS}
    elif name in STUDY_DIR:
        st = STUDY_DIR[name]
        out = {w: np.load(V04_OUT / "deep" / w / st / f"probs_{w}_cv.npy") for w in ALL_WELLS}
    else:
        raise KeyError(f"Unknown v04 model name: {name}")

    for w in ALL_WELLS:
        if out[w].shape != (len(well(w).y), 3):
            raise ValueError(f"{name}/{w}: probs {out[w].shape} vs {len(well(w).y)} labels")
    return out


def ensemble_pred(name: str) -> dict[str, np.ndarray]:
    return {w: p.argmax(1) for w, p in ensemble_probs(name).items()}


def ensemble_metrics(name: str, w: str | None = None) -> dict:
    """Pooled (55,155 samples) or single-well evaluation metrics."""
    p = ensemble_pred(name)
    pr = ensemble_probs(name)
    if w is None:
        return compute_metrics(
            y_true(),
            np.concatenate([p[x] for x in TEST_WELLS]),
            probs=np.concatenate([pr[x] for x in TEST_WELLS]),
        )
    return compute_metrics(well(w).y, p[w], probs=pr[w])


def seed_values(name: str, key: str, w: str | None = None) -> np.ndarray:
    """In v04, return the 6 outer held-out well metrics (for cross-well dots and mean ± SD)."""
    if w is not None:
        return np.array([ensemble_metrics(name, w)[key]])
    return np.array([ensemble_metrics(name, wk)[key] for wk in ALL_WELLS])


def compute_f1_kappa(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    m = compute_metrics(y, pred)
    return float(m["macro_f1"]), float(m["kappa"])


# ----------------------------------------------------------------------------------------------
# Existing v04 summaries & statistical significance
# ----------------------------------------------------------------------------------------------
def read_csv_table(name: str) -> list[dict]:
    with (V04_LIGHTNING / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


@functools.lru_cache(maxsize=None)
def summary_per_well() -> pd.DataFrame:
    rows = []
    for st_name in BACKBONES + [vdl(b) for b in BACKBONES]:
        for w in ALL_WELLS:
            m = ensemble_metrics(st_name, w)
            rows.append({"model": st_name, "well": w, "macro_f1": m["macro_f1"], "kappa": m["kappa"], "f1_Good": m["f1_Good"]})
    return pd.DataFrame(rows)


@functools.lru_cache(maxsize=None)
def sig_vdl_pairs() -> dict:
    """Adapt 05_statistical_significance.json paired_vdl_vs_curves to fig_04 expected keys."""
    raw = json.loads((V04_LIGHTNING / "05_statistical_significance.json").read_text(encoding="utf-8"))
    pv = raw["paired_vdl_vs_curves"]
    out = {}
    for b in BACKBONES:
        if b == "ModernTCN":
            r = pv["BondFusion-TCN (Ours) vs ModernTCN (Curves only)"]
        else:
            r = pv[f"{b} +VDL vs {b} (Curves only)"]
        out[f"{vdl(b)} vs {b}"] = {
            "ensemble_macro_f1_diff": r["delta_macro_f1"],
            "boot_macro_f1_diff_ci95": r["ci95_macro_f1"],
            "p_value_macro_f1": r["p_value_macro_f1"],
            "ensemble_kappa_diff": r["delta_kappa"],
            "boot_kappa_diff_ci95": r["ci95_kappa"],
            "p_value_kappa": r["p_value_kappa"],
        }
    return out


def hpo_best(name: str) -> dict:
    """Return 6-outer-fold mean inner-CV macro-F1 from verified v04 files."""
    if name in (REF_VDL, "BondFusion-TCN", "BondFusion-TCN (Ours)"):
        vals = []
        for w in ALL_WELLS:
            fvals = [json.loads((V04_OUT / "ablations" / w / "hilbert-bin20" / "cv" / str(k) / "fit.json").read_text())["best_val_macro_f1"] for k in range(5)]
            vals.append(float(np.mean(fvals)))
        return {"value": float(np.mean(vals)), "per_fold": vals}
    st = STUDY_DIR[name]
    vals = [json.loads((V04_OUT / "deep" / w / st / "hpo" / f"{st}.best.json").read_text())["value"] for w in ALL_WELLS]
    return {"value": float(np.mean(vals)), "per_fold": vals}


TUNED_MODELS = BACKBONES + [vdl(b) for b in BACKBONES] + [MEAN_POOL, "2D-CNN +VDL"]


def cv_vs_blind() -> pd.DataFrame:
    rows = []
    for m in TUNED_MODELS:
        h = hpo_best(m)
        em = ensemble_metrics(m)
        rows.append({
            "model": m,
            "display": display_name(m),
            "input_set": input_set(m),
            "cv_macro_f1": h["value"],
            "blind_macro_f1_mean": float(seed_values(m, "macro_f1").mean()),
            "blind_macro_f1_pooled": float(em["macro_f1"]),
            "blind_kappa_mean": float(seed_values(m, "kappa").mean()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Derived quantities shared by figures
# ----------------------------------------------------------------------------------------------
def class_envelope_stats(wells: list[str]) -> dict:
    env = np.concatenate([log_envelope(w) for w in wells])
    y = np.concatenate([well(w).y for w in wells])
    out = {}
    for k, c in enumerate(CLASSES):
        e = env[y == k]
        out[c] = {"n": int(len(e)), "median": np.median(e, 0), "q25": np.percentile(e, 25, 0),
                  "q75": np.percentile(e, 75, 0)}
    return out


def bin_centers() -> np.ndarray:
    return np.arange(N_KEEP // BIN) * BIN + (BIN - 1) / 2


def vdl_window(margin: float = 1.0) -> dict:
    st = class_envelope_stats(ALL_WELLS)
    mod, poor, good = st["Moderate"]["median"], st["Poor"]["median"], st["Good"]["median"]
    ok = (poor - mod > margin) & (good - mod > margin)
    runs = contiguous_runs(ok)
    if not runs:
        raise ValueError("no VDL bin satisfies the Moderate-gap rule")
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return {"bins": [a, b], "sample_start": a * BIN, "sample_end": b * BIN,
            "all_runs_samples": [[x * BIN, y * BIN] for x, y in runs], "margin_log_units": margin}


def pooled_confusion(models: list[str]) -> np.ndarray:
    """Sum of the held-out pooled confusion matrices across `models`."""
    return np.sum([np.array(ensemble_metrics(m)["confusion"]) for m in models], 0)


def good_zones(w: str) -> list[dict]:
    """Contiguous runs of class Good (y == 2) in well `w`, with depth & segment metadata."""
    d = well(w)
    zones = []
    for a, b in contiguous_runs(d.y == 2):
        seg = next(s for s in d.segments if s[0] <= a < s[1])
        zones.append({"start_idx": a, "end_idx": b, "n": b - a, "top_m": float(d.depth[a]),
                      "base_m": float(d.depth[b - 1]), "length_m": float(d.depth[b - 1] - d.depth[a]),
                      "segment_top_m": float(d.depth[seg[0]]), "segment_base_m": float(d.depth[seg[1] - 1]),
                      "at_segment_top": a == seg[0], "at_segment_base": b == seg[1],
                      "at_well_base": b == len(d.y)})
    return zones


# ----------------------------------------------------------------------------------------------
# Original Plot Style (share-sci-plot: Times New Roman, registry palette, normalize_fonts, A–E modes)
# ----------------------------------------------------------------------------------------------
_SKILL_REF = Path(__file__).resolve().parent / "_reproduction"
if not (_SKILL_REF / "figure_standards.py").is_file():
    _SKILL_REF = Path(__file__).resolve().parent / "_reproduction"
sys.path.insert(0, str(_SKILL_REF))
import figure_standards as fs  # noqa: E402

PAL4 = fs.default_palette(4)        # ['#376795', '#72BCD5', '#FFD06F', '#E76254']
PAL10 = fs.default_palette()        # warm -> cold
CLASS_COLORS = {"Poor": PAL4[3], "Moderate": PAL4[2], "Good": PAL4[1]}
INPUT_COLORS = {"Curves": "#9A9A9A", "Curves + VDL": "#5E3C99"}
INPUT_COLORS_LIGHT = {"Curves": "#CFCFCF", "Curves + VDL": "#B9A7DB"}
MODES = fs.FIGURE_MODES
MAX_WIDTH_IN = 7.25


def setup_style():
    import matplotlib as mpl
    import matplotlib.font_manager as fm
    try:
        fm.findfont("Times New Roman", fallback_to_default=False)
    except ValueError:
        for f in fm.findSystemFonts():
            if Path(f).name.lower() in {"times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"}:
                fm.fontManager.addfont(f)
        fm.findfont("Times New Roman", fallback_to_default=False)
    fs.apply_sci_style()
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold", "mathtext.bfit": "Times New Roman:bold:italic",
        "mathtext.sf": "Times New Roman", "mathtext.tt": "Times New Roman",
        "mathtext.cal": "Times New Roman:italic", "mathtext.fallback": None,
        "axes.unicode_minus": True, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.labelpad": 2.5,
    })


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "ukiyoe_div", [PAL10[0], PAL10[2], PAL10[4], "#FFFFFF", PAL10[5], PAL10[7], PAL10[9]])


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("ukiyoe_seq", ["#FFFFFF", PAL10[5], PAL10[7], PAL10[9]])


def class_cmap():
    from matplotlib.colors import ListedColormap
    return ListedColormap([CLASS_COLORS[c] for c in CLASSES])


def fmt_signed(v: float, nd: int = 2) -> str:
    """'+0.07', '\u22120.02', and '0.00' for values that round to zero (no '-0.00')."""
    r = round(float(v), nd)
    if r == 0:
        return f"{0:.{nd}f}"
    return f"{r:+.{nd}f}".replace("-", "\u2212")


def annotate_cells(ax, im, data, labels, fontsize=6.5):
    for (i, j), val in np.ndenumerate(data):
        t = ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=fontsize,
                    color=fs.heatmap_text_color(im.cmap(im.norm(val))))
        t.set_gid(fs.HEATMAP_CELL_GID)


def panel_label(ax, letter: str, x_pt: float = -24.0, y_pt: float = 4.0):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(x_pt, y_pt), textcoords="offset points",
                fontweight="bold", ha="left", va="bottom", annotation_clip=False)


def save(fig, stem: str, max_png_bytes: int = 1_000_000):
    import matplotlib.pyplot as plt
    fs.normalize_fonts(fig)
    for t in fig.findobj(lambda o: hasattr(o, "get_gid") and o.get_gid() == "small_text"):
        t.set_fontsize(6)
    out = FIG_DIR / stem
    kw = dict(dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), **kw)
    fig.savefig(out.with_suffix(".png"), pil_kwargs={"optimize": True}, **kw)
    plt.close(fig)
    png = out.with_suffix(".png")
    size = png.stat().st_size
    from PIL import Image
    note = ""
    if size > max_png_bytes:
        with Image.open(png) as im:
            q = im.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
            q.save(png, optimize=True, dpi=(300, 300))
        note = f" (quantised from {size / 1e6:.2f} MB)"
        size = png.stat().st_size
    with Image.open(png) as im:
        width_in = im.size[0] / 300
        print(f"saved {stem}.pdf/.png  {im.size[0]}x{im.size[1]} px ({width_in:.2f} x {im.size[1] / 300:.2f} in)"
              f"  {size / 1e6:.2f} MB{note}")
    if width_in > MAX_WIDTH_IN:
        raise RuntimeError(f"{stem}: exported width {width_in:.2f} in exceeds {MAX_WIDTH_IN} in (double column)")
    if size > max_png_bytes:
        raise RuntimeError(f"{stem}.png is {size / 1e6:.2f} MB > {max_png_bytes / 1e6:.2f} MB")
