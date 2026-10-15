"""Shared numerical data access and derived quantities for the v03 paper analysis.

Every number comes from existing v03 outputs (no training):
- labels / curves / VDL: `_data/raw/` via `da_reformer.data.load_well` and `da_reformer.vdl`;
- per-seed blind probabilities and metrics: `outputs/v03/final/<model>/seed*/`;
- summaries, significance, HPO, screen: `outputs/v03/{report,hpo,screen}/`;
- HPO selection-noise check: `_sandbox/2026-10-10-hpo-selection-noise/out/`.
Ensemble = mean of the 10 seeds' probabilities, then argmax (as step_06 / step_08).
Ensemble probabilities are cached under `_tmp/p04_paper_cache/` (keyed by seed-file size/mtime).
"""
from __future__ import annotations

import functools
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RELEASE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("BONDFUSION_DATA_ROOT", str(RELEASE_ROOT))).resolve()
PIPE = ROOT / "_pipelines" / "01_run_cement_bond_benchmark"
OUTPUTS = PIPE / "outputs"
V03 = OUTPUTS / "v03"
REPORT = V03 / "report"
FIG_DIR = RELEASE_ROOT / "_figures"
OUT_DIR = RELEASE_ROOT / "_outputs" / "p04_v03_paper"
CACHE = Path(os.environ.get("BONDFUSION_CACHE_DIR", str(RELEASE_ROOT / "_tmp" / "p04_paper_cache"))).resolve()
HPO_NOISE = ROOT / "_sandbox" / "2026-10-10-hpo-selection-noise" / "out" / "VDL-ModernTCN_(mean_pool).json"
RAW = ROOT / "_data" / "raw"
SUPPORT = RELEASE_ROOT / "results" / "support"
if not HPO_NOISE.is_file():
    HPO_NOISE = SUPPORT / "hpo_noise_runs.json"


def released_fallback(path: Path, fallback: Path) -> Path:
    """Prefer recomputed input files; otherwise read the released aggregate."""
    return path if path.is_file() else fallback


sys.path.insert(0, str(RELEASE_ROOT / "_code"))
from da_reformer.data import CLASS_NAMES, load_well  # noqa: E402
from da_reformer.experiment import build_safe_name, load_protocol, resolve_final_dir  # noqa: E402
from da_reformer.metrics import compute_metrics  # noqa: E402
from da_reformer.vdl import build_log_envelope, load_vdl  # noqa: E402

# step_08 helpers reused verbatim (compute_f1_kappa, load_seed_probs, load_seed_metric)
_spec = importlib.util.spec_from_file_location("p04_step08", RELEASE_ROOT / "_pipelines/01_run_cement_bond_benchmark/step_08_compute_significance.py")
step08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step08)
compute_f1_kappa = step08.compute_f1_kappa
load_seed_probs = step08.load_seed_probs
load_seed_metric = step08.load_seed_metric

PROTOCOL = load_protocol(RELEASE_ROOT / "_pipelines/01_run_cement_bond_benchmark/protocol_v03.yaml")
TRAIN_WELLS = list(PROTOCOL["train_wells"])
TEST_WELLS = list(PROTOCOL["test_wells"])
ALL_WELLS = TRAIN_WELLS + TEST_WELLS
GAP_M = PROTOCOL["gap_m"]
N_KEEP = PROTOCOL["vdl"]["n_keep"]
BIN = PROTOCOL["vdl"]["bin"]
CLASSES = list(CLASS_NAMES)  # index = label: 0 Poor, 1 Moderate, 2 Good

# ----------------------------------------------------------------------------------------------
# Model names (code) and display names (paper)
# ----------------------------------------------------------------------------------------------
BACKBONES = list(PROTOCOL["models"]["baselines"])
assert BACKBONES == ["LSTM", "GRU", "Transformer", "Reformer", "Crossformer", "TimesNet", "TimeMixer", "ModernTCN"]


def vdl(b: str) -> str:
    return f"{b} +VDL"


MEAN_POOL = PROTOCOL["models"]["proposed"]
assert MEAN_POOL == "VDL-ModernTCN (mean pool)"

# encoder designs (ModernTCN backbone, hyper-parameters tuned once for mean pooling)
DESIGN_DISPLAY = {
    "VDL-ModernTCN (mean pool)": "Mean pooling",
    "VDL-ModernTCN (static query)": "Static attention",
    "VDL-ModernTCN": "Curve-conditioned attention",
    "VDL-ModernTCN + Reformer": "Curve-conditioned attention + LSH layer",
    "VDL-ModernTCN w/o waveform encoder": "Block features (no encoder)",
    "VDL-ModernTCN w/o VDL": "No VDL",
}
DESIGNS = list(DESIGN_DISPLAY)
REF_CURVES = "ModernTCN"
REF_VDL = "ModernTCN +VDL"
REF_DISPLAY = {REF_CURVES: "ModernTCN, Curves", REF_VDL: "BondFusion-TCN"}

# design screen ids (protocol_v03.yaml `screen.candidates`). S0..S5 ran with the earlier transforms
# (CCL signed-log only, linear CBL); the *-logCBL rows add signed-log CBL (the final v03 transform).
SCREEN_DISPLAY = {
    "S0": "Curves (linear CBL)",
    "S0-logCBL": "Curves (log CBL)",
    "S1": "Block features (no encoder)",
    "S2": "Mean pooling",
    "S3": "Static attention",
    "S4": "Curve-conditioned attention",
    "S5": "Curve-conditioned attention + LSH layer",
    "S1-logCBL": "Block features (no encoder, log CBL)",
    "S2-logCBL": "Mean pooling (log CBL)",
}
SCREEN_ORDER = ["S0", "S0-logCBL", "S1", "S1-logCBL", "S2", "S2-logCBL", "S3", "S4", "S5"]

ALL_MODELS = DESIGNS + [vdl(b) for b in BACKBONES] + BACKBONES  # 22 distinct final runs


def display_name(name: str) -> str:
    if name in DESIGN_DISPLAY:
        return DESIGN_DISPLAY[name]
    if name.endswith(" +VDL"):
        return f"{name[:-5]}, Curves + VDL"
    if name in BACKBONES:
        return f"{name}, Curves"
    raise KeyError(name)


def input_set(name: str) -> str:
    """'Curves' or 'Curves + VDL' (designs: No VDL = Curves)."""
    if name == "VDL-ModernTCN w/o VDL" or name in BACKBONES:
        return "Curves"
    return "Curves + VDL"


# ----------------------------------------------------------------------------------------------
# Raw data
# ----------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def well(w: str):
    """WellData with .depth, .y, .segments (CBL only loaded; labels are what matter)."""
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
    """[N, 190] log-envelope (bin 10) as fed to the waveform encoder (before scaling)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    src = RAW / f"{w}_vdl.npy"
    f = CACHE / f"logenv_bin{BIN}_{w}.npz"
    sig = f"{src.stat().st_size}:{src.stat().st_mtime_ns}"
    if f.exists():
        z = np.load(f)
        if str(z["sig"]) == sig:
            return z["env"]
    env = build_log_envelope(vdl_raw(w), BIN)
    np.savez(f, env=env, sig=sig)
    return env


def y_true(wells=None) -> np.ndarray:
    return np.concatenate([well(w).y for w in (wells or TEST_WELLS)])


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(int(a), int(b)) for a, b in zip(e[::2], e[1::2])]


# ----------------------------------------------------------------------------------------------
# Model outputs
# ----------------------------------------------------------------------------------------------
def final_dir(name: str) -> Path:
    return resolve_final_dir(OUTPUTS, PROTOCOL, name)


@functools.lru_cache(maxsize=None)
def seed_metrics(name: str) -> list[dict]:
    d = final_dir(name)
    runs = [json.loads((p / "metrics.json").read_text()) for p in sorted(d.glob("seed*")) if (p / "metrics.json").exists()]
    if len(runs) != 10:
        raise ValueError(f"{name}: expected 10 seeds, found {len(runs)}")
    return runs


def seed_values(name: str, key: str, w: str | None = None) -> np.ndarray:
    return np.array([r["merged"][key] if w is None else r["per_well"][w][key] for r in seed_metrics(name)])


@functools.lru_cache(maxsize=None)
def ensemble_probs(name: str) -> dict[str, np.ndarray]:
    d = final_dir(name)
    files = sorted(d.glob("seed*/probs_*.npy"))
    sig = ";".join(f"{f.relative_to(d)}:{f.stat().st_size}:{f.stat().st_mtime_ns}" for f in files)
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"ens_{build_safe_name(name)}.npz"
    if cf.exists():
        z = np.load(cf)
        if str(z["sig"]) == sig:
            return {w: z[w] for w in TEST_WELLS}
    runs = load_seed_probs(d, TEST_WELLS)
    if len(runs) != 10:
        raise ValueError(f"{name}: expected 10 seed prob sets, found {len(runs)}")
    out = {w: np.mean([r[w] for r in runs], 0) for w in TEST_WELLS}
    for w in TEST_WELLS:
        if out[w].shape != (len(well(w).y), 3):
            raise ValueError(f"{name}/{w}: probs {out[w].shape} vs {len(well(w).y)} labels")
    np.savez(cf, sig=sig, **out)
    return out


def ensemble_pred(name: str) -> dict[str, np.ndarray]:
    return {w: p.argmax(1) for w, p in ensemble_probs(name).items()}


def ensemble_metrics(name: str, w: str | None = None) -> dict:
    p = ensemble_pred(name)
    if w is None:
        return compute_metrics(y_true(), np.concatenate([p[x] for x in TEST_WELLS]))
    return compute_metrics(well(w).y, p[w])


def seed_preds(name: str) -> list[np.ndarray]:
    """Per-seed merged (E then F) predictions."""
    return [np.concatenate([r[w].argmax(1) for w in TEST_WELLS]) for r in load_seed_probs(final_dir(name), TEST_WELLS)]


# ----------------------------------------------------------------------------------------------
# Existing summaries
# ----------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def summary_merged() -> pd.DataFrame:
    return pd.read_csv(released_fallback(REPORT / "summary_merged.csv", RELEASE_ROOT / "results/summary_merged.csv")).set_index("model")


@functools.lru_cache(maxsize=None)
def summary_per_well() -> pd.DataFrame:
    return pd.read_csv(released_fallback(REPORT / "summary_per_well.csv", RELEASE_ROOT / "results/summary_per_well.csv"))


@functools.lru_cache(maxsize=None)
def sig_vdl_pairs() -> dict:
    return json.loads(released_fallback(REPORT / "significance_vdl_pairs.json", RELEASE_ROOT / "results/significance_vdl_pairs.json").read_text())


@functools.lru_cache(maxsize=None)
def sig_vs_mean_pool() -> dict:
    return json.loads(released_fallback(REPORT / f"significance_vs_{build_safe_name(MEAN_POOL)}.json", SUPPORT / "significance_vs_mean_pool.json").read_text())


@functools.lru_cache(maxsize=None)
def reference_baselines() -> dict:
    return json.loads(released_fallback(V03 / "reference_baselines.json", SUPPORT / "reference_baselines.json").read_text())


REFERENCE_DISPLAY = {
    "CBL two-threshold rule": "CBL threshold rule",
    "HGB point-wise (GR,CBL,CCL)": "Point-wise gradient boosting, Curves",
    "HGB point-wise (GR,CBL,CCL + VDL blocks)": "Point-wise gradient boosting, Curves + VDL",
}


def hpo_best(name: str) -> dict:
    return json.loads(released_fallback(V03 / "hpo" / f"{build_safe_name(name)}.best.json", RELEASE_ROOT / "_pipelines/01_run_cement_bond_benchmark/outputs/v03/hpo" / f"{build_safe_name(name)}.best.json").read_text())


TUNED_MODELS = BACKBONES + [vdl(b) for b in BACKBONES] + [MEAN_POOL]


@functools.lru_cache(maxsize=None)
def screen_table() -> pd.DataFrame:
    """screen_summary.csv plus the per-seed CV macro-F1 recomputed from <id>_seed<k>.json."""
    s = pd.read_csv(released_fallback(V03 / "screen" / "screen_summary.csv", RELEASE_ROOT / "results/screen_summary.csv")).set_index("id")
    seeds = PROTOCOL["screen"]["seeds"]
    vals = {}
    for sid in s.index:
        v = []
        for k in seeds:
            r = json.loads(released_fallback(V03 / "screen" / f"{sid}_seed{k}.json", SUPPORT / "screen" / f"{sid}_seed{k}.json").read_text())
            if r["id"] != sid or r["seed"] != k:
                raise ValueError(f"screen file mismatch {sid} seed {k}")
            v.append(r["mean_macro_f1"])
        vals[sid] = v
    s["seed_values"] = pd.Series(vals)
    return s


# ----------------------------------------------------------------------------------------------
# Derived quantities shared by figures and paper_numbers.json
# ----------------------------------------------------------------------------------------------
def class_envelope_stats(wells: list[str]) -> dict:
    """Per class: median, q25, q75 of the log-envelope per bin, pooled over `wells`."""
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
    """Contiguous bins where the A–D Moderate median is below both Poor and Good medians by > margin.
    If several runs qualify, the longest is returned (all runs are listed)."""
    st = class_envelope_stats(TRAIN_WELLS)
    mod, poor, good = st["Moderate"]["median"], st["Poor"]["median"], st["Good"]["median"]
    ok = (poor - mod > margin) & (good - mod > margin)
    runs = contiguous_runs(ok)
    if not runs:
        raise ValueError("no VDL bin satisfies the Moderate-gap rule")
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return {"bins": [a, b], "sample_start": a * BIN, "sample_end": b * BIN,  # samples [start, end)
            "all_runs_samples": [[x * BIN, y * BIN] for x, y in runs], "margin_log_units": margin}


def window_class_medians(wells: list[str], bins: tuple[int, int]) -> dict:
    """Per class: range of per-bin medians inside the window, and the median of the window-mean envelope."""
    env = np.concatenate([log_envelope(w) for w in wells])[:, bins[0]:bins[1]]
    y = np.concatenate([well(w).y for w in wells])
    out = {}
    for k, c in enumerate(CLASSES):
        e = env[y == k]
        if len(e) == 0:
            out[c] = None
            continue
        med = np.median(e, 0)
        out[c] = {"n": int(len(e)), "bin_median_min": float(med.min()), "bin_median_max": float(med.max()),
                  "median_of_window_mean": float(np.median(e.mean(1)))}
    return out


def good_zones(w: str) -> list[dict]:
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


def pooled_confusion(models: list[str]) -> np.ndarray:
    """Sum of the ensemble confusion matrices (E+F merged) over `models`."""
    return np.sum([np.array(ensemble_metrics(m)["confusion"]) for m in models], 0)


def cv_vs_blind() -> pd.DataFrame:
    rows = []
    for m in TUNED_MODELS:
        h = hpo_best(m)
        rows.append({"model": m, "display": display_name(m), "input_set": input_set(m),
                     "cv_macro_f1": h["value"], "blind_macro_f1_mean": float(seed_values(m, "macro_f1").mean()),
                     "blind_kappa_mean": float(seed_values(m, "kappa").mean())})
    return pd.DataFrame(rows)


def hpo_noise() -> dict:
    """Mirror of _sandbox/2026-10-10-hpo-selection-noise/run.py summary (same formulas)."""
    recs = json.loads(HPO_NOISE.read_text())
    by, hv = {}, {}
    for r in recs:
        by.setdefault(r["trial"], []).append(r["cv_macro_f1"])
        hv[r["trial"]] = r["hpo_value"]
    trials = sorted(by, key=lambda t: -hv[t])
    means = {t: float(np.mean(by[t])) for t in trials}
    return {
        "trials": [{"trial": t, "optuna_cv": hv[t], "reeval": by[t], "reeval_mean": means[t],
                    "reeval_sd": float(np.std(by[t], ddof=1))} for t in trials],
        "between_config_sd": float(np.std(list(means.values()), ddof=1)),
        "pooled_seed_sd": float(np.sqrt(np.mean([np.var(v, ddof=1) for v in by.values() if len(v) > 1]))),
        "winners_curse": float(hv[trials[0]] - means[trials[0]]),
        "seeds": sorted({r["seed"] for r in recs}),
    }
