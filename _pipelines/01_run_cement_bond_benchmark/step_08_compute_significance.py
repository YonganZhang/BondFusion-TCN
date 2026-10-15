"""Step 08: is the proposed model's advantage larger than the noise?

For each comparison model, the seed-averaged probabilities of both models are
compared on the blind wells with a block bootstrap over depth segments (segments
resampled with replacement inside each well, so spatial autocorrelation is kept).
Reports the macro-F1 / kappa difference (proposed - other), its 95% interval and
the share of bootstrap draws where the difference is <= 0. Seed-level mean
differences are reported alongside.

With --vdl-pairs, every "<baseline> +VDL" is compared with its curves-only "<baseline>"
instead (effect of the VDL input per backbone).

Usage: python step_08_compute_significance.py [--protocol P] [--reference "ModernTCN +VDL"] [--n-boot 2000]
       python step_08_compute_significance.py --protocol protocol_v03.yaml --vdl-pairs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "_code"))

from da_reformer.data import load_well  # noqa: E402
from da_reformer.experiment import build_safe_name, load_protocol, resolve_final_dir  # noqa: E402


def load_seed_probs(model_dir: Path, wells: list[str]) -> list[dict[str, np.ndarray]]:
    seeds = sorted(model_dir.glob("seed*"))
    return [{w: np.load(s / f"probs_{w}.npy") for w in wells} for s in seeds if (s / "metrics.json").exists()]


def load_seed_metric(model_dir: Path, key: str) -> list[float]:
    return [json.loads((p / "metrics.json").read_text())["merged"][key]
            for p in sorted(model_dir.glob("seed*")) if (p / "metrics.json").exists()]


def compute_f1_kappa(y: np.ndarray, p: np.ndarray, n_classes: int = 3) -> tuple[float, float]:
    """Macro-F1 (classes absent from both y and p count as 0, as sklearn zero_division=0) and Cohen's kappa."""
    cm = np.bincount(y * n_classes + p, minlength=n_classes * n_classes).reshape(n_classes, n_classes).astype(float)
    tp, rows, cols, n = np.diag(cm), cm.sum(1), cm.sum(0), cm.sum()
    f1 = np.where(rows + cols > 0, 2 * tp / np.maximum(rows + cols, 1), 0.0)
    pe = (rows * cols).sum() / n ** 2
    return float(f1.mean()), float((tp.sum() / n - pe) / (1 - pe))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", type=Path, default=HERE / "protocol_v03.yaml")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--vdl-pairs", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)
    protocol = load_protocol(args.protocol)
    base = HERE / "outputs" / protocol["version"]
    wells = list(protocol["test_wells"])
    m = protocol["models"]
    if args.vdl_pairs:
        pairs = [(f"{b} +VDL", b) for b in m["baselines"]]
        out_name = "significance_vdl_pairs.json"
    else:
        ref = args.reference or m["proposed"]
        others = m["ablations"] + m.get("plugins", []) + m.get("vdl_baselines", []) + m["baselines"]
        pairs = [(ref, o) for o in others if o != ref]
        out_name = f"significance_vs_{build_safe_name(ref)}.json"
    wd = {w: load_well(ROOT / protocol["raw_dir"], w, ["CBL"], protocol["gap_m"]) for w in wells}
    y = {w: wd[w].y for w in wells}
    yt = np.concatenate([y[w] for w in wells])
    rng = np.random.default_rng(0)
    boot_idx = []
    for _ in range(args.n_boot):
        idx = []
        for w in wells:
            segs = wd[w].segments
            pick = rng.integers(0, len(segs), len(segs))
            idx.append([(w, a, b) for a, b in (segs[i] for i in pick)])
        boot_idx.append(idx)
    boot_y = [np.concatenate([y[w][a:b] for part in sample for w, a, b in part]) for sample in boot_idx]

    preds = {}

    def ensemble_pred(name: str):
        if name not in preds:
            runs = load_seed_probs(resolve_final_dir(HERE / "outputs", protocol, name), wells)
            preds[name] = {w: np.mean([r[w] for r in runs], 0).argmax(1) for w in wells} if runs else None
        return preds[name]

    def gather(pred: dict, sample) -> np.ndarray:
        return np.concatenate([pred[w][a:b] for part in sample for w, a, b in part])

    results = {}
    for a, b in pairs:
        pa, pb = ensemble_pred(a), ensemble_pred(b)
        if pa is None or pb is None:
            print(f"{a} vs {b}: missing final runs, skipped")
            continue
        fa, ka = compute_f1_kappa(yt, np.concatenate([pa[w] for w in wells]))
        fb, kb = compute_f1_kappa(yt, np.concatenate([pb[w] for w in wells]))
        d_f1, d_k = [], []
        for sample, yy in zip(boot_idx, boot_y):
            f1a, k1a = compute_f1_kappa(yy, gather(pa, sample))
            f1b, k1b = compute_f1_kappa(yy, gather(pb, sample))
            d_f1.append(f1a - f1b)
            d_k.append(k1a - k1b)
        dir_a, dir_b = (resolve_final_dir(HERE / "outputs", protocol, n) for n in (a, b))
        r = {
            "model": a, "other": b,
            "ensemble_macro_f1_diff": fa - fb,
            "boot_macro_f1_diff_ci95": [float(np.percentile(d_f1, 2.5)), float(np.percentile(d_f1, 97.5))],
            "boot_macro_f1_p_le_0": float(np.mean(np.array(d_f1) <= 0)),
            "ensemble_kappa_diff": ka - kb,
            "boot_kappa_diff_ci95": [float(np.percentile(d_k, 2.5)), float(np.percentile(d_k, 97.5))],
            "boot_kappa_p_le_0": float(np.mean(np.array(d_k) <= 0)),
            "seed_mean_macro_f1_diff": float(np.mean(load_seed_metric(dir_a, "macro_f1")) - np.mean(load_seed_metric(dir_b, "macro_f1"))),
            "seed_mean_kappa_diff": float(np.mean(load_seed_metric(dir_a, "kappa")) - np.mean(load_seed_metric(dir_b, "kappa"))),
        }
        results[f"{a} vs {b}"] = r
        print(f"{a:28s} vs {b:34s} ΔmacroF1 ens {r['ensemble_macro_f1_diff']:+.3f} CI {np.round(r['boot_macro_f1_diff_ci95'], 3)} "
              f"P(Δ<=0) {r['boot_macro_f1_p_le_0']:.2f} | Δkappa ens {r['ensemble_kappa_diff']:+.3f} CI {np.round(r['boot_kappa_diff_ci95'], 3)} "
              f"P(Δ<=0) {r['boot_kappa_p_le_0']:.2f} | seed ΔF1 {r['seed_mean_macro_f1_diff']:+.3f} Δκ {r['seed_mean_kappa_diff']:+.3f}",
              flush=True)
    out = base / "report"
    out.mkdir(parents=True, exist_ok=True)
    (out / out_name).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
