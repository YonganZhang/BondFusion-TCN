"""Experiment protocol: model factory, leave-one-well-out CV on training wells,
final multi-seed training on all training wells and held-out evaluation.

Model names and their inputs:
  "<Baseline>"               logging curves only (protocol `features`)
  "<Baseline> +VDL"          curves + coarse VDL log-envelope blocks as extra channels
  "VDL-ModernTCN<variant>"   curves + fine VDL log-envelope bins through the waveform encoder
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .data import (FeatureScaler, build_prepared, compute_class_counts, compute_class_weights, load_wells)
from .metrics import compute_metrics
from .train import TrainConfig, run_evaluate, run_seed_everything, run_training
from .vdl import build_vdl_channels

VDL_PREFIX = "VDL-ModernTCN"
# input: "bins" = waveform encoder on fine VDL bins; "blocks" = coarse VDL blocks as plain channels
# of the ModernTCN backbone; None = curves only. The last two are ablations of the proposed model
# (same tuned hyper-parameters, component removed).
VDL_VARIANTS = {
    "": dict(input="bins", pool="curve_attn", global_layer="none"),
    " (mean pool)": dict(input="bins", pool="mean", global_layer="none"),
    " (static query)": dict(input="bins", pool="static_attn", global_layer="none"),
    " + Reformer": dict(input="bins", pool="curve_attn", global_layer="reformer"),
    " w/o waveform encoder": dict(input="blocks"),
    " w/o VDL": dict(input=None),
}
VDL_BLOCK_SUFFIX = " +VDL"


def parse_vdl_name(name: str) -> dict | None:
    """'VDL-ModernTCN (mean pool)' -> variant flags; None for other models."""
    if not name.startswith(VDL_PREFIX):
        return None
    variant = name[len(VDL_PREFIX):]
    if variant not in VDL_VARIANTS:
        raise KeyError(f"unknown VDL-ModernTCN variant {variant!r}; expected one of {list(VDL_VARIANTS)}")
    return dict(VDL_VARIANTS[variant])


def resolve_vdl_mode(name: str) -> str | None:
    """'bins' (waveform encoder), 'blocks' (extra channels) or None (curves only)."""
    flags = parse_vdl_name(name)
    if flags is not None:
        return flags["input"]
    if name.endswith(VDL_BLOCK_SUFFIX):
        return "blocks"
    return None


TRAIN_KEYS = {"lr", "weight_decay", "batch_size"}


def build_safe_name(name: str) -> str:
    """File-system name of a model (used for study DBs and output directories)."""
    return name.replace(" ", "_").replace("/", "-").replace("+", "plus")


def load_protocol(path: Path) -> dict:
    p = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    p["_root"] = Path(path).resolve().parents[2]
    return p


def resolve_final_dir(outputs: Path, protocol: dict, name: str) -> Path:
    """outputs/<version>/final/<safe name>; models listed under protocol `reuse` point to the
    final runs of `reuse.from_version` (identical protocol for those models)."""
    reuse = protocol.get("reuse") or {}
    version = reuse["from_version"] if name in reuse.get("models", []) else protocol["version"]
    return Path(outputs) / version / "final" / build_safe_name(name)


def build_model(name: str, n_features: int, n_classes: int, seq_len: int, hp: dict, n_domains: int, class_counts,
                n_wave: int = 0):
    """n_features counts every input channel incl. the valid flag; n_wave = VDL bins among them."""
    from .models.baselines import TRAIN_HP_KEYS, build_baseline, default_hp
    vdl_flags = parse_vdl_name(name)
    if vdl_flags is not None:
        if vdl_flags.pop("input") == "bins":
            from .models.vdl_moderntcn import VdlModernTCN
            return VdlModernTCN(n_features - 1 - n_wave, n_wave, n_classes, seq_len, hp, **vdl_flags)
        keys = set(default_hp("ModernTCN")) | TRAIN_HP_KEYS
        return build_baseline("ModernTCN", n_features, n_classes, seq_len, {k: v for k, v in hp.items() if k in keys})
    if name.endswith(VDL_BLOCK_SUFFIX):
        return build_baseline(name[:-len(VDL_BLOCK_SUFFIX)], n_features, n_classes, seq_len, hp)
    return build_baseline(name, n_features, n_classes, seq_len, hp)


def build_train_config(protocol: dict, hp: dict, seed: int, stop_epoch: int | None = None) -> TrainConfig:
    t = dict(protocol["train"])
    t.update({k: hp[k] for k in TRAIN_KEYS if k in hp})
    return TrainConfig(**t, seed=seed, stop_epoch=stop_epoch)


def build_split(protocol: dict, train_names: list[str], eval_names: list[str], vdl_mode: str | None = None):
    """Prepared wells (curves scaled on train_names; with `vdl_mode`, VDL log-envelope channels are
    appended after the curve channels), curve scaler, training class counts and CE weights."""
    root = protocol["_root"]
    wells = load_wells(root / protocol["raw_dir"], train_names + eval_names, protocol["features"], protocol["gap_m"])
    tr = [wells[n] for n in train_names]
    scaler = FeatureScaler(protocol["features"], protocol.get("transforms"), protocol.get("clip", 8.0),
                           protocol.get("missing_indicators")).fit(tr)
    prepared = build_prepared(wells, scaler, train_names)
    if vdl_mode is not None:
        v = protocol["vdl"]
        size = {"bins": v["bin"], "blocks": v["block"]}[vdl_mode]
        extra = build_vdl_channels(root / protocol["raw_dir"], train_names + eval_names, train_names, v["n_keep"],
                                   size, protocol.get("clip", 8.0))
        for n, pw in prepared.items():
            pw.x = np.concatenate([pw.x, extra[n]], axis=1)
    counts = compute_class_counts(tr)
    weights = compute_class_weights(counts, protocol.get("class_weight_power", 0.5))
    return prepared, scaler, counts, weights


def run_fit_eval(name: str, hp: dict, protocol: dict, train_names: list[str], eval_names: list[str], device,
                 seed: int, validate: bool, stop_epoch: int | None = None, log_every: int = 0):
    prepared, scaler, counts, weights = build_split(protocol, train_names, eval_names, resolve_vdl_mode(name))
    cfg = build_train_config(protocol, hp, seed, stop_epoch)
    run_seed_everything(seed, cfg.deterministic)  # seed the parameter initialisation too
    n_in = prepared[train_names[0]].x.shape[1]
    model = build_model(name, n_in + 1, 3, cfg.seq_len, hp, len(train_names), counts,
                        n_wave=n_in - scaler.n_channels if resolve_vdl_mode(name) == "bins" else 0)
    tr = [prepared[n] for n in train_names]
    ev = [prepared[n] for n in eval_names]
    fit = run_training(model, tr, ev if validate else None, cfg, weights, device, log_every=log_every)
    metrics, probs = run_evaluate(model, ev, cfg, device)
    per_well = {n: compute_metrics(prepared[n].y, probs[n].argmax(1)) for n in eval_names}
    return {"fit": fit, "metrics": metrics, "per_well": per_well, "probs": probs, "model": model,
            "n_params": sum(p.numel() for p in model.parameters())}


def run_cv(name: str, hp: dict, protocol: dict, device, seed: int = 0, report=None) -> dict:
    """Leave-one-training-well-out CV. `report(fold_idx, running_mean)` may raise to prune."""
    train = list(protocol["train_wells"])
    folds = []
    for k, val in enumerate(train):
        rest = [w for w in train if w != val]
        r = run_fit_eval(name, hp, protocol, rest, [val], device, seed, validate=True)
        pw = r["per_well"][val]
        folds.append({"val_well": val, "best_epoch": r["fit"]["best_epoch"],
                      "macro_f1": r["fit"]["best_val_macro_f1"], "seconds": r["fit"]["seconds"],
                      **{f"f1_{c}": pw[f"f1_{c}"] for c in ("Poor", "Moderate", "Good")}})
        if report is not None:
            report(k, float(np.mean([f["macro_f1"] for f in folds])))
    return {"mean_macro_f1": float(np.mean([f["macro_f1"] for f in folds])),
            "stop_epoch": int(np.median([f["best_epoch"] for f in folds])) + 1, "folds": folds}
