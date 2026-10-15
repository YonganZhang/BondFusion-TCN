"""Optuna hyper-parameter search with an equal trial budget for every model.

Objective = mean validation macro-F1 over leave-one-training-well-out folds
(test wells E/F are never touched). TPE sampler + median pruner at fold level;
studies live in SQLite so an interrupted search resumes where it stopped.
"""
from __future__ import annotations

import json
from pathlib import Path

import optuna

from .experiment import VDL_BLOCK_SUFFIX, build_safe_name, parse_vdl_name, run_cv


def suggest_common(trial: optuna.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
    }


def suggest_wave(trial: optuna.Trial) -> dict:
    return {
        "d_wave": trial.suggest_categorical("d_wave", [16, 32, 64]),
        "wave_channels": trial.suggest_categorical("wave_channels", [16, 32, 64]),
    }


def suggest_all(name: str, trial: optuna.Trial) -> dict:
    """Common training keys + backbone space; VDL-ModernTCN searches the ModernTCN space plus the
    waveform encoder; '+VDL' baselines search the space of their base model."""
    from .models.baselines import suggest_hp
    if parse_vdl_name(name) is not None:
        return {**suggest_common(trial), **suggest_hp("ModernTCN", trial), **suggest_wave(trial)}
    if name.endswith(VDL_BLOCK_SUFFIX):
        return {**suggest_common(trial), **suggest_hp(name[:-len(VDL_BLOCK_SUFFIX)], trial)}
    return {**suggest_common(trial), **suggest_hp(name, trial)}


def run_study(name: str, protocol: dict, out_dir: Path, device, n_trials: int | None = None) -> optuna.Study:
    out_dir.mkdir(parents=True, exist_ok=True)
    h = protocol["hpo"]
    n_trials = n_trials or h["n_trials"]
    safe = build_safe_name(name)
    study = optuna.create_study(
        study_name=safe, storage=f"sqlite:///{out_dir / f'{safe}.db'}", load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=h.get("sampler_seed", 0)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=h.get("pruner_startup", 8), n_warmup_steps=1))

    def objective(trial: optuna.Trial) -> float:
        hp = suggest_all(name, trial)

        def report(step, value):
            trial.report(value, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        cv = run_cv(name, hp, protocol, device, seed=h.get("cv_seed", 0), report=report)
        trial.set_user_attr("hp", hp)
        trial.set_user_attr("stop_epoch", cv["stop_epoch"])
        trial.set_user_attr("folds", cv["folds"])
        return cv["mean_macro_f1"]

    done = len([t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)])
    if done < n_trials:
        study.optimize(objective, n_trials=n_trials - done, catch=(RuntimeError,))
    best = study.best_trial
    (out_dir / f"{safe}.best.json").write_text(json.dumps(
        {"model": name, "value": best.value, "trial": best.number, "hp": best.user_attrs["hp"],
         "stop_epoch": best.user_attrs["stop_epoch"], "folds": best.user_attrs["folds"],
         "n_trials": len(study.trials)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return study
