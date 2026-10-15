"""Training loop, sliding-window inference and early stopping (shared by all models)."""
from __future__ import annotations

import copy
import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import IGNORE_INDEX, PreparedWell, TrainWindows, build_window_tensor, build_windows
from .metrics import compute_metrics

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required for deterministic cuBLAS


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 150
    patience: int = 25
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    seq_len: int = 256
    train_stride: int = 64
    infer_stride: int = 128
    critic_lr: float = 2e-4
    dom_ramp_epochs: int = 20       # GRL weight reaches ~λ_max after this many epochs
    deterministic: bool = True
    seed: int = 0
    stop_epoch: int | None = None   # final training: stop after this epoch, no validation


def run_seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    # warn_only: ops without a deterministic CUDA kernel warn instead of failing
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def build_lr_lambda(warmup: int, total: int):
    def f(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        t = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
    return f


@torch.no_grad()
def run_window_average(model: nn.Module, pw: PreparedWell, seq_len: int, stride: int, device,
                       output: str = "prob", batch_size: int = 64) -> np.ndarray:
    """Average per-point outputs of overlapping windows (triangular weights favour window centres).
    output="prob": softmax probabilities; output="feature": representation h fed to the head.
    The RNG is re-seeded so that models with stochastic inference (LSH attention re-draws its
    random rotations on every call) give identical outputs for identical inputs; window order
    and batch composition are fixed (TimesNet selects periods per batch)."""
    model.eval()
    torch.manual_seed(20261010)
    n = len(pw.y)
    acc, wsum = None, np.zeros(n, np.float64)
    wins = build_windows(pw.segments, seq_len, stride)
    tri = np.minimum(np.arange(1, seq_len + 1), np.arange(seq_len, 0, -1)).astype(np.float64)
    for i in range(0, len(wins), batch_size):
        chunk = wins[i:i + batch_size]
        xb = torch.from_numpy(np.stack([build_window_tensor(pw, s, ln, seq_len)[0] for s, ln in chunk])).to(device)
        logits, h = model(xb)
        out = (torch.softmax(logits.float(), dim=-1) if output == "prob" else h.float()).cpu().numpy()
        if acc is None:
            acc = np.zeros((n, out.shape[-1]), np.float64)
        for (s, ln), o in zip(chunk, out):
            w = tri[:ln] if ln == seq_len else np.ones(ln)
            acc[s:s + ln] += o[:ln] * w[:, None]
            wsum[s:s + ln] += w
    if np.any(wsum == 0):
        raise RuntimeError(f"well {pw.name}: {int((wsum == 0).sum())} samples without prediction")
    return acc / wsum[:, None]


def run_predict_well(model: nn.Module, pw: PreparedWell, seq_len: int, stride: int, device) -> np.ndarray:
    return run_window_average(model, pw, seq_len, stride, device, "prob")


def run_extract_features(model: nn.Module, pw: PreparedWell, seq_len: int, stride: int, device) -> np.ndarray:
    return run_window_average(model, pw, seq_len, stride, device, "feature").astype(np.float32)


def run_evaluate(model, wells: list[PreparedWell], cfg: TrainConfig, device) -> tuple[dict, dict]:
    probs = {w.name: run_predict_well(model, w, cfg.seq_len, cfg.infer_stride, device) for w in wells}
    y = np.concatenate([w.y for w in wells])
    pred = np.concatenate([probs[w.name].argmax(1) for w in wells])
    return compute_metrics(y, pred), probs


def run_training(model: nn.Module, train_wells: list[PreparedWell], val_wells: list[PreparedWell] | None,
                 cfg: TrainConfig, class_weights: np.ndarray, device, report=None, log_every: int = 0) -> dict:
    """Train with AdamW + warmup/cosine. With `val_wells`, keep the state with best
    validation macro-F1 and stop after `patience` epochs without improvement;
    otherwise train until `cfg.stop_epoch` and keep the last state.
    `report(epoch, val_macro_f1)` may raise to prune (Optuna)."""
    run_seed_everything(cfg.seed, cfg.deterministic)
    model.to(device)
    ds = TrainWindows(train_wells, cfg.seq_len, cfg.train_stride, seed=cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)
    critic_params = list(model.critic_parameters()) if hasattr(model, "critic_parameters") else []
    critic_ids = {id(p) for p in critic_params}
    main_params = [p for p in model.parameters() if id(p) not in critic_ids]
    opt = torch.optim.AdamW(main_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    critic_opt = torch.optim.Adam(critic_params, lr=cfg.critic_lr, betas=(0.5, 0.9)) if critic_params else None
    sched = torch.optim.lr_scheduler.LambdaLR(opt, build_lr_lambda(cfg.warmup_epochs, cfg.max_epochs))
    cw = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    last_epoch = cfg.stop_epoch if cfg.stop_epoch is not None else cfg.max_epochs
    best = {"score": -1.0, "epoch": -1, "state": None}
    history, bad, t0 = [], 0, time.time()
    for epoch in range(last_epoch):
        ds.reshuffle(epoch)
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, generator=g, drop_last=len(ds) > cfg.batch_size)
        model.train()
        sums: dict[str, float] = {}
        for xb, yb, db in loader:
            xb, yb, db = xb.to(device), yb.to(device), db.to(device)
            logits, h = model(xb)
            loss_cls = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1), weight=cw, ignore_index=IGNORE_INDEX)
            losses = {"cls": loss_cls}
            if hasattr(model, "compute_aux_losses"):
                losses.update(model.compute_aux_losses(h, yb, yb != IGNORE_INDEX, db, epoch, cfg.dom_ramp_epochs))
            total = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(main_params, cfg.grad_clip)
            opt.step()
            if critic_opt is not None:
                lc = model.compute_critic_loss()
                if lc is not None:
                    critic_opt.zero_grad(set_to_none=True)
                    lc.backward()
                    critic_opt.step()
                    losses["critic"] = lc.detach()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
        sched.step()
        rec = {"epoch": epoch, **{k: v / max(1, len(loader)) for k, v in sums.items()}}
        if val_wells:
            m, _ = run_evaluate(model, val_wells, cfg, device)
            rec["val_macro_f1"] = m["macro_f1"]
            if m["macro_f1"] > best["score"]:
                best = {"score": m["macro_f1"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
                bad = 0
            else:
                bad += 1
            if report is not None:
                report(epoch, m["macro_f1"])
        history.append(rec)
        if log_every and epoch % log_every == 0:
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}, f"{time.time() - t0:.0f}s", flush=True)
        if val_wells and bad >= cfg.patience:
            break
    if val_wells:
        model.load_state_dict(best["state"])
    else:
        best = {"score": None, "epoch": last_epoch - 1, "state": None}
    return {"best_epoch": best["epoch"], "best_val_macro_f1": best["score"], "history": history,
            "seconds": time.time() - t0}
