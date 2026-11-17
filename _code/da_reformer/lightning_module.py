"""PyTorch Lightning Slot-2 container and training helper for cement-bond sequence classification.

Enforces the 4-slot AutoResearch training invariants:
  - Automatic ``weight_decay=0.0`` on all 1D parameters (biases, norms, scalar gates).
  - O(1) float64 numerator/denominator accumulation across validation batches for
    four domain slices (Poor free-pipe, Moderate trough, Good coupled, Boundary zone)
    plus exact pooled macro-F1 — never caches GPU prediction lists and never averages
    per-batch metrics across uneven slice counts.
"""
from __future__ import annotations

import logging
import math
from typing import Iterable

import lightning as L
from lightning.pytorch.callbacks import Callback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import IGNORE_INDEX, PreparedWell, TrainWindows, build_window_tensor
from .metrics import compute_metrics, transition_mask

logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("lightning.fabric").setLevel(logging.WARNING)

DOMAIN_SLICES = ("poor_free_pipe", "moderate_trough", "good_coupled", "boundary_transition")


def build_param_groups(module: nn.Module, weight_decay: float) -> list[dict]:
    """Split trainable parameters into weight-decayed (ndim >= 2) and non-decayed (ndim < 2)."""
    decay = [p for p in module.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in module.parameters() if p.requires_grad and p.ndim < 2]
    groups: list[dict] = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups or [{"params": list(module.parameters()), "weight_decay": float(weight_decay)}]


class _WindowDictDataset(TrainWindows):
    """Wrap TrainWindows to emit dictionary batches with 4 domain slice masks for Lightning."""

    def __init__(self, wells: list[PreparedWell], seq_len: int, stride: int, seed: int = 0) -> None:
        self.well_bmasks = [transition_mask(w.y, w.segments) for w in wells]
        super().__init__(wells, seq_len, stride, seed=seed)

    def __getitem__(self, k: int) -> dict:
        i, s, n = self.index[k]
        w = self.wells[i]
        x, y = build_window_tensor(w, s, n, self.seq_len)
        bt = np.zeros(self.seq_len, dtype=bool)
        bt[:n] = self.well_bmasks[i][s : s + n]
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        return {
            "x": xt,
            "y": yt,
            "slice_masks": {
                "poor_free_pipe": yt == 0,
                "moderate_trough": yt == 1,
                "good_coupled": yt == 2,
                "boundary_transition": torch.from_numpy(bt),
            },
        }


class BondFusionLightningModule(L.LightningModule):
    """Fixed Slot-2 Lightning container wrapping any [B, L, C] cement-bond classifier."""

    def __init__(
        self,
        model: nn.Module,
        class_weights: Iterable[float] | None = None,
        slice_names: tuple[str, ...] = DOMAIN_SLICES,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        aux_prior_weight: float = 0.25,
        warmup_epochs: int = 5,
        cosine_total_epochs: int = 100,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "class_weights"])
        self.model = model
        self.slice_names = tuple(slice_names)
        if class_weights is not None:
            cw = torch.as_tensor(list(class_weights), dtype=torch.float32)
        else:
            cw = torch.ones(3, dtype=torch.float32)
        self.register_buffer("class_weights", cw)
        self.register_buffer("_slice_correct", torch.zeros(len(self.slice_names), dtype=torch.float64), persistent=False)
        self.register_buffer("_slice_total", torch.zeros(len(self.slice_names), dtype=torch.float64), persistent=False)
        self.register_buffer("_val_cm", torch.zeros(3, 3, dtype=torch.float64), persistent=False)

    def forward(self, batch_or_x: torch.Tensor | dict) -> dict[str, torch.Tensor]:
        out = self.model(batch_or_x)
        if isinstance(out, dict):
            return out
        if isinstance(out, tuple):
            return {"logits": out[0], "pred": out[0], "h": out[1]}
        return {"logits": out, "pred": out}

    def _compute_loss(self, out: dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        logits = out["logits"]
        weights = self.class_weights.to(device=logits.device, dtype=torch.float32)
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            weight=weights,
            ignore_index=self.hparams.ignore_index,
        )
        prior_logits = out.get("prior_logits")
        if prior_logits is not None and self.hparams.aux_prior_weight > 0.0:
            prior_loss = F.cross_entropy(
                prior_logits.float().reshape(-1, prior_logits.shape[-1]),
                y.reshape(-1),
                weight=weights,
                ignore_index=self.hparams.ignore_index,
            )
            loss = loss + self.hparams.aux_prior_weight * prior_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected: {loss.item()}")
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        out = self.forward(batch)
        loss = self._compute_loss(out, batch["y"])
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["y"].size(0))
        return loss

    def on_train_epoch_start(self) -> None:
        dl = self.trainer.train_dataloader if self._trainer is not None else None
        ds = getattr(dl, "dataset", None)
        if ds is not None and hasattr(ds, "reshuffle"):
            ds.reshuffle(int(self.current_epoch))

    def on_validation_epoch_start(self) -> None:
        self._slice_correct.zero_()
        self._slice_total.zero_()
        self._val_cm.zero_()

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        out = self.forward(batch)
        y = batch["y"]
        loss = self._compute_loss(out, y)
        preds = out["logits"].detach().argmax(dim=-1)
        valid = y != self.hparams.ignore_index

        flat_y = y[valid].reshape(-1).to(torch.long)
        flat_p = preds[valid].reshape(-1).to(torch.long)
        if flat_y.numel() > 0:
            bins = torch.bincount(flat_y * 3 + flat_p, minlength=9).reshape(3, 3).to(torch.float64)
            self._val_cm += bins

        slice_masks = batch.get("slice_masks", {})
        for i, name in enumerate(self.slice_names):
            raw_mask = slice_masks.get(name)
            mask = valid if raw_mask is None else (valid & raw_mask.to(device=y.device, dtype=torch.bool))
            correct = torch.where(mask, (preds == y).to(torch.float64), torch.zeros_like(y, dtype=torch.float64))
            self._slice_correct[i] += correct.sum()
            self._slice_total[i] += mask.to(torch.float64).sum()

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=y.size(0), sync_dist=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        world = self.trainer.world_size if self._trainer is not None else 1
        correct = self.all_gather(self._slice_correct).sum(0) if world > 1 else self._slice_correct
        total = self.all_gather(self._slice_total).sum(0) if world > 1 else self._slice_total
        cm = self.all_gather(self._val_cm).sum(0) if world > 1 else self._val_cm

        tp = torch.diagonal(cm)
        denom = cm.sum(dim=0) + cm.sum(dim=1)
        f1_per_class = torch.where(denom > 0, 2.0 * tp / denom.clamp_min(1.0), torch.zeros_like(tp))
        macro_f1 = f1_per_class.mean().to(torch.float32)
        self.log("val/macro_f1", macro_f1, prog_bar=True, sync_dist=False)

        slice_scores = []
        for i, name in enumerate(self.slice_names):
            cnt = float(total[i].item())
            self.log(f"val/slice_{name}_count", cnt, prog_bar=False, sync_dist=False)
            if cnt > 0.0:
                acc = (correct[i] / total[i]).to(torch.float32)
                self.log(f"val/slice_{name}", acc, prog_bar=(i == 0), sync_dist=False)
                slice_scores.append(acc)
            else:
                self.log(f"val/slice_{name}", float("nan"), prog_bar=False, sync_dist=False)
        if slice_scores:
            self.log("val/primary_mean", torch.stack(slice_scores).mean(), prog_bar=True, sync_dist=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(build_param_groups(self.model, self.hparams.weight_decay), lr=self.hparams.lr)
        warmup = max(1, int(self.hparams.warmup_epochs))
        total = max(warmup + 1, int(self.hparams.cosine_total_epochs))

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return float(epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


class _BestMetricCallback(Callback):
    """Track best validation checkpoint & enforce post-warmup patience via well-level inference."""

    def __init__(
        self,
        val_wells: list[PreparedWell] | None = None,
        seq_len: int = 256,
        infer_stride: int = 128,
        monitor: str = "val/macro_f1",
        warmup_epochs: int = 5,
        patience: int = 25,
        early_stop_after_warmup: bool = True,
    ) -> None:
        super().__init__()
        self.val_wells = val_wells
        self.seq_len = seq_len
        self.infer_stride = infer_stride
        self.monitor = monitor
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.early_stop_after_warmup = early_stop_after_warmup
        self.best_score: float = -1.0
        self.best_epoch: int = -1
        self.best_state: dict[str, torch.Tensor] | None = None
        self.bad_epochs: int = 0

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        if self.val_wells:
            from .train import run_predict_well

            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            was_training = pl_module.model.training
            try:
                y_val = np.concatenate([w.y for w in self.val_wells])
                p_val = np.concatenate([
                    run_predict_well(pl_module.model, w, self.seq_len, self.infer_stride, pl_module.device)
                    for w in self.val_wells
                ])
                score = float(compute_metrics(y_val, p_val.argmax(1), probs=p_val)["macro_f1"])
            finally:
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
                pl_module.model.train(was_training)
        else:
            score_t = trainer.callback_metrics.get(self.monitor)
            if score_t is None:
                return
            score = float(score_t.item())

        active = (not self.early_stop_after_warmup) or (int(trainer.current_epoch) >= self.warmup_epochs)
        if active and math.isfinite(score):
            if score > self.best_score:
                self.best_score = score
                self.best_epoch = int(trainer.current_epoch)
                self.best_state = {k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()}
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= self.patience:
                    trainer.should_stop = True


def fit_lightning_with_val(
    model: nn.Module,
    train_wells: list[PreparedWell],
    val_wells: list[PreparedWell],
    class_weights: np.ndarray,
    seq_len: int = 256,
    train_stride: int = 64,
    infer_stride: int = 128,
    batch_size: int = 32,
    max_epochs: int = 150,
    patience: int = 25,
    warmup_epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    aux_prior_weight: float = 0.10,
    seed: int = 0,
    accelerator: str = "gpu",
    test_wells: list[PreparedWell] | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Train with post-warmup validation checkpointing and early stopping via L.Trainer."""
    from .train import run_predict_well

    L.seed_everything(seed, workers=False)
    lit = BondFusionLightningModule(
        model=model,
        class_weights=class_weights,
        lr=lr,
        weight_decay=weight_decay,
        aux_prior_weight=aux_prior_weight,
        warmup_epochs=warmup_epochs,
        cosine_total_epochs=max_epochs,
    )
    tr_ds = _WindowDictDataset(train_wells, seq_len, train_stride, seed=seed)
    va_ds = _WindowDictDataset(val_wells, seq_len, infer_stride, seed=seed)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=len(tr_ds) > batch_size)
    va_loader = DataLoader(va_ds, batch_size=64, shuffle=False)

    best_cb = _BestMetricCallback(
        val_wells=val_wells,
        seq_len=seq_len,
        infer_stride=infer_stride,
        monitor="val/macro_f1",
        warmup_epochs=warmup_epochs,
        patience=patience,
        early_stop_after_warmup=True,
    )
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=max_epochs,
        gradient_clip_val=1.0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[best_cb],
        deterministic="warn",
    )
    trainer.fit(lit, train_dataloaders=tr_loader, val_dataloaders=va_loader)
    if best_cb.best_state is not None:
        lit.load_state_dict(best_cb.best_state)
    device = torch.device("cuda:0" if accelerator == "gpu" and torch.cuda.is_available() else "cpu")
    lit.to(device).eval()

    val_probs = {w.name: run_predict_well(lit.model, w, seq_len, infer_stride, device) for w in val_wells}
    test_probs = {w.name: run_predict_well(lit.model, w, seq_len, infer_stride, device) for w in (test_wells or [])}
    info = {
        "best_val_macro_f1": best_cb.best_score if best_cb.best_state is not None else None,
        "best_epoch": best_cb.best_epoch,
        "epochs_run": int(trainer.current_epoch),
        "n_params": sum(p.numel() for p in lit.model.parameters() if p.requires_grad),
        "test_probs": test_probs,
    }
    return val_probs, info


def refit_and_predict_lightning(
    model: nn.Module,
    train_wells: list[PreparedWell],
    test_wells: list[PreparedWell],
    class_weights: np.ndarray,
    stop_epoch: int,
    seq_len: int = 256,
    train_stride: int = 64,
    infer_stride: int = 128,
    batch_size: int = 32,
    warmup_epochs: int = 5,
    cosine_total_epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    aux_prior_weight: float = 0.25,
    seed: int = 0,
    accelerator: str = "gpu",
) -> tuple[dict[str, np.ndarray], dict]:
    """Train on all 5 outer training wells for ``stop_epoch`` epochs via L.Trainer and predict test well."""
    from .train import run_predict_well

    L.seed_everything(seed, workers=False)
    lit = BondFusionLightningModule(
        model=model,
        class_weights=class_weights,
        lr=lr,
        weight_decay=weight_decay,
        aux_prior_weight=aux_prior_weight,
        warmup_epochs=warmup_epochs,
        cosine_total_epochs=max(cosine_total_epochs, stop_epoch),
    )
    tr_ds = _WindowDictDataset(train_wells, seq_len, train_stride, seed=seed)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=len(tr_ds) > batch_size)
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=stop_epoch,
        gradient_clip_val=1.0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic="warn",
    )
    trainer.fit(lit, train_dataloaders=tr_loader)
    device = torch.device("cuda:0" if accelerator == "gpu" and torch.cuda.is_available() else "cpu")
    lit.to(device).eval()

    test_probs = {w.name: run_predict_well(lit.model, w, seq_len, infer_stride, device) for w in test_wells}
    info = {
        "stop_epoch": int(stop_epoch),
        "n_params": sum(p.numel() for p in lit.model.parameters() if p.requires_grad),
    }
    return test_probs, info
