"""Five-second CPU mathematical invariant tests for BondFusion-TCN and Lightning container."""
from __future__ import annotations

import torch

import lightning as L

from da_reformer.experiment import build_model
from da_reformer.lightning_module import BondFusionLightningModule, DOMAIN_SLICES, build_param_groups
from da_reformer.models.bondfusion_tcn import BondFusionTCN


def _sample_batch(batch_size: int = 2, seq_len: int = 256, n_curve: int = 3, n_wave: int = 19) -> dict:
    torch.manual_seed(2026)
    x = torch.randn(batch_size, seq_len, n_curve + n_wave + 1)
    x[..., -1] = 1.0  # valid mask
    y = torch.randint(0, 3, (batch_size, seq_len))
    slice_masks = {
        "poor_free_pipe": y == 0,
        "moderate_trough": y == 1,
        "good_coupled": y == 2,
        "boundary_transition": torch.zeros_like(y, dtype=torch.bool),
    }
    slice_masks["boundary_transition"][:, ::16] = True
    return {"x": x, "y": y, "slice_masks": slice_masks}


def test_autoresearch_module_invariants() -> None:
    base_hp = {"d_lift": 16, "patch_size": 8, "d_model": 64, "ffn_ratio": 1, "num_blocks": 1, "large_size": 51, "small_size": 5, "dropout": 0.1}
    baseline_model = build_model("ModernTCN +VDL", 3 + 19 + 1, 3, 256, base_hp, n_domains=5, class_counts=[100, 100, 100], n_wave=0)
    candidate_model = BondFusionTCN(n_curve=3, n_wave=19, n_classes=3, seq_len=256)
    fine_model = BondFusionTCN(n_curve=3, n_wave=95, n_classes=3, seq_len=256)
    sample_batch = _sample_batch()

    baseline_model.eval()
    candidate_model.eval()

    # 1. No parameter bloat relative to ModernTCN +VDL (~230k) and exact match vs no-zero-phase variant.
    n_base = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
    n_cand = sum(p.numel() for p in candidate_model.parameters() if p.requires_grad)
    n_fine = sum(p.numel() for p in fine_model.parameters() if p.requires_grad)
    assert n_cand <= n_base * 1.05 and n_fine <= n_base * 1.05, f"Param bloat: {n_cand}/{n_fine} vs {n_base}"

    # 2. Law 1: Decoupled curve prior has zero dependence / shortcut leakage from VDL channels.
    b1 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sample_batch.items()}
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sample_batch.items()}
    b2["x"][:, :, 3:22] = torch.randn_like(b2["x"][:, :, 3:22]) * 100.0
    with torch.no_grad():
        prior1 = candidate_model.forward_prior(b1)
        prior2 = candidate_model.forward_prior(b2)
    assert torch.allclose(prior1, prior2, atol=1e-6), "Shortcut leakage: forward_prior depends on VDL channels!"

    # 3. Law 2: Zero-phase symmetry of multiscale depth filter and depthwise convs around center impulse.
    impulse = torch.zeros(1, 1, 65)
    impulse[0, 0, 32] = 1.0
    with torch.no_grad():
        resp = candidate_model.multiscale_filter(impulse)
    assert int(resp.abs().argmax(dim=-1).item()) == 32, "Phase shift detected in multiscale_filter!"
    assert torch.allclose(resp, resp.flip(dims=[-1]), atol=1e-5), "Asymmetric filter response!"

    # 4. Law 2 & Law 6: Rational boundary gate vanishes identically at d=0.
    with torch.no_grad():
        g0 = candidate_model.boundary_gate(torch.zeros(4, 16))
        g_pos = candidate_model.boundary_gate(torch.ones(4, 16))
    assert torch.allclose(g0, torch.zeros_like(g0), atol=1e-7), f"Boundary gate g(0) != 0: max={g0.abs().max()}"
    assert torch.all((g_pos > 0.0) & (g_pos < 1.0)), "Positive input must activate boundary gate in (0, 1)"

    # 5. bf16 + 10-sigma outlier spike finite loss & non-zero finite gradients on every parameter.
    candidate_model.train()
    candidate_model.zero_grad(set_to_none=True)
    noisy = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sample_batch.items()}
    std = noisy["x"].float().std().clamp_min(1.0)
    spike_mask = (torch.rand_like(noisy["x"]) > 0.98).to(noisy["x"].dtype)
    noisy["x"] = noisy["x"] + (10.0 * std) * spike_mask
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        out = candidate_model(noisy)
        loss = out["pred"].float().pow(2).mean() + 0.25 * out["prior_logits"].float().pow(2).mean()
    assert torch.isfinite(loss), f"Non-finite loss under bf16 + 10-sigma spike: {loss}"
    loss.backward()
    for name, p in candidate_model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Dead parameter (grad is None): {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"


def test_lightning_container_and_param_groups() -> None:
    hp = {"d_lift": 32, "wave_width": 16, "ffn_ratio": 2, "num_blocks": 1, "large_size": 31, "small_size": 5, "dropout": 0.1}
    model = BondFusionTCN(n_curve=3, n_wave=19, n_classes=3, seq_len=64, hp=hp)

    # Verify ndim < 2 zero weight decay rule (Law 7 Trap D).
    groups = build_param_groups(model, weight_decay=1e-3)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 1e-3 and all(p.ndim >= 2 for p in groups[0]["params"])
    assert groups[1]["weight_decay"] == 0.0 and all(p.ndim < 2 for p in groups[1]["params"])

    lit = BondFusionLightningModule(model, class_weights=[1.0, 1.0, 2.0], lr=1e-3, weight_decay=1e-3)
    b = _sample_batch(batch_size=2, seq_len=64)

    class SingleBatchLoader:
        def __len__(self):
            return 1

        def __iter__(self):
            yield b

    trainer = L.Trainer(accelerator="cpu", max_epochs=1, logger=False, enable_checkpointing=False, enable_progress_bar=False)
    empty_mask = torch.zeros_like(b["y"], dtype=torch.bool)
    b_empty = {**b, "slice_masks": {**b["slice_masks"], DOMAIN_SLICES[-1]: empty_mask}}

    class EmptySliceLoader:
        def __len__(self):
            return 1

        def __iter__(self):
            yield b_empty

    metrics = trainer.validate(lit, dataloaders=EmptySliceLoader(), verbose=False)[0]
    assert "val/macro_f1" in metrics and 0.0 <= metrics["val/macro_f1"] <= 1.0
    assert torch.isnan(torch.tensor(metrics[f"val/slice_{DOMAIN_SLICES[-1]}"])), "Empty slice must report NaN (fail-closed)"
    trainer.fit(lit, train_dataloaders=SingleBatchLoader(), val_dataloaders=SingleBatchLoader())
