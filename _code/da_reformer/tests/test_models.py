"""Shape / gradient / search-space smoke tests for the baseline registry.

Run: PYTHONPATH=_code python -m pytest _code/da_reformer/tests/test_models.py -q
"""
import optuna
import pytest
import torch

from da_reformer.models.backbones import ReformerEncoder
from da_reformer.models.baselines import BASELINE_NAMES, build_baseline, default_hp, suggest_hp

N_FEATURES, N_CLASSES, BATCH = 4, 3, 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_forward_backward(model, seq_len, n_features=N_FEATURES):
    model.to(DEVICE).train()
    x = torch.randn(BATCH, seq_len, n_features, device=DEVICE)
    logits, h = model(x)
    assert logits.shape == (BATCH, seq_len, N_CLASSES)
    assert h.dim() == 3 and h.shape[0] == BATCH and h.shape[1] == seq_len
    assert model.head.in_features == h.shape[2]
    assert torch.isfinite(logits).all() and torch.isfinite(h).all()
    logits.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_registry_names():
    assert BASELINE_NAMES == ["LSTM", "GRU", "Transformer", "Reformer", "Crossformer", "TimesNet", "TimeMixer",
                              "ModernTCN"]


@pytest.mark.parametrize("seq_len", [256, 128])
@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_default_hp_forward_backward(name, seq_len):
    torch.manual_seed(0)
    model = build_baseline(name, N_FEATURES, N_CLASSES, seq_len, default_hp(name))
    print(f"{name} L={seq_len} params={count_params(model):,}")
    run_forward_backward(model, seq_len)


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_suggest_hp_samples(name):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(3):
        trial = study.ask()
        hp = {**suggest_hp(name, trial), "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4}
        model = build_baseline(name, N_FEATURES, N_CLASSES, 256, hp)
        print(f"{name} hp={hp} params={count_params(model):,}")
        run_forward_backward(model, 256)
        study.tell(trial, 0.0)


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_five_features_and_l512(name):
    model = build_baseline(name, 5, N_CLASSES, 512, default_hp(name))
    run_forward_backward(model, 512, n_features=5)


def test_unknown_hp_key_raises():
    with pytest.raises(ValueError):
        build_baseline("LSTM", N_FEATURES, N_CLASSES, 256, {"hiden": 32})


def test_reformer_encoder_interface_odd_length():
    enc = ReformerEncoder(N_FEATURES).to(DEVICE)
    h = enc(torch.randn(2, 250, N_FEATURES, device=DEVICE))  # 250 % (4 * 2) != 0 -> fit_length pad + crop
    assert h.shape == (2, 250, 64)


def test_no_instance_normalization_scale_sensitivity():
    # Inputs that differ only by a per-window offset must give different outputs
    # (an instance-normalized model would map them to (almost) the same logits).
    torch.manual_seed(0)
    x = torch.randn(2, 128, N_FEATURES, device=DEVICE)
    for name in BASELINE_NAMES:
        model = build_baseline(name, N_FEATURES, N_CLASSES, 128, default_hp(name)).to(DEVICE).eval()
        with torch.no_grad():
            torch.manual_seed(1)  # fixes the LSH random rotations of Reformer for both calls
            a, _ = model(x)
            torch.manual_seed(1)
            b, _ = model(x + 3.0)
        assert (a - b).abs().max() > 1e-3, name
