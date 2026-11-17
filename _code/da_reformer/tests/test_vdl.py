"""Tests for the VDL input path and VDL-ModernTCN."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from da_reformer.experiment import (build_model, build_split, load_protocol, parse_vdl_name, resolve_vdl_mode,
                                    run_fit_eval)
from da_reformer.models.vdl_moderntcn import VdlModernTCN
from da_reformer.vdl import build_log_envelope, build_vdl_channels, load_vdl

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "_data" / "raw"
PROTOCOL = ROOT / "_pipelines" / "01_run_cement_bond_benchmark" / "protocol_v03.yaml"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMALL = {"d_lift": 8, "patch_size": 16, "d_model": 32, "d_wave": 16, "wave_channels": 16}


def test_vdl_rows_align_and_trailer_is_cut():
    for w in "ABCDEF":
        raw = np.load(RAW / f"{w}_vdl.npy")
        assert raw.dtype == np.int16 and raw.shape == (len(pd.read_csv(RAW / f"{w}.csv")), 2000)
        assert not (raw[:, :1900] == -32768).any()  # the int16 floor only appears in the trailer
        assert load_vdl(RAW, w, 1900).shape == (len(raw), 1900)


def test_log_envelope_bins_and_train_only_scaling():
    v = np.arange(40, dtype=np.float32).reshape(2, 20) - 10
    env = build_log_envelope(v, 10)
    assert env.shape == (2, 2) and np.isclose(env[0, 0], np.log1p(np.abs(v[0, :10]).mean()))
    a = build_vdl_channels(RAW, ["A", "B", "C"], ["A", "B"], 1900, 100)
    b = build_vdl_channels(RAW, ["A", "B", "C"], ["A", "B", "C"], 1900, 100)
    assert a["C"].shape[1] == 19 and np.isfinite(a["C"]).all() and np.abs(a["C"]).max() <= 8.0
    assert not np.allclose(a["C"], b["C"])


def test_name_parsing_and_input_modes():
    assert resolve_vdl_mode("VDL-ModernTCN") == "bins"
    assert resolve_vdl_mode("VDL-ModernTCN w/o waveform encoder") == "blocks"
    assert resolve_vdl_mode("VDL-ModernTCN w/o VDL") is None
    assert resolve_vdl_mode("GRU +VDL") == "blocks" and resolve_vdl_mode("GRU") is None
    assert parse_vdl_name("VDL-ModernTCN + Reformer")["global_layer"] == "reformer"
    with pytest.raises(KeyError):
        parse_vdl_name("VDL-ModernTCN (typo)")


def test_build_split_appends_vdl_channels():
    p = load_protocol(PROTOCOL)
    for mode, extra in ((None, 0), ("blocks", 19), ("bins", 190)):
        prep, sc, _, _ = build_split(p, ["A", "B"], ["C"], mode)
        assert prep["C"].x.shape == (len(prep["C"].y), sc.n_channels + extra)


@pytest.mark.parametrize("name", ["VDL-ModernTCN", "VDL-ModernTCN (mean pool)", "VDL-ModernTCN (static query)",
                                  "VDL-ModernTCN + Reformer"])
def test_variants_forward_and_gradients_reach_waveform_encoder(name):
    torch.manual_seed(0)
    m = build_model(name, 3 + 190 + 1, 3, 256, SMALL, 4, None, n_wave=190).to(DEV)
    x = torch.randn(2, 256, 194, device=DEV)
    logits, h = m(x)
    assert logits.shape == (2, 256, 3) and h.shape == (2, 256, 8)
    logits.square().mean().backward()
    assert m.wave.conv[0].weight.grad.abs().sum() > 0
    if "mean pool" not in name:
        assert torch.allclose(m.last_attention.sum(-1), torch.ones(2, 256, device=DEV), atol=1e-5)
    if name in ("VDL-ModernTCN", "VDL-ModernTCN + Reformer"):
        assert m.query[0].weight.grad.abs().sum() > 0


def test_curve_conditioned_attention_depends_on_curves_only_for_curve_query():
    torch.manual_seed(0)
    x = torch.randn(1, 256, 194, device=DEV)
    x2 = x.clone()
    x2[..., 1] += 3.0  # change CBL only
    for pool, should_change in (("curve_attn", True), ("static_attn", False)):
        m = VdlModernTCN(3, 190, 3, 256, SMALL, pool=pool).to(DEV).eval()
        with torch.no_grad():
            m(x)
            a1 = m.last_attention.clone()
            m(x2)
            a2 = m.last_attention
        assert (not torch.allclose(a1, a2)) == should_change


def test_ablation_variants_build_plain_moderntcn():
    from da_reformer.models.baselines import PointwiseClassifier
    hp = {**SMALL, "lr": 1e-3}
    assert isinstance(build_model("VDL-ModernTCN w/o waveform encoder", 3 + 19 + 1, 3, 256, hp, 4, None),
                      PointwiseClassifier)
    assert isinstance(build_model("VDL-ModernTCN w/o VDL", 4, 3, 256, hp, 4, None), PointwiseClassifier)


def test_run_fit_eval_vdl_reproducible_from_any_rng_state():
    p = load_protocol(PROTOCOL)
    res = []
    for k in range(2):
        torch.rand(555 * (k + 1))
        r = run_fit_eval("VDL-ModernTCN", SMALL, p, ["A", "B"], ["C"], DEV, seed=1, validate=False, stop_epoch=2)
        res.append(r["probs"]["C"])
    assert res[0].shape == (3994, 3) and np.allclose(res[0], res[1], atol=1e-6)
