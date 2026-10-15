"""Exercise the release on temporary synthetic wells; no paper result is computed."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_code"))

import numpy as np
import pandas as pd
import torch

from da_reformer.experiment import build_model, build_split, load_protocol, run_fit_eval
from da_reformer.vdl import build_log_envelope


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    torch.set_num_threads(1)
    p = load_protocol(ROOT / "_pipelines/01_run_cement_bond_benchmark/protocol_v03.yaml")
    hp = {"d_lift": 4, "patch_size": 4, "d_model": 8, "ffn_ratio": 1,
          "num_blocks": 1, "large_size": 7, "small_size": 3,
          "dropout": 0.0}
    p["train"].update(seq_len=32, batch_size=2, max_epochs=1, warmup_epochs=0,
                      train_stride=16, infer_stride=16)
    rng = np.random.default_rng(2026)
    with tempfile.TemporaryDirectory(prefix="bondfusion-synthetic-") as tmp:
        p["_root"] = Path(tmp)
        raw = Path(tmp) / p["raw_dir"]
        raw.mkdir(parents=True)
        for w in ["A", "B", "E"]:
            n = 48
            # Two segments exercise padding and prevent windows crossing a depth gap.
            depth = np.r_[np.arange(24) * 0.1, 10 + np.arange(24) * 0.1]
            pd.DataFrame({"DEPTH": depth, "GR": rng.uniform(1, 90, n),
                          "CBL": rng.uniform(1, 30, n), "CCL": rng.normal(0, 10, n),
                          "AC": np.full(n, -9999),
                          "label_raw": np.tile(["胶结差", "胶结中等", "胶结好"], 16)}).to_csv(
                raw / f"{w}.csv", index=False)
            np.save(raw / f"{w}_vdl.npy", rng.integers(-500, 501, (n, 2000), dtype=np.int16))

        probe = np.arange(-10, 10, dtype=np.float32).reshape(1, 20)
        env = build_log_envelope(probe, 10)
        assert env.shape == (1, 2)
        assert np.isclose(env[0, 0], np.log1p(np.abs(probe[0, :10]).mean()))

        prep, scaler, _, _ = build_split(p, ["A", "B"], ["E"], "blocks")
        train_before = prep["A"].x.copy()
        median_before = scaler.median.copy()
        frame = pd.read_csv(raw / "E.csv")
        frame["GR"] += 10000
        frame.to_csv(raw / "E.csv", index=False)
        eval_wave = np.load(raw / "E_vdl.npy")
        np.save(raw / "E_vdl.npy", eval_wave * 10)
        prep_after, scaler_after, _, _ = build_split(p, ["A", "B"], ["E"], "blocks")
        assert np.array_equal(train_before, prep_after["A"].x)
        assert np.array_equal(median_before, scaler_after.median)
        assert prep_after["E"].x.shape == (48, 22)
        assert not np.array_equal(prep["E"].x, prep_after["E"].x)

        r = run_fit_eval("ModernTCN +VDL", hp, p, ["A", "B"], ["E"],
                         torch.device("cpu"), seed=0, validate=False, stop_epoch=1)
        probs = r["probs"]["E"]
        assert probs.shape == (48, 3) and np.isfinite(probs).all()
        assert np.allclose(probs.sum(1), 1.0, atol=1e-6)
        assert len(r["fit"]["history"]) == 1
        assert np.isfinite(r["fit"]["history"][0]["cls"])

        wave_hp = {**hp, "d_wave": 8, "wave_channels": 8}
        model = build_model("VDL-ModernTCN (mean pool)", 194, 3, 32, wave_hp, 2, None, n_wave=190)
        logits, _ = model(torch.randn(2, 32, 194))
        assert logits.shape == (2, 32, 3)
        logits.square().mean().backward()
        assert model.wave.conv[0].weight.grad.abs().sum() > 0
    print(json.dumps({"status": "passed", "device": "cpu", "data": "temporary synthetic only",
                      "checks": ["envelope", "train-only scaling", "segment-aware training",
                                 "per-point probabilities", "waveform encoder gradients"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
