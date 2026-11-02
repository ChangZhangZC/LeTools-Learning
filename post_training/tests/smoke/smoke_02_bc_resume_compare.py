"""Compare uninterrupted vs checkpoint-resumed RL1002D BC training exactly.

After running the two workflows documented in BATCH02_RL100_BC.md:
  python post_training/tests/smoke/smoke_02_bc_resume_compare.py \
    --continuous outputs/train/rl100_bc_continuous/checkpoints/latest.ckpt \
    --resumed outputs/train/rl100_bc_resumed/checkpoints/latest.ckpt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import dill
import numpy as np
import torch


def assert_same(a, b, path="payload"):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or not torch.equal(a.cpu(), b.cpu()):
            raise AssertionError(f"Tensor mismatch: {path}")
    elif isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or not np.array_equal(a, b):
            raise AssertionError(f"NumPy mismatch: {path}")
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise AssertionError(f"Dict keys mismatch: {path}")
        for key in a:
            assert_same(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        if not isinstance(b, type(a)) or len(a) != len(b):
            raise AssertionError(f"Sequence mismatch: {path}")
        for i, (left, right) in enumerate(zip(a, b, strict=True)):
            assert_same(left, right, f"{path}[{i}]")
    elif a != b:
        raise AssertionError(f"Scalar mismatch: {path}: {a!r} != {b!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    args = parser.parse_args()
    continuous = torch.load(args.continuous, map_location="cpu", pickle_module=dill)
    resumed = torch.load(args.resumed, map_location="cpu", pickle_module=dill)
    for label, payload in (("continuous", continuous), ("resumed", resumed)):
        epoch = dill.loads(payload["pickles"]["epoch"])
        step = dill.loads(payload["pickles"]["global_step"])
        if (epoch, step) != (2, 4):
            raise AssertionError(f"{label}: expected (epoch=2,step=4), got {(epoch, step)}")
    assert_same(continuous["state_dicts"], resumed["state_dicts"], "state_dicts")
    for key in ("epoch", "global_step", "ema_optimization_step", "ema_decay",
                "rng_state"):
        if key not in continuous["pickles"] or key not in resumed["pickles"]:
            raise AssertionError(f"Missing continuation key {key}")
        assert_same(
            dill.loads(continuous["pickles"][key]),
            dill.loads(resumed["pickles"][key]),
            f"pickles.{key}",
        )
    # Earlier accepted checkpoints predate best-val selection; keep the
    # existing exact-resume test valid for those saved server artifacts.
    k = "best_val_loss"
    if (k in continuous["pickles"]) != (k in resumed["pickles"]):
        raise AssertionError("Only one run records best-val state")
    if k in continuous["pickles"]:
        assert_same(dill.loads(continuous["pickles"][k]),
                    dill.loads(resumed["pickles"][k]), "pickles.best_val_loss")
    for path in ("best.ckpt",):
        left = args.continuous.with_name(path)
        right = args.resumed.with_name(path)
        if left.exists() != right.exists():
            raise AssertionError(f"{path} exists only for one training run")
        if left.exists():
            a = torch.load(left, map_location="cpu", pickle_module=dill)
            b = torch.load(right, map_location="cpu", pickle_module=dill)
            assert_same(a["state_dicts"], b["state_dicts"], "best.state_dicts")
            for key in ("epoch", "global_step", "best_val_loss"):
                assert_same(dill.loads(a["pickles"][key]),
                            dill.loads(b["pickles"][key]), f"best.{key}")
    log_a = args.continuous.parents[1] / "metrics.jsonl"
    log_b = args.resumed.parents[1] / "metrics.jsonl"
    if log_a.exists() != log_b.exists():
        raise AssertionError("metrics.jsonl exists only for one run")
    if log_a.exists():
        entries_a = [json.loads(x) for x in log_a.read_text().splitlines() if x.strip()]
        entries_b = [json.loads(x) for x in log_b.read_text().splitlines() if x.strip()]
        assert_same(entries_a, entries_b, "metrics.jsonl")
    print("[PASS] Uninterrupted and epoch-boundary-resumed BC states match exactly")
    print("[PASS] Model, normalizer, optimizer, scheduler, EMA, step counters and RNG match")


if __name__ == "__main__":
    main()
