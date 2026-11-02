"""Validate native RL1002D best-validation checkpoint and metric history.

Usage:
  python post_training/tests/smoke/smoke_02_bc_best.py \
    --output-dir outputs/train/rl100_bc_validation_trial
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import dill
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    folder = args.output_dir
    entries = [
        json.loads(x) for x in (folder / "metrics.jsonl").read_text().splitlines()
        if x.strip()
    ]
    if not entries:
        raise AssertionError("No recorded training epochs")
    for i, row in enumerate(entries):
        if not math.isfinite(row["train_loss"]) or not math.isfinite(row["val_loss"]):
            raise AssertionError(f"Nonfinite loss at metrics row {i}")
        if i and row["global_step"] <= entries[i - 1]["global_step"]:
            raise AssertionError(f"Step counter did not advance at metrics row {i}")
    best_row = min(entries, key=lambda item: item["val_loss"])
    best_path = folder / "checkpoints" / "best.ckpt"
    latest_path = folder / "checkpoints" / "latest.ckpt"
    best = torch.load(best_path, map_location="cpu", pickle_module=dill)
    latest = torch.load(latest_path, map_location="cpu", pickle_module=dill)
    best_epoch = int(dill.loads(best["pickles"]["epoch"]))
    best_loss = float(dill.loads(best["pickles"]["best_val_loss"]))
    if best_epoch != int(best_row["epoch"]) + 1:
        raise AssertionError(f"Best epoch mismatch: {best_epoch} != {best_row['epoch'] + 1}")
    if not math.isclose(best_loss, best_row["val_loss"], abs_tol=1e-7):
        raise AssertionError(f"Best loss mismatch: {best_loss} != {best_row['val_loss']}")
    latest_epoch = int(dill.loads(latest["pickles"]["epoch"]))
    if latest_epoch != int(entries[-1]["epoch"]) + 1:
        raise AssertionError("Latest checkpoint doesn't match final logged epoch")
    if int(dill.loads(latest["pickles"]["global_step"])) != entries[-1]["global_step"]:
        raise AssertionError("Latest checkpoint global_step differs from metrics")
    if not math.isclose(
        float(dill.loads(latest["pickles"]["best_val_loss"])),
        best_row["val_loss"], abs_tol=1e-7,
    ):
        raise AssertionError("Latest checkpoint lost best-val tracking")
    for key in ("model", "ema_model", "optimizer", "lr_scheduler"):
        if key not in best["state_dicts"]:
            raise AssertionError(f"Missing checkpoint state: {key}")
    print(f"[PASS] metrics epochs={len(entries)}, latest_epoch={latest_epoch}")
    print(f"[PASS] best validation epoch={best_epoch}, loss={best_loss:.6f}")
    print("[PASS] best/latest Checkpoints contain native RL1002D and optimizer states")


if __name__ == "__main__":
    main()
