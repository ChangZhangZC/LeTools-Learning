"""Batch 02: verify the native BC-only optimizer checkpoint on real Zarr.

Run after the one-step BC trainer:
  python post_training/tests/smoke/smoke_02_bc_checkpoint.py \
    --checkpoint outputs/train/rl100_bc_batch02/checkpoints/latest.ckpt \
    --device cuda:0
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import dill
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "post_training" / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "rl100" / "RL-100"))

from arbim_rl100.data.offline_buffer import OfflineBuffer
from arbim_rl100.data.offline_dataset import OfflineDataset
from arbim_rl100.train_bc import (
    bc_batch, build_policy, fit_train_normalizer, load_checkpoint, shape_meta_from_buffer,
)
from rl_100.model.common.lr_scheduler import get_scheduler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", pickle_module=dill)
    config = payload["cfg"]
    if set(payload) != {"cfg", "state_dicts", "pickles"}:
        raise AssertionError("Checkpoint deviates from original RL-100 payload structure")
    if set(payload["state_dicts"]) != {"model", "ema_model", "optimizer", "lr_scheduler"}:
        raise AssertionError("Expected model, EMA, optimizer and LR scheduler")
    if not payload["state_dicts"]["optimizer"]["state"]:
        raise AssertionError("Optimizer has no state: no effective training step?")
    buffer = OfflineBuffer(torch.device("cpu"), gamma=0.99, use_depth=False)
    buffer.load_zarr(str(Path(config.data.zarr).expanduser().resolve()))
    data = OfflineDataset(
        buffer, horizon=int(config.data.horizon), pad_before=int(config.data.pad_before),
        pad_after=int(config.data.pad_after), seed=int(config.data.seed),
        val_ratio=float(config.data.val_ratio),
    )
    device = torch.device(args.device)
    model = build_policy(config, shape_meta_from_buffer(buffer))
    # RL-100 LinearNormalizer adds registered per-field parameters on fit.
    # Initialize identical structure before strict checkpoint state_dict loading.
    model.set_normalizer(fit_train_normalizer(buffer, data.train_mask))
    model.to(device)
    ema_model = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.training.lr),
        betas=tuple(float(v) for v in config.training.betas),
        eps=float(config.training.eps),
        weight_decay=float(config.training.weight_decay),
    )
    loader_len = (len(data) + int(config.training.batch_size) - 1) // int(config.training.batch_size)
    steps = min(loader_len, int(config.training.max_train_steps_per_epoch)) \
        if config.training.max_train_steps_per_epoch is not None else loader_len
    scheduler = get_scheduler(
        str(config.training.lr_scheduler), optimizer=optimizer,
        num_warmup_steps=int(config.training.lr_warmup_steps),
        num_training_steps=steps * int(config.training.epochs),
    )
    epoch, global_step = load_checkpoint(
        args.checkpoint, model, optimizer, ema_model, scheduler
    )
    if epoch != 1 or global_step != 1:
        raise AssertionError(f"Expected one-step checkpoint, got epoch={epoch}, global_step={global_step}")
    for key, expected in payload["state_dicts"]["model"].items():
        actual = model.state_dict()[key]
        if not torch.equal(actual.detach().cpu(), expected):
            raise AssertionError(f"Model/normalizer checkpoint mismatch: {key}")
    for key, expected in payload["state_dicts"]["ema_model"].items():
        if not torch.equal(ema_model.state_dict()[key].detach().cpu(), expected):
            raise AssertionError(f"EMA checkpoint mismatch: {key}")
    model.eval()
    sample = next(iter(DataLoader(data, batch_size=1, shuffle=False, num_workers=0)))
    batch = bc_batch(sample, int(config.data.n_obs_steps), device)
    with torch.no_grad():
        action = model.predict_action(batch["obs"], deterministic=True)["action"]
    expected_shape = (1, int(config.data.n_action_steps), int(buffer["action"].shape[-1]))
    if tuple(action.shape) != expected_shape or not torch.isfinite(action).all():
        raise AssertionError(f"Invalid restored Action Chunk: {tuple(action.shape)}")
    print(f"[PASS] checkpoint epoch={epoch}, global_step={global_step}")
    print("[PASS] strict RL1002D/Normalizer/EMA weights, optimizer and scheduler restored")
    print(f"[PASS] restored Action Chunk={tuple(action.shape)}")


if __name__ == "__main__":
    main()
