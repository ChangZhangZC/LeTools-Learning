"""Batch 01D: original ARBiM Zarr reader -> native RL1002D BC loss/backward.

Smoke test only: does NOT optimize weights, save a checkpoint, or infer reward.
Default source: data/zarr/icra_task1_10bags.zarr
Run:
  python post_training/tests/smoke/smoke_01_real_bc.py \
    --zarr data/zarr/icra_task1_10bags.zarr --device cuda:0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "post_training" / "src"))
RL100_ROOT = ROOT / "third_party" / "rl100" / "RL-100"
sys.path.insert(0, str(RL100_ROOT))

from arbim_rl100.data.batch_adapter import to_rl100_bc_batch
from arbim_rl100.data.offline_buffer import OfflineBuffer
from arbim_rl100.data.offline_dataset import OfflineDataset
from rl_100.model.common.normalizer import LinearNormalizer
from rl_100.model.vision.model_getter import get_resnet
from rl_100.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from rl_100.policy.rl100_2d import RL1002D
from rl_100.unidpg.diffusion_policy.diffusers_patch.ddim_with_logprob_dpok import (
    DDIMSchedulerExtended,
)

CAMERAS = {
    "rgb_head": "head_rgb",
    "rgb_left_hand": "wrist_left_rgb",
    "rgb_right_hand": "wrist_right_rgb",
}


def fit_train_normalizer(buffer: OfflineBuffer, train_mask: np.ndarray) -> LinearNormalizer:
    """Call RL-100's own fit(), using *train episodes only*."""
    boundaries = buffer.episode_ends
    chunks = []
    start = 0
    for use_episode, end in zip(train_mask, boundaries, strict=True):
        if use_episode:
            chunks.append(np.arange(start, int(end), dtype=np.int64))
        start = int(end)
    if not chunks:
        raise ValueError("There are no training episodes for normalizer fitting")
    indices = np.concatenate(chunks)
    normalizer = LinearNormalizer()
    normalizer.fit(
        {"agent_pos": buffer["state"][indices], "action": buffer["action"][indices]},
        mode="limits",
    )
    print(f"[PASS] RL-100 LinearNormalizer fit on {len(indices)} training frames")
    return normalizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr", type=Path, default=Path("data/zarr/icra_task1_10bags.zarr"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizon", type=int, default=9)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--n-action-steps", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--resize", type=int, default=84)
    args = parser.parse_args()

    if not args.zarr.is_dir():
        raise FileNotFoundError(
            f"Offline Zarr cache not found: {args.zarr}. "
            "Convert the local LeRobot dataset first; see data/README.md"
        )
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be in [0,1)")
    if args.horizon < 1 or not 1 <= args.n_obs_steps <= args.horizon:
        raise ValueError("Require 1 <= n_obs_steps <= horizon")
    if not 1 <= args.n_action_steps <= args.horizon - args.n_obs_steps + 1:
        raise ValueError("n_action_steps exceeds native RL1002D executable horizon")
    if args.resize < 32:
        raise ValueError("--resize must be at least 32")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    torch.manual_seed(123)
    torch.set_num_threads(min(torch.get_num_threads(), 4))

    buffer = OfflineBuffer(device=torch.device("cpu"), gamma=0.99, use_depth=False)
    buffer.load_zarr(str(args.zarr.resolve()))
    dataset = OfflineDataset(
        buffer, horizon=args.horizon, val_ratio=args.val_ratio, seed=42,
    )
    if not len(dataset):
        raise ValueError(
            f"No complete training windows of horizon={args.horizon}; "
            "check episode lengths or reduce --horizon"
        )
    val_dataset = dataset.get_validation_dataset()
    print(
        f"[PASS] Zarr frames={len(buffer)}, episodes={len(buffer.episode_ends)}, "
        f"train_windows={len(dataset)}, validation_windows={len(val_dataset)}"
    )

    state_dim = int(buffer["state"].shape[-1])
    action_dim = int(buffer["action"].shape[-1])
    shape_meta = {
        "obs": {
            key: {"shape": list(buffer[value].shape[1:]), "type": "rgb"}
            for key, value in CAMERAS.items()
        },
        "action": {"shape": [action_dim]},
    }
    shape_meta["obs"]["agent_pos"] = {
        "shape": [state_dim], "type": "low_dim"
    }

    normalizer = fit_train_normalizer(buffer, dataset.train_mask)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    sample = next(iter(loader))
    batch = to_rl100_bc_batch(sample, n_obs_steps=args.n_obs_steps)
    batch = {
        "obs": {key: value.to(device) for key, value in batch["obs"].items()},
        "action": batch["action"].to(device),
    }
    print(
        f"State={tuple(batch['obs']['agent_pos'].shape)} "
        f"Action={tuple(batch['action'].shape)} "
        f"RGB={tuple(batch['obs']['rgb_head'].shape)}"
    )

    # Reuse the exact native model construction pattern of Batch 00.
    obs_encoder = MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=get_resnet("resnet18", weights=None),
        resize_shape=(args.resize, args.resize),
        crop_shape=None,
        random_crop=False,
        use_group_norm=False,
        share_rgb_model=True,
        imagenet_norm=True,
        use_agent_pos=True,
        use_vib=False,
        use_recon=False,
    )
    scheduler = DDIMSchedulerExtended(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
        clip_std_min=0.01,
    )
    policy = RL1002D(
        shape_meta=shape_meta,
        obs_encoder=obs_encoder,
        cm_noise_scheduler=None,
        ddim_noise_scheduler=scheduler,
        scheduler_type="ddim",
        model="sample_dp3",
        feature_type="2D",
        w_pc=False,
        use_agent_pos=True,
        obs_as_global_cond=True,
        action_norm=True,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        num_inference_steps=2,
        ddim_inference_steps=2,
        down_dims=(32, 64, 128),
        diffusion_step_embed_dim=32,
        kernel_size=3,
        n_groups=8,
        use_aug=False,
    )
    policy.set_normalizer(normalizer)
    policy.to(device).train()

    loss, diagnostics = policy.compute_loss(batch)
    if loss.ndim != 0 or not torch.isfinite(loss).item():
        raise AssertionError(f"Invalid BC loss: {loss}")
    loss.backward()
    for label, module in (("U-Net", policy.model), ("RGB encoder", policy.obs_encoder)):
        grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
        if not grads or not all(torch.isfinite(g).all().item() for g in grads):
            raise AssertionError(f"No finite gradients for {label}")
        print(f"[PASS] {label} gradient")

    policy.eval()
    with torch.no_grad():
        prediction = policy.predict_action(batch["obs"], deterministic=True)["action"]
    expected = (batch["action"].shape[0], args.n_action_steps, action_dim)
    if tuple(prediction.shape) != expected or not torch.isfinite(prediction).all().item():
        raise AssertionError(
            f"Invalid predicted action, got {tuple(prediction.shape)}, expected {expected}"
        )
    print(f"[PASS] real-data RL1002D BC loss={loss.item():.6f}, diagnostics={diagnostics}")
    print(f"[PASS] predicted action={tuple(prediction.shape)}")
    print("[PASS] Batch 01D native RL1002D smoke")


if __name__ == "__main__":
    main()
