"""Batch 02 LeRobot-only native RL1002D policy smoke.

No Zarr import, OfflineBuffer, or RL post-training dataset involved.
Run after: pip install -e integrations/lerobot_policy_rl100_2d --no-deps
  python post_training/tests/smoke/smoke_02_lerobot_policy.py --device cuda:0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party" / "lerobot" / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "rl100" / "RL-100"))
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot_policy_rl100_2d import RL1002DConfig
from lerobot_policy_rl100_2d.modeling_rl100_2d import RL1002DPolicy


def features():
    return {
        "observation.state": PolicyFeature(FeatureType.STATE, (16,)),
        **{
            key: PolicyFeature(FeatureType.VISUAL, (3, 64, 64))
            for key in (
                "observation.images.head_cam_h",
                "observation.images.wrist_cam_l",
                "observation.images.wrist_cam_r",
            )
        },
    }


def stats(dim=16):
    return {
        name: {
            "min": torch.full((dim,), -1.0),
            "max": torch.full((dim,), 1.0),
            "mean": torch.zeros(dim),
            "std": torch.ones(dim),
        }
        for name in ("observation.state", "action")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if get_policy_class("rl100_2d") is not RL1002DPolicy:
        raise AssertionError("Native RL1002D policy not registered")
    cfg = RL1002DConfig(
        device=args.device, resize=84, down_dims=(32, 64, 128),
        diffusion_step_embed_dim=32, num_train_timesteps=10,
        num_inference_steps=2, resnet_weights=None,
        share_rgb_model=True, use_group_norm=False,
    )
    cfg.input_features = features()
    cfg.output_features = {"action": PolicyFeature(FeatureType.ACTION, (16,))}
    policy = RL1002DPolicy(cfg, dataset_stats=stats()).to(args.device)
    if any("zarr" in module for module in sys.modules if module.startswith("arbim_rl100")):
        raise AssertionError("LeRobot BC adapter loaded an ARBiM Zarr module")

    sample = {
        "observation.state": torch.zeros(1, 2, 16, device=args.device),
        "action": torch.zeros(1, 9, 16, device=args.device),
        **{
            key: torch.full((1, 2, 3, 64, 64), 128, dtype=torch.uint8, device=args.device)
            for key in (
                "observation.images.head_cam_h",
                "observation.images.wrist_cam_l",
                "observation.images.wrist_cam_r",
            )
        },
    }
    adapted = policy._to_native_batch(sample, with_time=True)
    assert adapted["rgb_head"].dtype == torch.float32
    assert adapted["rgb_head"].shape == (1, 2, 3, 64, 64)
    assert torch.allclose(adapted["rgb_head"].flatten()[:1], torch.tensor([128 / 255], device=args.device))
    loss, diagnostics = policy(sample)
    if not torch.isfinite(loss):
        raise AssertionError("Native RL1002D BC Loss is not finite")
    loss.backward()
    if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.native.model.parameters()):
        raise AssertionError("No native RL1002D U-Net gradient")
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=stats())
    if not preprocessor or not postprocessor:
        raise AssertionError("Plugin pre/postprocessor factory failed")
    policy.eval()
    policy.reset()
    inference = {k: v[:, -1] for k, v in sample.items() if k.startswith("observation.")}
    with torch.no_grad():
        action = policy.select_action(inference)
    if action.shape != (1, 16) or not torch.isfinite(action).all():
        raise AssertionError(f"Native inference action has invalid shape: {tuple(action.shape)}")
    print(f"[PASS] Native RL1002D LeRobot plugin forward/backward loss={loss.item():.6f}")
    print("[PASS] LeRobot three-camera tensor mapping and processor discovery")
    print("[PASS] LeRobot-only BC; no Zarr used")
    print(f"[PASS] Inference action shape={tuple(action.shape)}")


if __name__ == "__main__":
    main()
