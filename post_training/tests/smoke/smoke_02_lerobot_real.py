"""Real LeRobotDataset → native RL1002D BC adapter integration smoke (no Zarr).

Example, from LeTools repository root:
  python post_training/tests/smoke/smoke_02_lerobot_real.py \
    --lerobot-root /absolute/path/to/lerobot --device cuda:0
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party" / "lerobot" / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "rl100" / "RL-100"))

import torch
from torch.utils.data._utils.collate import default_collate
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.configs import FeatureType
from lerobot_policy_rl100_2d.configuration_rl100_2d import RL1002DConfig
from lerobot_policy_rl100_2d.modeling_rl100_2d import RL1002DPolicy, CAMERAS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video-backend", default="pyav")
    args = parser.parse_args()
    root = args.lerobot_root.expanduser().resolve()
    if not (root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Expected local LeRobot v3 dataset at {root}")
    meta = LeRobotDatasetMetadata(repo_id=root.name, root=root)
    features = dataset_to_policy_features(meta.features)
    config = RL1002DConfig(
        device=args.device, resize=84, down_dims=(32, 64, 128),
        diffusion_step_embed_dim=32, num_train_timesteps=10, num_inference_steps=2,
        share_rgb_model=True, use_group_norm=False,
    )
    config.input_features = {
        k: v for k, v in features.items() if v.type != FeatureType.ACTION
    }
    config.output_features = {
        k: v for k, v in features.items() if v.type == FeatureType.ACTION
    }
    config.validate_features()
    delta = {
        key: [t / meta.fps for t in (
            config.action_delta_indices if key == "action"
            else config.observation_delta_indices
        )]
        for key in ("action", "observation.state", *CAMERAS)
    }
    dataset = LeRobotDataset(
        repo_id=root.name, root=root, delta_timestamps=delta,
        video_backend=args.video_backend, return_uint8=True,
    )
    batch = default_collate([dataset[0]])
    assert tuple(batch["action"].shape[1:]) == (
        config.horizon, config.action_feature.shape[0],
    )
    assert batch["observation.state"].shape[1] == config.n_obs_steps
    for key in CAMERAS:
        if batch[key].shape[1] != config.n_obs_steps:
            raise AssertionError(f"LeRobot observation window mismatch: {key}")
    batch = {key: value.to(args.device) if torch.is_tensor(value) else value
             for key, value in batch.items()}
    policy = RL1002DPolicy(config, dataset_stats=meta.stats).to(args.device)
    loss, diagnostics = policy(batch)
    if not torch.isfinite(loss):
        raise AssertionError("Native RL1002D LeRobot loss is not finite")
    loss.backward()
    if not any(p.grad is not None for p in policy.native.model.parameters()):
        raise AssertionError("Missing native RL1002D backward gradient")
    with tempfile.TemporaryDirectory(prefix="rl100_letools_bc_") as temp:
        policy.save_pretrained(temp)
        restored = RL1002DPolicy.from_pretrained(temp, config=None)
        if restored.config.native_stats != policy.config.native_stats:
            raise AssertionError("LeRobot checkpoint lost native normalization stats")
        for key, value in policy.state_dict().items():
            if not torch.equal(value.cpu(), restored.state_dict()[key].cpu()):
                raise AssertionError(f"LeRobot policy save/load mismatch: {key}")
    print(f"[PASS] Direct LeRobot frames={len(dataset)} episodes={dataset.num_episodes}")
    print(f"[PASS] LeRobot delta timestamps: obs={config.n_obs_steps}, action={config.horizon}")
    print(f"[PASS] Native RL1002D real-data BC loss/backward={loss.item():.6f}")
    print("[PASS] LeRobot save_pretrained/from_pretrained and native normalizer stats")
    print("[PASS] No Zarr or OfflineBuffer used in BC")


if __name__ == "__main__":
    main()
