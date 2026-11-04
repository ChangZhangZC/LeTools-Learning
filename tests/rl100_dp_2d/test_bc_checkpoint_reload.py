"""Feature 03: independently rebuild an RL1002D checkpoint and infer real data.

Checks more than torch.load(): native actor construction from stored
shape_meta, strict model state restoration (including native normalizer),
and finite action-chunk inference on a LeRobot v3 observation.
"""
from __future__ import annotations

import argparse
import sys
from collections import UserDict
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "rl100" / "RL-100"))

from kuavo_model.rl100_dp_2d.bc_utils import _to_device  # noqa: E402
from kuavo_model.rl100_dp_2d.dataset import open_lerobot_bc_dataset  # noqa: E402
from kuavo_model.rl100_dp_2d.policy_factory import build_policy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = OmegaConf.create(payload["config"])
    saved_meta = payload["shape_meta"]
    if not saved_meta["obs"] or not saved_meta["action"]["shape"]:
        raise AssertionError("Checkpoint has no usable shape contract")

    # Native RL1002D recursively processes ordinary dict values in shape_meta;
    # the same UserDict compatibility wrapper as Feature 02 is required here.
    shape_meta = {
        "obs": {key: UserDict(spec) for key, spec in saved_meta["obs"].items()},
        "action": dict(saved_meta["action"]),
    }
    model = build_policy(cfg, SimpleNamespace(shape_meta=shape_meta))
    result = model.load_state_dict(payload["model"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"Not a strict reload: {result}")
    model.to(args.device).eval()

    # RL-100's DictOfTensorMixin reconstructs normalizer ParameterDict
    # dynamically; independently compare *every* checkpoint tensor.
    actual = model.state_dict()
    expected = payload["model"]
    if set(actual) != set(expected):
        raise AssertionError(f"State keys differ: missing={set(expected)-set(actual)}, "
                             f"unexpected={set(actual)-set(expected)}")
    for key, value in expected.items():
        if not torch.equal(actual[key].cpu(), value.cpu()):
            raise AssertionError(f"Checkpoint tensor mismatch: {key}")
    print(f"[PASS] strict RL1002D state reload: {len(expected)} tensors")

    state_dim = shape_meta["obs"]["agent_pos"]["shape"][0]
    action_dim = shape_meta["action"]["shape"][0]
    action_norm_key = "normalizer.params_dict.action.scale"
    state_norm_key = "normalizer.params_dict.agent_pos.scale"
    for key, dim in ((action_norm_key, action_dim), (state_norm_key, state_dim)):
        if key not in actual or actual[key].numel() != dim:
            raise AssertionError(f"Native normalizer missing or wrong shape: {key}")
    print(f"[PASS] restored native state/action normalizer: {state_dim}D / {action_dim}D")

    # Validate original metadata is sufficient to recover real observation
    # mapping, without inventing a robot- or end-effector-specific DOF table.
    dataset = open_lerobot_bc_dataset(
        str(cfg.data.repo_id), str(cfg.data.root),
        n_obs_steps=int(cfg.data.n_obs_steps),
        horizon=int(cfg.data.horizon),
        video_backend=str(cfg.data.video_backend),
    )
    if dict(dataset.contract.source_to_rgb) != payload["camera_map"]:
        raise AssertionError("Dataset cameras differ from the checkpoint contract")
    if dataset.contract.state_dim != state_dim or dataset.contract.action_dim != action_dim:
        raise AssertionError("Dataset dimensions differ from the checkpoint contract")
    if list(dataset.contract.state_names) != payload["state_names"]:
        raise AssertionError("Dataset state joint order differs from checkpoint")
    if list(dataset.contract.action_names) != payload["action_names"]:
        raise AssertionError("Dataset action joint order differs from checkpoint")

    sample = dataset[0]
    # Dataset now keeps RGB uint8 through DataLoader; use the same native
    # boundary adapter as training to supply float32 RGB in [0, 1].
    prepared = _to_device({
        "obs": {key: tensor.unsqueeze(0) for key, tensor in sample["obs"].items()},
        "action": sample["action"].unsqueeze(0),
    }, torch.device(args.device))
    with torch.inference_mode():
        out = model.predict_action(prepared["obs"], deterministic=True)
    actions = out["action"]
    expected_shape = (1, int(cfg.data.n_action_steps), action_dim)
    if tuple(actions.shape) != expected_shape or not torch.isfinite(actions).all().item():
        raise AssertionError(f"Invalid loaded policy inference: {tuple(actions.shape)}")
    print(f"[PASS] reloaded RL1002D inference: action={tuple(actions.shape)}")
    print("[PASS] Feature03 native checkpoint rebuild + inference")


if __name__ == "__main__":
    main()
