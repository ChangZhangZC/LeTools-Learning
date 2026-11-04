"""LeRobot v3 -> native RL100 DP-2D BC input; no robot-specific DOF tables.

LeTools' CvtRosbag2Lerobot owns robot/EEF selection and its state/action
ordering. This module only reads the resulting dataset features and returns
the exact keys/shapes expected by RL1002D.compute_loss().
"""
from __future__ import annotations

import re
import sys
from collections import UserDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

STATE_KEY = "observation.state"
ACTION_KEY = "action"
RGB_PREFIX = "observation.images."


@dataclass(frozen=True)
class RL100LeRobotContract:
    shape_meta: dict[str, Any]
    source_to_rgb: dict[str, str]
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]

    @property
    def state_dim(self) -> int:
        return self.shape_meta["obs"]["agent_pos"]["shape"][0]

    @property
    def action_dim(self) -> int:
        return self.shape_meta["action"]["shape"][0]


def _vector_dim(features: Mapping[str, Any], key: str) -> int:
    if key not in features:
        raise KeyError(f"LeRobot metadata has no {key!r}")
    feature = features[key]
    shape = feature.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 1:
        raise ValueError(f"{key} must have vector shape [D], got {shape!r}")
    dim = shape[0]
    if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
        raise ValueError(f"{key} has invalid dimension {dim!r}")
    return dim


def _feature_names(spec: Mapping[str, Any], key: str, dim: int) -> tuple[str, ...]:
    names = spec.get("names")
    if isinstance(names, Mapping):
        names = names.get("state_names" if key == STATE_KEY else "action_names")
    if names is None:
        return ()
    if not isinstance(names, (tuple, list)) or len(names) != dim:
        raise ValueError(f"{key} metadata joint names do not match dimension {dim}")
    return tuple(str(name) for name in names)


def contract_from_features(features: Mapping[str, Mapping[str, Any]]) -> RL100LeRobotContract:
    """Derive DP shapes *only* from LeRobot info.json features, not EEF labels.

    UserDict on each obs spec is a narrow compatibility workaround: the
    unmodified RL1002D uses a recursively descending dict_apply() that
    otherwise traverses plain nested shape metadata too deeply.
    """
    state_dim = _vector_dim(features, STATE_KEY)
    action_dim = _vector_dim(features, ACTION_KEY)
    source_to_rgb: dict[str, str] = {}
    obs = {"agent_pos": UserDict({"shape": [state_dim], "type": "low_dim"})}

    for source_key, feature in sorted(features.items()):
        if not source_key.startswith(RGB_PREFIX):
            continue
        if feature.get("dtype") not in ("image", "video"):
            continue
        shape = feature.get("shape")
        if (
            not isinstance(shape, (tuple, list))
            or len(shape) != 3
            or shape[0] != 3
            or any(not isinstance(x, int) or x <= 0 for x in shape)
        ):
            raise ValueError(f"{source_key} must be 3-channel CHW RGB, got {shape!r}")
        model_key = "rgb_" + re.sub(r"[^A-Za-z0-9_]", "_", source_key[len(RGB_PREFIX):])
        if model_key in obs:
            raise ValueError(f"Camera name collision in RL1002D mapping: {source_key!r}")
        source_to_rgb[source_key] = model_key
        obs[model_key] = UserDict({"shape": list(shape), "type": "rgb"})

    if not source_to_rgb:
        raise ValueError("No observation.images.* RGB image/video features in LeRobot metadata")
    return RL100LeRobotContract(
        shape_meta={"obs": obs, "action": {"shape": [action_dim]}},
        source_to_rgb=source_to_rgb,
        state_names=_feature_names(features[STATE_KEY], STATE_KEY, state_dim),
        action_names=_feature_names(features[ACTION_KEY], ACTION_KEY, action_dim),
    )


def build_delta_timestamps(
    fps: int,
    n_obs_steps: int,
    horizon: int,
    camera_keys: tuple[str, ...] | list[str],
) -> dict[str, list[float]]:
    """Align a LeRobot sample at time t with native DP horizons.

    Obs:    t-(To-1), ..., t
    Action: t-(To-1), ..., t+H-To
    RL1002D(no_pre_action=True) removes the first To-1 action entries.
    LeRobot handles episode boundaries and returns <key>_is_pad flags.
    """
    if not isinstance(fps, int) or fps <= 0:
        raise ValueError("dataset fps must be a positive integer")
    if not isinstance(n_obs_steps, int) or n_obs_steps <= 0:
        raise ValueError("n_obs_steps must be positive")
    if not isinstance(horizon, int) or horizon < n_obs_steps:
        raise ValueError("horizon must be >= n_obs_steps")
    obs_offsets = [step / fps for step in range(1 - n_obs_steps, 1)]
    action_offsets = [step / fps for step in range(1 - n_obs_steps, 1 - n_obs_steps + horizon)]
    return {
        **{key: list(obs_offsets) for key in (STATE_KEY, *camera_keys)},
        ACTION_KEY: action_offsets,
    }


def _float_vector(value: Any, expected: tuple[int, int], key: str) -> torch.Tensor:
    value = torch.as_tensor(value)
    if tuple(value.shape) != expected:
        raise ValueError(f"{key}: expected {expected}, got {tuple(value.shape)}")
    result = value.to(torch.float32)
    if not torch.isfinite(result).all().item():
        raise ValueError(f"{key} has non-finite values")
    return result


def _rgb_tensor(value: Any, expected: tuple[int, int, int, int], key: str) -> torch.Tensor:
    value = torch.as_tensor(value)
    if tuple(value.shape) != expected:
        raise ValueError(f"{key}: expected {expected}, got {tuple(value.shape)}")
    if value.dtype == torch.uint8:
        return value.to(torch.float32) / 255.0
    if not value.is_floating_point():
        raise TypeError(f"{key}: image must be uint8 or floating point")
    value = value.to(torch.float32)
    if not torch.isfinite(value).all().item() or value.min().item() < 0 or value.max().item() > 1:
        raise ValueError(f"{key}: floating RGB expected in [0, 1]")
    return value


class RL100LeRobotDataset(Dataset):
    """Thin view of a LeRobotDataset already configured with delta_timestamps."""

    def __init__(self, dataset: Any, contract: RL100LeRobotContract, n_obs_steps: int, horizon: int):
        self.dataset = dataset
        self.contract = contract
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        frame = self.dataset[idx]
        obs: dict[str, torch.Tensor] = {
            "agent_pos": _float_vector(
                frame[STATE_KEY], (self.n_obs_steps, self.contract.state_dim), STATE_KEY
            )
        }
        for source_key, target_key in self.contract.source_to_rgb.items():
            c, h, w = self.contract.shape_meta["obs"][target_key]["shape"]
            obs[target_key] = _rgb_tensor(
                frame[source_key], (self.n_obs_steps, c, h, w), source_key
            )
        action = _float_vector(
            frame[ACTION_KEY], (self.horizon, self.contract.action_dim), ACTION_KEY
        )
        pad_key = f"{ACTION_KEY}_is_pad"
        if pad_key not in frame:
            raise KeyError("LeRobot must supply action_is_pad from delta_timestamps")
        action_is_pad = torch.as_tensor(frame[pad_key], dtype=torch.bool)
        if tuple(action_is_pad.shape) != (self.horizon,):
            raise ValueError(f"{pad_key}: expected {(self.horizon,)}, got {tuple(action_is_pad.shape)}")

        # Extra pad flag is intentionally kept OUTSIDE the native RL1002D
        # batch['obs'] / batch['action'] contract. Feature 03 must explicitly
        # choose a boundary masking/filtering policy rather than silently train
        # on action targets replicated at episode ends.
        return {"obs": obs, "action": action, "action_is_pad": action_is_pad}


def open_lerobot_bc_dataset(
    repo_id: str,
    root: str | Path,
    *,
    n_obs_steps: int,
    horizon: int,
    episodes: list[int] | None = None,
    video_backend: str | None = None,
) -> RL100LeRobotDataset:
    """Open the native local LeRobot v3 dataset without Zarr or data copies."""
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root).expanduser().resolve()
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a local LeRobot dataset: {root}/meta/info.json")
    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    contract = contract_from_features(meta.features)
    delta_timestamps = build_delta_timestamps(
        meta.fps, n_obs_steps, horizon, list(contract.source_to_rgb)
    )
    dataset = LeRobotDataset(
        repo_id=repo_id, root=root, episodes=episodes,
        delta_timestamps=delta_timestamps, video_backend=video_backend,
    )
    return RL100LeRobotDataset(dataset, contract, n_obs_steps, horizon)
