"""Native RL1002D -> standard Kuavo ModelServerAdapter.

All model conversion stays in this new adapter. The only necessary LeTools
source edit is one lazy registration in kuavo_server/builtin_adapters.py.
The policy checkpoint is the Feature03 BC training state_dict, not a
Hugging Face or LeRobot PreTrainedPolicy.
"""
from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from collections import UserDict, deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..runtime import register_adapter
from .base import ModelServerAdapter, kuavo_repo_root


def _tensor_input(value: Any):
    """Keep the converter independent of the client transport's tensor type."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(np.ascontiguousarray(np.asarray(value)))


@register_adapter
class RL1002DKuavoAdapter(ModelServerAdapter):
    """Serve an unmodified RL1002D trained by post-rl/src/rl100_compat/train_bc.py."""

    name = "rl100_2d"

    def __init__(
        self,
        *,
        checkpoint: str,
        device: str = "cuda:0",
        execution_horizon: int = 0,
        which_arm: str = "both",
        binary_action_names: str = "",
        binary_threshold: float = 0.5,
        resize_images: bool = False,
    ) -> None:
        import torch
        from omegaconf import OmegaConf

        if execution_horizon < 0:
            raise ValueError("execution_horizon must be nonnegative")
        if which_arm not in ("left", "right", "both"):
            raise ValueError(f"Unsupported which_arm {which_arm!r}")
        if not 0 <= binary_threshold <= 1:
            raise ValueError("binary_threshold must be in [0,1]")
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Missing RL100 BC checkpoint: {self.checkpoint}")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA not available: {device}")

        # Lazy-import native RL100 and the Feature03 model factory only here.
        # 'serve.py --adapter rl100_2d --help' must not instantiate the model.
        root = kuavo_repo_root()
        for path in (root / "third_party" / "rl100" / "RL-100", root / "post-rl" / "src"):
            if not path.is_dir():
                raise FileNotFoundError(f"Missing RL100 integration path: {path}")
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from rl100_compat.train_bc import build_policy

        saved = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        for required in ("config", "shape_meta", "model", "camera_map", "state_names", "action_names"):
            if required not in saved:
                raise KeyError(f"RL100 checkpoint missing {required!r}")
        self.config = OmegaConf.create(saved["config"])
        raw_shape = saved["shape_meta"]
        if not isinstance(raw_shape, dict) or not isinstance(raw_shape.get("obs"), dict):
            raise ValueError("Checkpoint shape_meta has invalid schema")
        # RL1002D's native dict_apply descends only plain dicts. Recreate the
        # same atomic-leaf UserDict shape contract as the dataset adapter.
        self.shape_meta = {
            "obs": {key: UserDict(dict(spec)) for key, spec in raw_shape["obs"].items()},
            "action": dict(raw_shape["action"]),
        }
        self.state_dim = int(self.shape_meta["obs"]["agent_pos"]["shape"][0])
        self.action_dim = int(self.shape_meta["action"]["shape"][0])
        self.n_obs_steps = int(self.config.data.n_obs_steps)
        self.model_n_action_steps = int(self.config.data.n_action_steps)
        if self.n_obs_steps < 1 or self.state_dim < 1 or self.action_dim < 1:
            raise ValueError("Invalid saved policy dimensions")
        if self.model_n_action_steps < 1:
            raise ValueError("Checkpoint n_action_steps must be positive")
        if execution_horizon > self.model_n_action_steps:
            raise ValueError("execution_horizon exceeds trained n_action_steps")
        self.execution_horizon = execution_horizon or self.model_n_action_steps
        self.which_arm = which_arm
        self.resize_images = resize_images

        self.camera_map = dict(saved["camera_map"])
        if not self.camera_map:
            raise ValueError("Checkpoint has no camera mapping")
        self.expected_images: dict[str, tuple[int, int, int]] = {}
        for source, model_key in self.camera_map.items():
            if not source.startswith("observation.images."):
                raise ValueError(f"Invalid camera source key: {source}")
            if model_key not in self.shape_meta["obs"]:
                raise ValueError(f"Camera mapping missing model key: {model_key}")
            spec = self.shape_meta["obs"][model_key]
            if spec.get("type") != "rgb" or len(spec["shape"]) != 3 or spec["shape"][0] != 3:
                raise ValueError(f"Invalid RGB shape for {model_key}")
            self.expected_images[source] = tuple(int(v) for v in spec["shape"])

        self.state_names = tuple(saved["state_names"])
        self.action_names = tuple(saved["action_names"])
        if self.state_names and len(self.state_names) != self.state_dim:
            raise ValueError("Checkpoint state_names dimension mismatch")
        if self.action_names and len(self.action_names) != self.action_dim:
            raise ValueError("Checkpoint action_names dimension mismatch")
        names = [v.strip() for v in binary_action_names.split(",") if v.strip()]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate binary_action_names")
        missing = set(names) - set(self.action_names)
        if missing:
            raise ValueError(f"Unknown binary action names: {sorted(missing)}")
        if names and len(self.action_names) != self.action_dim:
            raise ValueError("Binary action mapping requires ordered checkpoint action_names")
        self.binary_action_indices = tuple(self.action_names.index(name) for name in names)
        self.binary_action_names = tuple(names)
        self.binary_threshold = float(binary_threshold)

        self.model = build_policy(self.config, SimpleNamespace(shape_meta=self.shape_meta))
        self.model.load_state_dict(saved["model"], strict=True)
        # RL100's DictOfTensorMixin recreates ParameterDict directly on CPU;
        # move the complete restored actor to the requested device afterwards.
        self.model.to(self.device).eval()
        actual = self.model.state_dict()
        if actual.keys() != saved["model"].keys():
            raise RuntimeError("Native RL100 state_dict keys differ after strict load")
        for key, weight in saved["model"].items():
            if not torch.equal(actual[key].cpu(), weight.cpu()):
                raise RuntimeError(f"Checkpoint weight did not restore exactly: {key}")
        for key, dim in (("agent_pos", self.state_dim), ("action", self.action_dim)):
            normalizer_key = f"normalizer.params_dict.{key}.scale"
            if normalizer_key not in actual or actual[normalizer_key].numel() != dim:
                raise RuntimeError(f"Missing or malformed native RL100 normalizer: {key}")

        self._history: deque[dict[str, Any]] = deque(maxlen=self.n_obs_steps)
        self._pending_actions: list[np.ndarray] = []

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--checkpoint", required=True, help="Feature03 latest.pt path")
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--execution_horizon", type=int, default=0,
                            help="0 uses trained n_action_steps; cannot exceed it")
        parser.add_argument("--which_arm", choices=["left", "right", "both"], default="both",
                            help="Metadata label only. Does NOT slice or remap checkpoint actions.")
        parser.add_argument("--binary_action_names", default="",
                            help="Comma-separated checkpoint action names that require 0/1 thresholding")
        parser.add_argument("--binary_threshold", type=float, default=0.5)
        parser.add_argument("--resize_images", action="store_true",
                            help="Explicitly resize incoming RGB to saved training shape")

    @classmethod
    def from_args(cls, args: Namespace) -> "RL1002DKuavoAdapter":
        return cls(
            checkpoint=args.checkpoint,
            device=args.device,
            execution_horizon=args.execution_horizon,
            which_arm=args.which_arm,
            binary_action_names=args.binary_action_names,
            binary_threshold=args.binary_threshold,
            resize_images=args.resize_images,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "adapter": self.name,
            "checkpoint": str(self.checkpoint),
            "which_arm": self.which_arm,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "n_obs_steps": self.n_obs_steps,
            "execution_horizon": self.execution_horizon,
            "camera_map": dict(self.camera_map),
            "state_names": list(self.state_names),
            "action_names": list(self.action_names),
            "binary_action_names": list(self.binary_action_names),
            "resize_images": self.resize_images,
        }

    def reset(self) -> dict[str, str]:
        self._pending_actions.clear()
        self._history.clear()
        return {"status": "ok", "message": "RL100 action and observation histories cleared"}

    def _state(self, value: Any):
        import torch

        state = _tensor_input(value)
        if state.ndim == 2 and state.shape[0] == 1:
            state = state[0]
        if state.ndim != 1 or state.numel() != self.state_dim:
            raise ValueError(f"Kuavo state must be [D] or [1,D] with D={self.state_dim}; "
                             f"got {tuple(state.shape)}")
        state = state.to(dtype=torch.float32)
        if not torch.isfinite(state).all().item():
            raise ValueError("Nonfinite Kuavo state")
        return state

    def _rgb(self, value: Any, source: str):
        import torch
        import torch.nn.functional as F

        rgb = _tensor_input(value)
        if rgb.ndim == 4 and rgb.shape[0] == 1:
            rgb = rgb[0]
        if rgb.ndim != 3:
            raise ValueError(f"{source}: expected CHW or HWC RGB, got {tuple(rgb.shape)}")
        if rgb.shape[0] == 3:
            pass
        elif rgb.shape[-1] == 3:
            rgb = rgb.permute(2, 0, 1)
        else:
            raise ValueError(f"{source}: RGB must have three channels")
        if rgb.dtype == torch.uint8:
            rgb = rgb.to(torch.float32).div(255.0)
        elif rgb.is_floating_point():
            rgb = rgb.to(torch.float32)
            if (not torch.isfinite(rgb).all().item()
                    or rgb.min().item() < 0 or rgb.max().item() > 1):
                raise ValueError(f"{source}: floating RGB must be finite in [0,1]")
        else:
            raise TypeError(f"{source}: image dtype must be uint8 or float [0,1]")

        expected = self.expected_images[source]
        if tuple(rgb.shape) != expected:
            if not self.resize_images:
                raise ValueError(
                    f"{source}: image shape {tuple(rgb.shape)} differs from saved {expected}. "
                    "Align Kuavo camera config or explicitly enable --resize_images."
                )
            rgb = F.interpolate(
                rgb[None], size=expected[1:], mode="bilinear", align_corners=False
            )[0]
        return rgb.contiguous()

    def _model_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        if "observation.state" not in obs:
            raise KeyError("Missing observation.state")
        frame = {"agent_pos": self._state(obs["observation.state"])}
        for source, model_key in self.camera_map.items():
            if source not in obs:
                raise KeyError(f"Missing required RGB camera: {source}")
            frame[model_key] = self._rgb(obs[source], source)

        # Initial observation context: replicate the first actual observation
        # until enough real frames have arrived. Never cross reset() boundaries.
        if not self._history:
            for _ in range(self.n_obs_steps - 1):
                self._history.append({key: value.clone() for key, value in frame.items()})
        self._history.append(frame)
        import torch
        return {
            key: torch.stack([past[key] for past in self._history], dim=0)
                      .unsqueeze(0).to(self.device)
            for key in frame
        }

    def _predict_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        import torch

        model_obs = self._model_obs(obs)
        with torch.inference_mode():
            result = self.model.predict_action(model_obs, deterministic=True)
        actions = result["action"]
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] != self.action_dim:
            raise ValueError(f"Native RL100 action has invalid shape: {tuple(actions.shape)}")
        chunk = actions[0, :self.execution_horizon].detach().float().cpu().numpy()
        if chunk.shape != (self.execution_horizon, self.action_dim):
            raise ValueError(f"Invalid truncated chunk shape: {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise ValueError("RL100 produced nonfinite actions")
        chunk = chunk.astype(np.float64, copy=True)
        if self.binary_action_indices:
            chunk[:, self.binary_action_indices] = (
                chunk[:, self.binary_action_indices] >= self.binary_threshold
            ).astype(np.float64)
        return chunk

    def select_action(self, obs: dict[str, Any]) -> np.ndarray:
        """Step-by-step fallback for direct select_action callers."""
        if not self._pending_actions:
            self._pending_actions.extend(self._predict_chunk(obs))
        return self._pending_actions.pop(0)

    def select_action_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        """Native Kuavo PolicyClient consumes/caches this entire [T,D] chunk."""
        self._pending_actions.clear()
        return self._predict_chunk(obs)
