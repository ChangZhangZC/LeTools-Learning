"""In-process RL100 DP-2D policy for LeTools' existing native Sim Eval.

Loads inference-only Safetensors actor bundle, not the original train/resume
checkpoint. The original RL1002D actor and normalizers remain unchanged.
"""
from __future__ import annotations

import json
from collections import UserDict, deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from .export_safetensors import MANIFEST, WEIGHTS
from .policy_factory import build_policy


def resolve_rl100_eval_output_dir(pretrained_path: str | Path,
                                  output_root: str | Path = "outputs/eval") -> Path:
    """Separate results per exported checkpoint without changing legacy flat bundles."""
    source = Path(pretrained_path).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    if source.name == "pretrained_model" and source.parent.parent.name == "checkpoints":
        checkpoint = source.parent
        run = checkpoint.parent.parent
        method = run.parent
        task = method.parent
        return root / task.name / method.name / run.name / checkpoint.name
    return root / source.name


class RL100DP2DEvalPolicy:
    def __init__(self, pretrained_path: str | Path, *,
                 device: str | torch.device = "cuda",
                 n_action_steps: int | None = None,
                 resize_images: bool = False) -> None:
        from safetensors.torch import load_file

        bundle = Path(pretrained_path).expanduser().resolve()
        manifest_path, weights_path = bundle / MANIFEST, bundle / WEIGHTS
        if not manifest_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(f"Expected {MANIFEST} and {WEIGHTS} inside {bundle}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("format") != "letools_rl100_dp_2d"
                or manifest.get("format_version") != 1):
            raise ValueError("Unsupported RL100 DP-2D export manifest version")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable: {self.device}")

        saved_config = OmegaConf.create(manifest["config"])
        raw_shape = manifest["shape_meta"]
        shape_meta = {
            "obs": {key: UserDict(dict(spec))
                    for key, spec in raw_shape["obs"].items()},
            "action": dict(raw_shape["action"]),
        }
        self.state_dim = int(shape_meta["obs"]["agent_pos"]["shape"][0])
        self.action_dim = int(shape_meta["action"]["shape"][0])
        self.n_obs_steps = int(saved_config.data.n_obs_steps)
        self.trained_action_steps = int(saved_config.data.n_action_steps)
        # None means conservative single-step eval; YAML can explicitly use 16.
        self.n_action_steps = 1 if n_action_steps is None else n_action_steps
        if (not isinstance(self.n_action_steps, int)
                or isinstance(self.n_action_steps, bool)
                or not 1 <= self.n_action_steps <= self.trained_action_steps):
            raise ValueError(f"Eval n_action_steps must be in [1, {self.trained_action_steps}]")
        if self.n_obs_steps < 1:
            raise ValueError("Invalid n_obs_steps in exported checkpoint")

        self.camera_map = dict(manifest["camera_map"])
        if not self.camera_map:
            raise ValueError("Export has no RGB cameras")
        self.expected_images = {}
        for source, key in self.camera_map.items():
            if not source.startswith("observation.images.") or key not in shape_meta["obs"]:
                raise ValueError(f"Invalid camera mapping {source!r} -> {key!r}")
            spec = shape_meta["obs"][key]
            expected = tuple(int(v) for v in spec["shape"])
            if len(expected) != 3 or expected[0] != 3 or spec.get("type") != "rgb":
                raise ValueError(f"Invalid RGB shape for {key}: {expected}")
            self.expected_images[source] = expected
        self.resize_images = bool(resize_images)
        self.state_names = list(manifest["state_names"])
        self.action_names = list(manifest["action_names"])
        if (self.state_names and len(self.state_names) != self.state_dim
                or self.action_names and len(self.action_names) != self.action_dim):
            raise ValueError("Exported names do not match state/action dimensions")

        self.model = build_policy(saved_config, SimpleNamespace(shape_meta=shape_meta))
        actor = load_file(str(weights_path), device="cpu")
        self.model.load_state_dict(actor, strict=True)
        self.model.to(self.device).eval()
        # The native normalizer must be restored from the actor, not refit
        # on the Eval data (which would change robot action values).
        for key, dim in (("agent_pos", self.state_dim), ("action", self.action_dim)):
            name = f"normalizer.params_dict.{key}.scale"
            if name not in actor or actor[name].numel() != dim:
                raise ValueError(f"Exported native normalizer missing/invalid: {name}")

        # Eval only accesses these LeRobot-like informational attributes.
        self.config = SimpleNamespace(
            type="rl100_dp_2d",
            n_obs_steps=self.n_obs_steps,
            input_features={source: self.expected_images[source]
                            for source in self.camera_map}
                           | {"observation.state": (self.state_dim,)},
            output_features={"action": (self.action_dim,)},
        )
        self._history: deque[dict[str, torch.Tensor]] = deque(maxlen=self.n_obs_steps)
        self._pending: deque[torch.Tensor] = deque()

    def reset(self) -> None:
        self._pending.clear()
        self._history.clear()

    def _state(self, value: Any) -> torch.Tensor:
        state = torch.as_tensor(value)
        if state.ndim == 2 and state.shape[0] == 1:
            state = state[0]
        if state.ndim != 1 or state.numel() != self.state_dim:
            raise ValueError(f"State expected [D] or [1,D], D={self.state_dim}; got {tuple(state.shape)}")
        state = state.to(dtype=torch.float32)
        if not torch.isfinite(state).all().item():
            raise ValueError("Nonfinite observation.state")
        return state

    def _rgb(self, value: Any, source: str) -> torch.Tensor:
        rgb = torch.as_tensor(value)
        if rgb.ndim == 4 and rgb.shape[0] == 1:
            rgb = rgb[0]
        if rgb.ndim != 3:
            raise ValueError(f"{source}: expected [C,H,W] or [H,W,C]")
        if rgb.shape[0] != 3:
            if rgb.shape[-1] != 3:
                raise ValueError(f"{source}: expected 3 RGB channels")
            rgb = rgb.permute(2, 0, 1)
        if rgb.dtype == torch.uint8:
            rgb = rgb.float().div(255.0)
        elif rgb.is_floating_point():
            rgb = rgb.float()
            if not torch.isfinite(rgb).all().item() or rgb.min() < 0 or rgb.max() > 1:
                raise ValueError(f"{source}: RGB floats must be finite in [0,1]")
        else:
            raise TypeError(f"{source}: RGB dtype must be uint8 or floating [0,1]")
        expected = self.expected_images[source]
        if tuple(rgb.shape) != expected:
            if not self.resize_images:
                raise ValueError(f"{source}: got {tuple(rgb.shape)}, trained {expected}; "
                                 "align env.image_size or set inference.rl100_resize_images")
            rgb = F.interpolate(rgb.unsqueeze(0), size=expected[1:],
                                mode="bilinear", align_corners=False)[0]
        return rgb.contiguous()

    def _update_history(self, observation: dict[str, Any]) -> None:
        if "observation.state" not in observation:
            raise KeyError("Missing observation.state")
        frame = {"agent_pos": self._state(observation["observation.state"])}
        for source, key in self.camera_map.items():
            if source not in observation:
                raise KeyError(f"Missing RGB camera: {source}")
            frame[key] = self._rgb(observation[source], source)
        if not self._history:
            for _ in range(self.n_obs_steps - 1):
                self._history.append({k: v.clone() for k, v in frame.items()})
        self._history.append(frame)

    def select_action(self, observation: dict[str, Any]) -> torch.Tensor:
        # This call happens every sim env.step, EVEN while cached actions are
        # being executed. Preserve consecutive frame history for n_obs_steps>1.
        self._update_history(observation)
        if self._pending:
            return self._pending.popleft()

        model_obs = {
            key: torch.stack([frame[key] for frame in self._history], 0)
                      .unsqueeze(0).to(self.device)
            for key in self._history[-1]
        }
        with torch.inference_mode():
            prediction = self.model.predict_action(model_obs, deterministic=True)["action"]
        if prediction.ndim != 3 or prediction.shape[0] != 1 or prediction.shape[2] != self.action_dim:
            raise ValueError(f"RL100 prediction must be [1,T,{self.action_dim}]")
        if prediction.shape[1] < self.n_action_steps:
            raise ValueError("RL100 predicted fewer actions than Eval n_action_steps")
        chunk = prediction[0, :self.n_action_steps].detach().float().cpu()
        if not torch.isfinite(chunk).all().item():
            raise ValueError("RL100 predicted nonfinite actions")
        self._pending.extend(chunk[i:i + 1] for i in range(self.n_action_steps))
        return self._pending.popleft()
