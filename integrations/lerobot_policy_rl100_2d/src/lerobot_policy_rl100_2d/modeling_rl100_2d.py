"""Adapter only: pinned RL1002D stays the trainable Diffusion Policy actor.

This module never imports OfflineBuffer, OfflineDataset or Zarr. The LeRobot
trainer handles frame-aligned windows via observation/action_delta_indices.
"""
from __future__ import annotations

from collections import UserDict, deque

import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from rl_100.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from rl_100.model.vision.model_getter import get_resnet
from rl_100.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from rl_100.policy.rl100_2d import RL1002D
from rl_100.unidpg.diffusion_policy.diffusers_patch.ddim_with_logprob_dpok import (
    DDIMSchedulerExtended,
)

from .configuration_rl100_2d import RL1002DConfig


CAMERAS = {
    "observation.images.head_cam_h": "rgb_head",
    "observation.images.wrist_cam_l": "rgb_left_hand",
    "observation.images.wrist_cam_r": "rgb_right_hand",
}
STATE_KEY = "observation.state"
ACTION_KEY = "action"


def _field_stats(dataset_stats, key, dim):
    if not dataset_stats or key not in dataset_stats:
        raise ValueError(f"Missing LeRobot dataset statistics for {key}")
    result = {}
    for name in ("min", "max", "mean", "std"):
        if name not in dataset_stats[key]:
            raise ValueError(f"LeRobot stats {key} missing {name}")
        value = torch.as_tensor(dataset_stats[key][name], dtype=torch.float32).flatten()
        if value.numel() != dim or not torch.isfinite(value).all():
            raise ValueError(f"Invalid LeRobot stats {key}.{name}: {tuple(value.shape)}")
        result[name] = value.tolist()
    return result


def _manual_rl100_normalizer(stats):
    normalizer = LinearNormalizer()
    for native_key, summary in stats.items():
        tensors = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in summary.items()}
        # Match native RL-100 _fit(..., mode='limits', output_min=-1, output_max=1):
        low, high = tensors["min"], tensors["max"]
        spread = high - low
        constant = spread < 1e-4
        spread = torch.where(constant, torch.full_like(spread, 2.0), spread)
        scale = 2.0 / spread
        offset = -1.0 - scale * low
        offset = torch.where(constant, -low, offset)
        normalizer[native_key] = SingleFieldLinearNormalizer.create_manual(
            scale=scale, offset=offset, input_stats_dict=tensors,
        )
    return normalizer


def _to_rgb(x: torch.Tensor, key: str, *, with_time: bool) -> torch.Tensor:
    if with_time and x.ndim != 5:
        raise ValueError(f"{key} must be [B,T,3,H,W], got {tuple(x.shape)}")
    if not with_time and x.ndim != 4:
        raise ValueError(f"{key} must be [B,3,H,W], got {tuple(x.shape)}")
    if x.shape[2 if with_time else 1] != 3:
        raise ValueError(f"{key} must have 3 RGB channels")
    if x.dtype == torch.uint8:
        return x.to(dtype=torch.float32).div(255.0)
    if not x.is_floating_point():
        raise TypeError(f"{key} expected uint8 or float RGB")
    # LeRobotDataset(return_uint8=True) should be float [0,1] only if
    # an upstream processor already converted it; no ImageNet norm here.
    if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError(f"{key} RGB range must be [0,1] (prevent double normalization)")
    return x.float()


class RL1002DPolicy(PreTrainedPolicy):
    """LeRobot interface only; all actor computation uses native RL1002D."""

    config_class = RL1002DConfig
    name = "rl100_2d"

    def __init__(self, config: RL1002DConfig, *, dataset_stats=None, dataset_meta=None, **kwargs):
        super().__init__(config)
        config.validate_features()
        state_dim = int(config.robot_state_feature.shape[0])
        action_dim = int(config.action_feature.shape[0])
        if not config.native_stats:
            config.native_stats = {
                "agent_pos": _field_stats(dataset_stats, STATE_KEY, state_dim),
                "action": _field_stats(dataset_stats, ACTION_KEY, action_dim),
            }
        else:
            for key, dim in (("agent_pos", state_dim), ("action", action_dim)):
                if key not in config.native_stats or len(config.native_stats[key]["min"]) != dim:
                    raise ValueError(f"Saved native normalizer does not match {key} dimension")

        shape_meta = {
            "obs": {
                native: UserDict({"shape": list(config.input_features[feature].shape), "type": "rgb"})
                for feature, native in CAMERAS.items()
            },
            "action": {"shape": [action_dim]},
        }
        shape_meta["obs"]["agent_pos"] = UserDict({"shape": [state_dim], "type": "low_dim"})
        encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model=get_resnet("resnet18", weights=config.resnet_weights),
            resize_shape=(config.resize, config.resize),
            crop_shape=None,
            random_crop=False,
            use_group_norm=config.use_group_norm,
            share_rgb_model=config.share_rgb_model,
            imagenet_norm=True,
            use_agent_pos=True,
            use_vib=config.use_vib,
            use_recon=config.use_recon,
        )
        noise_scheduler = DDIMSchedulerExtended(
            num_train_timesteps=config.num_train_timesteps,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type,
            clip_sample=True,
            clip_std_min=config.clip_std_min,
        )
        self.native = RL1002D(
            shape_meta=shape_meta,
            obs_encoder=encoder,
            cm_noise_scheduler=None,
            ddim_noise_scheduler=noise_scheduler,
            scheduler_type="ddim",
            model="sample_dp3",
            feature_type="2D",
            w_pc=False,
            use_agent_pos=True,
            obs_as_global_cond=True,
            action_norm=config.action_norm,
            horizon=config.horizon,
            n_obs_steps=config.n_obs_steps,
            n_action_steps=config.n_action_steps,
            num_inference_steps=config.num_inference_steps,
            ddim_inference_steps=config.num_inference_steps,
            down_dims=tuple(config.down_dims),
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
            use_aug=config.use_aug,
            use_recon=config.use_recon,
        )
        self.native.set_normalizer(_manual_rl100_normalizer(config.native_stats))
        # Original RL-100 constructs the optimizer before setting normalizer.
        # Freeze imported dataset statistics: do not let the LeRobot optimizer
        # accidentally train the normalization scale/offset parameters.
        self.native.normalizer.requires_grad_(False)
        self.reset()

    def get_optim_params(self):
        return self.native.parameters()

    def reset(self):
        self._frames = deque(maxlen=self.config.n_obs_steps)
        self._actions = deque()

    def _to_native_batch(self, batch, *, with_time):
        if STATE_KEY not in batch:
            raise KeyError(f"Missing LeRobot {STATE_KEY}")
        state = batch[STATE_KEY]
        if with_time:
            if state.ndim != 3 or state.shape[1] != self.config.n_obs_steps:
                raise ValueError(f"{STATE_KEY} expected [B,{self.config.n_obs_steps},D]")
        elif state.ndim != 2:
            raise ValueError(f"{STATE_KEY} expected [B,D]")
        obs = {"agent_pos": state.float()}
        for feature, name in CAMERAS.items():
            if feature not in batch:
                raise KeyError(f"Missing LeRobot camera {feature}")
            obs[name] = _to_rgb(batch[feature], feature, with_time=with_time)
        return obs

    def forward(self, batch):
        obs = self._to_native_batch(batch, with_time=True)
        if ACTION_KEY not in batch:
            raise KeyError("Missing LeRobot action chunk")
        action = batch[ACTION_KEY]
        if action.ndim != 3 or action.shape[1] != self.config.horizon:
            raise ValueError(f"Action must be [B,{self.config.horizon},D]")
        loss, metrics = self.native.compute_loss({"obs": obs, "action": action.float()})
        return loss, metrics

    def _stack_history(self):
        if not self._frames:
            raise RuntimeError("Call select_action at least once before inference")
        frames = list(self._frames)
        frames = [frames[0]] * (self.config.n_obs_steps - len(frames)) + frames
        return {key: torch.stack([frame[key] for frame in frames], dim=1)
                for key in frames[0]}

    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        if batch[STATE_KEY].ndim == 3:
            obs = self._to_native_batch(batch, with_time=True)
        else:
            obs = self._stack_history()
        return self.native.predict_action(obs, deterministic=kwargs.get("deterministic", False))["action"]

    @torch.no_grad()
    def select_action(self, batch, **kwargs):
        frame = self._to_native_batch(batch, with_time=False)
        if frame["agent_pos"].shape[0] != 1:
            raise ValueError("select_action currently supports one robot per policy instance")
        self._frames.append(frame)
        if not self._actions:
            chunk = self.predict_action_chunk(batch, **kwargs)
            self._actions.extend(chunk.unbind(dim=1))
        return self._actions.popleft()
