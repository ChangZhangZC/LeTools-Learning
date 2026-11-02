"""Configuration for an out-of-tree wrapper around the native RL1002D.

LeRobot owns dataset windowing/checkpoint lifecycle. RL-100 remains the actor
and keeps its own state/action normalizer (LeRobot processors use IDENTITY).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("rl100_2d")
@dataclass
class RL1002DConfig(PreTrainedConfig):
    n_obs_steps: int = 2
    horizon: int = 9
    n_action_steps: int = 4
    # Keep native RL-100 scale: data statistics are fitted into its internal
    # LinearNormalizer, not normalized by LeRobot a second time.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )
    resize: int = 224
    down_dims: tuple[int, ...] = (256, 512, 1024)
    diffusion_step_embed_dim: int = 256
    num_train_timesteps: int = 100
    num_inference_steps: int = 10
    share_rgb_model: bool = False
    use_group_norm: bool = True
    resnet_weights: str | None = None
    use_aug: bool = False
    use_recon: bool = False
    use_vib: bool = False
    action_norm: bool = True
    kernel_size: int = 5
    n_groups: int = 8
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"
    clip_std_min: float = 0.0067
    # Saved with policy config to reconstruct native normalizers at inference
    # without depending on a training dataset being mounted.
    native_stats: dict = field(default_factory=dict)

    optimizer_lr: float = 1.0e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1.0e-8
    optimizer_weight_decay: float = 1.0e-6
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()
        if not 1 <= self.n_obs_steps <= self.horizon:
            raise ValueError("require 1 <= n_obs_steps <= horizon")
        if not 1 <= self.n_action_steps <= self.horizon - self.n_obs_steps + 1:
            raise ValueError("action chunk exceeds horizon")
        if self.resize <= 0 or not self.down_dims:
            raise ValueError("resize and down_dims must be positive")

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self):
        return AdamWConfig(
            lr=self.optimizer_lr, betas=tuple(self.optimizer_betas),
            eps=self.optimizer_eps, weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=0.0,
        )

    def get_scheduler_preset(self):
        return DiffuserSchedulerConfig(
            name="cosine", num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self):
        if not self.robot_state_feature or not self.action_feature:
            raise ValueError("RL1002D requires observation.state and action")
        for camera in (
            "observation.images.head_cam_h",
            "observation.images.wrist_cam_l",
            "observation.images.wrist_cam_r",
        ):
            if camera not in self.image_features:
                raise ValueError(f"RL1002D requires RGB camera {camera}")
            shape = self.image_features[camera].shape
            if len(shape) != 3 or shape[0] != 3:
                raise ValueError(f"{camera} must have CHW RGB shape, got {shape}")
