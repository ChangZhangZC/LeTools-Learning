"""Construct the unmodified native RL1002D diffusion policy.

The interface adapts LeRobot metadata only; it does not replace RL-100.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RL100_ROOT = ROOT / "third_party" / "rl100" / "RL-100"
if str(RL100_ROOT) not in sys.path:
    sys.path.insert(0, str(RL100_ROOT))

from rl_100.model.vision.model_getter import get_resnet  # noqa: E402
from rl_100.model.vision.multi_image_obs_encoder import MultiImageObsEncoder  # noqa: E402
from rl_100.policy.rl100_2d import RL1002D  # noqa: E402
from rl_100.unidpg.diffusion_policy.diffusers_patch.ddim_with_logprob_dpok import (  # noqa: E402
    DDIMSchedulerExtended,
)


def build_policy(cfg, contract) -> RL1002D:
    """Original RL-100 components with dimension-only metadata adaptation."""
    p = cfg.policy
    encoder = MultiImageObsEncoder(
        shape_meta=contract.shape_meta,
        rgb_model=get_resnet("resnet18", weights=None),
        resize_shape=(int(p.resize), int(p.resize)),
        crop_shape=None,
        random_crop=False,
        use_group_norm=bool(p.use_group_norm),
        share_rgb_model=bool(p.share_rgb_model),
        imagenet_norm=True,
        use_agent_pos=True,
        use_vib=False,
        use_recon=False,
    )
    scheduler = DDIMSchedulerExtended(
        num_train_timesteps=int(p.num_train_timesteps),
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
        clip_std_min=0.01,
    )
    return RL1002D(
        shape_meta=contract.shape_meta,
        obs_encoder=encoder,
        cm_noise_scheduler=None,
        ddim_noise_scheduler=scheduler,
        scheduler_type="ddim",
        model="sample_dp3",
        feature_type="2D",
        w_pc=False,
        use_agent_pos=True,
        obs_as_global_cond=True,
        action_norm=True,
        horizon=int(cfg.data.horizon),
        n_obs_steps=int(cfg.data.n_obs_steps),
        n_action_steps=int(cfg.data.n_action_steps),
        num_inference_steps=int(p.num_inference_steps),
        ddim_inference_steps=int(p.num_inference_steps),
        down_dims=tuple(int(dim) for dim in p.down_dims),
        diffusion_step_embed_dim=int(p.diffusion_step_embed_dim),
        kernel_size=int(p.kernel_size),
        n_groups=int(p.n_groups),
        use_aug=False,
    )

