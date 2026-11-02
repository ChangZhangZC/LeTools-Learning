"""Convert ARBiM offline sequence samples to the native RL1002D BC batch.

Input:  samples produced by kdc_ARBiM's OfflineDataset, or their collated batch.
Output: {'obs': {rgb_head, rgb_left_hand, rgb_right_hand, agent_pos},
         'action': tensor[B, horizon, action_dim]}.

This module never estimates reward, pads episodes, normalizes joints/actions,
resizes images, or silently changes action horizon alignment.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch

SOURCE_TO_RL100 = {
    "head_rgb": "rgb_head",
    "wrist_left_rgb": "rgb_left_hand",
    "wrist_right_rgb": "rgb_right_hand",
}


def _tensor(value: Any, name: str) -> torch.Tensor:
    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Invalid {name}: not convertible to tensor") from exc


def _batched(value: Any, name: str, single_ndim: int) -> torch.Tensor:
    tensor = _tensor(value, name)
    if tensor.ndim == single_ndim:
        return tensor.unsqueeze(0)
    if tensor.ndim == single_ndim + 1:
        return tensor
    raise ValueError(
        f"{name} must have {single_ndim} (sample) or {single_ndim + 1} "
        f"(batch) dimensions, got shape {tuple(tensor.shape)}"
    )


def _rgb_to_float(image: torch.Tensor, name: str) -> torch.Tensor:
    if image.shape[2] != 3:
        raise ValueError(f"{name} must be RGB CHW, got shape {tuple(image.shape)}")
    if image.dtype == torch.uint8:
        return image.to(torch.float32).div(255.0)
    if not image.is_floating_point():
        raise TypeError(f"{name} must be uint8 [0,255] or float [0,1], got {image.dtype}")
    if image.numel() and (
        not torch.isfinite(image).all().item()
        or image.min().item() < 0
        or image.max().item() > 1
    ):
        raise ValueError(f"{name} floating RGB must be finite and within [0,1]")
    return image.to(torch.float32)


def to_rl100_bc_batch(
    sample: Mapping[str, Any],
    *,
    n_obs_steps: int,
) -> dict[str, Any]:
    """Adapt one sequence sample or a collated batch, without side effects.

    Required source:
        sample['obs']['state']: [T,Ds] / [B,T,Ds]
        sample['obs'][camera]:  [T,3,H,W] / [B,T,3,H,W]
        sample['action']:       [Horizon,Da] / [B,Horizon,Da]

    Returns only the inputs of RL1002D.compute_loss(); reward and
    next_obs stay the responsibility of the offline RL trainer.
    """
    if not isinstance(n_obs_steps, int) or n_obs_steps < 1:
        raise ValueError("n_obs_steps must be a positive integer")
    obs = sample.get("obs")
    if not isinstance(obs, Mapping):
        raise KeyError("sample['obs'] must be a mapping")
    if "state" not in obs or "action" not in sample:
        raise KeyError("Missing sample['obs']['state'] or sample['action']")

    state = _batched(obs["state"], "obs.state", 2)
    action = _batched(sample["action"], "action", 2)
    batch_size = action.shape[0]
    if (
        state.shape[0] != batch_size
        or state.shape[1] < n_obs_steps
        or action.shape[1] < n_obs_steps
        or state.shape[2] == 0
        or action.shape[2] == 0
    ):
        raise ValueError(
            "Batch/time/dim mismatch: require matching batch size, "
            f"state T>=n_obs_steps, action horizon>=n_obs_steps; "
            f"state={tuple(state.shape)}, action={tuple(action.shape)}, "
            f"n_obs_steps={n_obs_steps}"
        )
    if not state.is_floating_point() and state.dtype != torch.int64:
        raise TypeError("obs.state must be numeric")
    state = state.to(torch.float32)
    action = action.to(torch.float32)
    if not torch.isfinite(state).all().item() or not torch.isfinite(action).all().item():
        raise ValueError("State/action must be finite")

    output_obs = {"agent_pos": state[:, :n_obs_steps]}
    for source_key, target_key in SOURCE_TO_RL100.items():
        if source_key not in obs:
            raise KeyError(f"Missing camera obs.{source_key}")
        image = _batched(obs[source_key], f"obs.{source_key}", 4)
        if image.shape[0] != batch_size or image.shape[1] < n_obs_steps:
            raise ValueError(
                f"Camera batch/time mismatch: obs.{source_key}={tuple(image.shape)}"
            )
        output_obs[target_key] = _rgb_to_float(
            image[:, :n_obs_steps], f"obs.{source_key}"
        )

    return {"obs": output_obs, "action": action}
