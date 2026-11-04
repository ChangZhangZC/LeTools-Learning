"""BC data sampling, normalization, and checkpoint helpers for RL100 DP-2D.

These preserve the original tested LeTools integration behavior.
"""
from __future__ import annotations

import torch
from omegaconf import OmegaConf
from rl_100.model.common.normalizer import LinearNormalizer


def make_episode_sampler(view, n_obs_steps: int, horizon: int):
    """Native LeRobot sampling: no duplicated episode-tail target actions."""
    from lerobot.datasets.sampler import EpisodeAwareSampler

    if horizon < n_obs_steps:
        raise ValueError("horizon must be >= n_obs_steps")
    episodes = view.dataset.meta.episodes
    return EpisodeAwareSampler(
        [ep["dataset_from_index"] for ep in episodes],
        [ep["dataset_to_index"] for ep in episodes],
        episode_indices_to_use=view.dataset.episodes,
        drop_n_last_frames=horizon - n_obs_steps,
        shuffle=True,
    )


def assert_valid_action_targets(batch: dict, n_obs_steps: int) -> None:
    """Exclude padding only in the actions RL1002D will actually supervise."""
    actions, pad = batch["action"], batch["action_is_pad"]
    if actions.ndim != 3 or pad.shape != actions.shape[:2]:
        raise ValueError("action / action_is_pad shape mismatch")
    if not 1 <= n_obs_steps <= actions.shape[1]:
        raise ValueError("Invalid n_obs_steps")
    if pad[:, n_obs_steps - 1:].any().item():
        raise ValueError("Padded supervision action: check episode sampler")


def build_normalizer(dataset) -> LinearNormalizer:
    """Use RL-100 native limits normalization fitted to LeRobot extrema.

    Only min and max influence the limits mapping. Fitting two endpoint samples
    means input_stats mean/std are NOT full-dataset moments, so use native
    LeRobot stats for those diagnostics. RGB is not normalized here.
    """
    stats = dataset.meta.stats
    inputs = {}
    for source, target in (("observation.state", "agent_pos"), ("action", "action")):
        if source not in stats:
            raise KeyError(f"LeRobot stats missing {source}")
        entry = stats[source]
        dim = dataset.features[source]["shape"][0]
        bounds = []
        for name in ("min", "max"):
            if name not in entry:
                raise KeyError(f"LeRobot stats missing {source}/{name}")
            tensor = torch.as_tensor(entry[name], dtype=torch.float32).reshape(-1)
            if tensor.numel() != dim or not torch.isfinite(tensor).all().item():
                raise ValueError(f"Invalid {source}/{name} for dimension {dim}")
            bounds.append(tensor)
        if (bounds[1] < bounds[0]).any().item():
            raise ValueError(f"Invalid extrema for {source}: max < min")
        inputs[target] = torch.stack(bounds)
    normalizer = LinearNormalizer()
    normalizer.fit(inputs, mode="limits")
    return normalizer


def _to_device(sample, device):
    return {
        "obs": {key: tensor.to(device, non_blocking=True)
                for key, tensor in sample["obs"].items()},
        "action": sample["action"].to(device, non_blocking=True),
    }


def _payload(cfg, view, model, optimizer, lr_scheduler, step, ema):
    plain_shape_meta = {
        "obs": {key: dict(spec) for key, spec in view.contract.shape_meta["obs"].items()},
        "action": dict(view.contract.shape_meta["action"]),
    }
    return {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "ema_model": ema.averaged_model.state_dict() if ema is not None else None,
        "ema_step": ema.optimization_step if ema is not None else None,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "shape_meta": plain_shape_meta,
        "camera_map": dict(view.contract.source_to_rgb),
        "state_names": list(view.contract.state_names),
        "action_names": list(view.contract.action_names),
        "dataset_fps": int(view.dataset.fps),
    }


def _save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

