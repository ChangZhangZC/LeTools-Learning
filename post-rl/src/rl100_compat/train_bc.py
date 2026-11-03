"""Minimal BC training entry point for the *unmodified* RL1002D.

Read native LeRobot v3, not a custom replay buffer. This first version
supports fresh runs only (no exact resume / validation split).
"""
from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "post-rl" / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "rl100" / "RL-100"))

from rl100_compat.lerobot_data import open_lerobot_bc_dataset  # noqa: E402
from rl_100.model.common.lr_scheduler import get_scheduler  # noqa: E402
from rl_100.model.common.normalizer import LinearNormalizer  # noqa: E402
from rl_100.model.diffusion.ema_model import EMAModel  # noqa: E402
from rl_100.model.vision.model_getter import get_resnet  # noqa: E402
from rl_100.model.vision.multi_image_obs_encoder import MultiImageObsEncoder  # noqa: E402
from rl_100.policy.rl100_2d import RL1002D  # noqa: E402
from rl_100.unidpg.diffusion_policy.diffusers_patch.ddim_with_logprob_dpok import (  # noqa: E402
    DDIMSchedulerExtended,
)


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


def train(cfg, *, max_steps: int | None = None) -> Path:
    steps = int(max_steps if max_steps is not None else cfg.training.steps)
    obs_steps, horizon = int(cfg.data.n_obs_steps), int(cfg.data.horizon)
    action_steps = int(cfg.data.n_action_steps)
    if steps < 1 or int(cfg.training.batch_size) < 1:
        raise ValueError("steps and batch_size must be positive")
    if not 1 <= action_steps <= horizon - obs_steps + 1:
        raise ValueError("n_action_steps out of available action horizon")
    if int(cfg.training.save_every) < 1:
        raise ValueError("save_every must be positive")
    device = torch.device(str(cfg.training.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    seed = int(cfg.training.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    out = Path(str(cfg.output.dir)).expanduser().resolve()
    checkpoint = out / "checkpoints" / "latest.pt"
    if checkpoint.exists():
        raise FileExistsError(f"Checkpoint exists: {checkpoint}; choose a fresh output directory")
    view = open_lerobot_bc_dataset(
        str(cfg.data.repo_id), str(cfg.data.root),
        n_obs_steps=obs_steps, horizon=horizon,
        video_backend=str(cfg.data.video_backend),
    )
    sampler = make_episode_sampler(view, obs_steps, horizon)
    loader = DataLoader(
        view, sampler=sampler, batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.num_workers),
        pin_memory=device.type == "cuda",
    )
    model = build_policy(cfg, view.contract)
    model.set_normalizer(build_normalizer(view.dataset))
    model.to(device)
    ema = None
    if bool(cfg.training.use_ema):
        ema = EMAModel(model=copy.deepcopy(model), power=float(cfg.training.ema_power))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    lr_scheduler = get_scheduler(
        str(cfg.training.lr_scheduler), optimizer=optimizer,
        num_warmup_steps=int(cfg.training.lr_warmup_steps),
        num_training_steps=steps,
    )
    out.mkdir(parents=True, exist_ok=True)
    print(f"[BC] frames={len(view)}; anchors={len(sampler)}; "
          f"state={view.contract.state_dim}; action={view.contract.action_dim}; "
          f"cameras={list(view.contract.source_to_rgb)}")
    print(f"[BC] H={horizon}; To={obs_steps}; drop_last={horizon - obs_steps}")
    model.train()
    iterator = iter(loader)
    for step in range(1, steps + 1):
        try:
            sample = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            sample = next(iterator)
        assert_valid_action_targets(sample, obs_steps)
        batch = _to_device(sample, device)
        optimizer.zero_grad(set_to_none=True)
        loss, details = model.compute_loss(batch)
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise FloatingPointError(f"Invalid BC loss at step={step}: {loss}")
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if ema is not None:
            ema.step(model)
        print(f"[BC] step={step}/{steps} loss={loss.item():.6f} details={details}")
        if step % int(cfg.training.save_every) == 0 or step == steps:
            _save_checkpoint(checkpoint, _payload(
                cfg, view, model, optimizer, lr_scheduler, step, ema
            ))
            print(f"[PASS] saved {checkpoint}")
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.output_dir is not None:
        cfg.output.dir = str(args.output_dir)
    train(cfg, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
