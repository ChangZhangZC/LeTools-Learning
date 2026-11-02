"""Batch 02: BC-only entrypoint for the pinned, unmodified RL1002D.

Preserves RL-100/train.py stage-1 BC: policy.compute_loss, AdamW,
get_scheduler, EMAModel, and its checkpoint payload layout. Unlike the
upstream combined Workspace, does not initialize Critic/PPO/Dynamics/env.
Run from the LeTools-Learning repository root:
  python post_training/src/arbim_rl100/train_bc.py --config post_training/configs/bc/rl100_2d_bc.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
from itertools import islice
import random
import sys
from pathlib import Path

import dill
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "post_training" / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "rl100" / "RL-100"))

from arbim_rl100.data.batch_adapter import adapt_rl100_shape_meta, to_rl100_bc_batch
from arbim_rl100.data.offline_buffer import OfflineBuffer
from arbim_rl100.data.offline_dataset import OfflineDataset
from rl_100.model.common.lr_scheduler import get_scheduler
from rl_100.model.common.normalizer import LinearNormalizer
from rl_100.model.diffusion.ema_model import EMAModel
from rl_100.model.vision.model_getter import get_resnet
from rl_100.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from rl_100.policy.rl100_2d import RL1002D
from rl_100.unidpg.diffusion_policy.diffusers_patch.ddim_with_logprob_dpok import (
    DDIMSchedulerExtended,
)

CAMERAS = {
    "rgb_head": "head_rgb",
    "rgb_left_hand": "wrist_left_rgb",
    "rgb_right_hand": "wrist_right_rgb",
}


def fit_train_normalizer(buffer: OfflineBuffer, train_mask: np.ndarray) -> LinearNormalizer:
    # Same train-episode-only fitting contract validated in smoke_01_real_bc.py.
    indices, start = [], 0
    for selected, end in zip(train_mask, buffer.episode_ends, strict=True):
        if selected:
            indices.append(np.arange(start, int(end), dtype=np.int64))
        start = int(end)
    if not indices:
        raise ValueError("No training episodes available for normalizer fitting")
    selected = np.concatenate(indices)
    norm = LinearNormalizer()
    norm.fit({"agent_pos": buffer["state"][selected], "action": buffer["action"][selected]},
             mode="limits")
    return norm


def shape_meta_from_buffer(buffer: OfflineBuffer):
    meta = {"obs": {
        name: {"shape": list(buffer[source].shape[1:]), "type": "rgb"}
        for name, source in CAMERAS.items()
    }, "action": {"shape": [int(buffer["action"].shape[-1])]}}
    meta["obs"]["agent_pos"] = {
        "shape": [int(buffer["state"].shape[-1])], "type": "low_dim"
    }
    return adapt_rl100_shape_meta(meta)


def build_policy(cfg, shape_meta):
    # Identical actor configuration to the successfully verified Batch 01D
    # real-data smoke, with its numeric choices read from the BC YAML.
    obs_encoder = MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=get_resnet("resnet18", weights=None),
        resize_shape=(int(cfg.policy.resize), int(cfg.policy.resize)),
        crop_shape=None,
        random_crop=False,
        use_group_norm=bool(cfg.policy.use_group_norm),
        share_rgb_model=bool(cfg.policy.share_rgb_model),
        imagenet_norm=True,
        use_agent_pos=True,
        use_vib=False,
        use_recon=False,
    )
    scheduler = DDIMSchedulerExtended(
        num_train_timesteps=int(cfg.policy.num_train_timesteps),
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
        clip_std_min=0.01,
    )
    return RL1002D(
        shape_meta=shape_meta,
        obs_encoder=obs_encoder,
        cm_noise_scheduler=None,
        ddim_noise_scheduler=scheduler,
        scheduler_type="ddim",
        model="sample_dp3",
        feature_type="2D",
        w_pc=False,
        use_agent_pos=True,
        obs_as_global_cond=True,
        action_norm=bool(cfg.policy.action_norm),
        horizon=int(cfg.data.horizon),
        n_obs_steps=int(cfg.data.n_obs_steps),
        n_action_steps=int(cfg.data.n_action_steps),
        num_inference_steps=int(cfg.policy.num_inference_steps),
        ddim_inference_steps=int(cfg.policy.num_inference_steps),
        down_dims=tuple(int(x) for x in cfg.policy.down_dims),
        diffusion_step_embed_dim=int(cfg.policy.diffusion_step_embed_dim),
        kernel_size=3,
        n_groups=8,
        use_aug=False,
    )


def bc_batch(sample, n_obs_steps, device):
    adapted = to_rl100_bc_batch(sample, n_obs_steps=n_obs_steps)
    return {
        "obs": {k: v.to(device, non_blocking=True) for k, v in adapted["obs"].items()},
        "action": adapted["action"].to(device, non_blocking=True),
    }


def _rng_state():
    """Capture RNG after validation, at an epoch boundary."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint records CUDA RNG but CUDA is unavailable")
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; exact resume is not supported")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _check_resume_config(requested, saved):
    # Scheduler and BC sampling depend on these settings. Changing epochs
    # while resuming an already constructed cosine schedule is not valid.
    for section in ("data", "policy", "training"):
        new = OmegaConf.to_container(requested[section], resolve=True)
        old = OmegaConf.to_container(saved[section], resolve=True)
        if section == "training":
            new.pop("device", None)
            old.pop("device", None)
        if new != old:
            raise ValueError(
                f"Resume {section} config differs from checkpoint. "
                "Use the original config for exact continuation; use a new "
                "output directory for a new training experiment."
            )


def _save_checkpoint(path: Path, payload):
    """Atomic persistence; keep last valid checkpoint on failed writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary, pickle_module=dill)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_payload(cfg, model, optimizer, ema_model, lr_scheduler, epoch,
                       global_step, ema=None, best_val_loss=None):
    # Original RL-100 cfg/state_dicts/pickles shape, plus minimal BC-only
    # continuation state (LR schedule, EMA step and all RNG streams).
    states = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
              "lr_scheduler": lr_scheduler.state_dict()}
    if ema_model is not None:
        states["ema_model"] = ema_model.state_dict()
    pickles = {
        "epoch": dill.dumps(epoch),
        "global_step": dill.dumps(global_step),
        "rng_state": dill.dumps(_rng_state()),
    }
    if ema is not None:
        pickles["ema_optimization_step"] = dill.dumps(ema.optimization_step)
        pickles["ema_decay"] = dill.dumps(ema.decay)
    if best_val_loss is not None:
        pickles["best_val_loss"] = dill.dumps(float(best_val_loss))
    return {"cfg": cfg, "state_dicts": states, "pickles": pickles}


def load_checkpoint(path, model, optimizer, ema_model, lr_scheduler, *, ema=None,
                    requested_cfg=None, restore_rng=False, best_state=None):
    payload = torch.load(path, map_location="cpu", pickle_module=dill)
    if requested_cfg is not None:
        _check_resume_config(requested_cfg, payload["cfg"])
    states = payload["state_dicts"]
    if (ema_model is None) != ("ema_model" not in states):
        raise ValueError("Checkpoint EMA enabled state differs from training config")
    # RL-100 DictOfTensorMixin._load_from_state_dict REPLACES the normalizer
    # ParameterDict using checkpoint tensors on CPU (map_location="cpu").
    # Restore the entire model to its original device before loading AdamW.
    target_device = next(model.model.parameters()).device
    model.load_state_dict(states["model"], strict=True)
    model.to(target_device)
    optimizer.load_state_dict(states["optimizer"])
    lr_scheduler.load_state_dict(states["lr_scheduler"])
    if ema_model is not None:
        ema_device = next(ema_model.model.parameters()).device
        ema_model.load_state_dict(states["ema_model"], strict=True)
        ema_model.to(ema_device)
    saved = payload["pickles"]
    if best_state is not None:
        best_state["val_loss"] = (
            float(dill.loads(saved["best_val_loss"]))
            if "best_val_loss" in saved else float("inf")
        )
    epoch, global_step = (
        int(dill.loads(saved["epoch"])),
        int(dill.loads(saved["global_step"])),
    )
    if ema is not None:
        # Older validated single-step checkpoints did not store this counter.
        ema.optimization_step = (
            int(dill.loads(saved["ema_optimization_step"]))
            if "ema_optimization_step" in saved else global_step
        )
        if "ema_decay" in saved:
            ema.decay = float(dill.loads(saved["ema_decay"]))
        elif global_step > 0:
            ema.decay = ema.get_decay(global_step - 1)
    if restore_rng:
        if "rng_state" in saved:
            _restore_rng_state(dill.loads(saved["rng_state"]))
        else:
            print("[WARN] Legacy checkpoint has no RNG state: exact continuation not guaranteed")
    return epoch, global_step


def train(cfg, resume: Path | None = None, stop_after_epoch: int | None = None):
    seed = int(cfg.data.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(str(cfg.training.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    torch.set_num_threads(min(torch.get_num_threads(), 4))

    zarr_path = Path(cfg.data.zarr).expanduser()
    if not zarr_path.is_dir():
        raise FileNotFoundError(f"Missing offline Zarr: {zarr_path}")
    output = Path(cfg.output.dir).expanduser()
    checkpoint = output / "checkpoints" / "latest.ckpt"
    best_checkpoint = output / "checkpoints" / "best.ckpt"
    metrics_path = output / "metrics.jsonl"
    if resume is not None and resume.resolve() != checkpoint.resolve():
        raise ValueError("--resume must point to output.dir/checkpoints/latest.ckpt")
    if resume is None and (best_checkpoint.exists() or metrics_path.exists()):
        raise FileExistsError(
            f"Prior BC training artifacts exist under {output}; "
            "choose a fresh --output-dir or resume the matching checkpoint"
        )
    if checkpoint.exists() and resume is None:
        raise FileExistsError(
            f"{checkpoint} already exists; specify --resume to continue "
            "or choose another output.dir"
        )
    epochs = int(cfg.training.epochs)
    if epochs < 1 or int(cfg.training.batch_size) < 1:
        raise ValueError("epochs and batch_size must be positive")
    if int(cfg.training.checkpoint_every) != 1:
        raise ValueError(
            "Batch 02 only supports checkpoint_every=1: latest.ckpt must "
            "track every completed epoch for exact resume and metrics"
        )
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= epochs:
        raise ValueError("--stop-after-epoch must be in [1, training.epochs]")
    epoch_limit = epochs if stop_after_epoch is None else stop_after_epoch

    buffer = OfflineBuffer(device=torch.device("cpu"), gamma=0.99, use_depth=False)
    buffer.load_zarr(str(zarr_path.resolve()))
    dataset = OfflineDataset(
        buffer, horizon=int(cfg.data.horizon), pad_before=int(cfg.data.pad_before),
        pad_after=int(cfg.data.pad_after), seed=seed, val_ratio=float(cfg.data.val_ratio),
    )
    val_dataset = dataset.get_validation_dataset()
    if len(dataset) == 0:
        raise ValueError("No train windows; check horizon and episode lengths")
    train_loader = DataLoader(
        dataset, batch_size=int(cfg.training.batch_size), shuffle=True,
        num_workers=int(cfg.training.num_workers), pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=int(cfg.training.batch_size), shuffle=False,
        num_workers=int(cfg.training.num_workers), pin_memory=device.type == "cuda",
    )
    n_obs_steps = int(cfg.data.n_obs_steps)
    model = build_policy(cfg, shape_meta_from_buffer(buffer))
    model.set_normalizer(fit_train_normalizer(buffer, dataset.train_mask))
    model.to(device)
    ema_model = copy.deepcopy(model) if cfg.training.use_ema else None
    ema = None
    if ema_model is not None:
        ema = EMAModel(
            model=ema_model, update_after_step=0, inv_gamma=1.0,
            power=float(cfg.training.ema_power),
            min_value=0.0, max_value=float(cfg.training.ema_max_value),
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.training.lr),
        betas=tuple(float(v) for v in cfg.training.betas),
        eps=float(cfg.training.eps),
        weight_decay=float(cfg.training.weight_decay),
    )
    max_steps = cfg.training.max_train_steps_per_epoch
    steps_per_epoch = min(len(train_loader), int(max_steps)) if max_steps is not None else len(train_loader)
    if steps_per_epoch < 1:
        raise ValueError("max_train_steps_per_epoch must be positive")
    lr_scheduler = get_scheduler(
        str(cfg.training.lr_scheduler), optimizer=optimizer,
        num_warmup_steps=int(cfg.training.lr_warmup_steps),
        num_training_steps=steps_per_epoch * epochs,
    )
    start_epoch, global_step = (0, 0)
    best_state = {"val_loss": float("inf")}
    if resume is not None:
        start_epoch, global_step = load_checkpoint(
            resume, model, optimizer, ema_model, lr_scheduler,
            ema=ema, requested_cfg=cfg, restore_rng=True,
            best_state=best_state,
        )
        if start_epoch >= epoch_limit:
            raise ValueError(
                f"Checkpoint has completed epoch {start_epoch}; "
                f"requested stop is epoch {epoch_limit}. Nothing to train."
            )
        print(f"[PASS] Restored epoch={start_epoch}, global_step={global_step}; "
              f"EMA step={ema.optimization_step if ema else 'disabled'}")
    print(f"Train episodes={int(dataset.train_mask.sum())}; "
          f"train windows={len(dataset)}; val windows={len(val_dataset)}; "
          f"steps/epoch={steps_per_epoch}")
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, epoch_limit):
        model.train()
        train_losses = []
        for step, sample in enumerate(tqdm(
            islice(train_loader, steps_per_epoch),
            desc=f"Training epoch {epoch}", total=steps_per_epoch,
        )):

            batch = bc_batch(sample, n_obs_steps, device)
            raw_loss, details = model.compute_loss(batch)
            if not torch.isfinite(raw_loss).all().item():
                raise FloatingPointError(f"Nonfinite BC loss on epoch={epoch} step={step}")
            raw_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            if ema is not None:
                ema.step(model)
            global_step += 1
            train_losses.append(raw_loss.item())
        if not train_losses:
            raise RuntimeError("No BC training steps executed")
        result = {"train_loss": float(np.mean(train_losses)),
                  "global_step": global_step, "epoch": epoch}
        if len(val_loader) > 0 and cfg.training.max_val_steps != 0:
            model.eval()
            val_losses = []
            # Native diffusion validation samples random noise/timesteps.
            # Fix its RNG for comparable epochs without perturbing training.
            cuda_devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
            with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
                torch.manual_seed(seed + 1_000_003)
                for step, sample in enumerate(val_loader):
                    loss, _ = model.compute_loss(bc_batch(sample, n_obs_steps, device))
                    if not torch.isfinite(loss).all().item():
                        raise FloatingPointError(f"Nonfinite validation loss on epoch={epoch} step={step}")
                    val_losses.append(loss.item())
                    if cfg.training.max_val_steps is not None and step + 1 >= int(cfg.training.max_val_steps):
                        break
            result["val_loss"] = float(np.mean(val_losses))
        is_best = "val_loss" in result and result["val_loss"] < best_state["val_loss"]
        if is_best:
            best_state["val_loss"] = result["val_loss"]
        result["best_val_loss"] = (
            best_state["val_loss"] if np.isfinite(best_state["val_loss"]) else None
        )
        result["lr"] = float(optimizer.param_groups[0]["lr"])
        print(f"[BC] {result}")
        payload = checkpoint_payload(
            cfg, model, optimizer, ema_model, lr_scheduler,
            epoch + 1, global_step, ema=ema,
            best_val_loss=best_state["val_loss"],
        )
        _save_checkpoint(checkpoint, payload)
        print(f"[PASS] Saved BC checkpoint: {checkpoint}")
        if is_best:
            _save_checkpoint(best_checkpoint, payload)
            print(f"[PASS] Saved best validation BC checkpoint: {best_checkpoint}")
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    return checkpoint


def main():
    raise RuntimeError(
        "Zarr-based BC training is retired. Train the native RL1002D policy "
        "directly on LeRobot datasets via: "
        "python kuavo_model/train.py --policy rl100_2d. "
        "Zarr is reserved for RL post-training."
    )

if __name__ == "__main__":
    main()
