"""BC training entrypoint for LeTools native RL100 DP-2D baseline.

Uses the original RL1002D actor/loss/normalizer/checkpoint schema with episode-heldout
validation, train-only normalizer statistics, deterministic step sampling,
EMA-based model selection, resumable full training state, and JSONL metrics.
Neither original RL-100 nor existing LeTools ACT/DP training/runtime is edited.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "third_party" / "lerobot" / "src",
                  ROOT / "third_party" / "rl100" / "RL-100"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from kuavo_model.rl100_dp_2d.dataset import open_lerobot_bc_dataset  # noqa: E402
from kuavo_model.rl100_dp_2d.metrics import BCMetricLogger  # noqa: E402
from kuavo_model.rl100_dp_2d.bc_utils import (  # noqa: E402
    _payload, _save_checkpoint, _to_device, assert_valid_action_targets,
    make_episode_sampler,
)
from kuavo_model.rl100_dp_2d.policy_factory import build_policy  # noqa: E402
from rl_100.model.common.lr_scheduler import get_scheduler  # noqa: E402
from rl_100.model.common.normalizer import LinearNormalizer  # noqa: E402
from rl_100.model.diffusion.ema_model import EMAModel  # noqa: E402


def resolve_output_dir(path: str | Path) -> Path:
    """Resolve relative paths against LeTools repository root, like train.py."""
    selected = Path(path).expanduser()
    return (selected if selected.is_absolute() else ROOT / selected).resolve()


def training_limits(cfg) -> tuple[int | None, int | None]:
    """Step and complete-epoch caps are independently optional."""
    limits = []
    for key in ("steps", "epochs"):
        value = cfg.training.get(key)
        if value is not None and (isinstance(value, bool) or
                                  not isinstance(value, int) or value < 1):
            raise ValueError(f"training.{key} must be a positive integer or null")
        limits.append(value)
    if limits[0] is None and limits[1] is None:
        raise ValueError("At least one of training.steps or training.epochs is required")
    return limits[0], limits[1]


def resolve_training_budget(cfg, train_windows: int) -> tuple[int, int]:
    """Translate complete train-only epochs to steps using the existing sampler.

    The final partial batch counts as one optimizer step per epoch.
    """
    steps, epochs = training_limits(cfg)
    batch_size = int(cfg.training.batch_size)
    if train_windows < 1 or batch_size < 1:
        raise ValueError("train_windows and training.batch_size must be positive")
    steps_per_epoch = math.ceil(train_windows / batch_size)
    caps = [cap for cap in (steps, epochs * steps_per_epoch if epochs is not None else None)
            if cap is not None]
    return min(caps), steps_per_epoch


def check_config(cfg) -> None:
    """Reject incompatible horizons, nonfinite budgets, and missing features."""
    horizon, obs = int(cfg.data.horizon), int(cfg.data.n_obs_steps)
    action_steps = int(cfg.data.n_action_steps)
    if obs < 1 or horizon < obs or not 1 <= action_steps <= horizon - obs + 1:
        raise ValueError("Invalid observation/action horizon")
    down_dims = list(cfg.policy.down_dims)
    if not down_dims or any(int(x) < 1 for x in down_dims):
        raise ValueError("policy.down_dims must be positive")
    if (horizon - obs + 1) % (2 ** len(down_dims)):
        raise ValueError("Native RL100 U-Net denoising horizon must be divisible by "
                         "2**len(policy.down_dims)")
    if not 1 <= int(cfg.policy.num_inference_steps) <= int(cfg.policy.num_train_timesteps):
        raise ValueError("Invalid diffusion inference/training timestep count")
    training_limits(cfg)
    if int(cfg.training.batch_size) < 1:
        raise ValueError("training.batch_size must be positive")
    for name in ("num_workers", "lr_warmup_steps"):
        if int(cfg.training[name]) < 0:
            raise ValueError(f"training.{name} must be nonnegative")
    for key in ("save_every", "log_every", "val_every"):
        if int(cfg.training[key]) < 1:
            raise ValueError(f"training.{key} must be positive")
    if not math.isfinite(float(cfg.training.lr)) or float(cfg.training.lr) <= 0:
        raise ValueError("training.lr must be positive and finite")
    if float(cfg.training.weight_decay) < 0 or not math.isfinite(float(cfg.training.weight_decay)):
        raise ValueError("training.weight_decay must be finite and nonnegative")
    if not 0 < float(cfg.training.ema_power) <= 1:
        raise ValueError("training.ema_power must be in (0,1]")
    if float(cfg.training.max_grad_norm) < 0:
        raise ValueError("training.max_grad_norm must be >= 0 (0 disables clipping)")
    if not 0 < float(cfg.validation.fraction) < 1:
        raise ValueError("validation.fraction must be in (0,1)")
    if int(cfg.validation.batch_size) < 1 or int(cfg.validation.max_batches) < 0:
        raise ValueError("Invalid validation batch_size/max_batches")


def split_episode_ids(count: int, fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic disjoint episode split. Never split individual frames."""
    if count < 2:
        raise ValueError("At least two dataset episodes required for held-out validation")
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be in (0,1)")
    ids = list(range(count))
    random.Random(seed).shuffle(ids)
    n_val = max(1, min(count - 1, round(count * fraction)))
    return sorted(ids[n_val:]), sorted(ids[:n_val])


def episode_anchor_indices(view, episode_ids: list[int], obs_steps: int,
                           horizon: int) -> list[int]:
    """Use the SAME native LeRobot EpisodeAwareSampler as Feature03."""
    valid = make_episode_sampler(view, obs_steps, horizon).indices
    ranges = [
        (int(view.dataset.meta.episodes[i]["dataset_from_index"]),
         int(view.dataset.meta.episodes[i]["dataset_to_index"]))
        for i in episode_ids
    ]
    keep = [i for i in valid if any(lo <= i < hi for lo, hi in ranges)]
    if not keep:
        raise ValueError(f"No unpadded target windows in episodes {episode_ids}")
    return keep


class StepBatchSampler:
    """Step-addressable deterministic batches, independent of prefetch cursor.

    Every global optimizer step maps to one (epoch, batch) permutation. Resume
    can rebuild the exact next batch without serializing DataLoader workers.
    Models using stochastic dataset transforms should use num_workers=0 for
    strict reproducibility; worker-side random state is not checkpointed.
    """
    def __init__(self, anchors: list[int], batch_size: int, seed: int,
                 start_step: int, stop_step: int):
        if not anchors or batch_size < 1 or start_step < 0 or stop_step < start_step:
            raise ValueError("Invalid step sampler contract")
        self.anchors = tuple(anchors)
        self.batch_size = batch_size
        self.seed = seed
        self.start_step = start_step
        self.stop_step = stop_step
        self.batches_per_epoch = math.ceil(len(anchors) / batch_size)

    def __iter__(self):
        epoch = -1
        order = []
        for step in range(self.start_step, self.stop_step):
            next_epoch, batch = divmod(step, self.batches_per_epoch)
            if next_epoch != epoch:
                g = torch.Generator().manual_seed(self.seed + next_epoch)
                order = torch.randperm(len(self.anchors), generator=g).tolist()
                epoch = next_epoch
            offset = batch * self.batch_size
            yield [self.anchors[i] for i in order[offset:offset + self.batch_size]]

    def __len__(self) -> int:
        return self.stop_step - self.start_step


def build_train_only_normalizer(dataset, train_ids: list[int]) -> LinearNormalizer:
    """Fit native RL100 limits from raw TRAIN episode state/action (no RGB decode).

    Do not use dataset.meta.stats: those include the held-out validation episodes.
    The original native normalization math is unchanged.
    """
    from kuavo_model.rl100_dp_2d.dataset import ACTION_KEY, STATE_KEY
    raw = dataset.hf_dataset.select_columns([STATE_KEY, ACTION_KEY])
    dims = {
        STATE_KEY: int(dataset.features[STATE_KEY]["shape"][0]),
        ACTION_KEY: int(dataset.features[ACTION_KEY]["shape"][0]),
    }
    low = {k: torch.full((d,), float("inf")) for k, d in dims.items()}
    high = {k: torch.full((d,), -float("inf")) for k, d in dims.items()}
    rows = 0
    for ep in train_ids:
        info = dataset.meta.episodes[ep]
        for idx in range(int(info["dataset_from_index"]), int(info["dataset_to_index"])):
            sample = raw[idx]
            for key, dim in dims.items():
                value = torch.as_tensor(sample[key], dtype=torch.float32).reshape(-1)
                if value.numel() != dim or not torch.isfinite(value).all().item():
                    raise ValueError(f"Invalid raw {key} at train frame {idx}")
                low[key] = torch.minimum(low[key], value)
                high[key] = torch.maximum(high[key], value)
            rows += 1
    if rows == 0:
        raise ValueError("No training frames for normalizer")
    normalizer = LinearNormalizer()
    normalizer.fit({
        "agent_pos": torch.stack([low[STATE_KEY], high[STATE_KEY]]),
        "action": torch.stack([low[ACTION_KEY], high[ACTION_KEY]]),
    }, mode="limits")
    return normalizer


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    """weights_only=True-compatible RNG checkpoint (no numpy ndarray objects)."""
    n = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": n[0], "keys": n[1].tolist(),
            "pos": int(n[2]), "has_gauss": int(n[3]), "cached_gaussian": float(n[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((
        n["bit_generator"], np.asarray(n["keys"], dtype=np.uint32),
        n["pos"], n["has_gauss"], n["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint has CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["cuda"])


def _resume_signature(cfg) -> dict:
    """Optimizer schedule and data/action definition cannot drift on resume."""
    tr = cfg.training
    signature = {
        "data": OmegaConf.to_container(cfg.data, resolve=True),
        "policy": OmegaConf.to_container(cfg.policy, resolve=True),
        "training": {k: tr[k] for k in (
            "steps", "seed", "batch_size", "lr", "weight_decay",
            "lr_scheduler", "lr_warmup_steps", "use_ema",
            "ema_power", "max_grad_norm", "val_every",
        )},
        "validation": OmegaConf.to_container(cfg.validation, resolve=True),
    }
    # Preserve old step-only signature for previously saved training checkpoints.
    if tr.get("epochs") is not None:
        signature["training"]["epochs"] = tr.epochs
    return signature


def _write_metric(path: Path, **values) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(values, allow_nan=False) + "\n")


def _trim_metrics_after_resume(path: Path, last_step: int) -> None:
    """Avoid duplicate future log entries if an interrupted run wrote past latest.pt."""
    if not path.exists():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if int(entry["step"]) <= last_step:
            kept.append(line)
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def _sync_ema_buffers(ema: EMAModel, model: torch.nn.Module) -> None:
    """Native RL100 EMA averages parameters only; keep non-parameter buffers in sync."""
    with torch.no_grad():
        target = dict(ema.averaged_model.named_buffers())
        for name, buf in model.named_buffers():
            if name not in target:
                raise ValueError(f"EMA buffer missing {name}")
            target[name].copy_(buf)


def evaluate(model: torch.nn.Module, loader: DataLoader, obs_steps: int,
             device: torch.device, seed: int, max_batches: int) -> float:
    """Fixed diffusion-noise seed makes validation loss comparisons meaningful."""
    saved_rng = capture_rng_state()
    previous_mode = model.training
    total = 0.0
    count = 0
    try:
        _seed_all(seed)
        model.eval()
        with torch.no_grad():
            for batch_idx, sample in enumerate(loader):
                if max_batches and batch_idx >= max_batches:
                    break
                assert_valid_action_targets(sample, obs_steps)
                loss, _ = model.compute_loss(_to_device(sample, device))
                if loss.ndim != 0 or not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Invalid validation loss: {loss}")
                bs = sample["action"].shape[0]
                total += loss.item() * bs
                count += bs
    finally:
        model.train(previous_mode)
        restore_rng_state(saved_rng)
    if not count:
        raise ValueError("Validation produced no batches")
    return total / count


def _make_resume_payload(cfg, view, model, optimizer, scheduler, ema,
                         step: int, split: dict, best_loss: float | None,
                         best_step: int | None) -> dict:
    state = _payload(cfg, view, model, optimizer, scheduler, step, ema)
    state.update(
        checkpoint_role="training_latest",
        resume_signature=_resume_signature(cfg),
        split=split,
        normalizer_scope="train_episodes_only",
        best_val_loss=best_loss,
        best_step=best_step,
        rng_state=capture_rng_state(),
    )
    return state


def _make_best_payload(cfg, view, selected_model, step: int, split: dict,
                       val_loss: float, source: str) -> dict:
    """Deployment-friendly schema: 'model' IS the selected best/EMA actor."""
    shape = {
        "obs": {key: dict(spec) for key, spec in view.contract.shape_meta["obs"].items()},
        "action": dict(view.contract.shape_meta["action"]),
    }
    return {
        "step": step,
        "model": selected_model.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "shape_meta": shape,
        "camera_map": dict(view.contract.source_to_rgb),
        "state_names": list(view.contract.state_names),
        "action_names": list(view.contract.action_names),
        "dataset_fps": int(view.dataset.fps),
        "checkpoint_role": "deploy_best",
        "selected_weights": source,
        "best_val_loss": val_loss,
        "normalizer_scope": "train_episodes_only",
        "split": split,
    }


def train(cfg, *, stop_after: int | None = None) -> Path:
    check_config(cfg)
    device = torch.device(str(cfg.training.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = resolve_output_dir(str(cfg.output.dir))
    paths = {
        "latest": output / "checkpoints" / "latest.pt",
        "best": output / "checkpoints" / "best.pt",
        "metrics": output / "metrics.jsonl",
    }
    resume = bool(cfg.training.resume)
    if resume and not paths["latest"].is_file():
        raise FileNotFoundError(f"Missing resume checkpoint: {paths['latest']}")
    if not resume and (paths["latest"].exists() or paths["best"].exists()
                       or paths["metrics"].exists()):
        raise FileExistsError("Run outputs already exist; use a fresh output.dir or resume=true")

    _seed_all(int(cfg.training.seed))
    view = open_lerobot_bc_dataset(
        str(cfg.data.repo_id), str(cfg.data.root),
        n_obs_steps=int(cfg.data.n_obs_steps), horizon=int(cfg.data.horizon),
        video_backend=str(cfg.data.video_backend),
    )
    episodes = len(view.dataset.meta.episodes)
    train_ids, val_ids = split_episode_ids(
        episodes, float(cfg.validation.fraction), int(cfg.validation.split_seed),
    )
    split = {"train": train_ids, "val": val_ids}
    train_anchors = episode_anchor_indices(
        view, train_ids, int(cfg.data.n_obs_steps), int(cfg.data.horizon))
    val_anchors = episode_anchor_indices(
        view, val_ids, int(cfg.data.n_obs_steps), int(cfg.data.horizon))
    total_steps, steps_per_epoch = resolve_training_budget(cfg, len(train_anchors))
    if stop_after is not None and not 1 <= stop_after <= total_steps:
        raise ValueError("stop_after must be in [1, effective total_steps]")
    last_step = stop_after if stop_after is not None else total_steps
    saved = None
    if resume:
        saved = torch.load(paths["latest"], map_location="cpu", weights_only=True)
        if saved.get("checkpoint_role") != "training_latest":
            raise ValueError("Only production latest.pt is resumable (not deploy best.pt)")
        if saved.get("resume_signature") != _resume_signature(cfg) or saved.get("split") != split:
            raise ValueError("Checkpoint architecture, data, optimizer, or episode split differs")
        if saved.get("effective_total_steps", total_steps) != total_steps:
            raise ValueError("Checkpoint effective training budget differs")
        if dict(saved["camera_map"]) != dict(view.contract.source_to_rgb):
            raise ValueError("Camera metadata changed between training runs")
        if saved["state_names"] != list(view.contract.state_names) or saved["action_names"] != list(view.contract.action_names):
            raise ValueError("Robot state/action names changed since checkpoint")

    model = build_policy(cfg, view.contract)
    # IMPORTANT: the native DictOfTensorMixin creates new normalizer
    # ParameterDict tensors when loading a checkpoint. Restore those BEFORE
    # constructing the optimizer/EMA, or AdamW parameter-group counts differ.
    if resume:
        model.load_state_dict(saved["model"], strict=True)
    else:
        model.set_normalizer(build_train_only_normalizer(view.dataset, train_ids))
    model.to(device)
    ema = None
    if bool(cfg.training.use_ema):
        import copy
        ema = EMAModel(model=copy.deepcopy(model), power=float(cfg.training.ema_power))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    scheduler = get_scheduler(
        str(cfg.training.lr_scheduler), optimizer=optimizer,
        num_warmup_steps=int(cfg.training.lr_warmup_steps),
        num_training_steps=total_steps,
    )

    best_loss = None
    best_step = None
    start_step = 0
    if resume:
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["lr_scheduler"])
        if ema is not None:
            if saved["ema_model"] is None or saved["ema_step"] is None:
                raise ValueError("EMA weights missing from resumed checkpoint")
            ema.averaged_model.load_state_dict(saved["ema_model"], strict=True)
            ema.averaged_model.to(device)
            ema.optimization_step = int(saved["ema_step"])
        elif saved["ema_model"] is not None:
            raise ValueError("Unexpected EMA checkpoint for use_ema=false")
        start_step = int(saved["step"])
        best_loss, best_step = saved["best_val_loss"], saved["best_step"]
        if start_step < 0 or start_step > total_steps:
            raise ValueError("Checkpoint step incompatible with training budget")
        if best_step is not None:
            if not paths["best"].is_file():
                raise FileNotFoundError("Best deploy checkpoint missing on resume")
            best_saved = torch.load(paths["best"], map_location="cpu", weights_only=True)
            if best_saved.get("checkpoint_role") != "deploy_best" or best_saved.get("step") != best_step:
                raise ValueError("best.pt is inconsistent with latest.pt; inspect interrupted write")
        restore_rng_state(saved["rng_state"])
        _trim_metrics_after_resume(paths["metrics"], start_step)
    else:
        output.mkdir(parents=True, exist_ok=True)
        (output / "resolved_config.yaml").write_text(
            OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
        )

    train_loader = DataLoader(
        view,
        batch_sampler=StepBatchSampler(
            train_anchors, int(cfg.training.batch_size), int(cfg.training.seed),
            start_step, last_step,
        ),
        num_workers=int(cfg.training.num_workers),
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(int(cfg.training.seed)),
    )
    val_loader = DataLoader(
        view, sampler=val_anchors, batch_size=int(cfg.validation.batch_size),
        num_workers=int(cfg.training.num_workers),
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(int(cfg.validation.seed)),
    )
    print(f"[BC] episodes={episodes} train={train_ids} val={val_ids}; "
          f"anchors={len(train_anchors)}/{len(val_anchors)}")
    cap_steps, cap_epochs = training_limits(cfg)
    print(f"[BC] budget: steps={cap_steps}, epochs={cap_epochs}, "
          f"steps_per_epoch={steps_per_epoch}, effective_total_steps={total_steps}")
    print(f"[BC] state={view.contract.state_dim} action={view.contract.action_dim} "
          f"cameras={list(view.contract.source_to_rgb)}; "
          f"resume_step={start_step}, target={last_step}/{total_steps}")
    # Optional telemetry may initialize its own RNG. Keep training noise
    # identical with/without logging, including after an exact resume.
    logger_rng = capture_rng_state()
    try:
        metric_logger = BCMetricLogger(cfg, output, resumed=resume, resume_step=start_step)
    finally:
        restore_rng_state(logger_rng)
    try:
        model.train()
        start_time = time.monotonic()
        for step, sample in enumerate(train_loader, start=start_step + 1):
            assert_valid_action_targets(sample, int(cfg.data.n_obs_steps))
            batch = _to_device(sample, device)
            optimizer.zero_grad(set_to_none=True)
            loss, details = model.compute_loss(batch)
            if loss.ndim != 0 or not torch.isfinite(loss).item():
                raise FloatingPointError(f"Invalid BC loss at step={step}: {loss}")
            loss.backward()
            grad_limit = float(cfg.training.max_grad_norm)
            if grad_limit:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_limit, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.step(model)
                _sync_ema_buffers(ema, model)

            if step % int(cfg.training.log_every) == 0 or step == last_step:
                lr = optimizer.param_groups[0]["lr"]
                epoch_progress = step / steps_per_epoch
                _write_metric(paths["metrics"], phase="train", step=step,
                              epoch_equivalent=round(epoch_progress, 6),
                              loss=float(loss.item()), lr=float(lr),
                              elapsed_s=round(time.monotonic() - start_time, 3))
                metric_logger.train(step=step, loss=float(loss.item()),
                                    lr=float(lr), epoch=float(epoch_progress))
                print(f"[BC] step={step}/{total_steps} loss={loss.item():.6f} lr={lr:.3g}")

            if step % int(cfg.training.val_every) == 0 or step == total_steps:
                selected = ema.averaged_model if ema is not None else model
                val_loss = evaluate(
                    selected, val_loader, int(cfg.data.n_obs_steps),
                    device, int(cfg.validation.seed),
                    int(cfg.validation.max_batches),
                )
                _write_metric(paths["metrics"], phase="val", step=step,
                              loss=float(val_loss),
                              weights="ema" if ema is not None else "raw")
                print(f"[VAL] step={step} loss={val_loss:.6f}")
                if best_loss is None or val_loss < best_loss:
                    best_loss, best_step = float(val_loss), step
                    _save_checkpoint(paths["best"], _make_best_payload(
                        cfg, view, selected, step, split, best_loss,
                        "ema" if ema is not None else "raw",
                    ))
                    print(f"[PASS] best deploy checkpoint: step={step}, val={best_loss:.6f}")
                metric_logger.validation(step=step, loss=float(val_loss),
                                         best_loss=float(best_loss))

            if step % int(cfg.training.save_every) == 0 or step == last_step:
                latest = _make_resume_payload(
                    cfg, view, model, optimizer, scheduler, ema,
                    step, split, best_loss, best_step,
                )
                latest["effective_total_steps"] = total_steps
                _save_checkpoint(paths["latest"], latest)
                if bool(cfg.training.keep_step_checkpoints):
                    snapshot = output / "checkpoints" / f"step_{step:08d}.pt"
                    _save_checkpoint(snapshot, latest)
                print(f"[PASS] training checkpoint: step={step}")
        if start_step >= last_step:
            print(f"[BC] Nothing to do: checkpoint already at step={start_step}")
    finally:
        metric_logger.close()
    return paths["latest"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train" / "rl100_dp_2d.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume same run from latest.pt")
    parser.add_argument("--no-timestamp", action="store_true",
                        help="Use output.dir as-is, like LeTools train.py --no-timestamp")
    parser.add_argument("--stop-after", type=int, default=None,
                        help="Test interruption at this step; scheduler total remains unchanged")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.output_dir is not None:
        cfg.output.dir = str(args.output_dir)
    if args.resume:
        cfg.training.resume = True
    if not bool(cfg.training.resume) and not args.no_timestamp:
        output = Path(str(cfg.output.dir))
        cfg.output.dir = str(output.parent / (
            f"{output.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ))
    print(f"[BC] output_dir: {resolve_output_dir(str(cfg.output.dir))}")
    train(cfg, stop_after=args.stop_after)


if __name__ == "__main__":
    main()
