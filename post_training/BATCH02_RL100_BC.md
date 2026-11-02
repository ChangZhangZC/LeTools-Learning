# Batch 02 — native RL1002D BC/IL training (one-step acceptance passed)

## Boundary and source code

This batch uses the exact pinned RL-100 source at
`third_party/rl100/RL-100` (no submodule patch):

- `RL-100/train.py`, `TrainDP3Workspace.run()` stage 1-1
  (BC training), and its `save_checkpoint()/load_checkpoint()` payload
  layout;
- `RL-100/rl_100/policy/rl100_2d.py`:
  `RL1002D.compute_loss()`, `set_normalizer()`, `predict_action()`;
- `RL-100/rl_100/model/common/lr_scheduler.py`:
  original `get_scheduler()`;
- `RL-100/rl_100/model/diffusion/ema_model.py`:
  original `EMAModel.step()`;
- `RL-100/rl_100/config/rl100_2d_epsilon.yaml`: AdamW,
  EMA, scheduler and loader values.

The upstream `TrainDP3Workspace` unconditionally sets up PPO
(`BehaviorProximalPolicyOptimization`), Critic, Dynamics, and an
environment runner as part of its combined Stage-1/Stage-2 workflow.
Running that entire entrypoint would prematurely require later
batches' systems. Therefore the only new functional code is
`post_training/src/arbim_rl100/train_bc.py`, a thin **BC-only
orchestrator**. It does not reimplement policy losses, visual
encoders, schedulers, optimizers, or EMA. It connects previously
tested `OfflineDataset` + `to_rl100_bc_batch` with the original
classes. The `shape_meta` UserDict compatibility adapter remains
necessary for the pinned RL-100 constructor.

## What is deliberately different

- The default config `post_training/configs/bc/rl100_2d_bc.yaml`
  preserves the **Batch 01D real-data smoke actor architecture**
  (`resnet18`, resized RGB at 84×84, 9-step horizon, two observations,
  four predicted action steps, U-Net `down_dims=[32,64,128]`).
  **This is not the final RL-100 experiment configuration.**
- Uses the existing LeTools Zarr `OfflineBuffer` and
  `OfflineDataset`, not the upstream MetaWorld/Pusht dataset.
  Fits native RL-100 `LinearNormalizer` on train episodes **only**,
  as validated in Batch 01D. RGB scaling is unchanged.
- Original AdamW values are `lr=1e-4`,
  `betas=[0.95,0.999]`, `eps=1e-8`,
  `weight_decay=1e-6`. Cosine LR scheduler and native EMA are reused.
  `lr_warmup_steps=0` for the *one-step acceptance test*; the
  upstream default is 500 and should be restored when running a
  full schedule. Gradient accumulation is fixed at one step in this
  acceptance entrypoint to avoid introducing a new training variant.
- Skips unrelated IQL/PPO/Dynamics/online training and MuJoCo Runner
  construction. A small optional validation BC loss step is used
  instead of env rollout (the upstream Workspace has a validation
  code branch, but its `RUN_VALIDATION` is false by default).
- Checkpoint uses original `cfg/state_dicts/pickles` layout with
  `dill` and keys `model`, `optimizer`, `ema_model`, plus
  `lr_scheduler` for correct BC-only resume. It can be **strictly
  loaded by this BC entrypoint**. The original combined
  `TrainDP3Workspace.create_from_checkpoint()` **cannot yet be
  claimed compatible**, because this BC-only YAML omits Critic,
  Dynamics, and env runner config.
- This BC step consumes **no RL rewards**; the existing
  `terminal_proxy` real-data Zarr is acceptable for supervised
  reconstruction, *not* for Offline PPO.

## Server acceptance: first one optimizer update

Run from repository root in `letools_chang`, after fetching this
branch and with `data/zarr/icra_task1_10bags_parity.zarr` in place:

```bash
git fetch origin
git switch --track origin/feature/rl100-il

python post_training/src/arbim_rl100/train_bc.py \
  --config post_training/configs/bc/rl100_2d_bc.yaml

python post_training/tests/smoke/smoke_02_bc_checkpoint.py \
  --checkpoint outputs/train/rl100_bc_batch02/checkpoints/latest.ckpt \
  --device cuda:0
```

The first invocation performs a **real optimization update**, unlike
`smoke_01_real_bc.py`, which only computes gradients. The second
invocation reconstructs native RL1002D from the saved config, checks
optimizer state exists, strictly restores the model and EMA weights,
checks Normalizer weights, and runs a finite action prediction.

Observed on server (`letools_chang`): single optimizer update succeeded with
`train_loss=1.0632834434509277`, `val_loss=1.0436851978302002`,
`epoch=0`, `global_step=1`; saved `latest.ckpt`. In a fresh Python
process, strict model/normalizer/EMA restoration and optimizer/scheduler
state restoration passed, and restored `Action Chunk=(1,4,16)` was finite.
**This verifies one training step and restore, not convergence, full epochs,
rollout performance, nor compatibility with original combined Workspace.**

If a previous one-step checkpoint already exists, the training
command **refuses to overwrite it** without `--resume`; select a
new `output.dir` to repeat cleanly. In a new YAML for a longer
training run, set `training.max_train_steps_per_epoch: null`,
increase `training.epochs`, restore appropriate learning-rate
warmup, and leave other original RL-100 behavior unchanged.

## Multi-epoch Resume hardening (development; pending server acceptance)

**Development branch:** `gpt/rl100-il-resume` → integrate into
`feature/rl100-il` only after its regression tests pass. Only after
all IL functionality is validated may `feature/rl100-il` be
**Squash Merged** into `arbim/dev`. No direct `gpt/**` → dev merge.

The one-step BC checkpoint was verified previously, but was not an
exact multi-epoch resume contract. In particular, the original
`EMAModel` owns an `optimization_step` counter **outside** its
averaged-model `state_dict`. Recreating `EMAModel` without restoring
that counter changes the EMA decay schedule. DataLoader ordering and
diffusion noise also depend on Python/NumPy/Torch RNG states.

New backward-compatible BC-only checkpoint fields inside the existing
`pickles` map:

- `ema_optimization_step`, `ema_decay`: continue the **original**
  `EMAModel.step()` schedule; still uses original model weights.
- `rng_state`: Python `random`, NumPy, CPU Torch and all CUDA
  RNG streams captured **after** each epoch's validation.
- Existing `cfg/state_dicts/pickles` envelope, model/normalizer,
  optimizer, EMA state_dict and native LR scheduler state remain
  unchanged. Older single-step checkpoints without these fields
  can still load, but exact replay is not guaranteed.

The resumed `data`, `policy`, and `training` YAML must match the
saved configuration (except training device). This prevents accidentally
changing total epochs while reusing a cosine scheduler state. For a
different schedule, start a new training run rather than silently
extending an existing checkpoint. `--stop-after-epoch N` pauses
at a completed epoch **without changing the configured total epochs**.
`--output-dir` only changes the destination directory. Checkpoint
writes use a temporary file and atomic rename to avoid corrupting
the previous `latest.ckpt` if a save fails.

The previous one-step config is unchanged. The separate two-epoch
regression YAML has `epochs=2`, `batch_size=2`,
`max_train_steps_per_epoch=2`, `max_val_steps=1`, and `use_ema=true`.
This is an execution/resume test, **not a convergence experiment**.

### Server tests: interrupted vs uninterrupted

Run all from the repository root, `conda activate letools_chang`.
Each output directory must not already contain a checkpoint.

```bash
git fetch origin
git switch --track origin/gpt/rl100-il-resume

# Run two epochs straight: expected global_step=4.
python post_training/src/arbim_rl100/train_bc.py \
  --config post_training/configs/bc/rl100_2d_bc_resume_smoke.yaml \
  --output-dir outputs/train/rl100_bc_continuous

# Start another run but stop safely after epoch 1: expected global_step=2.
python post_training/src/arbim_rl100/train_bc.py \
  --config post_training/configs/bc/rl100_2d_bc_resume_smoke.yaml \
  --output-dir outputs/train/rl100_bc_resumed \
  --stop-after-epoch 1

# In a new process, continue same scheduled 2 epochs: global_step=4.
python post_training/src/arbim_rl100/train_bc.py \
  --config post_training/configs/bc/rl100_2d_bc_resume_smoke.yaml \
  --output-dir outputs/train/rl100_bc_resumed \
  --resume outputs/train/rl100_bc_resumed/checkpoints/latest.ckpt

# Reload finished weights and confirm real action inference.
python post_training/tests/smoke/smoke_02_bc_checkpoint.py \
  --checkpoint outputs/train/rl100_bc_resumed/checkpoints/latest.ckpt \
  --device cuda:0 --expected-epoch 2 --expected-step 4

# Compare full train, optimizer, EMA, scheduler, RNG and counters.
python post_training/tests/smoke/smoke_02_bc_resume_compare.py \
  --continuous outputs/train/rl100_bc_continuous/checkpoints/latest.ckpt \
  --resumed outputs/train/rl100_bc_resumed/checkpoints/latest.ckpt
```

If exact comparison fails, retain both checkpoints and logs; diagnose
the first mismatching field before adjusting any RL-100 upstream code.
Never treat a 2-epoch / 4-update smoke as successful IL convergence.

## Acceptance and branch lifecycle

1. One-step real-data train/optimizer/checkpoint smoke: **passed**.
2. Independent `smoke_02_bc_checkpoint.py` strict reload: **passed**.
3. Feature ownership: `feature/rl100-il` (not `feature/rl100-data`).
4. The validated one-step capability is currently owned by
   `feature/rl100-il`; the new `gpt/rl100-il-resume` regression must
   pass server tests before integration into that feature branch.
   When all IL requirements are validated, **Squash Merge**
   `feature/rl100-il` → `arbim/dev`; never merge `gpt/**` directly.
   PPO/IQL/Dynamics and online MuJoCo remain separate later batches.
