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

## Acceptance and branch lifecycle

1. One-step real-data train/optimizer/checkpoint smoke: **passed**.
2. Independent `smoke_02_bc_checkpoint.py` strict reload: **passed**.
3. Feature ownership: `feature/rl100-il` (not `feature/rl100-data`).
4. Squash merge this **validated BC-only training capability** into
   `arbim/dev`. Continued full-epoch training and deployment evaluation
   require their own tests and should not be reported as validated here.
   PPO/IQL/Dynamics and online MuJoCo remain separate later batches.
