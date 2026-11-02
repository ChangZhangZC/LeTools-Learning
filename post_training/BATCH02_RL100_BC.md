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

## Multi-epoch Resume hardening (server acceptance passed)

**Development branch:** `gpt/rl100-il-resume`, fast-forwarded into
`feature/rl100-il` as the **feature integration/testing branch**.
The multi-epoch resume regression **passed on the server**:
`epoch=2`, `global_step=4`; model, normalizer, optimizer,
scheduler, EMA, RNG and counters match the uninterrupted run exactly.
Only after all IL functionality is validated may `feature/rl100-il`
be **Squash Merged** into `arbim/dev`. No direct `gpt/**` → dev merge.

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

## Resume bug: custom RL-100 Normalizer device after CPU checkpoint load

**Observed:** interrupted training saved `epoch=1, global_step=2`
and `EMA step=2`; after loading on `cuda:0`, the next
`RL1002D.compute_loss()` failed on
`noisy_trajectory[condition_mask] = cond_data[condition_mask]` with
CPU/CUDA index mismatch. This occurred **after** successful strict
weights and optimizer/scheduler restoration.

**Source-based root cause:** The pinned
`rl_100.model.common.dict_of_tensor_mixin.DictOfTensorMixin._load_from_state_dict()`
reconstructs `self.params_dict` directly using tensors provided
by `torch.load(map_location="cpu")`. It does **not** preserve the
GPU placement of the previously fitted normalizer parameters.
`rl_100.model.common.normalizer._normalize()` moves action tensors
to `scale.device`, so restored actions become CPU tensors while
`LowdimMaskGenerator` stays on the model GPU.

**Minimal remedy outside upstream** in `load_checkpoint()`: after
each strict `model.load_state_dict()` / `ema_model.load_state_dict()`,
move that entire model back to the previously selected device;
restore AdamW state after the actor's device move.
Keep native RL1002D `compute_loss()`, mask generator,
normalization math, EMA and PPO code unchanged.
The updated `smoke_02_bc_checkpoint.py` now checks both normalizers'
devices **and computes native BC loss after reload**; testing
`predict_action()` alone was insufficient to detect this failure.

This fix **passed server verification**, including checking model/EMA
normalizers' target-device placement and computing the native BC loss
after restoring the checkpoint. The saved epoch-1
checkpoint from the failed attempt remains usable: the exception
happened before any optimizer step or checkpoint replacement.
After updating the feature branch:

```bash
# Run from repo root in letools_chang.
git fetch origin
git switch feature/rl100-il
git pull --ff-only origin feature/rl100-il

# First check the previously failing epoch-1 checkpoint independently.
python post_training/tests/smoke/smoke_02_bc_checkpoint.py \
  --checkpoint outputs/train/rl100_bc_resumed/checkpoints/latest.ckpt \
  --device cuda:0 --expected-epoch 1 --expected-step 2

# Continue epoch 1 (zero-indexed); no need to repeat earlier updates.
python post_training/src/arbim_rl100/train_bc.py \
  --config post_training/configs/bc/rl100_2d_bc_resume_smoke.yaml \
  --output-dir outputs/train/rl100_bc_resumed \
  --resume outputs/train/rl100_bc_resumed/checkpoints/latest.ckpt

python post_training/tests/smoke/smoke_02_bc_checkpoint.py \
  --checkpoint outputs/train/rl100_bc_resumed/checkpoints/latest.ckpt \
  --device cuda:0 --expected-epoch 2 --expected-step 4

# If uninterrupted run was already completed, compare them:
python post_training/tests/smoke/smoke_02_bc_resume_compare.py \
  --continuous outputs/train/rl100_bc_continuous/checkpoints/latest.ckpt \
  --resumed outputs/train/rl100_bc_resumed/checkpoints/latest.ckpt
```

## Stable validation and best-checkpoint selection (server smoke passed)

Implementation integrated into `feature/rl100-il` after review. **The real-Zarr 3-epoch smoke and best-checkpoint test passed on the server.** This does not yet establish production IL convergence or deployment performance. Do not Squash Merge into `arbim/dev` until remaining IL functionality acceptance is complete.

Pooled BC validation uses the original `RL1002D.compute_loss()`,
without changing the policy or converting validation loss into a task
success score. Diffusion `compute_loss()` samples noise/timesteps,
so a different random draw each epoch adds noise to model comparisons.
The BC-only orchestrator now encloses the validation loader/forward
pass in `torch.random.fork_rng()` and seeds it with
`data.seed + 1000003`. This fixes validation sampling across epochs
**without consuming training RNG state** and preserves deterministic
Resume behavior. Val split is still episode-level and no validation
frames are used for normalizer fitting.

- Record `train_loss`, `val_loss`, `best_val_loss`, `lr`,
  `global_step`, `epoch` in `output.dir/metrics.jsonl`.
- `checkpoints/latest.ckpt` is written **each completed epoch**,
  atomically, for safe epoch-boundary resume. Currently only
  `training.checkpoint_every=1` is supported (explicit error otherwise).
- `checkpoints/best.ckpt` is written atomically when deterministic
  held-out `val_loss` improves strictly. It retains the same native
  actor, EMA, optimizer, scheduler and normalizer states as the other
  BC-only checkpoints. The best value is also saved in
  `pickles.best_val_loss` for consistent Resume selection.
- Best BC loss is a supervised reconstruction/diffusion-noise metric,
  **not** task success, and `best.ckpt` is **not** evidence of real robot
  policy quality.

### Run a small validation trial on the real 10-bag cache

This `rl100_2d_bc_validation_trial.yaml` is a **functionality test**:
3 epochs × 12 optimizer steps, batch size 4, full validation windows,
warmup 10, 84×84 resized RGB. This is **not** the final IL recipe
and the 10-episode dataset is far too small for robust robot-task
performance conclusions. Do not change the existing validated
one-step and resume configs.

```bash
conda activate letools_chang
git fetch origin
git switch feature/rl100-il
git pull --ff-only origin feature/rl100-il

python post_training/src/arbim_rl100/train_bc.py \
  --config post_training/configs/bc/rl100_2d_bc_validation_trial.yaml

python post_training/tests/smoke/smoke_02_bc_best.py \
  --output-dir outputs/train/rl100_bc_validation_trial
```

**Observed server acceptance:** `epochs=3`, `steps_per_epoch=12`,
`global_step=36`; training losses `[1.1090234, 1.0637234, 1.0525346]`,
held-out `val_loss` `[1.0654494, 1.0489187, 1.0457117]`.
The smoke reported `[PASS] metrics epochs=3, latest_epoch=3`,
`[PASS] best validation epoch=3, loss=1.045712`, and
confirmed both Checkpoints contain native RL1002D and optimizer states.
The cosine LR reaching zero after its final scheduled step is
expected. These are only three short epochs on eight training
episodes; no success-rate claims are supported.

The existing 2-epoch exact Resume acceptance can be repeated on fresh
output directories with the new metric/best code, and
`smoke_02_bc_resume_compare.py` now also compares saved
`best.ckpt` states and per-epoch `metrics.jsonl`. Do not overwrite
the user's earlier acceptance checkpoints; use fresh
`--output-dir` for regressions.

**Gates before claiming production IL readiness:** the validation
trial must pass; data scale/train-val split, model resolution,
`max_train_steps_per_epoch`, action horizon and inference settings
must be separately selected for the actual IL experiment, and a
MuJoCo/Kuavo rollout benchmark must establish execution success.
The old `terminal_proxy` cache is acceptable for BC-only training,
not for PPO rewards.

## Acceptance and branch lifecycle

1. One-step real-data train/optimizer/checkpoint smoke: **passed**.
2. Independent `smoke_02_bc_checkpoint.py` strict reload: **passed**.
3. Feature ownership: `feature/rl100-il` (not `feature/rl100-data`).
4. One-step BC, exact multi-epoch Resume, and the new three-epoch
   Validation/Best-Checkpoint trial **passed server tests**. The
   combined new-validation + resumed-training equality regression
   remains to be rechecked on fresh outputs. Once IL functionality
   is fully accepted, **Squash Merge**
   `feature/rl100-il` → `arbim/dev`; never merge `gpt/**` directly.
   PPO/IQL/Dynamics and online MuJoCo remain separate later batches.
