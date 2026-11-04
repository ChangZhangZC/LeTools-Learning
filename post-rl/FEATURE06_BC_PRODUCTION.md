# Feature 06 — Production BC baseline for native RL1002D

Status: accepted on the pro6000 GPU using the 10-Episode subset (1,501 frames).
The native BC model and core trainer behavior were verified with real data,
EMA, training-state resume and adapter inference. Full 600-Episode training
and robot performance are NOT evaluated. Previous smoke scripts were cleaned
up on the subsequent maintenance branch. The model remains unmodified RL1002D.

## File and compatibility boundary

- NEW: post-rl/src/rl100_compat/train_bc_production.py
- NEW: post-rl/config/bc_train.yaml
- NEW: post-rl/test/test_bc_production.py
- Optional scalar logging: post-rl/src/rl100_compat/bc_logging.py
- This document: post-rl/FEATURE06_BC_PRODUCTION.md

UNCHANGED: original RL1002D, RL100 compute_loss(), DDIM scheduler,
ResNet/MultiImageObsEncoder, native EMA, native LeRobotDataset, native
EpisodeAwareSampler, kuavo_model/train.py, server/PolicyClient adapters,
Feature03 train_bc.py and bc.yaml, and all upstream gitlinks.

The new entrypoint reuses Feature03 build_policy, native compute_loss,
assert_valid_action_targets, _payload, _to_device, _save_checkpoint, and
make_episode_sampler. No alternative Diffusion Policy is introduced.

## BC contracts implemented

1. Episode-heldout validation: deterministic full-Episode train/val
   split without frame overlap. Native EpisodeAwareSampler drops
   horizon-n_obs_steps tail anchors in both splits. At least two episodes
   and one valid anchor per split are required. Validation BC denoising
   loss is NOT Kuavo task success rate.
2. Leak-free normalization: original RL100 LinearNormalizer.fit with
   mode=limits, using min/max from raw TRAIN episode state/action frames
   only. Held-out stats are excluded. Read the raw parquet columns
   without decoding all RGB frames for statistics.
3. Native training: RL100 compute_loss -> backward -> AdamW -> native LR
   scheduler -> optional native EMA. Optional gradient clipping.
   Deterministic minibatch order is a function of (seed, global step),
   so resume does not depend on DataLoader's prefetched cursor.
4. EMA validation: uses EMA actor when enabled. Native RL100 EMA averages
   parameters; the trainer synchronizes non-parameter buffers, including
   BatchNorm running statistics. A fixed validation RNG seed is restored
   after evaluation, for comparable diffusion-noise losses.
5. Two checkpoint roles: checkpoints/latest.pt contains full training
   state (actor+normalizer, optimizer, scheduler, EMA, EMA update counter,
   global step, Python/NumPy/Torch/CUDA RNG, config signature and
   train/validation Episode IDs). checkpoints/best.pt is inference-only,
   with its model state_dict set to validated EMA weights or raw policy
   when EMA is disabled. Its camera map and shape metadata match the
   unchanged RL1002D ModelServerAdapter contract. Do NOT resume best.pt.
6. Safe resume: resume=true requires latest.pt, validates model/data/
   optimizer/budget/split compatibility, and restores the native RL100
   normalizer before constructing AdamW parameter groups. This matters
   because RL100 DictOfTensorMixin dynamically rebuilds normalization
   tensors on load. The complete model/EMA is moved to the target GPU
   after state restoration. Logs beyond the restored step are trimmed.
   A cosine schedule's original total training step budget is immutable.
7. Logging: metrics.jsonl contains train loss, LR, elapsed time, and
   heldout validation loss with the selected weight type.
   resolved_config.yaml captures effective configuration.

This is a single-process trainer. It does not add DDP, AMP, online RL,
automatic Kuavo simulation, or robot task success metrics. Exact
bitwise reproducibility is not guaranteed with random transforms inside
multi-worker DataLoaders. Use num_workers=0 for strict resume tests.

## Config guidance

post-rl/config/bc.yaml remains Feature03's one-step test preset.
post-rl/config/bc_train.yaml is a starter production preset; its 100000
steps and other parameters are NOT tuned for Kuavo. It now uses
n_obs_steps=2, horizon=33 (32 modeled action steps), n_action_steps=16,
and disables Random Crop in the existing build_policy() helper.

Native RL1002D has no_pre_action=True:
  modeled_denoising_length = horizon - n_obs_steps + 1.

For len(down_dims) blocks, the modeled length must be divisible by
2**len(down_dims). The starter config uses horizon=33, To=2, resulting
in 32 modeled action steps, with n_action_steps=16. This differs from the
deployment adapter's execution_horizon=1 for consecutive observations.
Do not confuse diffusion timestep count with optimizer step count.

10 bags are sufficient for smoke but cannot support conclusions about
generalization. Before a real production run, choose your dataset,
total step budget, model size and GPU/batch budget separately.

## Current unit tests

The old standalone smoke scripts have been retired. Current maintained tests:

    PYTHONPATH=post-rl/src python -m unittest discover \
      -s post-rl/test -p test_bc_production.py -v

    PYTHONPATH=post-rl/src python -m unittest discover \
      -s post-rl/test -p test_bc_logging.py -v

Historical 10-bag/600-bag extraction notes are archival. Use the
production entrypoint and latest README for current commands.

## Full training (after testing and parameter decision)

    python post-rl/src/rl100_compat/train_bc_production.py \
      --config post-rl/config/bc_train.yaml

To resume interrupted training, preserve output.dir, keep all checked
settings identical, and use --resume (or training.resume=true in YAML).

Default output matches native LeTools ACT/DP naming. Fresh CLI runs
append _YYYYMMDD_HHMMSS to outputs/train/rl100_2d. For exact resume,
provide that run path using --output-dir and --resume.

Optional WandB and TensorBoard sinks (disabled by default) are configured
through the wandb and tensorboard YAML sections; JSONL remains always on.

Artifacts:

    <output.dir>/resolved_config.yaml
    <output.dir>/metrics.jsonl
    <output.dir>/checkpoints/latest.pt
    <output.dir>/checkpoints/best.pt
    <output.dir>/checkpoints/step_XXXXXXXX.pt (only if enabled)

Use best.pt for Kuavo ModelServerAdapter inference; selected EMA
weights are in its model field. This is NOT physical-robot readiness:
later Kuavo OpenSource simulation and EEF semantic checks are required.

## Integration status

Feature06 has already been Squash Merged into arbim/dev via PR #21.
The logging and smoke-cleanup maintenance branch is separate and remains
unmerged pending review and updated unit tests.
