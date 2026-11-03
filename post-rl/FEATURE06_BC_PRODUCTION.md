# Feature 06 — Production BC baseline for native RL1002D

Status: implementation prepared; **not yet verified on target GPU**.
This work upgrades the Feature03 minimal BC trainer without touching its
one-step smoke. The model algorithm stays the unmodified RL-100 RL1002D.

## File and compatibility boundary

- NEW: post-rl/src/rl100_compat/train_bc_production.py
- NEW: post-rl/config/bc_train.yaml
- NEW: post-rl/test/test_bc_production.py
- NEW: post-rl/test/smoke_bc_production.py
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
steps and other parameters are NOT tuned for Kuavo.

Native RL1002D has no_pre_action=True:
  modeled_denoising_length = horizon - n_obs_steps + 1.

For len(down_dims) blocks, the modeled length must be divisible by
2**len(down_dims). The starter config uses horizon=17, To=2, resulting
in 16 modeled action steps, with n_action_steps=8. This differs from the
deployment adapter's execution_horizon=1 for consecutive observations.
Do not confuse diffusion timestep count with optimizer step count.

10 bags are sufficient for smoke but cannot support conclusions about
generalization. Before a real production run, choose your dataset,
total step budget, model size and GPU/batch budget separately.

## Verify on GPU (run only on feature branch)

    cd /root/bayes-tmp/changzhang/LeTools-Learning
    conda activate letools_chang
    git fetch origin
    git switch feature/rl100-bc-production-v2
    git pull --ff-only origin feature/rl100-bc-production-v2
    git submodule update --init third_party/lerobot third_party/rl100
    mkdir -p post_traing/outputs
    set -o pipefail

    # Pure fake-data unit tests, deterministic full/partial resume
    PYTHONPATH=post-rl/src python -m unittest discover \
      -s post-rl/test -p test_bc_production.py -v \
      2>&1 | tee post_traing/outputs/feature06_unit.log

    # Real 10-bags: 2 optimizer steps -> resume to 3 -> best EMA deploy
    python post-rl/test/smoke_bc_production.py \
      --device cuda:0 \
      --output-dir post_traing/outputs/feature06_bc_smoke \
      2>&1 | tee post_traing/outputs/feature06_bc_smoke.log

    # Verify the original Feature03 BC trainer is unchanged
    python post-rl/test/smoke_bc_real.py \
      --config post-rl/config/bc.yaml \
      --output-dir post_traing/outputs/feature03_bc_smoke_f06_regression \
      2>&1 | tee post_traing/outputs/feature06_feature03_regression.log

The real-data smoke uses the dataset root currently configured in
post-rl/config/bc.yaml. A missing dataset or model dependencies will
fail explicitly. Each test uses a fresh output dir; reruns should choose
another directory, not overwrite a previous checkpoint.

## Full training (after testing and parameter decision)

    python post-rl/src/rl100_compat/train_bc_production.py \
      --config post-rl/config/bc_train.yaml

To resume interrupted training, preserve output.dir, keep all checked
settings identical, and use --resume (or training.resume=true in YAML).

Artifacts:

    <output.dir>/resolved_config.yaml
    <output.dir>/metrics.jsonl
    <output.dir>/checkpoints/latest.pt
    <output.dir>/checkpoints/best.pt
    <output.dir>/checkpoints/step_XXXXXXXX.pt (only if enabled)

Use best.pt for Kuavo ModelServerAdapter inference; selected EMA
weights are in its model field. This is NOT physical-robot readiness:
later Kuavo OpenSource simulation and EEF semantic checks are required.

## Merge policy

All GPU checks and original Feature03 regression must pass before
Squash Merge from feature/rl100-bc-production-v2 to arbim/dev.
The main and arbim/dev-deprecated branches remain untouched.
