# Feature 03 — Native RL1002D BC / IL

This feature adds a thin training entrypoint. No LeRobot-to-Zarr conversion,
Offline RL buffer, PPO, Critic, Dynamics, Kuavo robot mapping, or changes to
the pinned RL-100 submodule.

## Source responsibilities

- LeTools CVT: end-effector configuration, state/action order and dimensions.
- Feature 02: LeRobot v3 metadata -> shape_meta and native time windows.
- LeRobot: EpisodeAwareSampler for valid training anchors.
- RL1002D: original ResNet18/MultiImageObsEncoder, scheduler, diffusion UNet,
  LinearNormalizer, compute_loss() and optional EMAModel.
- This feature: sampling, native actor construction, optimizer, loop and checkpoint.

## Episode boundary contract

For n_obs_steps=To and horizon=H, actions are indexed from t-(To-1)
through t+H-To; RL1002D has no_pre_action=True and crops the first To-1.
The valid supervision interval is t through t+H-To.

The DataLoader uses LeRobot EpisodeAwareSampler with
drop_n_last_frames = H - To. Before any update, the batch is checked:
action_is_pad[:, To-1:] must contain no True entries. Padding in the
preceding observation window is valid, and the first (To-1) actions are
discarded inside the native actor. Unlike the LeRobot ACT, the original
RL1002D compute_loss does not use action_is_pad. Its condition_mask is a
different mask and must not be confused with episode boundaries.

For the smoke config H=9, To=2: 7 episode-end anchors are excluded.
The denoising action sequence has length 8, matching Feature01 testing.

## Normalizer

RL-100 native LinearNormalizer.fit(..., mode="limits") receives the
LeRobot meta/stats.json extrema for observation.state and action. This
ensures the 16D/26D/36D/etc joint input/output maps to native [-1,1]
without hardcoding robot DOFs. RGB stays in [0,1] and uses the original
MultiImageObsEncoder ImageNet transform. An unavailable or invalid state
or action minimum/maximum raises an error instead of silently clipping.

Since native fit is run on the two extrema vectors only, the normalizer
input_stats.mean/std are not meaningful data-set means/stds. Use LeRobot
metadata stats for reporting; min/max (and thus the normalization mapping)
are correct. The feature uses whole-dataset stats; no train/validation
split is introduced.

## Checkpoint

Saved as post_traing/outputs/<run>/checkpoints/latest.pt in the smoke case.
Includes model state_dict (with normalizer), optimizer, native LR scheduler,
optional native EMA, config, state/action names and camera key mapping.
It is a training checkpoint, not a LeTools deployment package yet.

This minimal feature intentionally does not claim exact resume, best-model
selection, train/val split, or deployment readiness. Run each trial with a
fresh output directory. The small 128px ResNet preprocessing, shared RGB
backbone, and reduced UNet in config/bc.yaml are SMOKE settings only.
Do not interpret one-step BC as a trained policy.

## Test gate: run only on feature/rl100-bc-v2

Use the existing 10-bag LeRobot v3.0 directory. All test scripts live in
post-rl/test, and logs remain under the previously selected
post_traing/outputs directory. No new dependencies are introduced here.

    git fetch origin
    git switch feature/rl100-bc-v2
    git pull --ff-only origin feature/rl100-bc-v2
    mkdir -p post_traing/outputs
    set -o pipefail

    PYTHONPATH=post-rl/src python -m unittest discover \
        -s post-rl/test -p test_bc_sampler.py -v \
        2>&1 | tee post_traing/outputs/feature03_unit.log

    python post-rl/test/smoke_bc_real.py \
        --config post-rl/config/bc.yaml \
        --output-dir post_traing/outputs/feature03_bc_smoke \
        2>&1 | tee post_traing/outputs/feature03_bc_smoke.log

The real smoke must show a finite BC loss, a completed optimizer step,
a successfully loaded checkpoint, nonempty native action normalizer and
metadata-derived state/action dimensions. Do not merge to arbim/dev until
the feature-branch runtime tests pass.
