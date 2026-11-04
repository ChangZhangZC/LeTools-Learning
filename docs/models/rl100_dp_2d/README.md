# RL100 DP-2D Baseline in LeTools

This is the native RL-100 RL1002D image-based Diffusion Policy,
not LeRobot's built-in Diffusion Policy and not an ACT model.
The original RL-100 source is pinned in third_party/rl100 and is not edited.

## Layout

| Component | Location |
| --- | --- |
| BC training YAML | configs/train/rl100_dp_2d.yaml |
| BC training entrypoint | kuavo_model/rl100_dp_2d/train.py |
| RL100 actor factory | kuavo_model/rl100_dp_2d/policy_factory.py |
| BC helpers | kuavo_model/rl100_dp_2d/bc_utils.py |
| LeRobot v3 dataset adapter | kuavo_model/rl100_dp_2d/dataset.py |
| Optional WandB/TensorBoard sinks | kuavo_model/rl100_dp_2d/metrics.py |
| Extra dependencies | kuavo_model/rl100_dp_2d/requirements.txt |
| Kuavo ModelServerAdapter | kuavo_server/adapters/rl100_dp_2d.py |
| Unit tests and subset tool | tests/rl100_dp_2d/ |
| Native upstream actor | third_party/rl100/RL-100/rl_100/policy/rl100_2d.py |

Future IQL/Critic/PPO/transition dynamics code belongs in post_training/,
not in the BC baseline package.

## BC Training

Run from the LeTools repository root:

    git submodule update --init third_party/rl100 third_party/lerobot

    python -m kuavo_model.rl100_dp_2d.train \
      --config configs/train/rl100_dp_2d.yaml

Update data.root to an existing local LeRobot v3 dataset before use.
Default on pro6000 points to the 600-Episode full dataset, not the
10-Episode smoke subset. Model settings: n_obs_steps=2, horizon=33,
no_pre_action=True implies 32 modeled actions, n_action_steps=16,
224 resize with no random crop. This is a configuration, not proof
of full-run convergence.

New CLI runs use the same timestamp convention as LeTools ACT/DP:

    outputs/train/rl100_dp_2d_YYYYMMDD_HHMMSS/
      resolved_config.yaml
      metrics.jsonl
      checkpoints/latest.pt
      checkpoints/best.pt
      tensorboard/           (when enabled)
      wandb_run_id.txt       (when enabled)

To resume use the exact old run path (no timestamp added on resume):

    python -m kuavo_model.rl100_dp_2d.train \
      --config configs/train/rl100_dp_2d.yaml \
      --resume \
      --output-dir outputs/train/rl100_dp_2d_YYYYMMDD_HHMMSS

--no-timestamp disables the automatic suffix for a fresh CLI run.
Only latest.pt contains optimizer, scheduler, EMA and RNG states.
best.pt contains the validated actor for inference, not resume.

## Logging

Always enabled: console, metrics.jsonl, resolved_config.yaml.
JSONL: train loss, LR, equivalent epoch, elapsed time; held-out
validation denoising loss (selected EMA/raw).

Optional YAML sections:

    wandb:
      enable: false
      project: "lerobot_rl100_dp_2d"
      entity: null
      mode: "online"
    tensorboard:
      enable: false

Install wandb/tensorboard separately if enabled. WandB keeps one Run ID
across resume and logs by optimizer step; TensorBoard uses purge_step
on resume. Disabled sinks require no extra packages. Validation loss
is NOT the Kuavo task success rate.

## Deployment

The canonical Kuavo ModelServerAdapter name is rl100_dp_2d
(historical name: rl100_2d). Use the existing serve.py entrypoint with
--adapter rl100_dp_2d and --checkpoint pointing to best.pt.

With n_obs_steps=2 the adapter still restricts execution_horizon to 1
because the PolicyClient caches actions without sending intermediate
observations. Changing n_action_steps to 16 does not change this.

## Verification before PR approval

    python -m unittest discover \
      -s tests/rl100_dp_2d -p 'test_*.py' -v

    python -m kuavo_model.rl100_dp_2d.train --help

The previous 10-Episode GPU smoke succeeded on the old paths. These
new paths and the refactored imports require regression checks on
pro6000 before merging the feature branch into arbim/dev.
For original integration decisions, see archive/ in this directory.
