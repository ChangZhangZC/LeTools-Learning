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

## Stop by Steps and/or Epochs

The baseline trainer supports two optional limits in
`configs/train/rl100_dp_2d.yaml`:

    training:
      steps: 100000  # Set null to disable optimizer-step cap.
      epochs: null   # Set a positive integer to enable full-data-pass cap.

- Steps only: epochs=null, training ends after the specified optimizer updates.
- Epochs only: steps=null, training ends after the specified number of complete
  passes through the **training subset's valid action windows**, NOT raw
  frames or held-out validation episodes.
- Both enabled: stops at whichever cap comes first. Both null is invalid.
- Epoch size is ceil(valid_train_windows / batch_size). The final partial
  batch of each epoch counts as an update and is not discarded.
- The chosen effective training step total is used for the original
  cosine/warmup scheduler, final validation, and final checkpoint. The
  equivalent epoch count is recorded in normal training metrics.
- Resume must keep both caps, data split and Batch Size unchanged: the
  scheduler budget cannot be redefined partway through an existing run.
  Previously saved steps-only checkpoints remain compatible when the
  epochs cap is unset.

Example: 1,001 training windows / batch_size=8 means 126 optimizer
steps per epoch. steps=500 and epochs=3 stops at min(500, 378)=378.
With a very small cap (e.g. 1 epoch), consider reducing the configured
lr_warmup_steps: otherwise the run may end entirely in warmup.

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
