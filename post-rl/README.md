# Native RL1002D — BC training and deployment (LeTools)

The native RL-100 diffusion actor is pinned under third_party/rl100.
LeRobot v3 loading, Kuavo deployment and BC training live in LeTools
compatibility code under post-rl/. The original RL100, LeRobot and
kuavo_model/train.py are not modified.

## Production entrypoints

- Config: post-rl/config/bc_train.yaml
- Trainer: post-rl/src/rl100_compat/train_bc_production.py
- Native RL1002D factory/helper functions: post-rl/src/rl100_compat/train_bc.py
- LeRobot dataset contract: post-rl/src/rl100_compat/lerobot_data.py
- Optional logging: post-rl/src/rl100_compat/bc_logging.py
- Kuavo model server adapter: kuavo_server/adapters/rl100_2d.py
- Selected-episode extraction utility: post-rl/test/extract_lerobot_test_subset.py
- Unit tests: post-rl/test/test_*.py

The historical standalone smoke_*.py programs were removed after
Feature01–06 acceptance; they are not part of the production workflow.
The original bc.yaml and train_bc.py remain for compatibility with
unit tests and the production model-construction helpers. Historical
Feature02/03/04 notes are archival and may mention removed programs.

## Dependency boundary

The pinned RL-100 submodule's nested submodules may contain restricted
URLs. Initialize required top-level checkouts, not recursively:

    git submodule update --init third_party/rl100 third_party/lerobot

Environment-specific additional requirements are tracked in
post-rl/requirements-rl100.txt. Do not downgrade the existing LeTools
PyTorch/CUDA stack just to match historical RL100 package versions.

## Train (working directory: repository root)

Update the LOCAL LeRobot v3 dataset root in bc_train.yaml first.
Current production model config: 2 observation frames, raw horizon=33,
32 modeled actions, 16 selected model actions, no random crop.
The current Kuavo adapter executes one action per inference when
two-frame history must remain consecutive.

    python post-rl/src/rl100_compat/train_bc_production.py \
      --config post-rl/config/bc_train.yaml

Default per-run output matches LeTools ACT/DP conventions:

    outputs/train/rl100_2d_YYYYMMDD_HHMMSS/
      resolved_config.yaml
      metrics.jsonl
      checkpoints/latest.pt
      checkpoints/best.pt
      tensorboard/          (only if tensorboard.enable=true)
      wandb_run_id.txt      (only if wandb.enable=true)

You may pass --no-timestamp to use the exact YAML output.dir.
The programmatic train(cfg) function also uses output.dir exactly,
which is important for test reproducibility.

Resume a specific run with the ORIGINAL settings and exact timestamped path:

    python post-rl/src/rl100_compat/train_bc_production.py \
      --config post-rl/config/bc_train.yaml \
      --resume \
      --output-dir outputs/train/rl100_2d_YYYYMMDD_HHMMSS

Do not resume best.pt: it is a deployment-only validation-selected
actor. Only latest.pt includes optimizer/scheduler/EMA/RNG state.
Do not change total training Steps/Batch Size or architecture when
resuming an existing run.

## Logging audit and options

- Always on: stdout + metrics.jsonl + resolved_config.yaml.
- JSONL train: optimizer step, loss, LR, equivalent epoch, elapsed time.
- JSONL val: held-out diffusion-denoising loss and EMA/raw selection.
- Optional WandB: set wandb.enable=true (requires installed wandb
  and login or mode=offline). Project is configured in YAML; a saved
  wandb_run_id.txt lets resume reuse the same WandB run.
  Train/val series use trainer_step as the x-axis.
- Optional TensorBoard: set tensorboard.enable=true (requires installed
  tensorboard). Use: tensorboard --logdir outputs/train
  The writer uses purge_step on resume so post-checkpoint entries are
  not double-counted.
- No task success metrics are produced without a Kuavo rollout or
  environment runner; validation loss measures BC denoising only.
- Neither logging package is imported if disabled.

LeTools ACT currently enables WandB by default; LeTools DP disables
WandB by default. BC production follows the DP default (disabled)
while allowing explicit enablement. Native ACT/DP do not provide a
built-in TensorBoard toggle in their examined YAML.

## Local unit checks (GPU not required for logging contracts)

    PYTHONPATH=post-rl/src python -m unittest discover \
      -s post-rl/test -p test_bc_logging.py -v

    PYTHONPATH=post-rl/src python -m unittest discover \
      -s post-rl/test -p test_bc_production.py -v

These test synthetic trainer/logging contracts. Actual full-model
VRAM usage, 600-episode training performance and Kuavo success rates
still require separate measurements before a full run.

## Historical context

Original native integration tasks and acceptance details are retained
in FEATURE02_LEROBOT.md, FEATURE03_BC.md, FEATURE04_DEPLOY.md
and FEATURE06_BC_PRODUCTION.md. Refer to the current production entry
above for live commands rather than earlier Smoke Test scripts.
