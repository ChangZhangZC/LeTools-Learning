# Local ARBiM / RL-100 data directory

This directory is the **single local dataset root** for the LeTools-Learning
integration. It is not a Git-tracked dataset repository: large raw data,
images, Zarr caches, staging rollouts, and generated videos are ignored by
`data/.gitignore`. Keep code/configs in their existing locations.

## Layout (create on the training host)

```text
data/
  lerobot/                        # authoritative LeRobot trajectories
    icra_task1_10bags/           # example: copy contents of lerobot_v3.0 here
  zarr/                           # derived, recreatable RL cache
    icra_task1_10bags.zarr/
  flywheel/                       # recorded rollouts and per-round provenance
    round_0001/
      staging/                    # raw T-action / T+1-observation rollout NPY
      lerobot/                    # curated LeRobot archive
      offline_rl.zarr/            # derived RL cache (only validated reward)
      manifest.json               # per-round provenance / reward status
```

`outputs/` remains separate and stores **run artifacts** (training
checkpoints, evaluation video/logs, smoke-test reports). Rollout staging and
LeRobot datasets are stored under `data/`, not under `outputs/test/`.

## Copy the existing 10-bag test dataset (server-side commands)

From the LeTools-Learning root on the same server:

```bash
mkdir -p data/lerobot/icra_task1_10bags data/zarr data/flywheel
rsync -a --info=progress2 \\
  /root/bayes-tmp/kuavo_dataset/icra_task1_lerobot_jiayi_10bags/lerobot_v3.0/ \\
  data/lerobot/icra_task1_10bags/
test -f data/lerobot/icra_task1_10bags/meta/info.json
```

This **copies**, rather than deletes/moves, the source dataset. Inspect disk
space before copying. After validating the destination, any cleanup of the
original is a separate deliberate action. If there is a prior successful
`outputs/test/icra_task1_10bags.zarr`, it may be moved manually to
`data/zarr/`; otherwise regenerate it from the copied LeRobot source.

## Current conversion example

```bash
python post_training/src/arbim_rl100/data/lerobot_to_zarr.py \\
  --lerobot-root data/lerobot/icra_task1_10bags \\
  --output data/zarr/icra_task1_10bags.zarr \\
  --reward-mode terminal_proxy
```

`terminal_proxy` is only for testing the conversion contract. Its Zarr
contains `reward_validated=false` and must not be consumed as a verified
reward dataset for offline PPO. For real RL, use `recorded_sparse` with a
validated per-episode reward JSON and retain the manifest.

## Data Wheel migration note

The old `kdc_ARBiM` system currently writes staging beneath its evaluation
output directory; that behavior has **not** been ported to LeTools. The
migration target is `data/flywheel/round_<id>/staging`, while
evaluation logs, videos and progress remain under `outputs/eval`.
See `post_training/FLYWHEEL_MIGRATION.md` for the source-code behavior and
integration plan. No actual old staging data was copied by this Git commit.
