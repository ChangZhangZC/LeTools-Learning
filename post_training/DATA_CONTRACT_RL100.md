# Batch 01 — ARBiM → RL1002D data contract (first sub-batch)

Status: **interface specification and synthetic tests only**. The converter for LeRobot 0.5.2, Zarr reader, episode sampler and real-data integration tests are separate work items. Do not treat a passing synthetic contract test as end-to-end migration.

## Project-wide migration rules (agreed 2026-11-02)

**Default: copy/reuse source code, not rewrite algorithms or redesign behavior.**
The native RL-100 code is pinned and kept unmodified in
`third_party/rl100`. Existing non-model utilities (reward, recorder,
samplers, dataset merge, reporting) should be copied or reused from
`ChangZhangZC/kdc_ARBiM` when their contracts are compatible.
LeTools' existing dataset/deploy entrypoints and other policy behavior
must remain intact.

The permitted change budget is intentionally narrow:

1. **Necessary adaptations only:** LeRobot version and naming differences,
   2D RGB observation/action dimensions and shapes, filesystem locations,
   LeTools deployment entrypoint differences, and demonstrable runtime/API
   incompatibilities. Use a thin adapter rather than modifying upstream
   algorithm behavior.
2. **No opportunistic additions/removals:** do not delete, silently simplify,
   or replace RL-100 or old ARBiM helper behavior; do not add an invented
   training algorithm, new loss, reward definition, or infrastructure feature.
   A tiny compatibility function, test, config or safety check is allowed
   **only when required** to get the preserved contract running.
3. **Minimal diff and traceability:** for every port, identify the original
   source path and ref, retain function/parameter/return semantics where
   possible, and document only the necessary changed lines and why.
   Do not reformat or refactor unrelated code during migration.
4. **Differential validation:** compare the port's outputs against original
   behavior on the same fixture when practical; then run isolated smoke
   tests, actual local data tests, and finally deployment tests as applicable.
5. **No premature environment migration:** RL-100's original simulator
   dependencies are not required for the current 2D/Offline pipeline.
   Restore them only when a specific exercised function needs them.
   Keep `third_party/rl100` unchanged.

**Execution order (Batch 00–08):**

- 00 native RL-100 runtime (integrated);
- 01 LeRobot/Zarr and original OfflineBuffer/SequenceSampler/Normalizer
  reuse (active);
- 02 native RL1002D BC training using upstream training logic;
- 03 existing LeTools simulation/deployment adaptation and IL baseline;
- 04 source-preserving IQL Critic and Dynamics;
- 05 source-preserving Offline PPO / offline post-training and evaluation;
- 06 port old ARBiM Data Wheel menu, progress, recorder, reward provenance,
  staging conversions, and merge/report (may be developed alongside 04–05
  after 03's simulation evaluation is available);
- 07 **last stage:** on-policy simulator Environment/Buffer integration;
- 08 **final batch:** Online PPO rollout, update, and evaluation.

Online PPO is **deferred until all prior offline and Data Wheel work is
validated** because it requires separate MuJoCo simulator step/reset,
termination/truncation, reward and on-policy rollout interfaces.
Do not modify MuJoCo online-environment contracts while carrying out
Batch 01–06. Ordinary simulator evaluation needed by Batch 03/06 remains
within scope.

**Repository layout:** keep existing scripts/modules in place unless
integration demonstrably requires relocation; configure through
`configs/post_training/` only when a YAML-backed entrypoint is needed.
Local persisted data lives under `data/{lerobot,zarr,flywheel}/`, whereas
run checkpoints, logs and evaluation media live under `outputs/`.
`data/` is ignored by Git.

## Canonical pipeline

```text
Kuavo Rosbag --LeTools/CvtRosbag2Lerobot.py--> LeRobot (canonical IL archive)
                                                   |
                              future: LeRobot 0.5.2 -> ARBiM Zarr
                                                   |
              Zarr/jpegs --OfflineBuffer/SequenceSampler--> ARBiM sequence
                                                   |
                                  to_rl100_bc_batch()
                                                   |
                          native RL1002D.compute_loss() / predict_action()
```

Source references (read-only): `ChangZhangZC/kdc_ARBiM@feature/flywheel`:
`post_training/DATA_CONTRACT.md`, `post_training/src/post_rl/data/{data_prepare,offline_buffer,offline_dataset}.py`.
Target is the unmodified `Lei-Kun/RL-100` commit `6b6afceb7eb12525398207b5d6e66908863dc385`.

## Explicit field mappings

| LeTools LeRobot field | Old ARBiM OfflineDataset field | RL1002D observation field |
| --- | --- | --- |
| `observation.state` | `obs.state` | `obs.agent_pos` |
| `observation.images.head_cam_h` | `obs.head_rgb` | `obs.rgb_head` |
| `observation.images.wrist_cam_l` | `obs.wrist_left_rgb` | `obs.rgb_left_hand` |
| `observation.images.wrist_cam_r` | `obs.wrist_right_rgb` | `obs.rgb_right_hand` |
| `action` | `action` | `action` |

Do not silently reinterpret image colorspace or action/state joint order. Preserve source metadata for state/action names, units, robot configuration, camera naming, FPS and preprocessing. State DOF and action DOF can be **different** (for example base velocity in action). **No hard-coded 16D or 28D**. Per-camera height/width can differ, but each configured model input shape must match its declared `shape_meta`.

## Tensor contract

The new `post_training/src/arbim_rl100/data/batch_adapter.py` accepts existing OfflineDataset-shaped samples or collated batches. For a batch of size `B`, observation history `To=n_obs_steps`, action horizon `H`, state dimension `Ds` and action dimension `Da`:

```text
Input:
  obs.state                [B,T,Ds]        float
  obs.head_rgb             [B,T,3,Hh,Wh]   uint8 or float[0,1]
  obs.wrist_left_rgb       [B,T,3,Hl,Wl]   uint8 or float[0,1]
  obs.wrist_right_rgb      [B,T,3,Hr,Wr]   uint8 or float[0,1]
  action                   [B,H,Da]        float

Output:
  obs.agent_pos            [B,To,Ds]       float32
  obs.rgb_head             [B,To,3,Hh,Wh]  float32[0,1]
  obs.rgb_left_hand        [B,To,3,Hl,Wl]  float32[0,1]
  obs.rgb_right_hand       [B,To,3,Hr,Wr]  float32[0,1]
  action                   [B,H,Da]        float32
```

The adapter **never** normalizes state/action, changes units, resizes/crops cameras, infers a joint mapping, drops action timesteps, calculates reward or crosses episode boundaries. `RL1002D` performs its own normalizer and internal `n_obs_steps-1` action slicing when `no_pre_action=True`; applying another slice here would be incorrect. The native vision encoder applies its own configured image transformations. `uint8 [0,255]` is converted exactly once to `float32 [0,1]`; floating-point inputs outside that range are rejected rather than guessed.

The existing ARBiM sequence sampler, not this adapter, is responsible for no-cross-episode windowing, beginning/end padding, train/validation split, and access to `next_obs`. Future BC Dataset and Offline PPO adapters must explicitly test padding and action-window alignment.

## Zarr and reward semantics: retained from old ARBiM

Old cache has `data/{state,next_state,action,next_action,reward,return,done,timeout,next_index}`, three `jpeg_bytes_v1` streams (`*_jpeg_data` and `*_jpeg_offsets`), optional Depth, and `meta/episode_ends`. The Zarr cache is **derived**, not the canonical archive. RGB-only remains the first training target; Depth archival policy must be decided independently when migrating the staging converter.

Old reward provenance distinguishes recorded versus computed `last_gripper_release_sparse` events and invalid reward labels. **Successful rollout does not imply valid RL reward**; do not auto-fill a missing/invalid reward. Existing `done` and `timeout` are both written as episode-end markers in the old converter; that is not a valid substitute for distinguishing true termination and truncation in future Online PPO. Retain this limitation in downstream adaptations.

Old `lerobot_v042_fast_merge.py` patch is version-specific. **Do not port it into LeRobot 0.5.2 without checking the current merger.** Avoid the deprecated `arbim_processed_npy_stream_v2` bridge when designing the next conversion step.

## Run the isolated contract test

```bash
python post_training/tests/smoke/smoke_01_rl100_batch_contract.py
```

This checks dynamic state/action dimensions, single/batched input, RGB layout/ranges, missing views, action horizon, and finite values. It does **not** test Zarr decoding, LeRobot dataset creation, reward labels, RL1002D loss or robot rollouts.

## Batch 01C — verbatim original ARBiM reader port

Source: `ChangZhangZC/kdc_ARBiM@feature/flywheel`,
`post_training/src/post_rl/data/{offline_buffer,sampler,offline_dataset}.py`.
Destination: `post_training/src/arbim_rl100/data/`, under the **same
filenames**. These three source files have identical Git blob SHA to
their old counterparts: no function, API, algorithm or behavior changed.

The old `OfflineBuffer` lazily decodes `jpeg_bytes_v1` RGB using
`next_index`; `SequenceSampler` retains `numba.jit` and its original
episode-aware window/padding/stride behavior; `OfflineDataset` retains
train/validation split, optional Depth, next observation, endpoint-only,
and latent-cache branches. The existing `batch_adapter.py` (a
separate necessary integration shim) maps dataset samples to native
RL1002D input tensors. No new ReplayBuffer or Sampler algorithm was
introduced.

```bash
# Run these after restoring the pinned Zarr 2.x environment.
python -c "import numba; print('numba', numba.__version__)"
python post_training/tests/smoke/smoke_01_lerobot_to_zarr.py
python post_training/tests/smoke/smoke_01_offline_reader.py
```

If `numba` is absent, install only that package with
`python -m pip install 'numba>=0.61.2,<0.63'`; it is recorded in
`post_training/requirements-rl100-extra.txt`. A passing synthetic
test verifies *loading and batching*, not complete model BC training,
nor validated reward suitability for Offline PPO. An extra
`RL1002D.compute_loss()` real-data integration test and native
`LinearNormalizer` fitting remain mandatory before closing Batch 01.

## Remaining Batch 01 work

1. `gpt/rl100-lerobot-zarr`: initial LeRobot 0.5.2 → Zarr converter and synthetic contract tests implemented (see `LEROBOT_TO_ZARR_RL100.md`); real-data validation, computed Reward V2 and performance benchmarking still pending.
2. `gpt/rl100-offline-reader`: original `offline_buffer.py`, `sampler.py`, `offline_dataset.py` copied verbatim from `kdc_ARBiM@feature/flywheel` (verified matching Git blob SHA); synthetic Zarr→Dataset→RL1002D batch-adapter test added, server validation pending. Native RL1002D `LinearNormalizer` fitting remains for subsequent integration.
3. `gpt/rl100-zarr-merge`: port metadata-aware merge with episode/offset/reward checks.
4. Real-data smoke tests, then promote `feature/rl100-data` to `arbim/dev`.
