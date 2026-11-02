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

## Batch 01D — original RL1002D BC integration smoke (pending server check)

After the 9 LeRobot-to-Zarr and 3 ARBiM OfflineReader synthetic checks passed
on the user's server, this batch adds **only a test entrypoint**:
`post_training/tests/smoke/smoke_01_real_bc.py`. It invokes original
`kdc_ARBiM` `OfflineBuffer` / `OfflineDataset`, existing
`to_rl100_bc_batch()`, the pinned `RL1002D` with the same
`MultiImageObsEncoder` and scheduler construction pattern as Batch 00,
and upstream `LinearNormalizer.fit()` / `set_normalizer()` /
`compute_loss()` / `predict_action()`.

No RL-100 model edits, new optimizer, new loss, checkpoint or training
framework. The script derives the state/action/camera dimensions from the
actual Zarr file, applies the **native vision encoder's resize** to 84x84
by default, fits the native min/max normalizer on **training episodes
only**, then checks BC loss, U-Net/RGB encoder gradients, and action
prediction for one sample.

Example (real local Zarr is required; not generated by this command):

```bash
python post_training/tests/smoke/smoke_01_real_bc.py \
  --zarr data/zarr/icra_task1_10bags.zarr --device cuda:0
```

The previously used `--reward-mode terminal_proxy` dataset is suitable
for a **BC format/gradient smoke**, not for Offline PPO with validated
reward. This smoke must pass before declaring the RL1002D BC input
contract integrated; full BC training, eval and checkpointing are
still Batch 02. Neither this smoke nor the synthetic tests prove
the user's real LeRobot 10-bag conversion ran successfully.

## Batch 01 source-preservation review (work branch, unvalidated)

Comparison base: `ChangZhangZC/kdc_ARBiM@feature/flywheel`
`post_training/src/post_rl/data/data_prepare.py`; integration work branch:
`gpt/rl100-data-source-parity`. Earlier synthetic suites passed (9+3),
**but this new expanded source-port still requires server tests**.

- `offline_buffer.py`, `offline_dataset.py`, `sampler.py`: no changes;
  destination/source Git Blob hashes are identical.
- `batch_adapter.py`: leave the already tested mandatory 3-RGB/name
  conversion as the only model-facing data adapter.
- `lerobot_to_zarr.py`: original ARBiM `_batch_column`,
  `_stack_video_batch`, `_load_lerobot_frame_batch`, original
  `_to_uint8_chw`, gripper-release Reward V2 and Staging/merged
  provenance functions restored; `Streaming build_db` progress restored;
  original optional Depth and `episode_end_shaped` processing restored.
- Necessary LeRobot 0.5.x differences: batching uses
  `dataset.reader._query_videos` (not old `dataset._query_videos`),
  supports stacked tensor columns, and keeps the pinned public item
  API as a fallback. `--video-backend pyav` remains the server's
  source-compatible workaround for TorchCodec binary errors.
- Existing safety contracts retained: no overwrite, temporary Zarr
  output, episode-safe transitions, explicit unvalidated
  `terminal_proxy` **for BC smoke only**, and prohibition of silent
  RL reward fabrication. The old `raw_to_npy` and `build_db` modes
  belong to a separate conversion path; Zarr merge is not part of
  this port and must reuse the old `zarr_merge.py` later.

Current real-data conversion remains unverified. Do not promote the
work branch to `feature/rl100-data` or `arbim/dev` until the expanded
synthetic tests and a short genuine LeRobot conversion run pass, and the
original `jpeg_bytes_v1` payload can be read by OfflineBuffer.

## Batch 01D — native RL1002D constructor compatibility (pending server validation)

Real LeRobot video conversion **passed on the user's server** with
`--video-backend pyav`: 1,482 frames / 10 episodes; 1,120 training and
282 validation windows; native `LinearNormalizer.fit()` used 1,184
training-only frames. The subsequent BC smoke stopped in the *pinned
RL-100 constructor*, before loss or gradient calculation:

```python
# third_party/rl100/RL-100/rl_100/policy/rl100_2d.py
obs_dict = dict_apply(obs_shape_meta, lambda x: x["shape"])
```

Its upstream `dict_apply` recursively descends into ordinary `dict`
feature specifications and then attempts to subscript the leaf
shape `list` with the string `"shape"`, raising `TypeError`.

**No upstream edits or runtime monkey-patching.** Existing
`batch_adapter.py` now exposes `adapt_rl100_shape_meta()`,
using stdlib picklable `collections.UserDict` to keep feature
specifications as non-`dict` Mapping leaves. The shape, type, original
input metadata and RL1002D mathematics remain unchanged. Both Batch 00
forward smoke and Batch 01D real Zarr smoke apply this adaptation.

**Tests to run in the same environment after pulling:**

```bash
python post_training/tests/smoke/smoke_01_rl100_batch_contract.py
python post_training/tests/smoke/smoke_00_rl100_runtime.py --level forward --device cuda:0
python post_training/tests/smoke/smoke_01_real_bc.py \
  --zarr data/zarr/icra_task1_10bags_parity.zarr --device cuda:0
```

The previous Batch 00 default `--level imports` checks imports
only, **not** constructor/forward or Backward; do not infer forward
success from the import-only test. Real-data normalization success
does not yet prove BC Loss or Backward. Separately inspect the
server-local `M third_party/rl100` submodule before deciding whether
any local upstream modifications need retaining.

## Batch 01E — original Zarr Merge (work branch; server tests pending)

**Original**: `ChangZhangZC/kdc_ARBiM@feature/flywheel`,
`post_training/src/post_rl/data/zarr_merge.py`.

**Destination**: `post_training/src/arbim_rl100/data/zarr_merge.py`.
The original 318-line script was copied without change *first*, then
modified only where new LeTools Zarr's Reward provenance needs preservation:

- No change to original `_reward_mode`, `_read_source`,
  `_check_compatible`, `_create_arrays`, index rebasing, JPEG-byte
  concatenation, Episode boundary construction or Return computation.
- Merged `reward_validated` is true **only if every input explicitly
  reports true**; if input metadata is absent, no inferred valid reward
  status is fabricated. Identical `reward_source` values propagate.
- `merge_inputs` now retains each source's original
  `reward_source`, `reward_validated`, `reward_event`, and
  `return_gamma` **with their local index semantics**. The script
  does not invent a global adjusted Reward V2 index.
- Original rejection of `terminal_proxy` is **intentional and
  unchanged**, even for valid BC-format Zarr caches. Do not use
  `icra_task1_10bags_parity.zarr` as an offline Reward dataset.
- Existing `overwrite=False`, source immutability, schema validation
  and temporary output logic are preserved. No new merge algorithm.

Synthetic acceptance:

```bash
git fetch origin
git switch --track origin/gpt/rl100-zarr-merge
python post_training/tests/smoke/smoke_01_zarr_merge.py
python post_training/tests/smoke/smoke_01_offline_reader.py
```

`smoke_01_zarr_merge.py` creates two small, **validated recorded-sparse
synthetic Zarr caches**, merges them using the original function, then
verifies Episode and Transition indices, JPEG offsets, Return,
OfflineBuffer compatibility, provenance, and expected rejection paths.
Tests have **not yet run on the user's LeTools server**. After they
pass, the independent work branch can be fast-forwarded into
`feature/rl100-data`. A real-data merge requires two compatible
Reward-valid input caches; the BC-only `terminal_proxy` cache cannot
satisfy this test.

## Remaining Batch 01 work

1. `gpt/rl100-data-source-parity`: review-phase port restores old batch decoder, computed/recorded Reward V2, optional Depth and progress; expanded tests and real-data performance/contract validation still pending. See `LEROBOT_TO_ZARR_RL100.md`.
2. `gpt/rl100-offline-reader`: original `offline_buffer.py`, `sampler.py`, `offline_dataset.py` copied verbatim from `kdc_ARBiM@feature/flywheel` (verified matching Git blob SHA); synthetic Zarr→Dataset→RL1002D batch-adapter test added, server validation pending. Native RL1002D `LinearNormalizer` fitting remains for subsequent integration.
3. `gpt/rl100-zarr-merge`: copied original `zarr_merge.py` and added only conservative Reward provenance propagation. Synthetic merge regressions added; **server acceptance pending**.
4. Real-data smoke tests, then promote `feature/rl100-data` to `arbim/dev`.
