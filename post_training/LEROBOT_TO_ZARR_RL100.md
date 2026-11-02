# Batch 01 — LeRobot 0.5.2 to ARBiM offline Zarr (sub-batch 02)

Initial development branch: `gpt/rl100-lerobot-zarr`.
Source-parity review branch: `gpt/rl100-data-source-parity`.
The earlier 9 Zarr and 3 reader **synthetic** tests passed on the server,
but the updated converter has not yet passed its new regressions or a
complete real-data conversion. The RL-100 submodule remains unmodified.

## Supported input and output

The converter uses the public LeRobot `LeRobotDataset(repo_id=...,
root=..., return_uint8=True)` constructor. To preserve original ARBiM's
batch decode, it reuses the old `_batch_column`,
`_stack_video_batch`, and `_load_lerobot_frame_batch` implementations.
The **only necessary private video-access relocation** is
`dataset._query_videos(...)` → `dataset.reader._query_videos(...)`
in the pinned LeRobot facade; the pre-existing indexed `dataset[i]`
path remains as fallback for synthetic or incompatible readers.
No deprecated NPY streaming bridge is used. Three RGB views are required:

- `observation.images.head_cam_h` → `head_rgb`;
- `observation.images.wrist_cam_l` → `wrist_left_rgb`;
- `observation.images.wrist_cam_r` → `wrist_right_rgb`.
- `observation.state` and `action` remain dynamic dimensional vectors.

Images are converted with the original ARBiM `_to_uint8_chw()`;
CHW/HWC RGB and finite float `[0,1]` or `[0,255]` are accepted.
Missing features or image dimension drift abort the output.
Optional `--use-depth` restores original three Depth field names
`observation.depth_h/l/r` and their dense `next_*` Zarr arrays.
Depth is not an input to the current RL1002D RGB-only BC smoke test.

The generated Zarr is a **derived cache** compatible with the old ARBiM
`jpeg_bytes_v1` layout:

```text
data/state, next_state, action, next_action
data/reward, return, done, timeout, next_index
data/{head_rgb,wrist_left_rgb,wrist_right_rgb}_jpeg_data
data/{head_rgb,wrist_left_rgb,wrist_right_rgb}_jpeg_offsets
meta/episode_ends
attrs: rgb_storage, rgb_shapes, rgb_dtype, reward_mode, reward_source,
       reward_validated, reward_event, return_gamma, source_manifest
```

For each episode, nonterminal transitions use `next_index=i+1`. Its last
frame uses a self-transition: `next_index=i`, `next_state=state`,
`next_action=action`. `done=timeout=True` on this last frame to *retain
the existing ARBiM offline cache contract*, but these flags **do not claim to
distinguish true termination and truncation**. Return is computed separately
for each episode with configurable discount (default 0.99). JPEG offset
streams have length `N+1` and end at the total byte count.

## Reward contract: no silent success inference

**Default: `--reward-mode recorded_sparse`.** You must provide
`--reward-json` with the validated sparse rewards. Supported input is
a JSON object with `episodes`, one record per LeRobot episode:

```json
{
  "episodes": [
    {
      "lerobot_episode_index": 0,
      "num_frames": 3,
      "reward_status": "valid",
      "reward_mode": "last_gripper_release_sparse",
      "reward_index": 1,
      "reward_event": {"frame_offset": 1}
    }
  ]
}
```

The reward offset must be **strictly before the episode's last frame**.
All episodes must have verified records; missing/invalid records abort.
An existing ARBiM Staging provenance JSON with those same episode fields
can be passed directly. The source-parity review also restores the original
`--reward-mode last_gripper_release_sparse` automatic detector:
it loads valid recorded Staging/Merge provenance when present, computes
missing Teleop rewards using the original gripper release detector,
and supports original `--reward-source auto|recorded|computed` selection.
The original `episode_end_shaped` mode is restored as an explicit option.
Gripper indices, open-side inference, dwell, expected cycles and
shaping parameters use the original defaults and are configurable via CLI.

For *format-only smoke testing*, users may explicitly set
`--reward-mode terminal_proxy` (does not require JSON). This assigns a
synthetic +1 at each episode end and sets `reward_validated=false`. It is
not a replacement for ARBiM Reward V2 and **must not feed PPO training as
if it were an observed success signal**. `episode_end_shaped` is also marked
`reward_validated=false` under the new conversion metadata contract; it is
not a verified sparse task-success label.

## Zarr 2.x environment requirement

This converter uses Zarr 2.x `Group.create_dataset`, Blosc compressor
arguments, and the existing `jpeg_bytes_v1` schema; it does not support
the Zarr 3 Python API. If imports succeed but the writer errors with
`'Group' object has no attribute 'create_dataset'`, check the runtime:

```bash
python -c "import zarr,numcodecs; print(zarr.__version__,numcodecs.__version__,zarr.__file__)"
python -m pip install "zarr==2.18.7" "numcodecs==0.15.1"
python -m pip check
python post_training/tests/smoke/smoke_01_lerobot_to_zarr.py
```

The code now fails early with an actionable error on incompatible versions.
Do not substitute `create_array` without redesigning and retesting the
storage/compression contract. LeRobot 0.5.2 requires Python 3.12 and
NumPy 2.x but does not itself pin Zarr; any incompatible upgrade should
be diagnosed separately.

## Local data location for this integration

The canonical local datasets and derived Zarr caches now belong under
`LeTools-Learning/data/` (see `data/README.md`). For the current
10-bag test dataset:

```bash
python post_training/src/arbim_rl100/data/lerobot_to_zarr.py \
  --lerobot-root data/lerobot/icra_task1_10bags \
  --output data/zarr/icra_task1_10bags.zarr \
  --reward-mode terminal_proxy
```

Run logs, metrics and videos still belong under `outputs/`. The Git
change only establishes the local directory contract; it does not
copy files from the user's server automatically. Existing valid
`outputs/test/*.zarr` may be moved to `data/zarr/` after verification.

## Video decoding on the Python 3.12 / PyTorch 2.9.1 server

The pinned LeRobot reader supports `video_backend="pyav"` directly.
When the default TorchCodec backend fails to load with missing
`libavutil.so.*` or an undefined PyTorch C++ symbol, **do not change
LeRobot or RL-100 source code**. Select the existing PyAV backend:

```bash
python post_training/src/arbim_rl100/data/lerobot_to_zarr.py \
  --lerobot-root data/lerobot/icra_task1_10bags \
  --output data/zarr/icra_task1_10bags.zarr \
  --reward-mode terminal_proxy \
  --video-backend pyav \
  --stream-batch-size 32
```

The `terminal_proxy` setting remains for **format/BC smoke tests only**,
not offline reward-based training. PyAV must be importable; it is an
existing LeRobot video dependency. This backend selection is not a
fix for broken TorchCodec binaries and does not imply any source changes.
Real-data conversion is not verified until this command completes on
the server.

## Source-preserving code review (pending tests)

**Unchanged copies** from `kdc_ARBiM@feature/flywheel`:
`offline_buffer.py`, `sampler.py`, `offline_dataset.py`
(identical Git blob SHA).

**Restored from** old `post_training/src/post_rl/data/data_prepare.py`:
`_to_uint8_chw`, `_batch_column`, `_stack_video_batch`,
`_load_lerobot_frame_batch`, `_stack_action_column`,
`_infer_gripper_thresholds`, `_classify_gripper`,
`_debounced_gripper_transitions`, `_last_gripper_release_offsets`,
`_validate_recorded_reward`, `_reward_offsets_for_subset`,
`_staging_reward_offsets`; plus original-shaped reward and progress.

**Necessary differences remaining:** LeRobot DatasetReader's private video
method location, the pre-existing `recorded_sparse` external-label
test contract, the explicit format-only `terminal_proxy` test mode,
and the already validated episode-contiguous Zarr/JPEG staging writer.
The old `raw_to_npy` / `build_db` NPY bridges are separate modes,
not the direct LeRobot-to-Zarr entrypoint. The old `merge_zarr` mode
remains scheduled for the next batch, not silently deleted.

```bash
# Work branch tests (NOT YET CONFIRMED on server):
git fetch origin
git switch gpt/rl100-data-source-parity
python post_training/tests/smoke/smoke_01_lerobot_to_zarr.py
python post_training/tests/smoke/smoke_01_offline_reader.py
```

## Run on the server

```bash
git fetch origin
git switch feature/rl100-data
git pull --ff-only origin feature/rl100-data

# Synthetic test first: no real LeRobot dataset required.
python post_training/tests/smoke/smoke_01_lerobot_to_zarr.py

# Then replace the sample paths with a real *local* LeRobot 0.5.2 dataset.
# A valid per-episode Reward V2 JSON is required for recorded_sparse.
python post_training/src/arbim_rl100/data/lerobot_to_zarr.py \
  --lerobot-root /path/to/lerobot_dataset \
  --output /path/to/new/offline.zarr \
  --reward-mode recorded_sparse \
  --reward-json /path/to/validated_rewards.json \
  --stream-batch-size 32

# Only if validating format without real reward labels:
python post_training/src/arbim_rl100/data/lerobot_to_zarr.py \
  --lerobot-root /path/to/lerobot_dataset \
  --output /path/to/new/proxy_test.zarr \
  --reward-mode terminal_proxy
```

No existing output path is overwritten. The converter writes to a temporary
sibling directory and renames it only after writing all episodes; errors
clean up the temporary directory. The source-parity branch
restores original batched video decode and `Streaming build_db` progress
via `tqdm`. Performance and byte-for-byte equivalence have **not** yet been
benchmarked on the real dataset. The original Zarr storage schema remains
the comparison target.

### Deferred work

- Server validation of the review branch's expanded synthetic regressions
  (batch decoder, computed/recorded Reward V2, shaped reward and Depth);
- End-to-end real LeRobot conversion with `--video-backend pyav`, followed
  by the RL1002D native BC Loss/Backward test;
- Zarr merge/Reward consistency validation using the original ARBiM code;
- Standardizing the original YAML entrypoint after source parity is verified.
