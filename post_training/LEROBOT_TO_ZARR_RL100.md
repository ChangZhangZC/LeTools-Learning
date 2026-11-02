# Batch 01 — LeRobot 0.5.2 to ARBiM offline Zarr (sub-batch 02)

Development branch: `gpt/rl100-lerobot-zarr`. This is **unvalidated on real
LeRobot data** until the server tests pass. The RL-100 submodule remains pinned
and unmodified.

## Supported input and output

The new reader uses the **public** LeRobot 0.5.2 `LeRobotDataset(repo_id=...,
root=..., return_uint8=True)` constructor and indexed `dataset[i]` decoding.
It does not call the old private `_query_videos` from LeRobot 0.4.2 and does
not use the deprecated NPY streaming bridge. All three RGB cameras are required:

- `observation.images.head_cam_h` → `head_rgb`;
- `observation.images.wrist_cam_l` → `wrist_left_rgb`;
- `observation.images.wrist_cam_r` → `wrist_right_rgb`.
- `observation.state` and `action` remain dynamic dimensional vectors.

Images are checked as CHW RGB. A `uint8` image is stored at its original
pixel scale before JPEG encoding. A float image must be finite in `[0,1]`;
it is converted to uint8 once. Camera dimension drift and missing features
abort the entire output. No Depth is exported in this first version.

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
can be passed directly, but merged-format reward records are not yet supported.

For *format-only smoke testing*, users may explicitly set
`--reward-mode terminal_proxy` (does not require JSON). This assigns a
synthetic +1 at each episode end and sets `reward_validated=false`. It is
not a replacement for ARBiM Reward V2 and **must not feed PPO training as
if it were an observed success signal**. Teleop-only computed Reward V2
detection and mixed recorded/computed rewards are **not yet migrated**.

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
clean up the temporary directory. The current public item-based video decode
may be slower than batched decode on a very large dataset. Performance
optimization requires a separate benchmark and must not change the data
contract.

### Deferred work

- Teleop Reward V2 computation and mixed-source/staging-merge provenance;
- LeRobot 0.5.2 decoder benchmarking and bounded batched access;
- OfflineBuffer + SequenceSampler/normalizer port and real end-to-end data test;
- Zarr merge/Reward consistency validation.
