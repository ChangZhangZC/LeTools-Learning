# Batch 01 — ARBiM → RL1002D data contract (first sub-batch)

Status: **interface specification and synthetic tests only**. The converter for LeRobot 0.5.2, Zarr reader, episode sampler and real-data integration tests are separate work items. Do not treat a passing synthetic contract test as end-to-end migration.

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

## Remaining Batch 01 work

1. `gpt/rl100-lerobot-zarr`: migrate and validate the real LeRobot 0.5.2 → Zarr converter.
2. `gpt/rl100-dataset-adapter`: port the lazy JPEG OfflineBuffer and episode sampler; build a proper RL100Dataset with validation split and fitted native `LinearNormalizer`.
3. `gpt/rl100-zarr-merge`: port metadata-aware merge with episode/offset/reward checks.
4. Real-data smoke tests, then promote `feature/rl100-data` to `arbim/dev`.
