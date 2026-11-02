# Feature 02 — LeRobot v3 to native RL1002D

The dimension contract is owned by **LeTools CvtRosbag2Lerobot**, not by DP.
`observation.state`, `action`, and `observation.images.*` are read from
`LeRobotDataset.meta.features` (`meta/info.json`). Never assume 14/16/26/28/36 DOF.

## Boundary

- LeTools Cvt decides robot platform, left/right/both, EEF DOF, joint ordering,
  any EEF compression, normalization and optional binarization.
- LeRobot v3 metadata carries the resulting **tensor shapes** and
  `state_names` / `action_names`; the two dimensions need not be equal.
- The new `rl100_compat.lerobot_data` only constructs `shape_meta` and
  uses LeRobot's own `delta_timestamps` to collect temporal windows.
- RL1002D receives `batch['obs']` with `agent_pos` and one RGB key per
  LeRobot camera, plus `batch['action']`; no source-code edits or Zarr conversion.
- Each new model is built for a **fixed** metadata contract. A trained
  checkpoint cannot automatically change action dimensions at inference.
- RGB keys are renamed (e.g. `observation.images.head_cam_h` ->
  `rgb_head_cam_h`) because native `nn.ModuleDict` does not allow dots
  in module keys. The mapping is stored on the contract.
- A lightweight `UserDict` wraps each observation spec to work around the
  upstream `RL1002D.dict_apply` recursion; all original model code remains intact.

## BC temporal alignment

At LeRobot anchor frame t:
- RGB and proprio: t-(n_obs_steps-1) through t;
- Action target: t-(n_obs_steps-1) through t+horizon-n_obs_steps.

This is compatible with native RL1002D `no_pre_action` cropping.
The LeRobot reader repeats boundary frames and returns `action_is_pad`.
The adapter **preserves** this mask as an extra batch key but does not
implement a BC loss or silently include/exclude padded targets.
Feature 03 must choose explicit filtering or masking before invoking
native `compute_loss()`.

## Binary EEF and semantics

The dimension-only contract does not reconstruct `eef_type` or
`is_binary`: current Cvt metadata has shapes and joint names, but does
not guarantee explicit EEF/binarization flags. Identical dimensions do
not imply compatible control semantics; preserve training deployment
settings/checkpoint metadata and validate this in later features.

## Run checks on the feature branch

All test scripts live under `post-rl/test/`. No new packages are required
beyond LeTools + Feature 01's environment.

```bash
git fetch origin
git switch feature/rl100-lerobot-data-v2
git pull --ff-only origin feature/rl100-lerobot-data-v2

PYTHONPATH=post-rl/src python -m unittest discover \
  -s post-rl/test -p test_lerobot_contract.py -v

python post-rl/test/smoke_lerobot_real.py \
  --repo-id local/kuavo_example --root /path/to/lerobot \
  --n-obs-steps 2 --horizon 5 --video-backend pyav
```

If TorchCodec reports `Could not load libtorchcodec` (shared-library or
PyTorch ABI mismatch), use the official LeRobot `--video-backend pyav` decoder
instead of changing Torch, CUDA, or RL100 source. The smoke script forwards
this choice to `LeRobotDataset(..., video_backend=...)`; the data adapter
already exposes that argument. PyAV may be slower on large videos.

The second command requires a **real** locally generated LeRobot v3 dataset.
Test metadata, shapes and episode boundary flags first; this feature does
not yet initialize RL1002D or train BC.
