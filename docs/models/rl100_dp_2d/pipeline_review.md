# Native RL100 DP-2D BC vs LeRobot Training Pipeline

Scope: LeTools `feature/rl100-bc-progress-eta`; upstream RL-100 pinned at
`6b6afceb7eb12525398207b5d6e66908863dc385`, LeRobot at
`a07f22e22ce88cddff1f6eddced9ea008fbfc37c`. The branch now
adopts LeRobot's compact uint8 video transport and MetricsTracker console style.
No native RL100 model/loss, optimizer, scheduler, EMA, sampler, validation
schedule or checkpoint format was changed.

## Comparison

| Stage | Original RL-100 | LeRobot ACT/Diffusion | Current native RL100 BC |
| --- | --- | --- | --- |
| Data | Zarr ReplayBuffer + SequenceSampler | LeRobot v3 MP4/Parquet | LeRobot v3 delta_timestamps + valid action windows |
| Video | Zarr arrays; no PyAV random seek | video_backend with uint8 transport | PyAV, LeRobotDataset(return_uint8=True) |
| Batch prep | Dataset mapping, device transfer | PolicyProcessorPipeline | Native mapping, uint8 over DataLoader IPC, float32 / 255 at _to_device boundary |
| Normalization | Actor-owned LinearNormalizer | NormalizerProcessorStep using dataset stats | **Actor-owned LinearNormalizer** |
| Model / loss | RL1002D.compute_loss(batch) | PreTrainedPolicy.forward(batch) | **RL1002D.compute_loss(batch)** |
| Update | Native RL100 optimizer/scheduler/EMA | Accelerate update_policy | Native RL100 optimizer/scheduler/EMA |
| Logging | epoch tqdm, WandB | MetricsTracker, tqdm | LeRobot AverageMeter/MetricsTracker + tqdm; JSONL, optional WandB/TB |
| Validation | Examined train_ddp.py BC path has RUN_VALIDATION=False | Configured evaluation | Held-out, every val_every optimizer steps |

Native RL1002D.compute_loss normalizes observation/action internally, slices
the first n_obs_steps-1 actions when no_pre_action=True, and encodes RGB with
the model's own resize/ImageNet-normalization. A LeRobot Diffusion/ACT
NormalizerProcessorStep would cause duplicate normalization and change the
input contract. The native normalizer is part of the actor state_dict and
must survive export and deployment.

LeRobot's general update_policy expects policy.forward(batch) and an
Accelerator, whereas this actor uses compute_loss(batch) and RL100-specific
training/checkpoint semantics. Do not substitute the entire LeRobot
trainer or make the native actor look like a LeRobot PreTrainedPolicy.

## Implemented adaptation on this branch

1. `open_lerobot_bc_dataset()` now requests
   `LeRobotDataset(return_uint8=True)` while preserving the configured
   PyAV backend, temporal alignment, train/val valid-window indices and
   camera metadata mapping.
2. `RL100LeRobotDataset.__getitem__()` keeps decoded uint8 RGB until
   DataLoader collation and worker-to-process transfer. State/action
   remain float32. `bc_utils._to_device()` converts each `rgb_*` tensor
   to float32 [0,1] at the model boundary; no new state/action
   normalization or RGB transforms are added.
3. `train.py` uses LeRobot's `AverageMeter` and `MetricsTracker` for
   console summaries alongside tqdm. Partial training batches are
   accounted for in samples/epoch display; the JSONL and optional
   WandB/TensorBoard logger remain intact.
4. `test_bc_checkpoint_reload.py` routes single-sample RGB through
   the same `_to_device()` conversion before calling native inference.

These are **data/telemetry** changes only. The native
`RL1002D.compute_loss(batch)`, actor-owned LinearNormalizer,
MultiImageObsEncoder resize/ImageNet normalization, original AdamW,
RL100 scheduler/EMA, checkpoint content, evaluation loop and
frequency are unchanged.

LeRobot's own train script converts uint8 -> float32 in its trainer
before the PolicyProcessorPipeline. Our adapter converts on the
target device during `_to_device()`. This is a deliberately narrow
optimization rather than a replacement with LeRobot's entire
Diffusion Policy preprocessor. PyAV random decoding itself is
unchanged; reduced data transport does **not** guarantee faster
decode or first-batch time.

## Deferred decisions (not in this change)

- Do not replace the RL100 native `compute_loss()` with LeRobot's
  `policy.forward()` or `update_policy()` / Accelerate.
- Do not use ACT/Diffusion `NormalizerProcessorStep`; it would
  duplicate or change actor-owned state/action normalization.
- Do not change PyAV to TorchCodec without a benchmark and compatibility check.
- Do not change validation cadence/held-out dataset/selection of best.pt.
  For 14,323 validation windows and validation batch size 8, a full
  validation is 1,791 batches; measuring its cost is a separate task.
- Native checkpoint / export and deployment remain unaffected.

## LeTools Diffusion training DataLoader settings (next change)

The native RL100 BC **training** loader now matches the pinned LeRobot
`lerobot_train.py` worker options: `num_workers: 4`, CUDA `pin_memory`,
`prefetch_factor: 4`, and `persistent_workers: true`. Values are exposed
under `training` in `configs/train/rl100_dp_2d.yaml`, and effective options
are printed at startup. For `num_workers: 0`, prefetch is disabled and workers
are nonpersistent (required by PyTorch). The held-out validation loader is
unchanged to avoid additional idle workers and out-of-scope validation edits.

Important: this is a **configuration parity experiment**, not a demonstrated
throughput fix. The same PyAV random-seek decoding and ~3-camera load remain.
Four workers × four prefetched batches may increase RAM and /dev/shm use,
particularly at batch_size=16 and large RGB frames. Check `df -h /dev/shm`,
container CPU quota, `data_s`, `updt_s`, OOM events and throughput before
a long run. If /dev/shm is restricted or workers are CPU-saturated, extra
prefetch may not help and can cause bus errors. Keep the current model
and validation/checkpoint selection behavior unchanged.

## Acceptance tests

- Same action-window indices, left/right/head RGB ordering and shapes
- One fixed batch: identical image range, state/action and sampled target
- Matching RL100 compute_loss at fixed RNG state and gradients to tolerance
- Same native actor, normalizer, checkpoint strict load and exact resume
- A/B throughput: startup, first batch, steady state, CPU RAM, GPU memory,
  data_s, update_s, and full validation time; test both PyAV and torchcodec
  only when the backend is available and pinned
- No changes to upstream RL100 or LeRobot submodules
