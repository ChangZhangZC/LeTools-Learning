# Native RL100 DP-2D BC vs LeRobot Training Pipeline

Scope: LeTools `feature/rl100-bc-progress-eta`; upstream RL-100 pinned at
`6b6afceb7eb12525398207b5d6e66908863dc385`, LeRobot at
`a07f22e22ce88cddff1f6eddced9ea008fbfc37c`. Review only:
no data, optimizer, model or checkpoint behavior changed here.

## Comparison

| Stage | Original RL-100 | LeRobot ACT/Diffusion | Current native RL100 BC |
| --- | --- | --- | --- |
| Data | Zarr ReplayBuffer + SequenceSampler | LeRobot v3 MP4/Parquet | LeRobot v3 delta_timestamps + valid action windows |
| Video | Zarr arrays; no PyAV random seek | video_backend with uint8 transport | PyAV; Dataset converts images to float32 |
| Batch prep | Dataset mapping, device transfer | PolicyProcessorPipeline | Custom LeRobot-to-RL100 mapping + device transfer |
| Normalization | Actor-owned LinearNormalizer | NormalizerProcessorStep using dataset stats | **Actor-owned LinearNormalizer** |
| Model / loss | RL1002D.compute_loss(batch) | PreTrainedPolicy.forward(batch) | **RL1002D.compute_loss(batch)** |
| Update | Native RL100 optimizer/scheduler/EMA | Accelerate update_policy | Native RL100 optimizer/scheduler/EMA |
| Logging | epoch tqdm, WandB | MetricsTracker, tqdm | JSONL, tqdm, optional WandB/TB |
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

## Recommended staged changes (proposals, not yet implemented)

1. **Data transport:** A/B-test LeRobotDataset(return_uint8=True), retaining
   uint8 across DataLoader worker IPC. At the custom RL100 adapter boundary,
   cast to float32 / 255 to preserve [0,1] input. LeRobot's own trainer
   performs its uint8 -> float32 conversion in the trainer process before
   the PolicyProcessorPipeline; conversion on the GPU would be an additional
   new optimization. Test torch.equal/allclose against the current batches,
   CHW/camera order, deterministic sampling, and CPU memory and throughput.
2. **Explicit light preprocessor:** Isolate key mapping, RGB casting,
   device transfer and batch validation. Do not normalize state/action
   outside the RL100 actor; do not move image resize/Imagenet transform out
   of the RL100 observation encoder.
3. **Metrics:** Retain existing BCMetricLogger, and augment interval averages
   and timing if needed; borrowing LeRobot's logging concepts does not
   require replacing its optimizer/step engine.
4. **Validation economics:** At 14,323 held-out windows and validation batch
   size 8, full validation has 1,791 batches. With 357,700 train steps and
   val_every=1000, 358 validation passes require 641,178 validation
   batches. Benchmark this overhead; a validation-frequency change or a
   max_batches cap changes best.pt selection, so treat it as a separate
   explicit experimental/configuration decision.

## Acceptance tests

- Same action-window indices, left/right/head RGB ordering and shapes
- One fixed batch: identical image range, state/action and sampled target
- Matching RL100 compute_loss at fixed RNG state and gradients to tolerance
- Same native actor, normalizer, checkpoint strict load and exact resume
- A/B throughput: startup, first batch, steady state, CPU RAM, GPU memory,
  data_s, update_s, and full validation time; test both PyAV and torchcodec
  only when the backend is available and pinned
- No changes to upstream RL100 or LeRobot submodules
