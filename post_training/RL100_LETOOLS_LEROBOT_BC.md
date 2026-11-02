# Native RL-100 2D Diffusion Policy → LeTools/LeRobot (Batch 02 repair)

## Data boundaries (binding)

- **BC/IL:** LeRobot Dataset v3 directly → LeRobot delta-timestamp windowing →
  `lerobot_policy_rl100_2d` → **pinned original `RL1002D.compute_loss`** →
  LeRobot trainer/optimizer/checkpoint. **No Zarr in this training path.**
- **RL Post-training:** original LeRobot dataset converted to ARBiM Zarr with
  Reward, Return, Done, Next State/Action → future Critic/Dynamics/PPO.
- No substitution with Hugging Face's *different* built-in Diffusion Policy.

## Why plugin, not another handwritten trainer?

This is LeRobot's official out-of-tree plugin mechanism for v0.5.x:
a named `lerobot_policy_*` distribution with `configuration_*`,
`modeling_*`, `processor_*` modules. `kuavo_model/train.py` remains
the LeTools thin launcher around `lerobot.scripts.lerobot_train`.
The `third_party/rl100` and `third_party/lerobot` submodules are
pinned and **unmodified**.

Implementation lives at `integrations/lerobot_policy_rl100_2d`.
LeRobot owns frame sampling, video decode, optimizer, scheduler and
native checkpoint folder layout; adapter owns only mapping model inputs
and native normalizer construction. Model uses original RL-100
`RL1002D`, `MultiImageObsEncoder`,
`DDIMSchedulerExtended` and `SampleConditionalUnet1D`, not
LeRobot DiffusionPolicy.

### Contract

LeRobot emits (each batch sample is centered on the current frame):

- `observation.state`: `[B, n_obs_steps, Ds]`
- `observation.images.head_cam_h`, `wrist_cam_l`, `wrist_cam_r`:
  `[B, n_obs_steps, 3, H, W]`
- `action`: `[B, horizon, Da]`, aligned to
  `range(1-n_obs_steps, 1-n_obs_steps+horizon)` relative to the
  current frame.

Adapter renames only `agent_pos`, `rgb_head`,
`rgb_left_hand`, `rgb_right_hand` and scales uint8 RGB to
float `[0,1]`. It does **not** generate rewards, pad transitions,
convert files, or rewrite native loss.

**Double normalization prevention:** LeRobot pre/postprocessors use
`IDENTITY` for VISUAL/STATE/ACTION. The native RL-100
`LinearNormalizer` is populated from LeRobot `meta.stats`
(min/max/mean/std); its stats are saved inside `policy.native_stats`
for inference without attaching the dataset. Original RL-100
`mode='limits'` scaling to [-1,1] is reproduced, including constant
dimensions. Images are normalized to ImageNet inside the original
native RGB encoder.

## Important equivalence constraints

This is a **LeRobot trainer integration**, not a claim that the
original all-in-one `TrainDP3Workspace` has been reproduced byte-for-byte.
The original Workspace includes its own EMA/rollout/TopK training
management and also creates Critic/PPO/Dynamics components. The LeRobot
training loop owns checkpoint saves and scheduling instead; it **does
not replicate RL-100 EMA**. Before full-scale training, decide
whether LeRobot-native training management is acceptable or an
adapter must explicitly preserve the original EMA/TopK semantics.
Do not silently conflate these implementations.

The initial plugin config defaults are close to RL-100 2D U-Net
sizes, but omit original experimental variants such as VIB/recon
and R3M downloads unless explicitly enabled. Reproduction requires
separately specifying those options and validating availability.

## Installation & first smoke (server)

From the LeTools repository root, using the existing `letools_chang`
conda environment and initialized pinned submodules:

```bash
conda activate letools_chang
git fetch origin
git switch --track origin/gpt/rl100-lerobot-policy
git submodule update --init third_party/lerobot third_party/rl100

# The plugin package is registered automatically by LeRobot's plugin scanner.
python -m pip install -e integrations/lerobot_policy_rl100_2d --no-deps

export PYTHONPATH="$PWD/third_party/lerobot/src:$PWD/third_party/rl100/RL-100:${PYTHONPATH:-}"

python post_training/tests/smoke/smoke_02_lerobot_policy.py --device cuda:0
python kuavo_model/train.py --policy rl100_2d --dry-run
```

For an actual local LeRobot dataset, edit
`configs/train/lerobot/rl100_2d.yaml`:
`dataset.root`, `dataset.repo_id`; then run one training step without
a Zarr file:

```bash
python kuavo_model/train.py --policy rl100_2d --no-timestamp \
  --steps=1 --save_freq=1 --batch_size=1
```

Use a fresh training output directory (or enable resume using the
LeTools/LeRobot checkpoint mechanism) to avoid overwriting earlier
experiments. Validate both loss/grad and LeRobot model
`save_pretrained()`/reload in a subsequent real-data acceptance.

## Old Zarr-based BC experiment

`post_training/src/arbim_rl100/train_bc.py` is the **legacy Batch 02
diagnostic**, saved previously in `arbim/dev` and reliant on Zarr.
**Do not use it to train BC going forward.** It should be retired after
this replacement has passed real LeRobot training and checkpoint tests.
The old diagnostic's success was evidence of RL1002D compute-loss
compatibility only, never evidence of the final data architecture.

**Current status:** developer branch only; server smoke not yet run.
No merge into `feature/**` or `arbim/dev` until acceptance.
