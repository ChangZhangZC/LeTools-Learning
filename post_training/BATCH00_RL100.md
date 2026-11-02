# Batch 00 — RL-100 submodule and runtime smoke test

Scope: integrate **unmodified** RL-100 as a pinned Git submodule, and verify basic imports and a synthetic 3-RGB RL1002D BC/prediction call. This batch does **not** implement dataset conversion, training, PPO or deployment.

## Pinned dependencies

- LeTools / LeRobot: existing `third_party/lerobot` gitlink, unchanged.
- RL-100: `third_party/rl100`, upstream `https://github.com/Lei-Kun/RL-100.git`, pinned to `6b6afceb7eb12525398207b5d6e66908863dc385`.
- Python package root inside the checkout: `third_party/rl100/RL-100` (containing `rl_100/`).
- RL-100 contains its own nested SSH submodule `third_party/rl100-state`. It is **not** needed for this basic 2D import smoke test; avoid recursive initialization until it is explicitly required and access is confirmed.

## On a machine with a working Python environment

From the LeTools-Learning repository root:

```bash
git fetch origin
git switch feature/rl100-submodule
git pull --ff-only origin feature/rl100-submodule
git submodule update --init third_party/rl100

python post_training/tests/smoke/smoke_00_rl100_runtime.py --level imports
python post_training/tests/smoke/smoke_00_rl100_runtime.py --level forward --device cpu
# Or run the forward check on a CUDA GPU:
python post_training/tests/smoke/smoke_00_rl100_runtime.py --level forward --device cuda:0
```

If running on a work branch before integration, replace `feature/rl100-submodule` with `gpt/rl100-import-smoke`.

The `imports` level checks RL1002D, MultiImageObsEncoder, custom DDIM scheduler, PPO, critic, and transition dynamics imports. It also asserts they come from **this** checkout, not a different installed RL-100 package.

The `forward` level additionally constructs an **untrained synthetic** RL1002D (3 RGB at 84×84, 16D robot state/action, 2 observation steps, 9-step training horizon, 4-step output chunk), checks finite BC loss/gradient and predicted action shape `[1,4,16]`.

The synthetic configuration intentionally uses a small U-Net, shared ResNet18 without downloaded pretrained weights, and `action_norm=False` for an isolated smoke test. **It must not be reused as an actual Kuavo training configuration.** Real image dimensions, state/action DOF, image transforms, normalizer and execution timing belong to later batches.

## Environment compatibility gate

LeTools' currently pinned LeRobot version is not the same as RL-100's documented environment. In particular, `pytorch3d.ops` is imported unconditionally by `rl100_2d.py`, even for 2D-only use. Do not assume successful git submodule initialization implies import or forward compatibility.

Do **not** blindly merge both projects' complete requirements into one environment. First record the Python / PyTorch / torchvision / diffusers / PyTorch3D versions in the actual target environment, run the two smoke levels above, and resolve incompatibilities in a reproducible environment manifest.

## Batch 00 acceptance

1. `.gitmodules` and the gitlink `third_party/rl100` reference the exact pinned commit.
2. Existing LeRobot gitlink remains unchanged.
3. The import smoke passes on the chosen training machine.
4. Synthetic 3-RGB RL1002D BC loss, backward and deterministic prediction pass.
5. Only after #3 and #4 have been confirmed should Batch 00 be merged from `feature/rl100-submodule` to `arbim/dev`.

Record run logs and environment versions before promotion. Git and Python syntax checks alone are **not** sufficient for runtime acceptance.
