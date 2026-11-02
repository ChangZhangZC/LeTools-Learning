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

## Incremental dependency installation

Keep extra RL-100 integration dependencies in `post_training/requirements-rl100-extra.txt`, separate from LeTools' base requirements. Run from the repository root after initializing the RL-100 submodule:

```bash
# Inspect the existing environment before compiling native extensions.
python -c "import torch; print('torch:', torch.__version__, 'torch CUDA:', torch.version.cuda, 'CUDA available:', torch.cuda.is_available())"
command -v nvcc && nvcc --version || echo "nvcc not found: build may fall back to CPU extension"
git submodule status third_party/rl100

# RL-100's bundled, reduced PyTorch3D implementation.
# --no-build-isolation exposes the already-installed torch during compilation.
MAX_JOBS=2 python -m pip install --no-build-isolation -r post_training/requirements-rl100-extra.txt

python -c "import pytorch3d, pytorch3d.ops, pytorch3d._C; print('PyTorch3D:', pytorch3d.__file__)"
python post_training/tests/smoke/smoke_00_rl100_runtime.py --level imports
python post_training/tests/smoke/smoke_00_rl100_runtime.py --level forward --device cuda:0
```

This is a genuine, compiled `pytorch3d` package containing the operators bundled by RL-100, **not** a compatibility shim. Do not run RL-100's full requirements installer in the existing LeTools environment. The source was originally validated upstream with Python 3.8 and PyTorch 2.4, not Python 3.12; if compilation fails, capture the full build log and check PyTorch/CUDA/compiler compatibility before changing versions. There is no claim that installation has been validated on the target server.

### Legacy Gym dependency encountered in RL-100 import chain

RL1002D -> `unidpg.critic` -> `unidpg.buffer` -> `unidpg.utils` imports the **`gym`** package, which is different from LeTools' `gymnasium`. The incremental requirements now pin `gym==0.26.2`, rather than installing the old vendored Gym 0.21 from RL-100's Python 3.8 environment.

Install just the newly discovered package if PyTorch3D is already working:

```bash
python -m pip install "gym==0.26.2"
python -c "import gym; print('gym', gym.__version__)"
python post_training/tests/smoke/smoke_00_rl100_runtime.py --level imports
```

**Compatibility limit:** gym 0.26 changed the reset/step API from RL-100's legacy environment assumptions. This fixes the import dependency only; it does not validate RL-100's original Gym environments or online rollout. Online training must use an adapted, tested environment contract. Avoid downgrading LeTools' existing Gymnasium or NumPy just to satisfy this import.

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
