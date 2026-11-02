# Native RL-100 DP-2D — Feature 01 bootstrap

Scope: pin unmodified `Lei-Kun/RL-100` as a Git Submodule and check that
its **native** 2D Actor imports and trains on a synthetic 3-RGB BC batch.
This feature does **not** convert LeRobot data, build an offline replay
buffer, implement a trainer, modify the algorithm, or integrate deployment.
Old implementation code remains available on `arbim/dev-deprecated` only as a reference.

## Source boundary

- RL-100: `third_party/rl100` gitlink pinned to
  `6b6afceb7eb12525398207b5d6e66908863dc385`.
- Python import root: `third_party/rl100/RL-100`, containing `rl_100/`.
- Existing `third_party/lerobot` gitlink: **unchanged**.
- No changes inside the RL-100 submodule or `kdc_ARBiM`.

The RL-100 upstream checkout contains nested submodules, including a
restricted/SSH URL. Initialize **only** the top-level RL-100 submodule for
this phase, not recursively:

```bash
git submodule update --init third_party/rl100
git submodule status third_party/rl100
```

## Preliminary environment adaptation

LeTools documents Python 3.12 and uses Torch 2.9.1; the upstream RL-100
environment is older. Do not install all upstream RL-100 dependencies
or downgrade LeTools' Torch/Gymnasium preemptively. In the target
training environment, record:

```bash
python -c "import sys,torch; print(sys.version); print(torch.__version__,torch.version.cuda)"
command -v nvcc && nvcc --version || true
```

The bundled *real* `pytorch3d_simplified` extension is imported by the
native DP-2D actor even when point clouds are disabled. Its compilation
must match the installed Torch/CUDA/compiler ABI. Do not replace it with
an import stub. One consolidated preliminary extra-dependency file is
`post-rl/requirements-rl100.txt`:

```bash
MAX_JOBS=2 python -m pip install --no-build-isolation -r post-rl/requirements-rl100.txt
python -m pip check
```

`gym==0.26.2` is a compatibility dependency for the Actor's indirect
imports; it does **not** establish support for upstream Gym environments.
`zarr==2.18.7` and `numcodecs==0.15.1` serve the native
`LinearNormalizer` import, **not** a request to add a Zarr conversion
pipeline. The upstream default visual config requests pretrained R3M
weights; this smoke test deliberately uses `weights=None`. R3M and
production configurations are **not** validated by Feature 01.

## Tests (all under `post-rl/test/`)

Run the complete checks **on the feature branch**, after code is integrated
and dependencies are available in the designated environment:

```bash
python post-rl/test/smoke_rl100_native_2d.py --level imports
python post-rl/test/smoke_rl100_native_2d.py --level forward --device cpu
# GPU on the intended training machine:
python post-rl/test/smoke_rl100_native_2d.py --level forward --device cuda:0
```

The synthetic check uses 3 x 84x84 RGB, 16D state/action, 2 observation
frames, 9-step sample horizon and a 4-step predicted action chunk.
It checks Actor and visual-encoder imports from this exact checkout,
finite BC loss, U-Net and encoder gradients, and `predict_action`
shape `[1,4,16]`. Its small, randomly initialized model is **not**
a Kuavo training recipe or a deployment checkpoint.

The smoke test uses a small `UserDict` leaf wrapper around
`shape_meta['obs']`: the unmodified RL1002D constructor calls recursive
`dict_apply` on that metadata, which otherwise descends into each
plain specification dict before it can retrieve `['shape']`.
This is an external input-contract shim, not a policy change.

## Promotion gate

Keep the feature off `arbim/dev` until the target machine confirms:
submodule SHA; imports; synthetic forward/backward; encoder and denoiser
gradients; inference output; and a reproducible environment version log.
No PPO/IQL/Dynamics, real-data BC or Kuavo rollout is claimed by these tests.
