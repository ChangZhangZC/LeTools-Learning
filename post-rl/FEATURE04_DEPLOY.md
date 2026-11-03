# Feature 04 — RL1002D native Kuavo ModelServerAdapter

## Change boundary

Only one native LeTools source file is changed:
`kuavo_server/builtin_adapters.py` receives exactly the lazy registration
`"rl100_2d": "kuavo_server.adapters.rl100_2d"`.

Our implementation is otherwise new code under
`kuavo_server/adapters/rl100_2d.py`, plus tests under `post-rl/test/`.
No changes to `kuavo_deploy`, `kuavo_server/runtime.py`, existing
adapters, CVT, LeRobot or native RL-100. Launch using the existing
`kuavo_server/serve.py` entrypoint directly; no launcher alias is added.

## Checkpoint and dimensions

Input is the Feature03 `latest.pt` checkpoint (a BC training state_dict).
The adapter rebuilds an original RL1002D via Feature03's build_policy,
recreates the UserDict-wrapped metadata shape specs, and loads every tensor
strictly, including native observation/action normalizer parameters.
It reads camera key mapping and state/action names from the checkpoint;
nothing in the model is hardcoded to 16D, 26D, 28D or 36D.
A different model/action dimension requires a matching new checkpoint.

The LeTools payload carries current-frame `observation.state` ([1,D] or
[D]) and `observation.images.*` ([1,3,H,W], CHW or HWC); the adapter
constructs the trained [1,To,D] / [1,To,3,H,W] windows. The first
observation is repeated during warm-up, and the history is cleared by
the existing server `reset` endpoint. Images must be uint8 or float
in [0,1], with the training RGB size saved in the checkpoint. By default
a size mismatch is an error, not a silent resize. Optional
`--resize_images` explicitly enables bilinear conversion to the
training metadata resolution, which can change the visual distribution.

Native `predict_action(...,deterministic=True)` returns [1,T,D] in
the original control units via the restored RL100 action normalizer.
`select_action_chunk` returns [T,D] float64 to the existing Kuavo
PolicyClient; PolicyClient handles chunk caching. `select_action`
implements a separate, stepwise fallback cache for direct endpoint users.
Both caches and the observation window reset across episodes.

## Control semantics: fail closed on unverified differences

The saved metadata contains *ordered* state/action names and dimensions,
but not a reliable flag indicating whether CVT originally binarized the
end-effector. This adapter will NEVER infer a robot/EEF type from D alone
and will NEVER automatically slice a both-arm action into a single arm.
`--which_arm` is a descriptive flag only: users must configure the
Kuavo runtime for the same action and state layout used during CVT.

For a **confirmed binarized** training dataset only, supply a comma-separated
list of exact action names to threshold at the default 0.5, e.g.:

    --binary_action_names left_claw,right_claw

Omit the option when the CVT dataset uses continuous EEF values.
Check the original CVT `is_binary`, robot arm mode, action ordering,
normalization ranges, direction convention, and EEF actuator contract
before any sim or hardware rollout. Neither the Checkpoint nor this
adapter can prove those settings match. An incorrectly configured binary
threshold or hand orientation can cause unsafe actuator commands.

The Feature03 one-step checkpoint is *untrained*. This feature validates
interface compatibility, not robotic success or safe control. Do not use
the one-step smoke weights for real robot actuation.

## Test gate — feature branch ONLY

    cd /root/bayes-tmp/changzhang/LeTools-Learning
    git fetch origin
    git switch feature/rl100-deploy-v2
    git pull --ff-only origin feature/rl100-deploy-v2

    mkdir -p post_traing/outputs
    set -o pipefail
    PYTHONPATH=. python -m unittest discover \
      -s post-rl/test -p test_deploy_adapter.py -v \
      2>&1 | tee post_traing/outputs/feature04_adapter_unit.log

    python kuavo_server/serve.py --adapter rl100_2d --help \
      2>&1 | tee post_traing/outputs/feature04_server_help.log

    python post-rl/test/smoke_deploy_rl100.py \
      --checkpoint post_traing/outputs/feature03_bc_smoke/checkpoints/latest.pt \
      --device cuda:0 \
      2>&1 | tee post_traing/outputs/feature04_checkpoint_server_smoke.log

After these pass, optionally start the *existing* server on the
training machine (no robot movement) to check its metadata/reset
endpoints:

    python kuavo_server/serve.py \
      --adapter rl100_2d \
      --checkpoint post_traing/outputs/feature03_bc_smoke/checkpoints/latest.pt \
      --device cuda:0 --host 127.0.0.1 --port 5555

Do not change `kuavo_server/launch.py` or `configs/server/launcher.yaml`;
the direct `serve.py` command uses the official registration.

Feature04 acceptance requires passing unit tests and the actual
Checkpoint -> adapter -> real RGB -> action chunk -> reset/serialization
smoke. End-to-end sim rollout is a separate subsequent feature.
