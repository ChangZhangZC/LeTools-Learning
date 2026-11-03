# Feature 05 — local sim_auto_test.py integration (NO simulator)

The user cannot currently start Kuavo OpenSource simulation. This feature
validates the ORIGINAL kuavo_deploy/src/eval/sim_auto_test.py CLIENT
inference path without running ROS, an OpenSource simulator or any robot.
All new files are under post-rl; no edits to LeTools, KuavoSimEnv,
PolicyClient, serve.py, original RL-100 or native algorithms.

## What actually runs

The local test imports sim_auto_test.py (no copied eval logic) using
local-process stubs for ROS and simulation-only modules. It calls the
actual setup_policy(policy_type="client") and run_single_episode(...)
with gym.make("Kuavo-Sim") replaced by a fake in-memory gym environment.
That fake returns REAL 10-bag LeRobot v3 RGB/state observations. Its
step() only records actions, and never actuates a robot or simulator.

A fresh native serve.py --adapter rl100_2d process runs on loopback
127.0.0.1 with an allocated port. The original PolicyClient performs
actual ZMQ/TorchSerializer requests and returns its normal action tensor.
The test changes the hardcoded client host/port only within the
sim_auto_test module function namespace in this test process.

A two-episode three-step smoke verifies RPC metadata, native checkpoint
inference, RGB/state/action shapes, PolicyClient calls for every step,
reset, original episode loop and cleanup. It records RPC latency.

Video encoding and ROS success/start interfaces are mocked. This keeps
the same loop but avoids requiring FFmpeg/ROS or a simulator and means no
real task success/failure is observed.

## Server test commands

    cd /root/bayes-tmp/changzhang/LeTools-Learning
    conda activate letools_chang
    git fetch origin
    git switch feature/rl100-local-sim-v2
    git pull --ff-only origin feature/rl100-local-sim-v2
    mkdir -p post_traing/outputs
    set -o pipefail

    # Real source import and fake env unit checks:
    PYTHONPATH=. python -m unittest discover \
      -s post-rl/test -p test_sim_auto_local.py -v \
      2>&1 | tee post_traing/outputs/feature05_local_unit.log

    # True local ZMQ server/client + sim_auto_test episode loop:
    python post-rl/test/smoke_sim_auto_test_local.py \
      --checkpoint post_traing/outputs/feature03_bc_smoke/checkpoints/latest.pt \
      --device cuda:0 \
      --episodes 2 --steps 3 \
      --output-dir post_traing/outputs/feature05_local \
      2>&1 | tee post_traing/outputs/feature05_local.log

The second command creates:
- post_traing/outputs/feature05_local/native_server.log
- post_traing/outputs/feature05_local/metrics.json

Expected PASS markers:
- loaded ACTUAL sim_auto_test.py with isolated ROS stubs
- real native serve.py listening
- native PolicyClient->ZMQ metadata
- actual sim_auto_test run_single_episode
- PolicyClient/ZMQ/Checkpoint/Reset/fake-env interface

## Python import isolation

The sim_auto_test.py loader injects temporary ROS/simulator **named modules**
into sys.modules, and restores only those named keys afterwards. It must
leave all unrelated Python imports cached. In particular, the original
sim_auto_test.py imports torchvision; clearing that import and re-importing
LeRobot's dataset utilities in the same process raises a duplicate
`torchvision::roi_align` Meta-kernel registration RuntimeError. The local
unit test now imports native LeRobotDatasetMetadata after sim_auto_test.py
to catch regression. No dependency reinstallation or torchvision patching
is required for this test-harness problem.

## Limitations and acceptance

This is a LOCAL library and network integration test, not an evaluation
of Kuavo OpenSource simulation, ROS topic correctness, actuator direction,
EEF binary settings, physical limits, 10 Hz control stability or robotic
task success. The 1-step BC smoke weights are not a trained policy.
The result must record task_success=null rather than invent a success rate.

With n_obs_steps=2, our Feature04 adapter returns one action per RPC to
preserve adjacent observation frames. The latency p95 in metrics.json
helps assess if client/server DP inference can potentially fit the
10 Hz target. It does not measure real ROS sensor/control delay.

Await actual server-side test logs before squash merging Feature05 to
arbim/dev. The future simulator-connected test remains a separate gate.
