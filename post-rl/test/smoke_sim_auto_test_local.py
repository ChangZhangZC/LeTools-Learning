"""Local-only Feature05 test that ACTUALLY runs sim_auto_test.py client path.

There is deliberately no Kuavo simulator, ROS master, robot, or env.step()
actuation. The original sim_auto_test.setup_policy and run_single_episode
functions execute against a fake local gym environment and ROS service,
but the REAL PolicyClient sends requests via TCP ZMQ to a REAL native
serve.py process with the Feature03 RL1002D checkpoint.

This is an interface/latency test, NEVER a robot-success evaluation.
"""
from __future__ import annotations

import argparse
import json
import logging
import runpy
import socket
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "post-rl" / "src"))

from kuavo_deploy.kuavo_service.client import PolicyClient, TorchSerializer  # noqa: E402
from rl100_compat.lerobot_data import open_lerobot_bc_dataset  # noqa: E402

SCRIPT = ROOT / "kuavo_deploy" / "src" / "eval" / "sim_auto_test.py"


def _module(name, **attributes):
    result = ModuleType(name)
    for key, value in attributes.items():
        setattr(result, key, value)
    return result


class _Subscription:
    def unregister(self):
        return None


class _TriggerResponse:
    def __init__(self, success=True, message="local fake ROS response"):
        self.success = success
        self.message = message


class _FakeROSManager:
    def register_subscriber(self, topic, msg_cls, callback):
        if topic != "/simulator/success":
            raise AssertionError(f"Unexpected ROS subscriber: {topic}")

    def close(self):
        return None


def _fake_import_modules():
    """Replace ONLY sim/ROS dependencies while importing sim_auto_test.py."""
    noop_logger = lambda name: logging.getLogger(f"feature05.{name}")
    fake_rospy = _module(
        "rospy",
        Subscriber=lambda *a, **k: _Subscription(),
        ServiceProxy=lambda name, service: (
            lambda *a, **k: _TriggerResponse()
        ) if name == "/simulator/start" else (_raise_unexpected_ros(name)),
        ServiceException=RuntimeError,
    )
    trigger = type("Trigger", (), {})
    request = type("TriggerRequest", (), {})
    modules = {
        "rospy": fake_rospy,
        "std_srvs": _module("std_srvs"),
        "std_srvs.srv": _module(
            "std_srvs.srv", Trigger=trigger, TriggerRequest=request,
            TriggerResponse=_TriggerResponse,
        ),
        "std_msgs": _module("std_msgs"),
        "std_msgs.msg": _module("std_msgs.msg", Bool=type("Bool", (), {})),
        "geometry_msgs": _module("geometry_msgs"),
        "geometry_msgs.msg": _module(
            "geometry_msgs.msg", PoseStamped=type("PoseStamped", (), {})
        ),
        "lerobot_patches": _module(
            "lerobot_patches", custom_patches=_module("custom_patches")
        ),
        "kuavo_deploy.config": _module(
            "kuavo_deploy.config", KuavoConfig=type("KuavoConfig", (), {})
        ),
        "kuavo_deploy.utils.logging_utils": _module(
            "kuavo_deploy.utils.logging_utils", setup_logger=noop_logger
        ),
        "kuavo_deploy.utils.policy_loader": _module(
            "kuavo_deploy.utils.policy_loader",
            inject_task_prompt=lambda obs, prompt: obs,
            load_native_policy_bundle=lambda **k: _raise_unexpected_ros("native_policy"),
            resolve_eval_output_dir=lambda *a, **k: Path("unused"),
        ),
        "kuavo_deploy.kuavo_env.KuavoSimEnv": _module(
            "kuavo_deploy.kuavo_env.KuavoSimEnv",
            KuavoSimEnv=type("KuavoSimEnv", (), {}),
        ),
        "kuavo_deploy.kuavo_env.KuavoRealEnv": _module(
            "kuavo_deploy.kuavo_env.KuavoRealEnv",
            KuavoRealEnv=type("KuavoRealEnv", (), {}),
        ),
        "kuavo_deploy.utils.ros_manager": _module(
            "kuavo_deploy.utils.ros_manager", ROSManager=_FakeROSManager
        ),
    }
    return modules


def _raise_unexpected_ros(name):
    raise RuntimeError(f"Local-only test forbids ROS/native call: {name}")


@contextmanager
def _temporary_import_stubs(stubs):
    """Replace only named ROS/sim modules; retain unrelated real imports.

    unittest.mock.patch.dict(sys.modules, ...) restores the ENTIRE module
    dictionary on exit. Imports made by sim_auto_test.py (notably torchvision)
    then disappear while their PyTorch operator registrations persist,
    causing a duplicate torchvision::roi_align Meta kernel on re-import.
    """
    missing = object()
    originals = {name: sys.modules.get(name, missing) for name in stubs}
    try:
        sys.modules.update(stubs)
        yield
    finally:
        for name, original in originals.items():
            if original is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def load_real_sim_auto_test_without_ros():
    if not SCRIPT.is_file():
        raise FileNotFoundError(SCRIPT)
    with _temporary_import_stubs(_fake_import_modules()):
        namespace = runpy.run_path(str(SCRIPT), run_name="arbim_sim_auto_test_local")
    for name in ("setup_policy", "run_single_episode"):
        if name not in namespace:
            raise RuntimeError(f"Missing original sim_auto_test function: {name}")
        if namespace[name].__code__.co_filename != str(SCRIPT):
            raise AssertionError(f"{name} does not originate from sim_auto_test.py")
    return namespace


class FakeKuavoGymEnv:
    """Returns REAL LeRobot RGB/state frames and records actions in memory."""

    def __init__(self, dataset, camera_map, action_dim, steps):
        self.dataset = dataset
        self.camera_map = camera_map
        self.action_dim = action_dim
        self.steps = steps
        self.ros_rate = 10
        self.average_sleep_time = 0.0
        self.unwrapped = self
        self.actions = []
        self.frame_idx = 0
        self.closed = False

    def _frame(self, index):
        sample = self.dataset[index]["obs"]
        frame = {"observation.state": sample["agent_pos"][-1].unsqueeze(0)}
        for source, model_key in self.camera_map.items():
            frame[source] = sample[model_key][-1].unsqueeze(0)
        return frame

    def reset(self, *, seed=None):
        self.frame_idx = 0
        self.actions.clear()
        return self._frame(0), {"fake_env": True}

    def step(self, action):
        action = np.asarray(action)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise AssertionError(f"Invalid Kuavo action: {action.shape}")
        self.actions.append(action.copy())  # No robot or ROS actuation!
        self.frame_idx += 1
        return self._frame(self.frame_idx), 0.0, False, self.frame_idx >= self.steps, {}

    def close(self):
        self.closed = True


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(proc, port, log_path, timeout=90.0):
    """Probe the actual REP socket; never use the PolicyClient's blocking recv."""
    import zmq

    context = zmq.Context()
    try:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Native serve.py exited {proc.returncode}; "
                    f"see {log_path}"
                )
            req = context.socket(zmq.REQ)
            req.setsockopt(zmq.LINGER, 0)
            req.setsockopt(zmq.RCVTIMEO, 600)
            req.setsockopt(zmq.SNDTIMEO, 600)
            try:
                req.connect(f"tcp://127.0.0.1:{port}")
                req.send(TorchSerializer.to_bytes({"endpoint": "ping"}))
                response = TorchSerializer.from_bytes(req.recv())
                if isinstance(response, dict) and response.get("status") == "ok":
                    return
            except zmq.Again:
                pass
            finally:
                req.close(linger=0)
            time.sleep(0.2)
    finally:
        context.term()
    raise TimeoutError(f"Native RL100 Server did not become ready; see {log_path}")


def _summary(latencies, all_actions, metadata, n_episodes, steps):
    values = np.asarray(latencies, dtype=float)
    if len(values) != n_episodes * steps:
        raise AssertionError("Not every simulator control step called the real PolicyClient")
    return {
        "status": "pass",
        "test_type": "local fake Kuavo env + real sim_auto_test client flow",
        "simulator_running": False,
        "robot_actuation": False,
        "model": metadata["adapter"],
        "state_dim": metadata["state_dim"],
        "action_dim": metadata["action_dim"],
        "cameras": list(metadata["camera_map"]),
        "execution_horizon": metadata["execution_horizon"],
        "episodes": n_episodes,
        "control_steps": len(values),
        "rpc_latency_ms": {
            "mean": float(values.mean() * 1000),
            "median": float(np.median(values) * 1000),
            "p95": float(np.percentile(values, 95) * 1000),
            "max": float(values.max() * 1000),
        },
        "actions_observed": len(all_actions),
        "ros_rate_target_hz": 10,
        "latency_under_100ms_for_all_steps": bool(np.all(values <= 0.1)),
        "task_success": None,
    }


def run(args):
    if args.steps < 1 or args.episodes < 1:
        raise ValueError("steps and episodes must be >=1")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing Feature03 BC checkpoint: {checkpoint}")

    # No ROS master and no simulator are accessed. These fake imports exist
    # only in this local test process; native repo source remains unchanged.
    sim = load_real_sim_auto_test_without_ros()
    print(f"[PASS] loaded ACTUAL {SCRIPT.relative_to(ROOT)} with isolated ROS stubs")

    port = _unused_local_port()
    server_log = output / "native_server.log"
    cmd = [
        sys.executable, "-u", str(ROOT / "kuavo_server" / "serve.py"),
        "--adapter", "rl100_2d", "--checkpoint", str(checkpoint),
        "--device", args.device, "--host", "127.0.0.1", "--port", str(port),
    ]
    proc = None
    try:
        with server_log.open("w", encoding="utf-8") as stream:
            proc = subprocess.Popen(
                cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
            )
            _wait_for_server(proc, port, server_log)
            print(f"[PASS] real native serve.py listening on 127.0.0.1:{port}")

            # Execute the actual setup_policy implementation, replacing ONLY
            # its hardcoded default PolicyClient host/port in this test process.
            setup = sim["setup_policy"]
            setup.__globals__["PolicyClient"] = lambda task_prompt: PolicyClient(
                host="127.0.0.1", port=port, task_prompt=task_prompt
            )
            policy, pre, post, _ = setup(
                checkpoint, "client", device=torch.device(args.device),
                task_prompt="local library smoke",
            )
            metadata = policy.policy.call_endpoint("metadata", requires_input=False)
            if metadata.get("adapter") != "rl100_2d" or metadata.get("status") != "ok":
                raise AssertionError(f"Unexpected RPC metadata: {metadata}")
            if int(metadata["execution_horizon"]) != 1:
                raise AssertionError("To=2 must request one action per observation")
            print(f"[PASS] native PolicyClient->ZMQ metadata: state={metadata['state_dim']}, "
                  f"action={metadata['action_dim']}, cameras={len(metadata['camera_map'])}")

            # Real images and ordered joint observations, but fake env.step.
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
            cfg = ckpt["config"]
            dataset = open_lerobot_bc_dataset(
                cfg["data"]["repo_id"], cfg["data"]["root"],
                n_obs_steps=int(cfg["data"]["n_obs_steps"]),
                horizon=int(cfg["data"]["horizon"]),
                video_backend=cfg["data"]["video_backend"],
            )
            first_episode_end = int(dataset.dataset.meta.episodes[0]["dataset_to_index"])
            if first_episode_end <= args.steps:
                raise ValueError("First dataset episode too short for selected --steps")
            if dataset.contract.state_dim != metadata["state_dim"]:
                raise AssertionError("Client state dimension differs from checkpoint")
            if dataset.contract.action_dim != metadata["action_dim"]:
                raise AssertionError("Client action dimension differs from checkpoint")
            if dataset.contract.source_to_rgb != metadata["camera_map"]:
                raise AssertionError("Client camera map differs from checkpoint")

            environments = []
            def fake_gym_make(name, **kwargs):
                if name != "Kuavo-Sim":
                    raise AssertionError(f"Unexpected requested environment: {name}")
                env = FakeKuavoGymEnv(
                    dataset, metadata["camera_map"], int(metadata["action_dim"]),
                    args.steps,
                )
                environments.append(env)
                return env

            real_select_action = policy.select_action
            timings = []
            def timed_select_action(observation):
                start = time.perf_counter()
                answer = real_select_action(observation)
                timings.append(time.perf_counter() - start)
                if tuple(answer.shape) != (1, metadata["action_dim"]):
                    raise AssertionError(f"PolicyClient returned {tuple(answer.shape)}")
                return answer

            policy.select_action = timed_select_action
            run_episode = sim["run_single_episode"]
            # Disable *only* video writing/ROS: original loop and model/network
            # calls remain untouched. Small placeholders allow cleanup logic
            # in the original sim_auto_test.py to execute unchanged.
            run_episode.__globals__["save_rollout_video"] = lambda *a, **k: None
            run_episode.__globals__["check_control_signals"] = lambda: True
            run_episode.__globals__["success_evt"].clear()

            with ExitStack() as stack:
                stack.enter_context(patch.object(sim["gym"], "make", side_effect=fake_gym_make))
                stack.enter_context(patch.object(
                    sim["imageio"], "imwrite",
                    side_effect=lambda path, img: Path(path).write_bytes(b"x")
                ))
                stack.enter_context(patch.object(
                    sim["imageio"], "imread",
                    side_effect=lambda path: np.zeros((4, 6, 3), dtype=np.uint8),
                ))
                conf = SimpleNamespace(
                    env=SimpleNamespace(env_name="Kuavo-Sim"),
                    inference=SimpleNamespace(
                        policy_type="client", seed=42, task_prompt="local library smoke",
                        max_episode_steps=args.steps,
                    ),
                )
                for episode in range(args.episodes):
                    # The original run_single_episode() resets the client and
                    # calls env.reset -> PolicyClient.select_action -> env.step.
                    status = run_episode(
                        conf, policy, pre, post, episode, output,
                    )
                    if status != 0:
                        raise AssertionError(
                            "Fake environment must not report simulator success"
                        )
                    last = environments[-1]
                    if not last.closed or len(last.actions) != args.steps:
                        raise AssertionError("Original sim_auto_test loop did not finish")
                    if policy._pending_actions:
                        raise AssertionError("PolicyClient unexpectedly cached extra actions")
                    print(f"[PASS] actual sim_auto_test run_single_episode: "
                          f"episode={episode}, env_steps={len(last.actions)}, "
                          f"no physical simulator")

            actions = [a for env in environments for a in env.actions]
            summary = _summary(timings, actions, metadata, args.episodes, args.steps)
            result_file = output / "metrics.json"
            result_file.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print("[PASS] PolicyClient/ZMQ/Checkpoint/Reset/fake-env interface")
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            print(f"[PASS] local library test metrics: {result_file}")
            return summary
    finally:
        if proc is not None and proc.poll() is None:
            # Prefer the official kill endpoint; fallback to OS terminate.
            try:
                # Use a finite-timeout socket so failure cannot block cleanup.
                import zmq
                context = zmq.Context()
                req = context.socket(zmq.REQ)
                req.setsockopt(zmq.LINGER, 0)
                req.setsockopt(zmq.RCVTIMEO, 1000)
                req.connect(f"tcp://127.0.0.1:{port}")
                req.send(TorchSerializer.to_bytes({"endpoint": "kill"}))
                req.recv()
                req.close(linger=0)
                context.term()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default="post_traing/outputs/feature03_bc_smoke/checkpoints/latest.pt"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output-dir", default="post_traing/outputs/feature05_local")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
