"""Feature04 smoke: native BC checkpoint -> ModelServerAdapter -> action chunk.

Run only on the independent feature branch. This does not drive a robot.
It uses an already verified local 10-bag LeRobot v3 dataset as observation
source and validates the official server serialization contract.
"""
from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "post-rl" / "src"))

from kuavo_server.builtin_adapters import ensure_adapter_loaded  # noqa: E402
from kuavo_server.runtime import TorchSerializer, get_adapter_class  # noqa: E402
from rl100_compat.lerobot_data import open_lerobot_bc_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    ensure_adapter_loaded("rl100_2d")
    adapter_cls = get_adapter_class("rl100_2d")
    adapter = adapter_cls.from_args(Namespace(
        checkpoint=str(args.checkpoint),
        device=args.device,
        execution_horizon=0,
        which_arm="both",
        binary_action_names="",
        binary_threshold=0.5,
        resize_images=False,
    ))
    md = adapter.metadata()
    assert md["state_dim"] == len(md["state_names"])
    assert md["action_dim"] == len(md["action_names"])
    assert md["camera_map"] and md["execution_horizon"] > 0
    print(f"[PASS] official ModelServerAdapter registered: {md['adapter']}")
    print(f"[PASS] checkpoint contract: state={md['state_dim']}, "
          f"action={md['action_dim']}, cameras={list(md['camera_map'])}")

    cfg = adapter.config
    dataset = open_lerobot_bc_dataset(
        str(cfg.data.repo_id), str(cfg.data.root),
        n_obs_steps=int(cfg.data.n_obs_steps),
        horizon=int(cfg.data.horizon),
        video_backend=str(cfg.data.video_backend),
    )
    example = dataset[0]["obs"]
    payload = {
        "observation.state": example["agent_pos"][-1].unsqueeze(0),
    }
    for source, target in adapter.camera_map.items():
        payload[source] = example[target][-1].unsqueeze(0)
    payload["prompt"] = "test only"

    chunk = adapter.select_action_chunk(payload)
    expected = (md["execution_horizon"], md["action_dim"])
    assert isinstance(chunk, np.ndarray) and chunk.shape == expected
    assert chunk.dtype == np.float64 and np.isfinite(chunk).all()
    wire_value = TorchSerializer.from_bytes(TorchSerializer.to_bytes(chunk))
    assert isinstance(wire_value, np.ndarray) and wire_value.shape == expected
    print(f"[PASS] real-data server chunk + serialization: {chunk.shape}")

    one = adapter.select_action(payload)
    assert one.shape == (md["action_dim"],)
    reset = adapter.reset()
    assert reset["status"] == "ok"
    chunk2 = adapter.select_action_chunk(payload)
    assert chunk2.shape == expected and np.isfinite(chunk2).all()
    print("[PASS] single-step fallback, observation history and reset")
    print("[PASS] Feature04 RL1002D model serving smoke (no robot rollout)")


if __name__ == "__main__":
    main()
