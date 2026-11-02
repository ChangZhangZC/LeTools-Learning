"""Feature 02 real-data smoke: requires an existing local LeTools LeRobot v3 dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rl100_compat import open_lerobot_bc_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Repo identifier of the local dataset")
    parser.add_argument("--root", required=True, help="Path with meta/info.json")
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    view = open_lerobot_bc_dataset(
        args.repo_id, args.root,
        n_obs_steps=args.n_obs_steps, horizon=args.horizon
    )
    print(f"Frames: {len(view)}; FPS: {view.dataset.fps}")
    print(f"State dim: {view.contract.state_dim}; Action dim: {view.contract.action_dim}")
    print(f"Camera map: {view.contract.source_to_rgb}")
    print(f"State names: {view.contract.state_names}")
    print(f"Action names: {view.contract.action_names}")
    print(f"Shape meta: {view.contract.shape_meta}")
    for idx in [0, min(len(view) // 2, len(view) - 1), len(view) - 1][:args.samples]:
        item = view[idx]
        print(f"index={idx}; state={tuple(item['obs']['agent_pos'].shape)}; "
              f"action={tuple(item['action'].shape)}; "
              f"action_pad={int(item['action_is_pad'].sum())}")
        for key, tensor in item["obs"].items():
            if key != "agent_pos":
                print(f"  {key}: {tuple(tensor.shape)} {tensor.dtype}")
        assert item["action"].dtype == torch.float32
    print("[PASS] LeRobot v3 -> RL1002D BC shape contract (not training)")


if __name__ == "__main__":
    main()
