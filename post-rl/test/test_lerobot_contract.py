"""Feature 02 contract tests: run together after merging code to feature branch."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rl100_compat.lerobot_data import (  # noqa: E402
    RL100LeRobotDataset,
    build_delta_timestamps,
    contract_from_features,
)


def features(state_dim=16, action_dim=16, camera_names=("head_cam_h", "wrist_cam_l", "wrist_cam_r")):
    result = {
        "observation.state": {
            "dtype": "float32", "shape": [state_dim],
            "names": {"state_names": [f"joint_{i}" for i in range(state_dim)]},
        },
        "action": {
            "dtype": "float32", "shape": [action_dim],
            "names": {"action_names": [f"action_{i}" for i in range(action_dim)]},
        },
    }
    for name in camera_names:
        result[f"observation.images.{name}"] = {
            "dtype": "video", "shape": [3, 8, 8]
        }
    return result


class FakeLeRobot:
    def __init__(self, metadata, obs_steps, horizon):
        self.features = metadata
        self.obs_steps = obs_steps
        self.horizon = horizon

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        frame = {
            "observation.state": torch.zeros(self.obs_steps, self.features["observation.state"]["shape"][0]),
            "action": torch.zeros(self.horizon, self.features["action"]["shape"][0]),
            "action_is_pad": torch.tensor([False] * (self.horizon - 1) + [idx == 1]),
        }
        for key, spec in self.features.items():
            if key.startswith("observation.images."):
                frame[key] = torch.zeros(self.obs_steps, *spec["shape"], dtype=torch.uint8)
        return frame


class ContractTests(unittest.TestCase):
    def test_multiple_eef_shapes_are_inferred_not_hardcoded(self):
        for state_dim, action_dim in ((16, 16), (26, 26), (36, 36), (36, 39)):
            with self.subTest(state=state_dim, action=action_dim):
                source = features(state_dim, action_dim)
                contract = contract_from_features(source)
                self.assertEqual((contract.state_dim, contract.action_dim), (state_dim, action_dim))
                self.assertEqual(len(contract.source_to_rgb), 3)
                self.assertEqual(len(contract.state_names), state_dim)
                self.assertEqual(len(contract.action_names), action_dim)
                self.assertTrue(all(not isinstance(spec, dict) for spec in contract.shape_meta["obs"].values()))

    def test_temporal_alignment(self):
        keys = ("observation.images.head_cam_h",)
        delta = build_delta_timestamps(10, 2, 5, keys)
        self.assertEqual(delta["observation.state"], [-0.1, 0.0])
        self.assertEqual(delta["action"], [-0.1, 0.0, 0.1, 0.2, 0.3])
        self.assertEqual(delta[keys[0]], [-0.1, 0.0])

    def test_collated_batch_matches_native_loss_contract(self):
        from torch.utils.data import DataLoader

        metadata = features(16, 36, ("head_cam_h", "wrist_cam_l"))
        contract = contract_from_features(metadata)
        view = RL100LeRobotDataset(FakeLeRobot(metadata, 2, 5), contract, 2, 5)
        batch = next(iter(DataLoader(view, batch_size=2)))
        self.assertEqual(tuple(batch["obs"]["agent_pos"].shape), (2, 2, 16))
        self.assertEqual(tuple(batch["action"].shape), (2, 5, 36))
        self.assertEqual(tuple(batch["action_is_pad"].shape), (2, 5))
        self.assertTrue(batch["action_is_pad"][1, -1])
        self.assertEqual(len(contract.source_to_rgb), 2)

    def test_invalid_metadata_rejected(self):
        bad = features()
        bad["action"]["shape"] = [16, 2]
        with self.assertRaises(ValueError):
            contract_from_features(bad)
        missing = features(camera_names=())
        with self.assertRaises(ValueError):
            contract_from_features(missing)


if __name__ == "__main__":
    unittest.main()
