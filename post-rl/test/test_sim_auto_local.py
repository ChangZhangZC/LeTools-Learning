"""Unit checks for the ROS-free sim_auto_test.py local integration harness."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from smoke_sim_auto_test_local import (  # noqa: E402
    FakeKuavoGymEnv,
    LEROBOT_SRC,
    SCRIPT,
    load_real_sim_auto_test_without_ros,
)


class FakeDataset:
    def __getitem__(self, idx):
        return {"obs": {
            "agent_pos": torch.full((2, 4), float(idx)),
            "rgb_head_cam_h": torch.zeros((2, 3, 4, 6), dtype=torch.float32),
        }}


class LocalSimAutoTestTests(unittest.TestCase):
    def test_loads_real_sim_auto_functions_without_ros(self):
        namespace = load_real_sim_auto_test_without_ros()
        self.assertEqual(namespace["setup_policy"].__code__.co_filename, str(SCRIPT))
        self.assertEqual(
            namespace["run_single_episode"].__code__.co_filename, str(SCRIPT)
        )
        # Importing sim_auto_test loads torchvision. It must remain cached
        # after the temporary ROS stubs are removed; otherwise reimport
        # re-registers torchvision::roi_align Meta and fails in PyTorch.
        self.assertTrue((LEROBOT_SRC / "lerobot").is_dir())
        self.assertIn(str(LEROBOT_SRC), sys.path)
        self.assertIn("torchvision", sys.modules)
        self.assertIn("torchvision._meta_registrations", sys.modules)
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        self.assertIsNotNone(LeRobotDatasetMetadata)

    def test_fake_environment_only_records_actions(self):
        env = FakeKuavoGymEnv(
            FakeDataset(),
            {"observation.images.head_cam_h": "rgb_head_cam_h"},
            action_dim=4, steps=2,
        )
        first, info = env.reset(seed=42)
        self.assertEqual(tuple(first["observation.state"].shape), (1, 4))
        self.assertEqual(tuple(first["observation.images.head_cam_h"].shape),
                         (1, 3, 4, 6))
        self.assertTrue(info["fake_env"])
        _, _, terminated, truncated, _ = env.step(np.zeros(4))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        _, _, terminated, truncated, _ = env.step(np.ones(4))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(len(env.actions), 2)
        with self.assertRaisesRegex(AssertionError, "Invalid Kuavo action"):
            env.step(np.zeros(3))
        env.close()
        self.assertTrue(env.closed)


if __name__ == "__main__":
    unittest.main()
