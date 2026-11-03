"""Minimal checks for the standalone v3 10-episode extraction helper."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_lerobot_test_subset import _frame_for_writer  # noqa: E402
from lerobot.utils.constants import DEFAULT_FEATURES  # noqa: E402


class TestExtractEpisodeSubset(unittest.TestCase):
    def test_frame_preserves_actual_state_action_images_and_task(self):
        metadata_features = {
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
            "observation.images.head_cam_h": {
                "dtype": "video", "shape": [3, 4, 6],
            },
            "episode_index": {"dtype": "int64", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
        }
        row = {
            "observation.state": torch.tensor([0., 1.], dtype=torch.float32),
            "action": torch.tensor([1., 2.], dtype=torch.float32),
            "observation.images.head_cam_h": torch.zeros(3, 4, 6),
            "episode_index": torch.tensor(8),
            "timestamp": torch.tensor(0.1),
            "task": "toy sorting",
        }
        frame = _frame_for_writer(row, metadata_features)
        self.assertEqual(set(frame), {
            "observation.state", "action", "observation.images.head_cam_h", "task",
        })
        self.assertEqual(frame["task"], "toy sorting")
        self.assertEqual(frame["observation.state"].dtype, np.float32)
        self.assertEqual(frame["observation.images.head_cam_h"].shape, (3, 4, 6))
        self.assertIn("episode_index", DEFAULT_FEATURES)

    def test_missing_state_rejected(self):
        with self.assertRaisesRegex(KeyError, "observation.state"):
            _frame_for_writer({"task": "test"}, {
                "observation.state": {"dtype": "float32", "shape": [2]},
            })


if __name__ == "__main__":
    unittest.main()
