"""Batch 02 LeRobot-only padding contract: keep every episode-tail BC sample.

This exercises the pinned LeRobot 0.5.x implementation, not a copied sampler:
  python post_training/tests/smoke/smoke_02_lerobot_padding.py

The RL1002D adapter must configure drop_n_last_frames=0 and use
LeRobot's delta_timestamps to obtain copy-padded action windows.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party" / "lerobot" / "src"))

from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.sampler import EpisodeAwareSampler


class LeRobotPaddingContractTest(unittest.TestCase):
    @staticmethod
    def reader():
        # Two consecutive episodes [0, 5) and [5, 7); action horizon=4.
        reader = object.__new__(DatasetReader)
        reader._meta = SimpleNamespace(episodes=[
            {"dataset_from_index": 0, "dataset_to_index": 5},
            {"dataset_from_index": 5, "dataset_to_index": 7},
        ])
        reader.delta_indices = {
            "action": [0, 1, 2, 3],
            "observation.state": [-1, 0],
        }
        return reader

    def test_every_episode_frame_is_eligible(self):
        sampler = EpisodeAwareSampler(
            dataset_from_indices=[0, 5],
            dataset_to_indices=[5, 7],
            drop_n_first_frames=0,
            drop_n_last_frames=0,
            shuffle=False,
        )
        self.assertEqual(list(sampler), list(range(7)))

    def test_last_frame_repeated_without_crossing_episode(self):
        reader = self.reader()
        indices, masks = reader._get_query_indices(abs_idx=4, ep_idx=0)
        self.assertEqual(indices["action"], [4, 4, 4, 4])
        self.assertEqual(masks["action_is_pad"].tolist(), [False, True, True, True])
        self.assertNotIn(5, indices["action"])

    def test_short_episode_is_not_dropped(self):
        reader = self.reader()
        first, mask_first = reader._get_query_indices(abs_idx=5, ep_idx=1)
        self.assertEqual(first["action"], [5, 6, 6, 6])
        self.assertEqual(mask_first["action_is_pad"].tolist(), [False, False, True, True])
        self.assertEqual(first["observation.state"], [5, 5])
        self.assertEqual(
            mask_first["observation.state_is_pad"].tolist(), [True, False]
        )
        last, mask_last = reader._get_query_indices(abs_idx=6, ep_idx=1)
        self.assertEqual(last["action"], [6, 6, 6, 6])
        self.assertEqual(mask_last["action_is_pad"].tolist(), [False, True, True, True])

    def test_regular_in_episode_window_is_not_padded(self):
        reader = self.reader()
        indices, masks = reader._get_query_indices(abs_idx=0, ep_idx=0)
        self.assertEqual(indices["action"], [0, 1, 2, 3])
        self.assertEqual(masks["action_is_pad"].tolist(), [False] * 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
