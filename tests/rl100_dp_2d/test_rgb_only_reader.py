"""The RL100 RGB-only reader must never query unused depth streams."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lerobot.datasets.dataset_reader import DatasetReader  # noqa: E402
from kuavo_model.rl100_dp_2d.rgb_only_reader import RGBOnlyDatasetReader  # noqa: E402


class RGBOnlyReaderTests(unittest.TestCase):
    def test_reader_get_item_queries_rgb_only_and_preserves_nonvideo_data(self):
        rgb_keys = (
            "observation.images.head_cam_h",
            "observation.images.wrist_cam_l",
            "observation.images.wrist_cam_r",
        )
        depth_keys = ("observation.depth_h", "observation.depth_l", "observation.depth_r")
        source = DatasetReader.__new__(DatasetReader)
        source._meta = SimpleNamespace(
            video_keys=[*depth_keys, *rgb_keys],
            tasks=SimpleNamespace(iloc=[SimpleNamespace(name="test_task")]),
            features={},
            subtasks=None,
        )
        source.delta_indices = None
        source._image_transforms = None
        source.hf_dataset = [{
            "episode_index": torch.tensor(0),
            "index": torch.tensor(0),
            "timestamp": torch.tensor(0.5),
            "task_index": torch.tensor(0),
            "observation.state": torch.tensor([1.0, 2.0]),
            "action": torch.tensor([3.0, 4.0]),
        }]
        source._absolute_to_relative_idx = None

        reader = RGBOnlyDatasetReader(source, rgb_keys)
        reader._query_videos = Mock(return_value={
            key: torch.full((3, 2, 2), i, dtype=torch.uint8)
            for i, key in enumerate(rgb_keys)
        })
        result = reader.get_item(0)

        requested, episode = reader._query_videos.call_args.args
        self.assertEqual(episode, 0)
        self.assertEqual(set(requested), set(rgb_keys))
        self.assertTrue(all(times == [0.5] for times in requested.values()))
        self.assertEqual(source._meta.video_keys, [*depth_keys, *rgb_keys])
        for key in depth_keys:
            self.assertNotIn(key, result)
        for key in rgb_keys:
            self.assertIn(key, result)
        torch.testing.assert_close(result["observation.state"], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(result["action"], torch.tensor([3.0, 4.0]))
        self.assertEqual(result["task"], "test_task")

    def test_rgb_selection_does_not_modify_underlying_metadata(self):
        source = DatasetReader.__new__(DatasetReader)
        source._meta = SimpleNamespace(video_keys=[
            "observation.depth_h", "observation.images.head_cam_h"
        ])
        source.delta_indices = None
        reader = RGBOnlyDatasetReader(source, ["observation.images.head_cam_h"])
        selected = reader._get_query_timestamps(1.5)
        self.assertEqual(selected, {"observation.images.head_cam_h": [1.5]})
        self.assertEqual(len(source._meta.video_keys), 2)


if __name__ == "__main__":
    unittest.main()
