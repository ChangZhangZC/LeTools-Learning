"""Synthetic episode-safe LeRobot -> Zarr round-trip tests.

Run from repository root:
  python post_training/tests/smoke/smoke_01_lerobot_to_zarr.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
import zarr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "post_training" / "src"))
from arbim_rl100.data.lerobot_to_zarr import convert_dataset, episode_ranges


class _Dataset:
    def __init__(self, episodes=(0, 0, 0, 1, 1)):
        self.hf_dataset = {"episode_index": list(episodes)}
        self.frames = []
        for i in range(len(episodes)):
            self.frames.append({
                "observation.state": torch.tensor([i, 10 + i], dtype=torch.float32),
                "action": torch.tensor([100 + i, 200 + i, 300 + i], dtype=torch.float32),
                "observation.images.head_cam_h": torch.full((3, 8, 12), i * 30, dtype=torch.uint8),
                "observation.images.wrist_cam_l": torch.full((3, 8, 12), 70, dtype=torch.uint8),
                "observation.images.wrist_cam_r": torch.full((3, 8, 12), 0.5, dtype=torch.float32),
            })

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        return self.frames[i]


def _recorded_json(path: Path):
    obj = {
        "episodes": [
            {
                "lerobot_episode_index": 0, "num_frames": 3,
                "reward_status": "valid",
                "reward_mode": "last_gripper_release_sparse",
                "reward_index": 1, "reward_event": {"frame_offset": 1},
            },
            {
                "lerobot_episode_index": 1, "num_frames": 2,
                "reward_status": "valid",
                "reward_mode": "last_gripper_release_sparse",
                "reward_index": 0, "reward_event": {"frame_offset": 0},
            },
        ]
    }
    path.write_text(json.dumps(obj), encoding="utf-8")


class LeRobotToZarrTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.labels = self.path / "reward.json"
        _recorded_json(self.labels)

    def test_episode_ranges_and_offsets(self):
        dataset = _Dataset()
        self.assertEqual(episode_ranges(dataset.hf_dataset["episode_index"]), [
            (0, 0, 3), (1, 3, 5)
        ])
        out = convert_dataset(
            dataset, self.path / "cache.zarr",
            reward_json=self.labels, gamma=0.9, stream_batch_size=2,
        )
        root = zarr.open_group(str(out), mode="r")
        d = root["data"]
        self.assertEqual(list(root["meta"]["episode_ends"][:]), [3, 5])
        self.assertEqual(list(d["next_index"][:]), [1, 2, 2, 4, 4])
        np.testing.assert_array_equal(d["done"][:, 0], [0, 0, 1, 0, 1])
        np.testing.assert_array_equal(d["timeout"][:, 0], [0, 0, 1, 0, 1])
        np.testing.assert_array_equal(d["reward"][:, 0], [0, 1, 0, 1, 0])
        np.testing.assert_allclose(d["return"][:, 0], [0.9, 1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(d["state"][2], d["next_state"][2])
        np.testing.assert_array_equal(d["next_state"][3], d["state"][4])
        np.testing.assert_array_equal(d["action"][4], d["next_action"][4])
        self.assertEqual(root.attrs["rgb_storage"], "jpeg_bytes_v1")
        self.assertTrue(root.attrs["reward_validated"])
        self.assertEqual(root.attrs["rgb_shapes"]["head_rgb"], [3, 8, 12])
        for key in ("head_rgb", "wrist_left_rgb", "wrist_right_rgb"):
            raw = d[f"{key}_jpeg_data"]
            offsets = d[f"{key}_jpeg_offsets"][:]
            self.assertEqual(len(offsets), 6)
            self.assertEqual(offsets[0], 0)
            self.assertEqual(offsets[-1], len(raw))
            self.assertTrue(np.all(np.diff(offsets) > 0))
            first = bytes(raw[offsets[0]:offsets[1]])
            decoded = cv2.imdecode(np.frombuffer(first, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape, (8, 12, 3))
            if key == "wrist_right_rgb":
                self.assertLess(abs(decoded.mean() - 128), 4)

    def test_refuse_unlabeled_reward(self):
        with self.assertRaises(ValueError):
            convert_dataset(_Dataset(), self.path / "cache.zarr")
        self.assertFalse((self.path / "cache.zarr").exists())

    def test_reject_invalid_reward(self):
        obj = json.loads(self.labels.read_text())
        obj["episodes"][0]["reward_status"] = "invalid"
        self.labels.write_text(json.dumps(obj))
        with self.assertRaises(ValueError):
            convert_dataset(
                _Dataset(), self.path / "cache.zarr", reward_json=self.labels
            )
        self.assertFalse((self.path / "cache.zarr").exists())

    def test_reject_terminal_sparse_reward(self):
        obj = json.loads(self.labels.read_text())
        obj["episodes"][0]["reward_index"] = 2
        obj["episodes"][0]["reward_event"]["frame_offset"] = 2
        self.labels.write_text(json.dumps(obj))
        with self.assertRaises(ValueError):
            convert_dataset(
                _Dataset(), self.path / "cache.zarr", reward_json=self.labels
            )

    def test_reject_interleaved_episodes(self):
        with self.assertRaises(ValueError):
            episode_ranges([0, 1, 0, 1])

    def test_reject_wrong_image_scale_and_cleanup(self):
        dataset = _Dataset()
        dataset.frames[3]["observation.images.head_cam_h"] = (
            dataset.frames[3]["observation.images.head_cam_h"].float()
        )
        with self.assertRaises(ValueError):
            convert_dataset(dataset, self.path / "cache.zarr", reward_json=self.labels)
        self.assertFalse((self.path / "cache.zarr").exists())
        self.assertEqual(list(self.path.glob("cache.zarr.partial.*")), [])

    def test_no_overwrite(self):
        out = self.path / "cache.zarr"
        out.mkdir()
        with self.assertRaises(FileExistsError):
            convert_dataset(_Dataset(), out, reward_json=self.labels)

    def test_explicit_proxy_is_marked_invalid_for_rl(self):
        out = convert_dataset(
            _Dataset(), self.path / "proxy.zarr", reward_mode="terminal_proxy"
        )
        root = zarr.open_group(str(out), mode="r")
        self.assertFalse(root.attrs["reward_validated"])
        self.assertEqual(root.attrs["reward_source"], "terminal_proxy_not_success_label")
        np.testing.assert_array_equal(
            root["data"]["reward"][:, 0], [0, 0, 1, 0, 1]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
