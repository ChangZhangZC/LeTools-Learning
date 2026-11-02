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
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import zarr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "post_training" / "src"))
from arbim_rl100.data.lerobot_to_zarr import convert_dataset, episode_ranges, _rgb, RGB_FEATURES


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

    def test_legacy_float_255_image_supported(self):
        image = np.full((3, 8, 12), 127.0, dtype=np.float32)
        np.testing.assert_array_equal(
            _rgb(image, "rgb"), np.full((3, 8, 12), 127, dtype=np.uint8)
        )
        hwc = np.transpose(image, (1, 2, 0))
        np.testing.assert_array_equal(_rgb(hwc, "rgb"), _rgb(image, "rgb"))

    def test_original_reward_v2_computation(self):
        episodes = [0] * 12 + [1] * 12
        dataset = _Dataset(episodes=episodes)
        for index, frame in enumerate(dataset.frames):
            action = torch.zeros(16, dtype=torch.float32)
            local_step = index % 12
            if 3 <= local_step < 7:
                action[7] = action[15] = 1.0
            frame["action"] = action
        out = convert_dataset(
            dataset, self.path / "computed.zarr",
            reward_mode="last_gripper_release_sparse",
        )
        root = zarr.open_group(str(out), mode="r")
        self.assertTrue(root.attrs["reward_validated"])
        self.assertEqual(root.attrs["reward_source"], "computed_last_gripper_release")
        reward = root["data"]["reward"][:, 0]
        self.assertEqual(int(np.count_nonzero(reward)), 2)
        self.assertEqual(reward[[7, 19]].tolist(), [1.0, 1.0])

    def test_original_optional_depth_arrays(self):
        dataset = _Dataset()
        depth_names = ("observation.depth_h", "observation.depth_l", "observation.depth_r")
        for i, frame in enumerate(dataset.frames):
            for j, key in enumerate(depth_names):
                frame[key] = np.full((1, 5, 7), i * 10 + j, dtype=np.uint16)
        out = convert_dataset(
            dataset, self.path / "depth.zarr", reward_json=self.labels,
            use_depth=True,
        )
        root = zarr.open_group(str(out), mode="r")
        self.assertTrue(root.attrs["use_depth"])
        data = root["data"]
        self.assertEqual(data["head_depth"].dtype, np.dtype("uint16"))
        self.assertTrue(np.all(data["head_depth"][0] == 0))
        self.assertTrue(np.all(data["next_head_depth"][0] == 10))
        self.assertTrue(np.all(data["next_head_depth"][2] == 20))

    def test_reused_legacy_batch_video_decoder(self):
        dataset = _Dataset()
        dataset.meta = SimpleNamespace(video_keys=list(RGB_FEATURES))
        dataset.hf_dataset.update({
            "observation.state": torch.stack([f["observation.state"] for f in dataset.frames]),
            "action": torch.stack([f["action"] for f in dataset.frames]),
            "timestamp": torch.tensor([0.0, 0.1, 0.2, 0.0, 0.1]),
        })
        lookup = {
            (int(dataset.hf_dataset["episode_index"][i]), round(float(dataset.hf_dataset["timestamp"][i]), 3)): i
            for i in range(len(dataset))
        }
        calls = []
        class FakeReader:
            def _query_videos(self, query_timestamps, episode_id):
                calls.append(len(next(iter(query_timestamps.values()))))
                return {
                    key: torch.stack([
                        dataset.frames[lookup[(episode_id, round(float(t), 3))]][key]
                        for t in ts
                    ])
                    for key, ts in query_timestamps.items()
                }
        dataset.reader = FakeReader()
        output = convert_dataset(dataset, self.path / "batch.zarr", reward_json=self.labels,
                                 stream_batch_size=2)
        self.assertEqual(calls, [2, 1, 2])
        root = zarr.open_group(str(output), mode="r")
        self.assertEqual(root["meta"]["episode_ends"][:].tolist(), [3, 5])

    def test_original_shaped_reward_mode(self):
        dataset = _Dataset()
        out = convert_dataset(
            dataset, self.path / "shaped.zarr",
            reward_mode="episode_end_shaped",
            reward_config={"lambda_penalty": 0.05, "smooth_penalty": 0.0, "max_episode_len": 2000},
        )
        root = zarr.open_group(str(out), mode="r")
        self.assertEqual(root.attrs["reward_mode"], "episode_end_shaped")
        self.assertFalse(root.attrs["reward_validated"])
        actual = root["data"]["reward"][:, 0]
        expected = [0, 0, 1 - 0.05 * 3 / 2000, 0, 1 - 0.05 * 2 / 2000]
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_original_recorded_and_computed_reward_mixing(self):
        dataset = _Dataset(episodes=[0] * 12 + [1] * 12)
        for index, frame in enumerate(dataset.frames):
            action = torch.zeros(16, dtype=torch.float32)
            if 3 <= (index % 12) < 7:
                action[7] = action[15] = 1.0
            frame["action"] = action
        source = self.path / "lerobot_source"
        meta = source / "meta"
        meta.mkdir(parents=True)
        manifest = {
            "format": "arbim_lerobot_staging_provenance_v1",
            "episodes": [
                {
                    "lerobot_episode_index": 0, "num_frames": 12,
                    "reward_mode": "last_gripper_release_sparse",
                    "reward_status": "valid",
                    "reward_index": 5, "reward_event": {"frame_offset": 5},
                },
                {
                    "lerobot_episode_index": 1, "num_frames": 12,
                    "reward_mode": "last_gripper_release_sparse",
                    "reward_status": "valid",
                    "reward_index": 7, "reward_event": {"frame_offset": 7},
                },
            ],
        }
        (meta / "arbim_staging_provenance.json").write_text(json.dumps(manifest))
        out = convert_dataset(
            dataset, self.path / "recorded.zarr",
            reward_mode="last_gripper_release_sparse", source_path=source,
        )
        root = zarr.open_group(str(out), mode="r")
        self.assertEqual(root.attrs["reward_source"], "staging_recorded_v1")
        np.testing.assert_array_equal(
            root["data"]["reward"][:, 0][[5, 19]], [1, 1]
        )
        self.assertEqual(int(np.count_nonzero(root["data"]["reward"][:])), 2)

        manifest["episodes"][1]["reward_status"] = "invalid"
        manifest["episodes"][1]["reward_index"] = None
        manifest["episodes"][1]["reward_event"] = None
        (meta / "arbim_staging_provenance.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "invalid recorded Reward V2"):
            convert_dataset(
                dataset, self.path / "bad_recorded.zarr",
                reward_mode="last_gripper_release_sparse", source_path=source,
            )
        (meta / "arbim_staging_provenance.json").unlink()
        merged = {
            "format": "arbim_lerobot_merge_provenance_v1",
            "episodes": [
                {"output_episode_index": 0, "frames": 12,
                 "staging_reward": {
                     "reward_status": "valid",
                     "reward_mode": "last_gripper_release_sparse",
                     "reward_index": 5, "reward_event": {"frame_offset": 5},
                 }},
                {"output_episode_index": 1, "frames": 12,
                 "staging_reward": None},
            ],
        }
        (meta / "arbim_merge_provenance.json").write_text(json.dumps(merged))
        out = convert_dataset(
            dataset, self.path / "mixed.zarr",
            reward_mode="last_gripper_release_sparse", source_path=source,
        )
        mixed = zarr.open_group(str(out), mode="r")
        self.assertEqual(mixed.attrs["reward_source"], "mixed_lerobot_reward_v2")
        self.assertTrue(mixed.attrs["reward_validated"])
        reward = mixed["data"]["reward"][:, 0]
        self.assertEqual(int(np.count_nonzero(reward)), 2)
        np.testing.assert_array_equal(reward[[5, 19]], [1, 1])

    def test_reject_zarr3_with_actionable_error(self):
        from unittest.mock import patch

        with patch.object(zarr, "__version__", "3.4.2"):
            with self.assertRaisesRegex(RuntimeError, "zarr==2.18.7"):
                convert_dataset(
                    _Dataset(), self.path / "cache.zarr", reward_json=self.labels
                )
        self.assertFalse((self.path / "cache.zarr").exists())

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
        self.assertEqual(root.attrs["reward_event"]["episodes"], [
            {"episode_index": 0, "reward_offset": 1},
            {"episode_index": 1, "reward_offset": 0},
        ])
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
            dataset.frames[3]["observation.images.head_cam_h"].float() + 300
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
