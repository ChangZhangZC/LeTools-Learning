"""Batch 01E: original ARBiM jpeg_bytes_v1 Zarr merge regression.

The production zarr_merge.py was copied from kdc_ARBiM@feature/flywheel,
with only necessary Reward-provenance metadata propagation changes.
Run from the LeTools repo root:
  python post_training/tests/smoke/smoke_01_zarr_merge.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import zarr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "post_training" / "src"))

from arbim_rl100.data.lerobot_to_zarr import convert_dataset
from arbim_rl100.data.offline_buffer import OfflineBuffer
from arbim_rl100.data.zarr_merge import merge_zarr_datasets
from smoke_01_lerobot_to_zarr import _Dataset, _recorded_json


class ZarrMergeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        labels = self.root / "reward.json"
        _recorded_json(labels)
        self.first = convert_dataset(
            _Dataset(), self.root / "first.zarr",
            reward_json=labels, stream_batch_size=2, gamma=0.9,
        )
        self.second = convert_dataset(
            _Dataset(), self.root / "second.zarr",
            reward_json=labels, stream_batch_size=2, gamma=0.9,
        )
        self.output = self.root / "merged.zarr"

    def merge(self, sources=None, **settings):
        cfg = {
            "zarr_sources": sources if sources is not None else [str(self.first), str(self.second)],
            "zarr_output_path": str(self.output),
            "overwrite": False,
            "return_gamma": 0.9,
        }
        cfg.update(settings)
        return merge_zarr_datasets(cfg)

    def test_merge_offsets_next_index_and_reward(self):
        result = self.merge()
        self.assertEqual(result["num_frames"], 10)
        self.assertEqual(result["num_episodes"], 4)
        root = zarr.open_group(str(self.output), mode="r")
        d = root["data"]
        np.testing.assert_array_equal(root["meta"]["episode_ends"][:], [3, 5, 8, 10])
        np.testing.assert_array_equal(d["next_index"][:], [1, 2, 2, 4, 4, 6, 7, 7, 9, 9])
        np.testing.assert_array_equal(d["reward"][:, 0], [0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
        np.testing.assert_allclose(
            d["return"][:, 0], [0.9, 1, 0, 1, 0, 0.9, 1, 0, 1, 0], rtol=1e-6
        )
        for name in ("head_rgb", "wrist_left_rgb", "wrist_right_rgb"):
            offsets = d[f"{name}_jpeg_offsets"][:]
            self.assertEqual(len(offsets), 11)
            self.assertEqual(int(offsets[0]), 0)
            self.assertEqual(int(offsets[-1]), len(d[f"{name}_jpeg_data"]))
            self.assertTrue(np.all(np.diff(offsets) > 0))
            original_offsets = zarr.open_group(str(self.first), mode="r")["data"][f"{name}_jpeg_offsets"][:]
            self.assertEqual(int(offsets[5]), int(original_offsets[-1]))
            self.assertEqual(int(offsets[6] - offsets[5]), int(original_offsets[1] - original_offsets[0]))

        buffer = OfflineBuffer(device=torch.device('cpu'), gamma=0.9)
        buffer.load_zarr(str(self.output))
        self.assertEqual(len(buffer), 10)
        self.assertEqual(buffer["head_rgb"][5].shape, (3, 8, 12))
        self.assertEqual(buffer.episode_ends.tolist(), [3, 5, 8, 10])

    def test_preserves_reward_provenance(self):
        self.merge()
        root = zarr.open_group(str(self.output), mode="r")
        self.assertEqual(root.attrs["reward_mode"], "last_gripper_release_sparse")
        self.assertTrue(root.attrs["reward_validated"])
        self.assertEqual(root.attrs["reward_source"], "recorded_validated")
        inputs = root.attrs["merge_inputs"]
        self.assertEqual(len(inputs), 2)
        self.assertEqual([item["frames"] for item in inputs], [5, 5])
        for item in inputs:
            self.assertTrue(item["reward_validated"])
            self.assertEqual(item["reward_event"]["episodes"], [
                {"episode_index": 0, "reward_offset": 1},
                {"episode_index": 1, "reward_offset": 0},
            ])
        # Per-source reward events are deliberately local, not falsely rebased.
        self.assertEqual(root.attrs["source_manifest"], [
            *zarr.open_group(str(self.first), mode="r").attrs["source_manifest"],
            *zarr.open_group(str(self.second), mode="r").attrs["source_manifest"],
        ])

    def test_refuses_terminal_proxy_as_verified_reward(self):
        proxy = convert_dataset(
            _Dataset(), self.root / "proxy.zarr", reward_mode="terminal_proxy"
        )
        with self.assertRaisesRegex(ValueError, "unsupported Reward mode"):
            self.merge(sources=[str(self.first), str(proxy)])
        self.assertFalse(self.output.exists())

    def test_reject_incompatible_action_shape_and_duplicate(self):
        second = zarr.open_group(str(self.second), mode="a")
        second["data"]["action"].resize((5, 4))
        second["data"]["next_action"].resize((5, 4))
        with self.assertRaisesRegex(ValueError, "incompatible action dtype or trailing dimensions"):
            self.merge()
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError, "duplicate Zarr source"):
            self.merge(sources=[str(self.first), str(self.first)])

    def test_no_overwrite_and_original_sources_unchanged(self):
        before = [
            zarr.open_group(str(path), mode="r")["data"]["next_index"][:].copy()
            for path in (self.first, self.second)
        ]
        self.merge()
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            self.merge()
        for path, old in zip((self.first, self.second), before, strict=True):
            np.testing.assert_array_equal(
                zarr.open_group(str(path), mode="r")["data"]["next_index"][:], old
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
