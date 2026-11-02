"""Original kdc_ARBiM OfflineBuffer/Sampler/Dataset -> native RL1002D adapter.

Uses the existing synthetic LeRobot fixture to build a genuine JPEG Zarr.
Run from LeTools repo root:
  python post_training/tests/smoke/smoke_01_offline_reader.py
"""
from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "post_training" / "src"))
from arbim_rl100.data.batch_adapter import to_rl100_bc_batch
from arbim_rl100.data.lerobot_to_zarr import convert_dataset
from arbim_rl100.data.offline_buffer import OfflineBuffer
from arbim_rl100.data.offline_dataset import OfflineDataset
from smoke_01_lerobot_to_zarr import _Dataset, _recorded_json


class OfflineReaderIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        labels = self.root / "reward.json"
        _recorded_json(labels)
        self.zarr_path = convert_dataset(
            _Dataset(), self.root / "source.zarr",
            reward_json=labels, gamma=0.9, stream_batch_size=2,
        )
        self.buffer = OfflineBuffer(device=torch.device("cpu"), gamma=0.9)
        self.buffer.load_zarr(str(self.zarr_path))

    def test_original_buffer_and_rl100_adapter(self):
        self.assertEqual(len(self.buffer), 5)
        self.assertEqual(self.buffer.episode_ends.tolist(), [3, 5])
        self.assertEqual(self.buffer["head_rgb"].shape, (5, 3, 8, 12))
        self.assertEqual(self.buffer["head_rgb"][0].dtype, np.uint8)
        self.assertEqual(self.buffer["next_head_rgb"][0].shape, (3, 8, 12))
        self.assertEqual(self.buffer["next_head_rgb"][2].shape, (3, 8, 12))
        np.testing.assert_array_equal(self.buffer["next_state"][2], self.buffer["state"][2])
        np.testing.assert_array_equal(self.buffer["next_state"][3], self.buffer["state"][4])
        np.testing.assert_allclose(self.buffer["return"][:, 0], [0.9, 1, 0, 1, 0], rtol=1e-6)

        dataset = OfflineDataset(self.buffer, horizon=2, val_ratio=0.0)
        self.assertEqual(len(dataset), 3)
        expected = [[0, 1], [1, 2], [3, 4]]
        for i, indices in enumerate(expected):
            sample = dataset[i]
            self.assertEqual(sample["action"].shape, (2, 3))
            np.testing.assert_array_equal(
                sample["obs"]["state"][:, 0].numpy(), indices
            )
            self.assertEqual(sample["next_obs"]["head_rgb"].shape, (2, 3, 8, 12))
            batch = to_rl100_bc_batch(sample, n_obs_steps=2)
            self.assertEqual(batch["obs"]["agent_pos"].shape, (1, 2, 2))
            self.assertEqual(batch["obs"]["rgb_head"].shape, (1, 2, 3, 8, 12))
            self.assertEqual(batch["action"].shape, (1, 2, 3))
            self.assertEqual(batch["obs"]["rgb_head"].dtype, torch.float32)

    def test_episode_split_and_padding_preserved(self):
        dataset = OfflineDataset(
            self.buffer, horizon=2, val_ratio=0.5, seed=1
        )
        validation = dataset.get_validation_dataset()
        self.assertEqual(len(dataset) + len(validation), 3)
        self.assertEqual(int(dataset.train_mask.sum()), 1)
        self.assertEqual(int(dataset.val_mask.sum()), 1)
        self.assertFalse(np.any(dataset.train_mask & dataset.val_mask))

        padded = OfflineDataset(
            self.buffer, horizon=4, pad_before=1, pad_after=1
        )
        self.assertEqual(len(padded), 3)
        starts = [item["obs"]["state"][:, 0].tolist() for item in padded]
        self.assertEqual(starts, [[0, 0, 1, 2], [0, 1, 2, 2], [3, 3, 4, 4]])

    def test_lazy_jpeg_handle_can_reopen_after_pickle(self):
        view = pickle.loads(pickle.dumps(self.buffer["head_rgb"]))
        self.assertEqual(view[0].shape, (3, 8, 12))
        self.assertEqual(view[[0, 1, 3]].shape, (3, 3, 8, 12))
        self.assertEqual(view[:0].shape, (0, 3, 8, 12))


if __name__ == "__main__":
    unittest.main(verbosity=2)
