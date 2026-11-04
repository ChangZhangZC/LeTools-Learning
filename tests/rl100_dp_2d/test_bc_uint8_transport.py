"""Verify compact uint8 transport preserves the native RL100 float32 batch."""
import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kuavo_model.rl100_dp_2d.bc_utils import _to_device
from kuavo_model.rl100_dp_2d.dataset import RL100LeRobotDataset, contract_from_features


class FakeSource:
    features = {
        "observation.state": {"dtype": "float32", "shape": [2]},
        "action": {"dtype": "float32", "shape": [2]},
        "observation.images.head_cam_h": {"dtype": "video", "shape": [3, 2, 2]},
        "observation.images.wrist_cam_l": {"dtype": "video", "shape": [3, 2, 2]},
    }

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "observation.state": torch.ones(2, 2),
            "action": torch.ones(5, 2),
            "action_is_pad": torch.zeros(5, dtype=torch.bool),
            **{
                key: torch.arange(24, dtype=torch.uint8).reshape(2, 3, 2, 2)
                for key in self.features if key.startswith("observation.images.")
            },
        }


class Uint8TransportTests(unittest.TestCase):
    def test_dataloader_keeps_uint8_and_actor_receives_original_float32(self):
        raw = FakeSource()
        view = RL100LeRobotDataset(raw, contract_from_features(raw.features), 2, 5)
        sample = next(iter(DataLoader(view, batch_size=2, num_workers=0)))
        for key, tensor in sample["obs"].items():
            if key.startswith("rgb_"):
                self.assertEqual(tensor.dtype, torch.uint8)
        batch = _to_device(sample, torch.device("cpu"))
        for key, tensor in batch["obs"].items():
            if key.startswith("rgb_"):
                self.assertEqual(tensor.dtype, torch.float32)
                torch.testing.assert_close(tensor, sample["obs"][key].float() / 255,
                                           rtol=0, atol=0)
        torch.testing.assert_close(batch["action"], sample["action"])
        torch.testing.assert_close(batch["obs"]["agent_pos"], sample["obs"]["agent_pos"])
        self.assertNotIn("action_is_pad", batch)

    def test_non_uint8_rgb_rejected_or_preserved_as_float(self):
        raw = FakeSource()
        view = RL100LeRobotDataset(raw, contract_from_features(raw.features), 2, 5)
        sample = next(iter(DataLoader(view, batch_size=1, num_workers=0)))
        key = "rgb_head_cam_h"
        original = sample["obs"][key]
        sample["obs"][key] = original.float() / 255
        batch = _to_device(sample, torch.device("cpu"))
        torch.testing.assert_close(batch["obs"][key], sample["obs"][key], rtol=0, atol=0)
        sample["obs"][key] = original.short()
        with self.assertRaises(TypeError):
            _to_device(sample, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
