"""Synthetic contract checks for ARBiM offline batches -> native RL1002D.

No rosbag, LeRobot dataset, GPU, or training checkpoint required.
Run: python post_training/tests/smoke/smoke_01_rl100_batch_contract.py
"""
from __future__ import annotations

import pickle
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "post_training" / "src"))

from arbim_rl100.data.batch_adapter import adapt_rl100_shape_meta, to_rl100_bc_batch


class RL100DataContractTest(unittest.TestCase):
    def make_sample(self, batched: bool = True):
        obs = {
            "state": torch.zeros(2, 7, 16, dtype=torch.float32),
            "head_rgb": torch.full((2, 7, 3, 8, 8), 255, dtype=torch.uint8),
            "wrist_left_rgb": torch.zeros(2, 7, 3, 6, 8, dtype=torch.uint8),
            "wrist_right_rgb": torch.full((2, 7, 3, 8, 6), 128, dtype=torch.uint8),
        }
        action = torch.zeros(2, 7, 18, dtype=torch.float32)
        if not batched:
            obs = {key: value[0] for key, value in obs.items()}
            action = action[0]
        return {"obs": obs, "action": action, "reward": torch.zeros(2, 7, 1)}

    def test_rl100_shape_meta_compat_is_picklable_and_nonmutating(self):
        from collections import UserDict

        metadata = {
            "obs": {
                "rgb_head": {"shape": [3, 84, 84], "type": "rgb"},
                "agent_pos": {"shape": [16], "type": "low_dim"},
            },
            "action": {"shape": [16]},
        }
        adapted = adapt_rl100_shape_meta(metadata)
        self.assertEqual(adapted["obs"]["agent_pos"]["shape"], [16])
        self.assertIsInstance(adapted["obs"]["rgb_head"], UserDict)
        self.assertFalse(isinstance(adapted["obs"]["rgb_head"], dict))
        self.assertIsInstance(metadata["obs"]["rgb_head"], dict)
        self.assertEqual(pickle.loads(pickle.dumps(adapted)), adapted)

    def test_batched(self):
        src = self.make_sample()
        original = src["obs"]["head_rgb"].clone()
        result = to_rl100_bc_batch(src, n_obs_steps=2)
        self.assertEqual(set(result["obs"]), {
            "agent_pos", "rgb_head", "rgb_left_hand", "rgb_right_hand"
        })
        self.assertEqual(tuple(result["obs"]["agent_pos"].shape), (2, 2, 16))
        self.assertEqual(tuple(result["obs"]["rgb_head"].shape), (2, 2, 3, 8, 8))
        self.assertEqual(tuple(result["obs"]["rgb_left_hand"].shape), (2, 2, 3, 6, 8))
        self.assertEqual(tuple(result["action"].shape), (2, 7, 18))
        self.assertEqual(result["obs"]["rgb_head"].dtype, torch.float32)
        self.assertEqual(result["obs"]["rgb_head"].max().item(), 1.0)
        self.assertAlmostEqual(result["obs"]["rgb_right_hand"][0, 0, 0, 0, 0].item(), 128 / 255, places=6)
        self.assertTrue(torch.equal(src["obs"]["head_rgb"], original))
        self.assertEqual(set(result), {"obs", "action"})

    def test_unbatched(self):
        result = to_rl100_bc_batch(self.make_sample(False), n_obs_steps=2)
        self.assertEqual(tuple(result["action"].shape), (1, 7, 18))
        self.assertEqual(tuple(result["obs"]["agent_pos"].shape), (1, 2, 16))

    def test_reject_hwc(self):
        src = self.make_sample()
        src["obs"]["head_rgb"] = src["obs"]["head_rgb"].permute(0, 1, 3, 4, 2)
        with self.assertRaises(ValueError):
            to_rl100_bc_batch(src, n_obs_steps=2)

    def test_reject_missing_camera(self):
        src = self.make_sample()
        del src["obs"]["wrist_left_rgb"]
        with self.assertRaises(KeyError):
            to_rl100_bc_batch(src, n_obs_steps=2)

    def test_reject_short_horizon(self):
        src = self.make_sample()
        src["action"] = src["action"][:, :1]
        with self.assertRaises(ValueError):
            to_rl100_bc_batch(src, n_obs_steps=2)

    def test_reject_float_rgb_255(self):
        src = self.make_sample()
        src["obs"]["head_rgb"] = src["obs"]["head_rgb"].float()
        with self.assertRaises(ValueError):
            to_rl100_bc_batch(src, n_obs_steps=2)

    def test_reject_nonfinite_action(self):
        src = self.make_sample()
        src["action"][0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            to_rl100_bc_batch(src, n_obs_steps=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
