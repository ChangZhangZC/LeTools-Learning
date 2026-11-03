"""Feature03: verify no padded action supervision enters native RL100 loss."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "post-rl" / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "rl100" / "RL-100"))
from rl100_compat.train_bc import assert_valid_action_targets, make_episode_sampler  # noqa: E402


def fake_view(ends):
    starts = [0, *ends[:-1]]
    return SimpleNamespace(dataset=SimpleNamespace(
        meta=SimpleNamespace(episodes=[
            {"dataset_from_index": a, "dataset_to_index": b}
            for a, b in zip(starts, ends, strict=True)
        ]),
        episodes=None,
    ))


class BCSamplerTests(unittest.TestCase):
    def test_disallow_tail_padding_without_changing_native_loss(self):
        sampler = make_episode_sampler(fake_view([12, 24]), 2, 9)
        self.assertEqual(len(sampler), 10)
        self.assertEqual(set(sampler.indices), set(range(0, 5)) | set(range(12, 17)))

    def test_leading_padded_action_is_cropped_by_native_actor(self):
        batch = {
            "action": torch.zeros(2, 5, 16),
            "action_is_pad": torch.tensor([
                [True, False, False, False, False],
                [False, False, False, False, False],
            ]),
        }
        assert_valid_action_targets(batch, 2)
        batch["action_is_pad"][0, -1] = True
        with self.assertRaisesRegex(ValueError, "Padded supervision"):
            assert_valid_action_targets(batch, 2)

    def test_short_episodes_skipped(self):
        sampler = make_episode_sampler(fake_view([4, 20]), 2, 9)
        self.assertEqual(len(sampler), 9)
        self.assertEqual(set(sampler.indices), set(range(4, 13)))


if __name__ == "__main__":
    unittest.main()
