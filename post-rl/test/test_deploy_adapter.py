"""Feature04: isolated adapter behavior checks; no server/robot required."""
from __future__ import annotations

import sys
import unittest
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kuavo_server.adapters.rl100_2d import RL1002DKuavoAdapter  # noqa: E402
from kuavo_server.builtin_adapters import ensure_adapter_loaded  # noqa: E402
from kuavo_server.runtime import get_adapter_class  # noqa: E402


class FakePolicy:
    def __init__(self):
        self.calls = 0
        self.last_obs = None

    def predict_action(self, obs, deterministic=False):
        self.calls += 1
        self.last_obs = obs
        assert deterministic
        row = torch.tensor([0.1, 0.4, 0.7, 0.9])
        return {"action": row.repeat(1, 3, 1)}


def fake_adapter(*, obs_steps=2, execution_horizon=1):
    obj = object.__new__(RL1002DKuavoAdapter)
    obj.state_dim = 4
    obj.action_dim = 4
    obj.n_obs_steps = obs_steps
    obj.model_n_action_steps = 3
    obj.execution_horizon = execution_horizon
    obj.device = torch.device("cpu")
    obj.camera_map = {
        "observation.images.head_cam_h": "rgb_head_cam_h",
        "observation.images.wrist_cam_l": "rgb_wrist_cam_l",
    }
    obj.expected_images = {key: (3, 4, 6) for key in obj.camera_map}
    obj.state_names = tuple(f"state_{i}" for i in range(4))
    obj.action_names = ("joint_1", "left_claw", "joint_2", "right_claw")
    obj.binary_action_names = ()
    obj.binary_action_indices = ()
    obj.binary_threshold = 0.5
    obj.resize_images = False
    obj.which_arm = "both"
    obj.checkpoint = Path("fake.pt")
    obj.model = FakePolicy()
    obj._history = deque(maxlen=obs_steps)
    obj._pending_actions = []
    return obj


def obs(state=1, *, hwc=False, width=6):
    image = torch.zeros(1, 3, 4, width, dtype=torch.float32)
    if hwc:
        image = (image[0].permute(1, 2, 0) * 255).to(torch.uint8)
    return {
        "observation.state": torch.full((1, 4), float(state)),
        "observation.images.head_cam_h": image,
        "observation.images.wrist_cam_l": image,
    }


class DeployAdapterUnitTests(unittest.TestCase):
    def test_registered_only_via_builtin_map(self):
        ensure_adapter_loaded("rl100_2d")
        self.assertIs(get_adapter_class("rl100_2d"), RL1002DKuavoAdapter)

    def test_image_and_state_history_reset(self):
        adapter = fake_adapter()
        chunk = adapter.select_action_chunk(obs(1, hwc=True))
        self.assertEqual(chunk.shape, (1, 4))
        self.assertEqual(chunk.dtype, np.float64)
        self.assertEqual(tuple(adapter.model.last_obs["agent_pos"].shape), (1, 2, 4))
        self.assertEqual(tuple(adapter.model.last_obs["rgb_head_cam_h"].shape), (1, 2, 3, 4, 6))
        self.assertTrue(torch.all(adapter.model.last_obs["agent_pos"] == 1))

        adapter.select_action_chunk(obs(2))
        self.assertTrue(torch.all(adapter.model.last_obs["agent_pos"][0, 0] == 1))
        self.assertTrue(torch.all(adapter.model.last_obs["agent_pos"][0, 1] == 2))
        adapter.reset()
        adapter.select_action_chunk(obs(7))
        self.assertTrue(torch.all(adapter.model.last_obs["agent_pos"] == 7))

    def test_stepwise_fallback_caches_one_chunk(self):
        adapter = fake_adapter(obs_steps=1, execution_horizon=3)
        self.assertEqual(adapter.model.calls, 0)
        for _ in range(3):
            step = adapter.select_action(obs(1))
            self.assertEqual(step.shape, (4,))
        self.assertEqual(adapter.model.calls, 1)
        adapter.select_action(obs(2))
        self.assertEqual(adapter.model.calls, 2)
        adapter.reset()
        adapter.select_action(obs(9))
        self.assertEqual(adapter.model.calls, 3)

    def test_binary_action_explicit_by_saved_name(self):
        adapter = fake_adapter()
        adapter.binary_action_names = ("left_claw", "right_claw")
        adapter.binary_action_indices = (1, 3)
        chunk = adapter.select_action_chunk(obs())
        np.testing.assert_allclose(chunk[0], [0.1, 0.0, 0.7, 1.0], atol=1e-6)

    def test_fail_on_missing_camera_state_or_resolution(self):
        adapter = fake_adapter()
        with self.assertRaises(KeyError):
            adapter.select_action_chunk({"observation.state": torch.zeros(1, 4)})
        with self.assertRaisesRegex(ValueError, "state"):
            adapter.select_action_chunk({**obs(), "observation.state": torch.zeros(1, 3)})
        with self.assertRaisesRegex(ValueError, "image shape"):
            adapter.select_action_chunk(obs(width=8))

    def test_resolution_conversion_requires_explicit_opt_in(self):
        adapter = fake_adapter()
        adapter.resize_images = True
        adapter.select_action_chunk(obs(width=8))
        self.assertEqual(tuple(adapter.model.last_obs["rgb_head_cam_h"].shape), (1, 2, 3, 4, 6))


if __name__ == "__main__":
    unittest.main()
