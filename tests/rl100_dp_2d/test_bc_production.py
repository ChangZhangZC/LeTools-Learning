"""Feature06 BC trainer contracts; no LeRobot files, GPU, ROS, or simulator needed."""
from __future__ import annotations

import copy
import io
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "rl100" / "RL-100"))

from kuavo_model.rl100_dp_2d import train as prod  # noqa: E402
from rl_100.model.common.normalizer import LinearNormalizer  # noqa: E402


class FakeTable:
    def __init__(self, rows):
        self.rows = rows

    def select_columns(self, names):
        return self

    def __getitem__(self, index):
        return self.rows[index]


class FakeRaw:
    def __init__(self, episode_count=4, episode_length=12):
        self.fps = 10
        self.meta = SimpleNamespace(episodes=[
            {
                "dataset_from_index": i * episode_length,
                "dataset_to_index": (i + 1) * episode_length,
            } for i in range(episode_count)
        ])
        self.episodes = None
        self.features = {
            "observation.state": {"shape": [2]},
            "action": {"shape": [2]},
        }
        self.hf_dataset = FakeTable([
            {"observation.state": [float(i), -float(i)],
             "action": [float(i) / 10, -float(i) / 10]}
            for i in range(episode_count * episode_length)
        ])


class FakeView(torch.utils.data.Dataset):
    def __init__(self):
        self.dataset = FakeRaw()
        self.contract = SimpleNamespace(
            shape_meta={
                "obs": {"agent_pos": {"shape": [2], "type": "low_dim"}},
                "action": {"shape": [2]},
            },
            source_to_rgb={"observation.images.head_cam_h": "rgb_head_cam_h"},
            state_names=("s0", "s1"),
            action_names=("a0", "a1"),
            state_dim=2,
            action_dim=2,
        )

    def __len__(self):
        return 48

    def __getitem__(self, index):
        row = self.dataset.hf_dataset[index]
        state = torch.as_tensor(row["observation.state"], dtype=torch.float32)
        action = torch.as_tensor(row["action"], dtype=torch.float32)
        return {
            "obs": {"agent_pos": state.repeat(2, 1)},
            "action": action.repeat(9, 1),
            "action_is_pad": torch.zeros(9, dtype=torch.bool),
        }


class TinyActor(torch.nn.Module):
    """Minimal differentiable surrogate for trainer-only resume checks."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)
        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        state = self.normalizer["agent_pos"].normalize(batch["obs"]["agent_pos"][:, -1])
        target = self.normalizer["action"].normalize(batch["action"][:, 1])
        prediction = self.layer(state)
        # Controlled stochasticity tests restoration of torch RNG across resume.
        prediction = prediction + 0.01 * torch.randn_like(prediction)
        return (prediction - target).square().mean(), {}


def config_for_test(output):
    cfg = OmegaConf.load(ROOT / "configs" / "train" / "rl100_dp_2d.yaml")
    cfg.data.horizon = 9
    cfg.data.n_action_steps = 4
    cfg.data.repo_id = "fake/robot"
    cfg.data.root = "fake"
    cfg.training.steps = 3
    cfg.training.batch_size = 2
    cfg.training.num_workers = 0
    cfg.training.device = "cpu"
    cfg.training.seed = 9
    cfg.training.use_ema = True
    cfg.training.ema_power = 0.75
    cfg.training.resume = False
    cfg.training.log_every = 1
    cfg.training.val_every = 1
    cfg.training.save_every = 1
    cfg.training.keep_step_checkpoints = False
    cfg.training.max_grad_norm = 1.0
    cfg.validation = {
        "fraction": 0.25, "split_seed": 9, "seed": 100,
        "batch_size": 2, "max_batches": 0,
    }
    cfg.output.dir = str(output)
    return cfg


class ProductionBCUnitTests(unittest.TestCase):
    def test_canonical_bc_defaults_32_predicted_16_selected(self):
        cfg = OmegaConf.load(ROOT / "configs/train/rl100_dp_2d.yaml")
        prod.check_config(cfg)
        self.assertEqual(int(cfg.data.n_obs_steps), 2)
        self.assertEqual(int(cfg.data.horizon) - int(cfg.data.n_obs_steps) + 1, 32)
        self.assertEqual(int(cfg.data.n_action_steps), 16)
        self.assertEqual(str(cfg.output.dir), "outputs/train/rl100_dp_2d")
        self.assertFalse(bool(cfg.wandb.enable))
        self.assertFalse(bool(cfg.tensorboard.enable))

    def test_split_by_episode_never_by_frame(self):
        first = prod.split_episode_ids(10, .2, 42)
        self.assertEqual(first, prod.split_episode_ids(10, .2, 42))
        self.assertEqual(len(first[0]), 8)
        self.assertEqual(len(first[1]), 2)
        self.assertEqual(set(first[0]) & set(first[1]), set())
        self.assertEqual(set(first[0]) | set(first[1]), set(range(10)))
        with self.assertRaises(ValueError):
            prod.split_episode_ids(1, .2, 42)

    def test_unpadded_episode_sampler_excludes_holdout(self):
        view = FakeView()
        anchors = prod.episode_anchor_indices(view, [0, 2], 2, 9)
        self.assertEqual(anchors, list(range(5)) + list(range(24, 29)))
        self.assertFalse(set(anchors) & set(range(12, 24)))

    def test_restarted_step_batches_match_uninterrupted(self):
        anchors = list(range(11))
        args = (anchors, 3, 42)
        whole = list(prod.StepBatchSampler(*args, start_step=0, stop_step=15))
        head = list(prod.StepBatchSampler(*args, start_step=0, stop_step=7))
        tail = list(prod.StepBatchSampler(*args, start_step=7, stop_step=15))
        self.assertEqual(whole, head + tail)
        self.assertEqual(len(whole), 15)
        with self.assertRaises(ValueError):
            list(prod.StepBatchSampler([], 3, 42, 0, 1))

    def test_training_only_stats_do_not_expose_val_outliers(self):
        ds = FakeRaw(episode_count=3, episode_length=3)
        normalizer = prod.build_train_only_normalizer(ds, [0, 1])
        stats = normalizer.get_input_stats()
        np.testing.assert_allclose(stats["agent_pos"]["min"].tolist(), [0, -5])
        np.testing.assert_allclose(stats["agent_pos"]["max"].tolist(), [5, 0])
        np.testing.assert_allclose(stats["action"]["min"].tolist(), [0, -.5])
        self.assertLess(stats["agent_pos"]["max"][0].item(), 8)

    def test_config_rejects_invalid_denoising_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config_for_test(tmp)
            prod.check_config(cfg)
            cfg.data.horizon = 10  # internal 9 steps not divisible by 8
            with self.assertRaisesRegex(ValueError, "denoising horizon"):
                prod.check_config(cfg)

    def test_rng_checkpoint_uses_safe_torch_load(self):
        prod._seed_all(100)
        state = prod.capture_rng_state()
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=True)
        baseline = (random.random(), float(np.random.rand()), float(torch.rand(())))
        prod.restore_rng_state(restored)
        actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
        self.assertEqual(baseline, actual)

    def test_ema_buffers_follow_model(self):
        actor = torch.nn.BatchNorm1d(2)
        from rl_100.model.diffusion.ema_model import EMAModel
        ema = EMAModel(model=copy.deepcopy(actor))
        actor.running_mean.fill_(3.0)
        prod._sync_ema_buffers(ema, actor)
        self.assertTrue(torch.all(ema.averaged_model.running_mean == 3.0))

    def test_resume_restores_model_optimizer_scheduler_ema_and_rng(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            resumed_cfg = config_for_test(root / "resumed")
            full_cfg = config_for_test(root / "uninterrupted")

            def fake_open(*args, **kwargs):
                return FakeView()

            def fake_policy(cfg, contract):
                return TinyActor()

            with (
                patch.object(prod, "open_lerobot_bc_dataset", side_effect=fake_open),
                patch.object(prod, "build_policy", side_effect=fake_policy),
            ):
                prod.train(resumed_cfg, stop_after=2)
                latest_path = root / "resumed" / "checkpoints" / "latest.pt"
                intermediate = torch.load(latest_path, weights_only=True)
                self.assertEqual(intermediate["step"], 2)
                self.assertEqual(intermediate["checkpoint_role"], "training_latest")
                self.assertEqual(intermediate["normalizer_scope"], "train_episodes_only")
                resumed_cfg.training.resume = True
                prod.train(resumed_cfg)
                prod.train(full_cfg)

            resumed = torch.load(root / "resumed" / "checkpoints" / "latest.pt",
                                 map_location="cpu", weights_only=True)
            baseline = torch.load(root / "uninterrupted" / "checkpoints" / "latest.pt",
                                  map_location="cpu", weights_only=True)
            self.assertEqual(resumed["step"], baseline["step"])
            self.assertEqual(resumed["split"], baseline["split"])
            self.assertEqual(resumed["ema_step"], baseline["ema_step"])
            for key, weight in baseline["model"].items():
                self.assertTrue(torch.equal(resumed["model"][key], weight), key)
            for key, weight in baseline["ema_model"].items():
                self.assertTrue(torch.equal(resumed["ema_model"][key], weight), key)
            best = torch.load(root / "resumed" / "checkpoints" / "best.pt",
                              map_location="cpu", weights_only=True)
            self.assertEqual(best["checkpoint_role"], "deploy_best")
            self.assertEqual(best["selected_weights"], "ema")
            self.assertTrue(best["camera_map"])
            self.assertIn("model", best)
            self.assertIn("config", best)
            self.assertNotIn("optimizer", best)
            # Both train and held-out datasets must remain disjoint on resume.
            self.assertFalse(set(best["split"]["train"]) & set(best["split"]["val"]))


    def test_optional_wandb_init_cannot_perturb_training_rng(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with_wb = config_for_test(root / "with_logging")
            without_wb = config_for_test(root / "without_logging")
            with_wb.wandb = {
                "enable": True, "project": "test-project",
                "entity": None, "mode": "offline",
            }

            class FakeRun:
                id = "test-run-123"
                def define_metric(self, *args, **kwargs):
                    pass
                def log(self, values):
                    pass
                def finish(self):
                    pass

            wandb = types.ModuleType("wandb")
            def init_wandb(**kwargs):
                random.seed(123456)
                np.random.seed(123456)
                torch.manual_seed(123456)
                return FakeRun()
            wandb.init = init_wandb
            with (
                patch.dict(sys.modules, {"wandb": wandb}),
                patch.object(prod, "open_lerobot_bc_dataset", return_value=FakeView()),
                patch.object(prod, "build_policy", side_effect=lambda cfg, contract: TinyActor()),
            ):
                prod.train(with_wb)
                prod.train(without_wb)
            one = torch.load(root / "with_logging/checkpoints/latest.pt", weights_only=True)
            two = torch.load(root / "without_logging/checkpoints/latest.pt", weights_only=True)
            for key in one["model"]:
                self.assertTrue(torch.equal(one["model"][key], two["model"][key]), key)
            for key in one["ema_model"]:
                self.assertTrue(torch.equal(one["ema_model"][key], two["ema_model"][key]), key)

    def test_relative_output_paths_resolve_at_repo_root(self):
        self.assertEqual(
            prod.resolve_output_dir("outputs/train/rl100_dp_2d"),
            (ROOT / "outputs/train/rl100_dp_2d").resolve(),
        )

    def test_cli_timestamp_matches_lerobot_and_resume_preserves_path(self):
        import re
        with tempfile.TemporaryDirectory() as temp:
            config_path = Path(temp) / "bc_train.yaml"
            cfg = config_for_test(Path(temp) / "unused")
            cfg.output.dir = str(Path(temp) / "outputs/train/rl100_dp_2d")
            OmegaConf.save(cfg, config_path)
            with (
                patch.object(prod, "train") as runner,
                patch.object(sys, "argv", [
                    "train.py", "--config", str(config_path),
                ]),
            ):
                prod.main()
            first = runner.call_args.args[0]
            self.assertRegex(
                str(first.output.dir), r"rl100_2d_\d{8}_\d{6}$",
            )

            exact = Path(temp) / "outputs/train/rl100_dp_2d_20261104_090000"
            with (
                patch.object(prod, "train") as runner,
                patch.object(sys, "argv", [
                    "train.py", "--config", str(config_path),
                    "--resume", "--output-dir", str(exact),
                ]),
            ):
                prod.main()
            self.assertEqual(str(runner.call_args.args[0].output.dir), str(exact))
            self.assertTrue(runner.call_args.args[0].training.resume)

            with (
                patch.object(prod, "train") as runner,
                patch.object(sys, "argv", [
                    "train.py", "--config", str(config_path),
                    "--no-timestamp",
                ]),
            ):
                prod.main()
            self.assertEqual(
                str(runner.call_args.args[0].output.dir), str(cfg.output.dir),
            )


if __name__ == "__main__":
    unittest.main()
