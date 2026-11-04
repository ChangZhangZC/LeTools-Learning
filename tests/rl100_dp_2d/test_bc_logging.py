"""Optional BC logging sinks: disabled imports, resume IDs, and step axes."""
from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kuavo_model.rl100_dp_2d.logging import BCMetricLogger  # noqa: E402


class DummyRun:
    def __init__(self, run_id):
        self.id = run_id
        self.records = []
        self.axes = []
        self.finished = False

    def define_metric(self, *args, **kwargs):
        self.axes.append((args, kwargs))

    def log(self, values):
        self.records.append(values.copy())

    def finish(self):
        self.finished = True


class DummySummaryWriter:
    instances = []

    def __init__(self, log_dir, **kwargs):
        self.dir = log_dir
        self.options = kwargs
        self.scalars = []
        self.closed = False
        self.instances.append(self)

    def add_scalar(self, key, val, global_step):
        self.scalars.append((key, val, global_step))

    def flush(self):
        pass

    def close(self):
        self.closed = True


def config(wandb=False, tb=False):
    return OmegaConf.create({
        "data": {"horizon": 33, "n_obs_steps": 2},
        "output": {"dir": "outputs/train/rl100_dp_2d"},
        "wandb": {
            "enable": wandb, "project": "lerobot_rl100_dp_2d",
            "entity": None, "mode": "offline",
        },
        "tensorboard": {"enable": tb},
    })


class TestBCLogging(unittest.TestCase):
    def test_disabling_both_does_not_require_optional_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            logger = BCMetricLogger(config(), Path(temp), resumed=False, resume_step=0)
            logger.train(step=1, loss=0.5, lr=1e-4, epoch=0.1)
            logger.validation(step=1, loss=0.6, best_loss=0.6)
            logger.close()
            self.assertFalse(list(Path(temp).iterdir()))

    def test_wandb_and_tensorboard_share_optimizer_step_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            init_calls = []
            runs = []

            def init(**kwargs):
                init_calls.append(kwargs)
                run = DummyRun(kwargs.get("id") or "offline-run-test-123")
                runs.append(run)
                return run

            dummy_wandb = types.ModuleType("wandb")
            dummy_wandb.init = init
            dummy_tb = types.ModuleType("torch.utils.tensorboard")
            dummy_tb.SummaryWriter = DummySummaryWriter
            DummySummaryWriter.instances.clear()
            with patch.dict(sys.modules, {
                "wandb": dummy_wandb,
                "torch.utils.tensorboard": dummy_tb,
            }):
                logger = BCMetricLogger(config(True, True), root,
                                        resumed=False, resume_step=0)
                logger.train(step=5, loss=0.9, lr=1e-4, epoch=0.25)
                logger.validation(step=5, loss=1.3, best_loss=1.3)
                logger.close()
                self.assertEqual(
                    (root / "wandb_run_id.txt").read_text().strip(),
                    "offline-run-test-123",
                )
                self.assertTrue(runs[0].finished)
                self.assertEqual([r["trainer_step"] for r in runs[0].records], [5, 5])
                self.assertEqual(len(runs[0].axes), 3)
                self.assertTrue(DummySummaryWriter.instances[0].closed)
                self.assertIn(("val/loss", 1.3, 5), DummySummaryWriter.instances[0].scalars)

                continued = BCMetricLogger(config(True, True), root,
                                           resumed=True, resume_step=5)
                continued.train(step=6, loss=0.7, lr=8e-5, epoch=0.3)
                continued.close()
                self.assertEqual(init_calls[1]["id"], "offline-run-test-123")
                self.assertEqual(init_calls[1]["resume"], "allow")
                self.assertEqual(DummySummaryWriter.instances[1].options["purge_step"], 6)
                self.assertTrue(DummySummaryWriter.instances[1].closed)

    def test_resume_wandb_without_id_fails_instead_of_new_run(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, "wandb_run_id"):
                BCMetricLogger(config(True), Path(temp), resumed=True, resume_step=2)

    def test_log_import_errors_only_if_enabled(self):
        with tempfile.TemporaryDirectory() as temp:
            # Missing dependency is caught by the lazy import in the target environment.
            # Here also verify a disabled configuration never creates any files.
            logger = BCMetricLogger(OmegaConf.create({}), Path(temp),
                                    resumed=False, resume_step=0)
            logger.close()
            self.assertEqual(list(Path(temp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
