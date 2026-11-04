"""Regression for RL100 actor-only export and in-process Eval chunk execution."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from kuavo_model.rl100_dp_2d.export_safetensors import export_checkpoint, export_run_checkpoints
from kuavo_model.rl100_dp_2d.eval_policy import RL100DP2DEvalPolicy, resolve_rl100_eval_output_dir


class TinyActor(torch.nn.Module):
    """Minimal actor with native RL100 normalizer state_dict key layout."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        normalizer = torch.nn.Module()
        params_dict = torch.nn.ModuleDict()
        for key in ("agent_pos", "action"):
            entry = torch.nn.Module()
            entry.register_buffer("scale", torch.ones(2))
            params_dict[key] = entry
        normalizer.params_dict = params_dict
        self.normalizer = normalizer
        self.observed_histories = []

    def predict_action(self, obs, deterministic=True):
        self.observed_histories.append(obs["agent_pos"][0, :, 0].tolist())
        return {"action": torch.arange(8).view(1, 4, 2).float().to(obs["agent_pos"].device)}


def sample_payload():
    return {
        "model": TinyActor().state_dict(),
        "config": {"data": {"n_obs_steps": 2, "n_action_steps": 4}},
        "shape_meta": {
            "obs": {
                "agent_pos": {"shape": [2], "type": "low_dim"},
                "rgb_head_cam_h": {"shape": [3, 4, 5], "type": "rgb"},
            },
            "action": {"shape": [2]},
        },
        "camera_map": {"observation.images.head_cam_h": "rgb_head_cam_h"},
        "state_names": ["joint_0", "joint_1"],
        "action_names": ["action_0", "action_1"],
        "dataset_fps": 10,
        "step": 9,
        "checkpoint_role": "deploy_best",
    }


def obs(step):
    return {
        "observation.state": torch.tensor([[float(step), float(-step)]]),
        "observation.images.head_cam_h": torch.zeros((1, 3, 4, 5), dtype=torch.uint8),
    }


class NativeEvalExportTests(unittest.TestCase):
    def test_eval_yaml_uses_native_policy_and_runtime_chunk_length(self):
        from kuavo_deploy.config import load_kuavo_config, ConfigInference

        root = Path(__file__).resolve().parents[2]
        profile = root / "configs/deploy/rl100_dp_2d_eval.yaml"
        cfg = load_kuavo_config(str(profile))
        self.assertEqual(cfg.env.inference_env, "sim")
        self.assertEqual(cfg.inference.policy_type, "rl100_dp_2d")
        self.assertEqual(cfg.inference.n_action_steps, 16)
        with self.assertRaisesRegex(ValueError, "n_action_steps"):
            ConfigInference(policy_type="rl100_dp_2d", n_action_steps=0).validate()

    def test_export_state_dict_roundtrip_and_original_checkpoint_intact(self):
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "best.pt"
            original = sample_payload()
            torch.save(original, source)
            exported = export_checkpoint(source, root / "bundle")
            manifest = json.loads((exported / "rl100_manifest.json").read_text())
            self.assertEqual(manifest["format"], "letools_rl100_dp_2d")
            self.assertEqual(manifest["source_step"], 9)
            self.assertEqual(manifest["camera_map"], original["camera_map"])
            weights = load_file(str(exported / "model.safetensors"))
            self.assertEqual(set(weights), set(original["model"]))
            for name, tensor in original["model"].items():
                self.assertTrue(torch.equal(tensor, weights[name]), name)
            self.assertTrue(source.is_file())
            with self.assertRaises(FileExistsError):
                export_checkpoint(source, exported)

    def test_post_rl_actor_can_use_bc_reference_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bc = root / "bc_best.pt"
            post = root / "post_rl_actor.pt"
            torch.save(sample_payload(), bc)
            torch.save({"model": TinyActor().state_dict(), "step": 123}, post)
            result = export_checkpoint(post, root / "post_export", metadata_from=bc)
            meta = json.loads((result / "rl100_manifest.json").read_text())
            self.assertEqual(meta["source_step"], 123)
            self.assertEqual(meta["config"]["data"]["n_obs_steps"], 2)
            with self.assertRaisesRegex(ValueError, "metadata"):
                export_checkpoint(post, root / "bad_export")

    def test_native_eval_cached_chunk_keeps_consecutive_observation_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "best.pt"
            torch.save(sample_payload(), source)
            bundle = export_checkpoint(source, root / "bundle")
            with patch("kuavo_model.rl100_dp_2d.eval_policy.build_policy",
                       side_effect=lambda cfg, contract: TinyActor()):
                policy = RL100DP2DEvalPolicy(bundle, device="cpu", n_action_steps=3)
            self.assertEqual(policy.config.type, "rl100_dp_2d")
            outputs = [policy.select_action(obs(i)) for i in range(7)]
            self.assertTrue(all(a.shape == (1, 2) for a in outputs))
            # Replan at 0, 3, 6: history is current + immediately previous step,
            # even though cached chunk actions were returned at 1, 2, 4, 5.
            self.assertEqual(policy.model.observed_histories,
                             [[0.0, 0.0], [2.0, 3.0], [5.0, 6.0]])
            policy.reset()
            policy.select_action(obs(12))
            self.assertEqual(policy.model.observed_histories[-1], [12.0, 12.0])

    def test_eval_rejects_runtime_horizon_outside_saved_actor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "best.pt"
            torch.save(sample_payload(), source)
            bundle = export_checkpoint(source, root / "bundle")
            with patch("kuavo_model.rl100_dp_2d.eval_policy.build_policy",
                       side_effect=lambda cfg, contract: TinyActor()):
                with self.assertRaisesRegex(ValueError, "n_action_steps"):
                    RL100DP2DEvalPolicy(bundle, device="cpu", n_action_steps=5)
                single = RL100DP2DEvalPolicy(bundle, device="cpu")
            self.assertEqual(single.n_action_steps, 1)
            with self.assertRaisesRegex(ValueError, "trained"):
                single.select_action({
                    **obs(1),
                    "observation.images.head_cam_h": torch.zeros(
                        (1, 3, 6, 6), dtype=torch.uint8),
                })


    def test_named_batch_export_preserves_original_sources_and_separates_checkpoints(self):
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpts = root / "training_run_123" / "checkpoints"
            ckpts.mkdir(parents=True)
            best = ckpts / "best.pt"
            step = ckpts / "step_00005000.pt"
            best_payload = sample_payload()
            step_payload = sample_payload()
            step_payload["step"] = 5000
            torch.save(best_payload, best)
            torch.save(step_payload, step)
            exported = export_run_checkpoints(
                [best, step], output_root=root / "export", task="icra_task1",
                method="rl100_dp_2d_bc",
            )
            base = root / "export" / "icra_task1" / "rl100_dp_2d_bc" / "training_run_123" / "checkpoints"
            self.assertEqual(exported, [
                base / "best_step_00000009" / "pretrained_model",
                base / "step_00005000" / "pretrained_model",
            ])
            for source, bundle in zip((best, step), exported):
                self.assertTrue(source.is_file())
                self.assertTrue((bundle / "model.safetensors").is_file())
                meta = json.loads((bundle / "rl100_manifest.json").read_text())
                self.assertEqual(meta["source_filename"], source.name)
                self.assertEqual(meta["source_run"], "training_run_123")
                self.assertEqual(meta["format"], "letools_rl100_dp_2d")
                self.assertIn("normalizer.params_dict.action.scale",
                              load_file(str(bundle / "model.safetensors")))
            # Strict no-overwrite behavior protects previously exported runs.
            with self.assertRaises(FileExistsError):
                export_run_checkpoints([best, step], output_root=root / "export",
                                       task="icra_task1")

    def test_named_batch_export_preflights_duplicate_destinations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            best = root / "best.pt"
            torch.save(sample_payload(), best)
            with self.assertRaisesRegex(ValueError, "same export directory"):
                export_run_checkpoints([best, best], output_root=root / "export",
                                       task="task")
            self.assertFalse((root / "export").exists())
            with self.assertRaisesRegex(ValueError, "Invalid task"):
                export_run_checkpoints([best], output_root=root / "export",
                                       task="../escape")

    def test_native_eval_output_is_separated_by_checkpoint_and_supports_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            one = root / "export/task/rl100_dp_2d_bc/run_123/checkpoints/best/pretrained_model"
            two = root / "export/task/rl100_dp_2d_bc/run_123/checkpoints/step_00005000/pretrained_model"
            expected = root / "eval/task/rl100_dp_2d_bc/run_123"
            self.assertEqual(resolve_rl100_eval_output_dir(one, root / "eval"),
                             expected / "best")
            self.assertEqual(resolve_rl100_eval_output_dir(two, root / "eval"),
                             expected / "step_00005000")
            self.assertEqual(resolve_rl100_eval_output_dir(root / "flat_bundle", root / "eval"),
                             root / "eval/flat_bundle")


if __name__ == "__main__":
    unittest.main()
