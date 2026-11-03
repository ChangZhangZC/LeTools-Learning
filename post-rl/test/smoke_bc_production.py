"""Feature06 on-target smoke: real LeRobot RGB + production trainer + EMA.

Four essential gates with a small Feature03 network:
(1) train with TRAIN-only stats and held-out episode val,
(2) write safe latest.pt + best.pt,
(3) resume from latest.pt and continue optimizer/scheduler/EMA,
(4) load best.pt in the unchanged Kuavo RL1002D adapter and infer an action.
No ROS, Kuavo simulator, or RL post-training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "post-rl" / "src", ROOT / "third_party" / "rl100" / "RL-100"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from kuavo_server.adapters.rl100_2d import RL1002DKuavoAdapter  # noqa: E402
from rl100_compat.lerobot_data import open_lerobot_bc_dataset  # noqa: E402
from rl100_compat.train_bc_production import train  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-config", default="post-rl/config/bc.yaml",
                        help="Feature03 small network config; NOT the production-size model")
    parser.add_argument("--output-dir", default="post_traing/outputs/feature06_bc_smoke")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-root", default=None,
                        help="Override LeRobot v3 root (e.g. extracted 10-episode subset)")
    parser.add_argument("--repo-id", default=None,
                        help="Override local LeRobot repo identifier")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.smoke_config)
    if args.dataset_root is not None:
        cfg.data.root = args.dataset_root
    if args.repo_id is not None:
        cfg.data.repo_id = args.repo_id
    cfg.training.device = args.device
    cfg.training.steps = 3
    cfg.training.batch_size = 1
    cfg.training.num_workers = 0
    cfg.training.lr_warmup_steps = 0
    cfg.training.save_every = 1
    cfg.training.log_every = 1
    cfg.training.val_every = 1
    cfg.training.max_grad_norm = 1.0
    cfg.training.resume = False
    cfg.training.use_ema = True
    cfg.training.keep_step_checkpoints = False
    cfg.validation = {
        "fraction": 0.2, "split_seed": 42, "seed": 1234,
        "batch_size": 4, "max_batches": 1,
    }
    cfg.output.dir = str(Path(args.output_dir))
    output = Path(cfg.output.dir)
    if (output / "checkpoints" / "latest.pt").exists():
        raise FileExistsError(f"Smoke run already exists: {output}; pick a new --output-dir")

    latest = train(cfg, stop_after=2)
    partial = torch.load(latest, map_location="cpu", weights_only=True)
    assert partial["step"] == 2 and partial["ema_step"] == 2
    assert partial["normalizer_scope"] == "train_episodes_only"
    print("[PASS] production BC 2 updates + held-out loss + EMA checkpoint")

    cfg.training.resume = True
    train(cfg)
    resumed = torch.load(latest, map_location="cpu", weights_only=True)
    assert resumed["step"] == 3 and resumed["ema_step"] == 3
    assert resumed["optimizer"] and resumed["lr_scheduler"]
    assert not set(resumed["split"]["train"]) & set(resumed["split"]["val"])
    print("[PASS] strict resumed optimizer/scheduler/model/EMA and episode split")

    best_path = output / "checkpoints" / "best.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    assert best["checkpoint_role"] == "deploy_best"
    assert best["selected_weights"] == "ema"
    assert best["model"] and best["camera_map"]
    adapter = RL1002DKuavoAdapter(checkpoint=str(best_path), device=args.device)
    md = adapter.metadata()
    dataset = open_lerobot_bc_dataset(
        str(cfg.data.repo_id), str(cfg.data.root),
        n_obs_steps=int(cfg.data.n_obs_steps),
        horizon=int(cfg.data.horizon),
        video_backend=str(cfg.data.video_backend),
    )
    sample = dataset[0]["obs"]
    observation = {"observation.state": sample["agent_pos"][-1].unsqueeze(0)}
    for src, key in md["camera_map"].items():
        observation[src] = sample[key][-1].unsqueeze(0)
    chunk = adapter.select_action_chunk(observation)
    assert chunk.shape == (md["execution_horizon"], md["action_dim"])
    assert np.isfinite(chunk).all()
    adapter.reset()
    print(f"[PASS] unmodified ModelServerAdapter loads EMA best.pt: action={chunk.shape}")
    print("[PASS] Feature06 production BC smoke: real data, val, resume, EMA deploy")


if __name__ == "__main__":
    main()
