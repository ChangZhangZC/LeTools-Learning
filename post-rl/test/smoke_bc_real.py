"""Feature03: one native RL1002D BC update, then verify checkpoint contents."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "post-rl" / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "rl100" / "RL-100"))
from rl100_compat.train_bc import train  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="post-rl/config/bc.yaml")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.output.dir = args.output_dir
    path = train(cfg, max_steps=1)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["step"] == 1
    assert payload["model"] and payload["shape_meta"]
    assert payload["shape_meta"]["action"]["shape"][0] > 0
    assert len(payload["camera_map"]) >= 1
    assert any("normalizer" in key and "action.scale" in key
               for key in payload["model"]), "Missing native action normalizer in checkpoint"
    print(f"[PASS] Native RL1002D real-data BC update + checkpoint: {path}")


if __name__ == "__main__":
    main()
