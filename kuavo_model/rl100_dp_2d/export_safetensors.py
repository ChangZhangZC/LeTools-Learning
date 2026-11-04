"""Export a RL100 DP-2D actor to an inference-only Safetensors bundle.

Source .pt/.pth checkpoints are never modified. The bundle is intentionally
NOT an ordinary LeRobot PreTrainedPolicy checkpoint: RL1002D requires its own
policy factory and RGB / temporal observation adapter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

MANIFEST = "rl100_manifest.json"
WEIGHTS = "model.safetensors"
REQUIRED = ("config", "shape_meta", "camera_map", "state_names", "action_names")


def export_checkpoint(source: str | Path, output_dir: str | Path, *,
                      actor_key: str = "model",
                      metadata_from: str | Path | None = None) -> Path:
    """Export only an actor state_dict; Critic/optimizer/RNG remain in the .pt.

    For a future RL checkpoint without the BC metadata fields, pass a
    compatible BC/previous exported-role .pt via metadata_from. A strict
    actor load during inference verifies architecture compatibility.
    """
    from safetensors.torch import save_file

    source = Path(source).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("RL100 checkpoint must contain a dictionary")
    actor = payload.get(actor_key)
    if not isinstance(actor, dict) or not actor:
        raise ValueError(f"Missing/non-dict actor state_dict at key {actor_key!r}")

    reference = payload
    if metadata_from is not None:
        refpath = Path(metadata_from).expanduser().resolve()
        reference = torch.load(refpath, map_location="cpu", weights_only=True)
        if not isinstance(reference, dict):
            raise TypeError("metadata_from must be a RL100 checkpoint dictionary")
    missing = [k for k in REQUIRED if k not in reference]
    if missing:
        raise ValueError(f"Checkpoint missing RL100 deployment metadata {missing}; "
                         "supply --metadata-from compatible_bc_best.pt")
    if not isinstance(reference["shape_meta"], dict):
        raise ValueError("Invalid RL100 shape_meta")

    tensors = {}
    for name, value in actor.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError(f"Actor state_dict must be named tensors; invalid {name!r}")
        tensors[name] = value.detach().cpu().contiguous()
    # Native RL100 normalizers are part of actor.state_dict; dropping them
    # would make exported actions incorrectly scaled.
    for key in ("agent_pos", "action"):
        norm_key = f"normalizer.params_dict.{key}.scale"
        if norm_key not in tensors:
            raise ValueError(f"Actor state_dict lacks {norm_key}; cannot export")
    obs = reference["shape_meta"].get("obs", {})
    action = reference["shape_meta"].get("action", {})
    if (not isinstance(obs, dict) or "agent_pos" not in obs
            or not isinstance(action, dict)):
        raise ValueError("Invalid RL100 observation/action shape metadata")
    expected_state = int(obs["agent_pos"]["shape"][0])
    expected_action = int(action["shape"][0])
    if (len(reference["state_names"]) != expected_state
            or len(reference["action_names"]) != expected_action):
        raise ValueError("Actor state/action names do not match metadata dimensions")

    manifest = {
        "format": "letools_rl100_dp_2d",
        "format_version": 1,
        "actor_key": actor_key,
        "source_checkpoint_role": payload.get("checkpoint_role"),
        "source_step": payload.get("step"),
        "config": reference["config"],
        "shape_meta": reference["shape_meta"],
        "camera_map": reference["camera_map"],
        "state_names": reference["state_names"],
        "action_names": reference["action_names"],
        "dataset_fps": reference.get("dataset_fps"),
    }
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    weights_path, manifest_path = output_dir / WEIGHTS, output_dir / MANIFEST
    if weights_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Export files already exist in {output_dir}; use a fresh directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        save_file(tensors, str(weights_path))
        manifest_path.write_text(manifest_text, encoding="utf-8")
    except Exception:
        weights_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Native RL100 .pt/.pth checkpoint, including best.pt/latest.pt")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--actor-key", default="model",
                        help="Actor state_dict key: 'model' (best or latest), 'ema_model', etc.")
    parser.add_argument("--metadata-from", type=Path, default=None,
                        help="Compatible RL100 BC .pt supplying metadata for post-RL actor")
    args = parser.parse_args()
    dest = export_checkpoint(args.checkpoint, args.output_dir,
                             actor_key=args.actor_key, metadata_from=args.metadata_from)
    print(f"Exported RL100 DP-2D actor bundle: {dest}")


if __name__ == "__main__":
    main()
