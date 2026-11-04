"""Export a RL100 DP-2D actor to an inference-only Safetensors bundle.

Source .pt/.pth checkpoints are never modified. The bundle is intentionally
NOT an ordinary LeRobot PreTrainedPolicy checkpoint: RL1002D requires its own
policy factory and RGB / temporal observation adapter.
"""
from __future__ import annotations

import argparse
import json
import re
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
        tensors[name] = value.detach().cpu().contiguous().clone()
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
    if ((reference["state_names"] and len(reference["state_names"]) != expected_state)
            or (reference["action_names"] and len(reference["action_names"]) != expected_action)):
        raise ValueError("Actor state/action names do not match metadata dimensions")

    manifest = {
        "format": "letools_rl100_dp_2d",
        "format_version": 1,
        "actor_key": actor_key,
        "source_checkpoint_role": payload.get("checkpoint_role"),
        "source_step": payload.get("step"),
        "source_filename": source.name,
        "source_run": source.parent.parent.name if source.parent.name == "checkpoints" else source.parent.name,
        "selected_weights": payload.get("selected_weights", "ema" if actor_key == "ema_model" else "raw"),
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


def _safe_name(value: str, field: str) -> str:
    """Validate a single folder name, not an arbitrary filesystem path."""
    name = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) or name in {".", ".."}:
        raise ValueError(f"Invalid {field}: {value!r}")
    return name


def export_run_checkpoints(sources: list[str | Path], *, output_root: str | Path,
                           task: str, method: str = "rl100_dp_2d_bc",
                           run_name: str | None = None, actor_key: str = "model",
                           metadata_from: str | Path | None = None) -> list[Path]:
    """Export one native RL100 actor bundle per checkpoint, preserving the .pt files.

    Layout: <output_root>/<task>/<method>/<run>/checkpoints/<label>/pretrained_model/
    No synthetic LeRobot processor files: the native actor already includes its
    normalization buffers, and eval_policy.py performs the RGB/state adaptation.
    """
    if not sources:
        raise ValueError("No checkpoints to export")
    task = _safe_name(task, "task")
    method = _safe_name(method, "method")
    root = Path(output_root).expanduser()
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root

    jobs: list[tuple[Path, Path]] = []
    for entry in sources:
        source = Path(entry).expanduser().resolve()
        # Common RL100 BC layout: run/checkpoints/{best,latest,step_XXXXXXXX}.pt.
        derived_run = source.parent.parent.name if source.parent.name == "checkpoints" else source.parent.name
        run = _safe_name(run_name or derived_run, "run_name")
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise TypeError(f"Checkpoint must contain a dictionary: {source}")
        label = source.stem
        step = payload.get("step")
        if label in {"best", "latest"} and isinstance(step, int):
            label = f"{label}_step_{step:08d}"
        if actor_key != "model":
            label += f"_{_safe_name(actor_key, 'actor_key')}"
        label = _safe_name(label, "checkpoint label")
        destination = root / task / method / run / "checkpoints" / label / "pretrained_model"
        jobs.append((source, destination))

    # Preflight every destination before writing anything in a batch.
    destinations = [dst for _, dst in jobs]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Multiple sources resolve to the same export directory")
    for destination in destinations:
        if destination.exists():
            raise FileExistsError(f"Export already exists: {destination}")

    return [export_checkpoint(source, destination, actor_key=actor_key,
                              metadata_from=metadata_from)
            for source, destination in jobs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, nargs="+",
                        help="One or more native RL100 .pt/.pth checkpoints")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Legacy exact target directory for one checkpoint")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/export"),
                        help="New layout root (default: outputs/export)")
    parser.add_argument("--task", default=None,
                        help="Required for task/method/run/checkpoint layout")
    parser.add_argument("--method", default="rl100_dp_2d_bc")
    parser.add_argument("--run-name", default=None,
                        help="Defaults to the source checkpoint's run directory")
    parser.add_argument("--actor-key", default="model",
                        help="Actor state_dict key: model, ema_model, etc.")
    parser.add_argument("--metadata-from", type=Path, default=None,
                        help="BC .pt supplying metadata for post-RL actor exports")
    args = parser.parse_args()
    if args.output_dir is not None:
        if len(args.checkpoint) != 1:
            parser.error("--output-dir accepts only one checkpoint; use --output-root for batch export")
        exported = [export_checkpoint(args.checkpoint[0], args.output_dir,
                                      actor_key=args.actor_key,
                                      metadata_from=args.metadata_from)]
    else:
        if not args.task:
            parser.error("--task is required for task/method/run checkpoint export")
        exported = export_run_checkpoints(args.checkpoint, output_root=args.output_root,
                                          task=args.task, method=args.method,
                                          run_name=args.run_name, actor_key=args.actor_key,
                                          metadata_from=args.metadata_from)
    for destination in exported:
        print(f"Exported RL100 DP-2D actor bundle: {destination}")


if __name__ == "__main__":
    main()
