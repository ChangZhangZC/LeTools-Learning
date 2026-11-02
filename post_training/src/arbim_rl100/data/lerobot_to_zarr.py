"""Stream a LeRobot 0.5.2 archive into the ARBiM jpeg_bytes_v1 Zarr cache.

No RL-100 source modification, no staging-NPY bridge, and no implicit reward.
Run from the LeTools repository root:
  python post_training/src/arbim_rl100/data/lerobot_to_zarr.py \
    --lerobot-root /path/to/local_dataset --output /path/to/offline.zarr \
    --reward-mode recorded_sparse --reward-json /path/to/reward.json

The input is an episode-contiguous local LeRobot archive, and the output is a
*derived* offline cache. Video frames are decoded and JPEG-encoded once.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import zarr
from numcodecs import Blosc

RGB_FEATURES = {
    "observation.images.head_cam_h": "head_rgb",
    "observation.images.wrist_cam_l": "wrist_left_rgb",
    "observation.images.wrist_cam_r": "wrist_right_rgb",
}
CORE = ("state", "next_state", "action", "next_action", "reward", "return",
        "done", "timeout", "next_index")


def episode_ranges(episode_column: Any) -> list[tuple[int, int, int]]:
    """Read episode boundaries without video decoding; reject interleaving."""
    ep = np.asarray(episode_column)
    if ep.ndim != 1 or ep.size == 0 or ep.dtype.kind not in "iu":
        raise ValueError("episode_index must be a nonempty integer column")
    if np.any(np.diff(ep.astype(np.int64)) < 0):
        raise ValueError("episode_index must be sorted and episode-contiguous")
    boundaries = np.flatnonzero(ep[1:] != ep[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, ep.size]
    return [(int(ep[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def _numpy(value: Any, field: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    array = np.asarray(value)
    if array.dtype.kind not in "uif" or not array.size:
        raise ValueError(f"{field}: expected a nonempty numeric tensor")
    return array


def _vector(value: Any, field: str, dim: int | None = None) -> np.ndarray:
    arr = _numpy(value, field)
    if arr.ndim != 1 or (dim is not None and arr.size != dim):
        raise ValueError(f"{field}: invalid vector shape {arr.shape}, expected {(dim,)}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{field}: contains nonfinite values")
    return arr.astype(np.float32)


def _rgb(value: Any, field: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    arr = _numpy(value, field)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"{field}: expected RGB [3,H,W], got {arr.shape}")
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{field}: shape drift {arr.shape}, expected {shape}")
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    if arr.dtype.kind != "f" or not np.isfinite(arr).all():
        raise ValueError(f"{field}: expected uint8 or finite float RGB")
    if arr.min() < 0 or arr.max() > 1:
        raise ValueError(f"{field}: float RGB must be in [0,1], not [0,255]")
    return np.ascontiguousarray(np.rint(arr * 255).astype(np.uint8))


def _jpeg(value: np.ndarray, quality: int) -> bytes:
    bgr = cv2.cvtColor(np.moveaxis(value, 0, -1), cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(
        ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not success:
        raise RuntimeError("Failed to JPEG-encode an RGB frame")
    return encoded.tobytes()


def reward_offsets_from_json(
    reward_json: Path, episodes: list[tuple[int, int, int]]
) -> dict[int, int]:
    """Require verified sparse reward at a nonterminal frame of each episode."""
    with Path(reward_json).open("r", encoding="utf-8") as fp:
        obj = json.load(fp)
    if not isinstance(obj, dict) or not isinstance(obj.get("episodes"), list):
        raise ValueError("Reward JSON requires an 'episodes' array")
    offsets: dict[int, int] = {}
    for row in obj["episodes"]:
        if not isinstance(row, dict):
            raise ValueError("Malformed reward episode record")
        episode_id = row.get("lerobot_episode_index", row.get("episode_index"))
        if type(episode_id) is not int or episode_id in offsets:
            raise ValueError("Duplicate or invalid reward episode_index")
        if row.get("reward_status") != "valid":
            raise ValueError(f"Episode {episode_id}: Reward is not validated")
        if row.get("reward_mode") != "last_gripper_release_sparse":
            raise ValueError(f"Episode {episode_id}: unsupported Reward mode")
        offset = row.get("reward_index")
        event = row.get("reward_event")
        if type(offset) is not int or not isinstance(event, dict):
            raise ValueError(f"Episode {episode_id}: missing reward event/index")
        if event.get("frame_offset") != offset:
            raise ValueError(f"Episode {episode_id}: inconsistent reward event")
        offsets[episode_id] = offset
    expected = {ep for ep, _, _ in episodes}
    if set(offsets) != expected:
        raise ValueError(f"Reward episodes mismatch; expected {sorted(expected)}")
    for episode_id, start, end in episodes:
        if end - start < 2 or not 0 <= offsets[episode_id] < end - start - 1:
            raise ValueError(
                f"Episode {episode_id}: reward index must precede terminal"
            )
        row = next(r for r in obj["episodes"] if r.get(
            "lerobot_episode_index", r.get("episode_index")
        ) == episode_id)
        n = row.get("num_frames")
        if n is not None and n != end - start:
            raise ValueError(f"Episode {episode_id}: reward frame count mismatch")
    return offsets


class _ZarrWriter:
    def __init__(
        self, path: Path, count: int, first: dict, jpeg_quality: int,
        stream_batch_size: int,
    ):
        self.root = zarr.open_group(str(path), mode="w")
        self.data = self.root.create_group("data")
        self.meta = self.root.create_group("meta")
        self.count = count
        self.jpeg_quality = jpeg_quality
        self.batch_size = stream_batch_size
        self.position = 0
        self.pending: list[dict] = []
        self.byte_positions = {key: 0 for key in RGB_FEATURES.values()}
        self.shapes = {k: list(v.shape) for k, v in first["rgb"].items()}
        state_dim = first["state"].size
        action_dim = first["action"].size
        comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
        n_chunk = min(stream_batch_size, count)
        dims = {
            "state": ((count, state_dim), "f4"),
            "next_state": ((count, state_dim), "f4"),
            "action": ((count, action_dim), "f4"),
            "next_action": ((count, action_dim), "f4"),
            "reward": ((count, 1), "f4"),
            "return": ((count, 1), "f4"),
            "done": ((count, 1), "bool"),
            "timeout": ((count, 1), "bool"),
            "next_index": ((count,), "i8"),
        }
        self.arrays = {
            k: self.data.create_dataset(
                k, shape=shape, chunks=(n_chunk, *shape[1:]),
                dtype=dtype, compressor=comp
            )
            for k, (shape, dtype) in dims.items()
        }
        for name in RGB_FEATURES.values():
            self.arrays[f"{name}_jpeg_data"] = self.data.create_dataset(
                f"{name}_jpeg_data", shape=(0,), chunks=(4 * 1024 * 1024,),
                dtype="u1", compressor=None,
            )
            self.arrays[f"{name}_jpeg_offsets"] = self.data.create_dataset(
                f"{name}_jpeg_offsets", shape=(count + 1,),
                chunks=(min(count + 1, stream_batch_size + 1),),
                dtype="i8", compressor=comp,
            )
            self.arrays[f"{name}_jpeg_offsets"][0] = 0
        self.root.attrs.update({
            "rgb_storage": "jpeg_bytes_v1",
            "rgb_shapes": self.shapes,
            "rgb_dtype": "uint8",
            "state_dim": state_dim,
            "action_dim": action_dim,
            "use_depth": False,
        })

    def append(self, transition: dict):
        self.pending.append(transition)
        if len(self.pending) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        start = self.position
        end = start + len(self.pending)
        if end > self.count:
            raise RuntimeError("Too many input frames")
        for key in CORE:
            values = [item[key] for item in self.pending]
            self.arrays[key][start:end] = np.asarray(values, dtype=self.arrays[key].dtype)
        for name in RGB_FEATURES.values():
            payloads = [_jpeg(item["rgb"][name], self.jpeg_quality) for item in self.pending]
            payload = b"".join(payloads)
            old = self.byte_positions[name]
            new = old + len(payload)
            data = self.arrays[f"{name}_jpeg_data"]
            data.resize((new,))
            if payload:
                data[old:new] = np.frombuffer(payload, dtype=np.uint8)
            ends = old + np.cumsum(
                [len(item) for item in payloads], dtype=np.int64
            )
            self.arrays[f"{name}_jpeg_offsets"][start + 1:end + 1] = ends
            self.byte_positions[name] = new
        self.position = end
        self.pending.clear()


def _frame(raw: dict, state_dim=None, action_dim=None, rgb_shapes=None) -> dict:
    required = ("observation.state", "action", *RGB_FEATURES.keys())
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"Missing LeRobot feature(s): {missing}")
    result = {
        "state": _vector(raw["observation.state"], "observation.state", state_dim),
        "action": _vector(raw["action"], "action", action_dim),
        "rgb": {},
    }
    for feature, name in RGB_FEATURES.items():
        expected = None if rgb_shapes is None else tuple(rgb_shapes[name])
        result["rgb"][name] = _rgb(raw[feature], feature, expected)
    return result


def _require_zarr_v2() -> None:
    """This cache contract uses Zarr 2's create_dataset and Blosc APIs."""
    version = getattr(zarr, "__version__", "unknown")
    if version.split(".", 1)[0] != "2" or not hasattr(zarr.group(), "create_dataset"):
        raise RuntimeError(
            f"ARBiM jpeg_bytes_v1 converter requires zarr==2.18.7 "
            f"(installed: {version}; module: {getattr(zarr, '__file__', '?')}). "
            "Install with: python -m pip install 'zarr==2.18.7' "
            "'numcodecs==0.15.1'"
        )


def convert_dataset(
    dataset: Any, output: str | Path, *, reward_mode: str = "recorded_sparse",
    reward_json: str | Path | None = None, gamma: float = 0.99,
    jpeg_quality: int = 95, stream_batch_size: int = 32,
    source_path: str | Path | None = None,
) -> Path:
    """Convert a local LeRobotDataset-like object; never overwrite output."""
    _require_zarr_v2()
    if not (0 < gamma <= 1):
        raise ValueError("gamma must lie in (0,1]")
    if not 1 <= jpeg_quality <= 100 or stream_batch_size < 1:
        raise ValueError("Invalid JPEG quality or batch size")
    if reward_mode not in {"recorded_sparse", "terminal_proxy"}:
        raise ValueError("Unsupported reward mode")
    if reward_mode == "recorded_sparse" and reward_json is None:
        raise ValueError("recorded_sparse requires --reward-json")
    if reward_mode == "terminal_proxy" and reward_json is not None:
        raise ValueError("Reward JSON requires recorded_sparse mode")

    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing Zarr: {output}")
    episodes = episode_ranges(dataset.hf_dataset["episode_index"])
    total = sum(end - start for _, start, end in episodes)
    if total != len(dataset):
        raise ValueError("Episode index column and dataset length differ")
    reward_offsets = (
        reward_offsets_from_json(Path(reward_json), episodes)
        if reward_json is not None else {}
    )
    first = _frame(dataset[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".partial.", dir=output.parent))
    try:
        writer = _ZarrWriter(staging, total, first, jpeg_quality, stream_batch_size)
        writer.root.attrs.update({
            "reward_mode": (
                "last_gripper_release_sparse"
                if reward_mode == "recorded_sparse" else "terminal_proxy"
            ),
            "reward_source": (
                "recorded_validated" if reward_mode == "recorded_sparse"
                else "terminal_proxy_not_success_label"
            ),
            "reward_validated": reward_mode == "recorded_sparse",
            "reward_event": {
                "source": "external_verified_episode_json",
                "episodes": [
                    {"episode_index": ep, "reward_offset": reward_offsets[ep]}
                    for ep, _, _ in episodes
                ],
            } if reward_mode == "recorded_sparse" else None,
            "return_gamma": gamma,
            "source_manifest": [{
                "kind": "lerobot",
                "path": str(source_path) if source_path is not None else "",
                "format": "lerobot_direct_v1",
                "reward_mode": reward_mode,
                "reward_json": str(reward_json) if reward_json is not None else None,
            }],
        })
        episode_ends = []
        for episode_id, start, end in episodes:
            pending = None
            episode_rewards = []
            for index in range(start, end):
                current = first if index == 0 else _frame(
                    dataset[index], first["state"].size, first["action"].size,
                    writer.shapes,
                )
                step = index - start
                terminal = index == end - 1
                reward = (
                    float(step == reward_offsets[episode_id])
                    if reward_mode == "recorded_sparse" else float(terminal)
                )
                episode_rewards.append(reward)
                if pending is not None:
                    writer.append({
                        "state": pending["state"], "next_state": current["state"],
                        "action": pending["action"], "next_action": current["action"],
                        "rgb": pending["rgb"], "reward": [episode_rewards[-2]],
                        "return": [0.0], "done": [False], "timeout": [False],
                        "next_index": index,
                    })
                pending = current
            assert pending is not None
            writer.append({
                "state": pending["state"], "next_state": pending["state"],
                "action": pending["action"], "next_action": pending["action"],
                "rgb": pending["rgb"], "reward": [episode_rewards[-1]],
                "return": [0.0], "done": [True], "timeout": [True],
                "next_index": end - 1,
            })
            writer.flush()
            if writer.position != end:
                raise RuntimeError("Write position disagrees with episode end")
            running = 0.0
            episode_returns = np.empty((end - start, 1), dtype=np.float32)
            for step in range(end - start - 1, -1, -1):
                running = episode_rewards[step] + gamma * running
                episode_returns[step, 0] = running
            writer.arrays["return"][start:end] = episode_returns
            episode_ends.append(end)
        if writer.position != total:
            raise RuntimeError("Incomplete Zarr output")
        writer.meta.create_dataset(
            "episode_ends", data=np.asarray(episode_ends, dtype=np.int64),
            chunks=(min(128, len(episode_ends)),),
        )
        os.replace(staging, output)
        return output
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reward-mode", choices=("recorded_sparse", "terminal_proxy"),
                        default="recorded_sparse")
    parser.add_argument("--reward-json", type=Path)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--stream-batch-size", type=int, default=32)
    parser.add_argument("--video-backend", default=None)
    args = parser.parse_args()
    root = args.lerobot_root.expanduser().resolve()
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"LeRobot meta/info.json missing: {root}")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        repo_id=root.name, root=root, return_uint8=True,
        video_backend=args.video_backend,
    )
    output = convert_dataset(
        ds, args.output, reward_mode=args.reward_mode,
        reward_json=args.reward_json, gamma=args.gamma,
        jpeg_quality=args.jpeg_quality,
        stream_batch_size=args.stream_batch_size, source_path=root,
    )
    print(f"[PASS] Created offline Zarr cache: {output}")


if __name__ == "__main__":
    main()
