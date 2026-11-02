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
from tqdm import tqdm

DEPTH_FEATURE_TO_BUFFER = {
    "observation.depth_h": "head_depth",
    "observation.depth_l": "wrist_left_depth",
    "observation.depth_r": "wrist_right_depth",
}

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


def _to_uint8_chw(image):
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"RGB image must be 3D, got shape={image.shape}")
    if image.shape[0] in (1, 3, 4):
        chw = image
    elif image.shape[-1] in (1, 3, 4):
        chw = np.moveaxis(image, -1, 0)
    else:
        raise ValueError(f"cannot infer RGB channel axis from shape={image.shape}")
    if chw.shape[0] != 3:
        raise ValueError(f"RGB image must have 3 channels, got shape={chw.shape}")
    if chw.dtype == np.uint8:
        return np.ascontiguousarray(chw)
    if np.issubdtype(chw.dtype, np.floating):
        min_value = float(np.nanmin(chw))
        max_value = float(np.nanmax(chw))
        if min_value >= -1e-6 and max_value <= 1.0 + 1e-6:
            chw = chw * 255.0
        elif min_value < -1e-6 or max_value > 255.0 + 1e-6:
            raise ValueError(
                f"RGB float image must be in [0,1] or [0,255], got [{min_value}, {max_value}]"
            )
    chw = np.clip(np.rint(chw), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(chw)




def _rgb(value: Any, field: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    arr = _numpy(value, field)
    if not np.isfinite(arr).all():
        raise ValueError(f"{field}: contains nonfinite RGB values")
    chw = _to_uint8_chw(arr)
    if shape is not None and chw.shape != shape:
        raise ValueError(f"{field}: shape drift {chw.shape}, expected {shape}")
    return chw

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


def _as_numpy(value, feature_name, frame_index):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if (
        array.size == 0
        or array.dtype == object
        or not np.issubdtype(array.dtype, np.number)
    ):
        raise ValueError(
            f"frame {frame_index} has an invalid {feature_name!r} value "
            f"(shape={array.shape}, dtype={array.dtype})"
        )
    return array



def _batch_column(dataset, key, indices):
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is None:
        raise AttributeError("LeRobot dataset does not expose hf_dataset")
    try:
        values = hf_dataset[key][indices]
    except (KeyError, TypeError, IndexError):
        values = hf_dataset[indices][key]
    if not isinstance(values, (list, tuple)):
        values = [values]
    return values


def _stack_video_batch(frames, count, feature_name):
    if count == 1 and getattr(frames, "ndim", None) == 3:
        frames = frames.unsqueeze(0)
    if len(frames) != count:
        raise RuntimeError(
            f"batched video decode for {feature_name!r} returned {len(frames)} frames, "
            f"expected {count}"
        )
    return frames


def _load_lerobot_frame_batch(dataset, episode_id, frame_indices, use_depth):
    """Decode one episode slice in batches so RAM is bounded by stream_batch_size."""
    if not frame_indices:
        return []
    count = len(frame_indices)
    state_values = _batch_column(dataset, "observation.state", frame_indices)
    action_values = _batch_column(dataset, "action", frame_indices)
    items = [
        {
            "agent_pos": _as_numpy(state_values[i], "observation.state", frame_indices[i]),
            "action": _as_numpy(action_values[i], "action", frame_indices[i]),
            "rgb": {},
        }
        for i in range(count)
    ]
    if use_depth:
        for item in items:
            item["depth"] = {}

    video_keys = set(getattr(dataset.meta, "video_keys", []))
    requested_video_keys = [
        key
        for key in [
            *RGB_FEATURES.keys(),
            *(DEPTH_FEATURE_TO_BUFFER.keys() if use_depth else []),
        ]
        if key in video_keys
    ]
    if requested_video_keys:
        timestamp_values = _batch_column(dataset, "timestamp", frame_indices)
        timestamps = [
            float(value.item() if hasattr(value, "item") else value)
            for value in timestamp_values
        ]
        video_batch = dataset.reader._query_videos(
            {key: timestamps for key in requested_video_keys}, episode_id
        )
        for key in requested_video_keys:
            frames = _stack_video_batch(video_batch[key], count, key)
            target = "rgb" if key in RGB_FEATURES else "depth"
            for i in range(count):
                items[i][target][key] = _as_numpy(frames[i], key, frame_indices[i])

    for key in [key for key in RGB_FEATURES if key not in video_keys]:
        values = _batch_column(dataset, key, frame_indices)
        if len(values) != count:
            raise RuntimeError(
                f"batched column read for {key!r} returned {len(values)} frames, expected {count}"
            )
        for i, value in enumerate(values):
            items[i]["rgb"][key] = _as_numpy(value, key, frame_indices[i])

    if use_depth:
        for key in [key for key in DEPTH_FEATURE_TO_BUFFER if key not in video_keys]:
            values = _batch_column(dataset, key, frame_indices)
            if len(values) != count:
                raise RuntimeError(
                    f"batched column read for {key!r} returned {len(values)} frames, expected {count}"
                )
            for i, value in enumerate(values):
                items[i]["depth"][key] = _as_numpy(value, key, frame_indices[i])

    for i, item in enumerate(items):
        missing_rgb = [key for key in RGB_FEATURES if key not in item["rgb"]]
        if missing_rgb:
            raise KeyError(f"frame {frame_indices[i]} is missing RGB feature(s): {missing_rgb}")
        if use_depth:
            missing_depth = [key for key in DEPTH_FEATURE_TO_BUFFER if key not in item["depth"]]
            if missing_depth:
                raise KeyError(
                    f"frame {frame_indices[i]} is missing depth feature(s): {missing_depth}"
                )
    return items


def _can_batch_decode(dataset):
    return (getattr(dataset, "reader", None) is not None and
            hasattr(dataset.reader, "_query_videos") and
            getattr(dataset, "hf_dataset", None) is not None)



def _stack_action_column(dataset):
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is not None:
        try:
            values = hf_dataset["action"]
            action = np.stack(
                [
                    _as_numpy(value, "action", index)
                    for index, value in enumerate(values)
                ],
                axis=0,
            )
        except (KeyError, TypeError, ValueError):
            action = None
        if action is not None:
            if action.ndim != 2:
                raise ValueError(f"Expected action column [N,D], got {action.shape}")
            return action.astype(np.float32, copy=False)

    action = np.stack(
        [
            _as_numpy(dataset[index]["action"], "action", index)
            for index in range(len(dataset))
        ],
        axis=0,
    )
    if action.ndim != 2:
        raise ValueError(f"Expected action column [N,D], got {action.shape}")
    return action.astype(np.float32, copy=False)


def _infer_gripper_thresholds(
    values,
    episode_frames,
    *,
    open_side,
    start_frames,
    low_threshold=None,
    high_threshold=None,
):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Gripper action contains no finite values")

    q10, q90 = np.quantile(finite, [0.10, 0.90]).tolist()
    span = q90 - q10
    if span <= 1e-8:
        raise ValueError(
            f"Gripper action has almost no dynamic range: q10={q10}, q90={q90}"
        )

    low = float(low_threshold) if low_threshold is not None else q10 + 0.35 * span
    high = float(high_threshold) if high_threshold is not None else q10 + 0.65 * span
    if not low < high:
        raise ValueError(
            f"Expected low gripper threshold < high threshold, got {low} >= {high}"
        )

    starts = []
    for frames in episode_frames:
        take = frames[: min(int(start_frames), len(frames))]
        starts.extend(values[np.asarray(take, dtype=np.int64)].tolist())
    start_median = float(np.median(np.asarray(starts, dtype=np.float64)))

    if open_side == "auto":
        open_side = (
            "low"
            if abs(start_median - q10) <= abs(start_median - q90)
            else "high"
        )
    if open_side not in {"low", "high"}:
        raise ValueError(
            f"reward gripper open_side must be auto/low/high, got {open_side!r}"
        )

    return {
        "low": float(low),
        "high": float(high),
        "open_side": open_side,
        "q10": float(q10),
        "q90": float(q90),
        "start_median": start_median,
    }


def _classify_gripper(value, thresholds):
    value = float(value)
    if value <= thresholds["low"]:
        side = "low"
    elif value >= thresholds["high"]:
        side = "high"
    else:
        return None
    return "open" if side == thresholds["open_side"] else "closed"


def _debounced_gripper_transitions(values, thresholds, min_dwell):
    state = None
    candidate = None
    candidate_start = None
    candidate_count = 0
    transitions = []

    for index, value in enumerate(np.asarray(values, dtype=np.float64)):
        classified = _classify_gripper(value, thresholds)
        if classified is None:
            continue

        if state is None:
            if candidate == classified:
                candidate_count += 1
            else:
                candidate = classified
                candidate_start = index
                candidate_count = 1
            if candidate_count >= min_dwell:
                state = classified
                candidate = None
                candidate_start = None
                candidate_count = 0
            continue

        if classified == state:
            candidate = None
            candidate_start = None
            candidate_count = 0
            continue

        if candidate == classified:
            candidate_count += 1
        else:
            candidate = classified
            candidate_start = index
            candidate_count = 1

        if candidate_count >= min_dwell:
            transitions.append(
                {
                    "frame_offset": int(candidate_start),
                    "from_state": state,
                    "to_state": classified,
                }
            )
            state = classified
            candidate = None
            candidate_start = None
            candidate_count = 0

    return transitions


def _last_gripper_release_offsets(dataset, episodes, config):
    action = _stack_action_column(dataset)
    left_index = int(config.get("reward_left_gripper_index", 7))
    right_index = int(config.get("reward_right_gripper_index", 15))
    max_index = max(left_index, right_index)
    if action.shape[1] <= max_index:
        raise ValueError(
            f"Action dim={action.shape[1]} cannot access reward gripper index {max_index}"
        )

    start_frames = int(config.get("reward_open_infer_frames", 5))
    min_dwell = int(config.get("reward_min_dwell", 3))
    expected_cycles = int(config.get("reward_expected_cycles", 1))
    if start_frames < 1 or min_dwell < 1 or expected_cycles < 1:
        raise ValueError(
            "reward_open_infer_frames, reward_min_dwell, and "
            "reward_expected_cycles must all be >= 1"
        )

    episode_frames = list(episodes.values())
    left_thresholds = _infer_gripper_thresholds(
        action[:, left_index],
        episode_frames,
        open_side=config.get("reward_left_open_side", "auto"),
        start_frames=start_frames,
        low_threshold=config.get("reward_left_low_threshold"),
        high_threshold=config.get("reward_left_high_threshold"),
    )
    right_thresholds = _infer_gripper_thresholds(
        action[:, right_index],
        episode_frames,
        open_side=config.get("reward_right_open_side", "auto"),
        start_frames=start_frames,
        low_threshold=config.get("reward_right_low_threshold"),
        high_threshold=config.get("reward_right_high_threshold"),
    )

    release_offsets = {}
    for episode_id, frames in episodes.items():
        frame_array = np.asarray(frames, dtype=np.int64)
        left_transitions = _debounced_gripper_transitions(
            action[frame_array, left_index],
            left_thresholds,
            min_dwell,
        )
        right_transitions = _debounced_gripper_transitions(
            action[frame_array, right_index],
            right_thresholds,
            min_dwell,
        )
        left_release = [
            event["frame_offset"]
            for event in left_transitions
            if event["from_state"] == "closed" and event["to_state"] == "open"
        ]
        right_release = [
            event["frame_offset"]
            for event in right_transitions
            if event["from_state"] == "closed" and event["to_state"] == "open"
        ]
        if len(left_release) != expected_cycles or len(right_release) != expected_cycles:
            raise ValueError(
                f"Episode {episode_id!r} reward event detection failed: "
                f"left_release={left_release}, right_release={right_release}, "
                f"expected_cycles={expected_cycles}"
            )
        reward_offset = max(left_release[-1], right_release[-1])
        if reward_offset >= len(frames) - 1:
            raise ValueError(
                f"Episode {episode_id!r} final gripper release offset={reward_offset} "
                f"must occur before recording end={len(frames) - 1}"
            )
        release_offsets[episode_id] = int(reward_offset)

    metadata = {
        "left_index": left_index,
        "right_index": right_index,
        "left_thresholds": left_thresholds,
        "right_thresholds": right_thresholds,
        "open_infer_frames": start_frames,
        "min_dwell": min_dwell,
        "expected_cycles": expected_cycles,
    }
    return release_offsets, metadata



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
        self.use_depth = "depth" in first
        self.depth_shapes = {
            k: list(v.shape) for k, v in first.get("depth", {}).items()
        }
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
        if self.use_depth:
            for name, value in first["depth"].items():
                for key in (name, f"next_{name}"):
                    self.arrays[key] = self.data.create_dataset(
                        key, shape=(count, *value.shape),
                        chunks=(n_chunk, *value.shape),
                        dtype=value.dtype, compressor=comp,
                    )
        self.root.attrs.update({
            "rgb_storage": "jpeg_bytes_v1",
            "rgb_shapes": self.shapes,
            "rgb_dtype": "uint8",
            "state_dim": state_dim,
            "action_dim": action_dim,
            "use_depth": self.use_depth,
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
        if self.use_depth:
            for name in DEPTH_FEATURE_TO_BUFFER.values():
                for field, data_key in (("depth", name), ("next_depth", f"next_{name}")):
                    values = [item[field][name] for item in self.pending]
                    self.arrays[data_key][start:end] = np.stack(values)
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


def _frame(
    raw: dict, state_dim=None, action_dim=None, rgb_shapes=None,
    use_depth: bool = False, depth_shapes=None,
) -> dict:
    required = (
        "observation.state", "action", *RGB_FEATURES.keys(),
        *(DEPTH_FEATURE_TO_BUFFER.keys() if use_depth else ()),
    )
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
    if use_depth:
        result["depth"] = {}
        for feature, name in DEPTH_FEATURE_TO_BUFFER.items():
            value = _numpy(raw[feature], feature)
            expected = None if depth_shapes is None else tuple(depth_shapes[name])
            if expected is not None and value.shape != expected:
                raise ValueError(f"{feature}: depth shape drift {value.shape}, expected {expected}")
            result["depth"][name] = np.ascontiguousarray(value)
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
    reward_config: dict | None = None, use_depth: bool = False,
) -> Path:
    """Convert a local LeRobotDataset-like object; never overwrite output."""
    _require_zarr_v2()
    if not (0 < gamma <= 1):
        raise ValueError("gamma must lie in (0,1]")
    if not 1 <= jpeg_quality <= 100 or stream_batch_size < 1:
        raise ValueError("Invalid JPEG quality or batch size")
    if reward_mode not in {"recorded_sparse", "last_gripper_release_sparse", "terminal_proxy"}:
        raise ValueError("Unsupported reward mode")
    if reward_mode == "recorded_sparse" and reward_json is None:
        raise ValueError("recorded_sparse requires --reward-json")
    if reward_mode != "recorded_sparse" and reward_json is not None:
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
    computed_metadata = None
    if reward_mode == "last_gripper_release_sparse":
        episode_indices = {
            episode_id: list(range(start, end))
            for episode_id, start, end in episodes
        }
        # Original kdc_ARBiM Reward V2 detector, not a new reward definition.
        reward_offsets, computed_metadata = _last_gripper_release_offsets(
            dataset, episode_indices, reward_config or {}
        )
    # Original kdc_ARBiM batch decode; new LeRobot's private method moved
    # from the facade onto DatasetReader. Synthetic fixtures use item fallback.
    batch_decode = _can_batch_decode(dataset)
    first = _frame(dataset[0], use_depth=use_depth)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".partial.", dir=output.parent))
    try:
        writer = _ZarrWriter(staging, total, first, jpeg_quality, stream_batch_size)
        writer.root.attrs.update({
            "reward_mode": (
                "terminal_proxy" if reward_mode == "terminal_proxy"
                else "last_gripper_release_sparse"
            ),
            "reward_source": (
                "recorded_validated" if reward_mode == "recorded_sparse"
                else "computed_last_gripper_release"
                if reward_mode == "last_gripper_release_sparse"
                else "terminal_proxy_not_success_label"
            ),
            "reward_validated": reward_mode != "terminal_proxy",
            "reward_event": {
                "source": (
                    "external_verified_episode_json" if reward_mode == "recorded_sparse"
                    else "computed_last_gripper_release"
                ),
                "calibration": computed_metadata,
                "episodes": [
                    {"episode_index": ep, "reward_offset": reward_offsets[ep]}
                    for ep, _, _ in episodes
                ],
            } if reward_mode != "terminal_proxy" else None,
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
        progress = tqdm(total=total, desc="Streaming build_db")
        for episode_id, start, end in episodes:
            pending = None
            episode_rewards = []
            for batch_start in range(start, end, stream_batch_size):
                batch_indices = list(range(batch_start, min(batch_start + stream_batch_size, end)))
                if batch_decode:
                    raw_batch = _load_lerobot_frame_batch(dataset, episode_id, batch_indices, use_depth)
                    raw_batch = [
                        {"observation.state": row["agent_pos"],
                         "action": row["action"], **row["rgb"],
                         **(row.get("depth", {}) if use_depth else {})}
                        for row in raw_batch
                    ]
                else:
                    raw_batch = [dataset[i] for i in batch_indices]
                for index, raw in zip(batch_indices, raw_batch, strict=True):
                    current = first if index == 0 else _frame(
                        raw, first["state"].size, first["action"].size,
                        writer.shapes, use_depth=use_depth,
                        depth_shapes=writer.depth_shapes,
                    )
                    step = index - start
                    terminal = index == end - 1
                    reward = (
                        float(step == reward_offsets[episode_id])
                        if reward_mode != "terminal_proxy" else float(terminal)
                    )
                    episode_rewards.append(reward)
                    if pending is not None:
                        writer.append({
                            "state": pending["state"], "next_state": current["state"],
                            "action": pending["action"], "next_action": current["action"],
                            "rgb": pending["rgb"],
                            **({
                                "depth": pending["depth"], "next_depth": current["depth"]
                            } if use_depth else {}),
                            "reward": [episode_rewards[-2]],
                            "return": [0.0], "done": [False], "timeout": [False],
                            "next_index": index,
                        })
                    pending = current
                    progress.update(1)
            assert pending is not None
            writer.append({
                "state": pending["state"], "next_state": pending["state"],
                "action": pending["action"], "next_action": pending["action"],
                "rgb": pending["rgb"],
                **({
                    "depth": pending["depth"], "next_depth": pending["depth"]
                } if use_depth else {}),
                "reward": [episode_rewards[-1]],
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
    finally:
        if "progress" in locals():
            progress.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reward-mode", choices=(
        "recorded_sparse", "last_gripper_release_sparse", "terminal_proxy"
    ),
                        default="recorded_sparse")
    parser.add_argument("--reward-json", type=Path)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--stream-batch-size", type=int, default=32)
    parser.add_argument("--video-backend", default=None)
    parser.add_argument("--use-depth", action="store_true")
    parser.add_argument("--reward-left-gripper-index", type=int, default=7)
    parser.add_argument("--reward-right-gripper-index", type=int, default=15)
    parser.add_argument("--reward-left-open-side", choices=("auto", "low", "high"), default="auto")
    parser.add_argument("--reward-right-open-side", choices=("auto", "low", "high"), default="auto")
    parser.add_argument("--reward-open-infer-frames", type=int, default=5)
    parser.add_argument("--reward-min-dwell", type=int, default=3)
    parser.add_argument("--reward-expected-cycles", type=int, default=1)
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
        use_depth=args.use_depth,
        reward_config={
            "reward_left_gripper_index": args.reward_left_gripper_index,
            "reward_right_gripper_index": args.reward_right_gripper_index,
            "reward_left_open_side": args.reward_left_open_side,
            "reward_right_open_side": args.reward_right_open_side,
            "reward_open_infer_frames": args.reward_open_infer_frames,
            "reward_min_dwell": args.reward_min_dwell,
            "reward_expected_cycles": args.reward_expected_cycles,
        },
    )
    print(f"[PASS] Created offline Zarr cache: {output}")


if __name__ == "__main__":
    main()
