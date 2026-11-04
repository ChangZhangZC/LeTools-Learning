"""Copy a small, self-contained LeRobot v3 episode subset for BC diagnostics.

Use the PINNED native LeRobot reader/writer, not ad hoc MP4/Parquet copying:
LeRobot v3 can co-pack several episodes in one video and data chunk. Only
selected episodes are decoded and written; original metadata and dataset
are never modified. Output videos are re-encoded by native LeRobot.

Run from repo root; see docs/models/rl100_dp_2d/README.md.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = ROOT / "third_party" / "lerobot" / "src"
if not (LEROBOT_SRC / "lerobot").is_dir():
    raise FileNotFoundError(
        f"Missing pinned LeRobot source: {LEROBOT_SRC}; "
        "git submodule update --init third_party/lerobot"
    )
sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.utils.constants import DEFAULT_FEATURES  # noqa: E402


def _frame_for_writer(row: dict, features: dict) -> dict:
    """Strip auto-generated index/timestamps; keep original robot feature data."""
    frame = {}
    for key in features:
        if key in DEFAULT_FEATURES:
            continue
        if key not in row:
            raise KeyError(f"Source row lacks required feature {key!r}")
        value = row[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.generic):
            value = value.item()
        frame[key] = value
    if "task" not in row:
        raise KeyError("Source reader did not return the episode task string")
    frame["task"] = str(row["task"])
    return frame


def extract(source: Path, destination: Path, *, count: int, start: int,
            source_repo_id: str, output_repo_id: str, codec: str,
            video_backend: str) -> dict:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Missing LeRobot v3 metadata: {source / 'meta/info.json'}")
    if destination.exists():
        raise FileExistsError(
            f"Target already exists: {destination}; use a new path. "
            "Never overwrite or delete an existing dataset automatically."
        )
    if count < 2 or start < 0:
        raise ValueError("Select at least two episodes and a nonnegative --start")
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    meta = LeRobotDatasetMetadata(source_repo_id, root=source)
    total = int(meta.total_episodes)
    if start + count > total:
        raise ValueError(f"Requested episodes [{start}, {start + count}) outside 0:{total}")
    ids = list(range(start, start + count))
    if not meta.camera_keys:
        raise ValueError("The source dataset contains no visual features")
    if meta.subtasks is not None:
        raise ValueError(
            "Source has subtask metadata; this small extractor does not "
            "rebuild subtask tables. Refusing to silently drop subtask semantics."
        )
    print(f"[DATA] source={source} episodes={total}, extracting {ids}", flush=True)
    print(f"[DATA] video_keys={meta.video_keys}, image_keys={meta.image_keys}", flush=True)
    src = LeRobotDataset(
        repo_id=source_repo_id, root=source, episodes=ids,
        video_backend=video_backend,
    )
    if int(src.num_episodes) != count:
        raise AssertionError(f"Source reader selected {src.num_episodes} episodes, expected {count}")

    out = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=int(meta.fps),
        features=copy.deepcopy(meta.features),
        root=destination,
        robot_type=meta.robot_type,
        use_videos=bool(meta.video_keys),
        video_backend=video_backend,
        vcodec=codec,
        image_writer_processes=0,
        image_writer_threads=0,
    )
    current_ep = None
    written = 0
    completed = 0
    per_episode = []
    try:
        for index in range(len(src)):
            row = src[index]  # Reader decodes only the chosen episode frames.
            episode_id = int(torch.as_tensor(row["episode_index"]).item())
            if episode_id not in ids:
                raise AssertionError(f"Unexpected episode id {episode_id}")
            if current_ep is None:
                current_ep = episode_id
            elif episode_id != current_ep:
                out.save_episode(parallel_encoding=False)
                completed += 1
                print(f"[PASS] copied episode {current_ep} "
                      f"({per_episode[-1]['frames']} frames)", flush=True)
                current_ep = episode_id
            if len(per_episode) == completed:
                per_episode.append({"source_episode": episode_id, "frames": 0})
            out.add_frame(_frame_for_writer(row, meta.features))
            written += 1
            per_episode[-1]["frames"] += 1
        if current_ep is not None:
            out.save_episode(parallel_encoding=False)
            completed += 1
            print(f"[PASS] copied episode {current_ep} "
                  f"({per_episode[-1]['frames']} frames)", flush=True)
        out.finalize()
    except Exception:
        print(f"[ERROR] extraction interrupted. Partial output may remain at {destination}; "
              "inspect it before re-running with a NEW destination path.", flush=True)
        raise
    if completed != count:
        raise AssertionError(f"Extracted {completed} episodes; expected {count}")
    if written != len(src):
        raise AssertionError(f"Extracted {written} frames; source subset has {len(src)}")

    # Re-open the exact destination to check LeRobot's new metadata and indexing.
    checked = LeRobotDataset(repo_id=output_repo_id, root=destination,
                             video_backend=video_backend)
    if int(checked.meta.total_episodes) != count or len(checked) != written:
        raise AssertionError(
            "Extracted dataset episode/frame count differs from source subset"
        )
    if checked.features != meta.features:
        raise AssertionError("Extracted feature schemas differ from source")
    if not checked.meta.camera_keys:
        raise AssertionError("Extracted dataset lost its cameras")
    first, last = checked[0], checked[len(checked) - 1]
    for key in ("observation.state", "action", *meta.camera_keys):
        if key not in first or key not in last:
            raise AssertionError(f"Missing output data key {key}")
    report = {
        "source": str(source), "destination": str(destination),
        "source_total_episodes": total, "selected_source_episodes": ids,
        "output_repo_id": output_repo_id, "extracted_episodes": completed,
        "extracted_frames": written, "camera_keys": list(meta.camera_keys),
        "fps": int(meta.fps), "episode_mapping": per_episode,
    }
    (destination / "extraction_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[PASS] independent LeRobot dataset: {completed} episodes, "
          f"{written} frames at {destination}", flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-repo-id", default="local/icra_task1_full")
    p.add_argument("--repo-id", default="local/icra_task1_test10")
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--video-backend", default="pyav")
    p.add_argument("--codec", default="auto",
                   help="LeRobot native video encoder, e.g. auto or h264")
    a = p.parse_args()
    extract(a.source, a.output, count=a.count, start=a.start,
            source_repo_id=a.source_repo_id, output_repo_id=a.repo_id,
            codec=a.codec, video_backend=a.video_backend)


if __name__ == "__main__":
    main()
