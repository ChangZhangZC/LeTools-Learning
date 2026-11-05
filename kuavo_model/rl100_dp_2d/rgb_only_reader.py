"""Keep the pinned LeRobot dataset reader, but decode only RL100 Actor RGB streams.

The source LeRobot v3 DatasetReader queries all metadata video keys even when
the policy uses only RGB, causing unnecessary depth-video decoding per sample.
This adapter leaves metadata, parquet indices, and native get_item() unchanged.
"""
from __future__ import annotations

from collections.abc import Collection

from lerobot.datasets.dataset_reader import DatasetReader


class RGBOnlyDatasetReader(DatasetReader):
    """Reuse an initialized LeRobot reader without reloading its datasets."""

    def __init__(self, source: DatasetReader, rgb_video_keys: Collection[str]) -> None:
        self.__dict__.update(source.__dict__)
        self._rgb_video_keys = frozenset(rgb_video_keys)

    def _get_query_timestamps(self, current_ts, query_indices=None):
        queries = super()._get_query_timestamps(current_ts, query_indices)
        return {key: times for key, times in queries.items() if key in self._rgb_video_keys}
