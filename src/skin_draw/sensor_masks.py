"""Immutable sensor validity in original hardware coordinates; no ADC heuristics."""

from functools import lru_cache
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def default_mask_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "sensor_masks.json"


@lru_cache(maxsize=256)
def all_valid(shape: tuple[int, int]) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    return np.frombuffer(mask.tobytes(), dtype=bool).reshape(shape)


class SensorMasks:
    """Updates create a new registry, safe to share with live snapshots."""

    def __init__(self):
        self._masks = {}

    def get(self, block_id, shape):
        mask = self._masks.get(block_id)
        return mask if mask is not None and mask.shape == tuple(shape) else all_valid(tuple(shape))

    def mismatched(self, block_id, shape):
        mask = self._masks.get(block_id)
        return mask is not None and mask.shape != tuple(shape)

    def with_mask(self, block_id, mask):
        array = np.asarray(mask)
        if type(block_id) is not int or not 1 <= block_id <= 16:
            raise ValueError("BLOCK ID 必须为 1–16 的整数")
        if array.dtype != np.bool_ or array.ndim != 2 or any(not 1 <= n <= 16 for n in array.shape):
            raise ValueError("有效点位必须是 1–16 行、1–16 列的布尔矩阵")
        result = SensorMasks()
        result._masks = self._masks.copy()
        result._masks[block_id] = np.frombuffer(array.tobytes(), dtype=bool).reshape(array.shape)
        return result

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["version"] != 1 or data["coordinates"] != "raw_row_col" or not isinstance(data["blocks"], dict):
                raise ValueError("不支持的点位配置格式")
            result = cls()
            for key, record in data["blocks"].items():
                block_id = int(key)
                if str(block_id) != key:
                    raise ValueError("BLOCK ID 格式错误")
                result = result.with_mask(block_id, record["valid"])
            return result
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"有效点位配置无效：{error}") from error

    def save(self, path):
        """Replace only after a complete write; no acquisition-thread file IO."""
        path = Path(path)
        payload = {"version": 1, "coordinates": "raw_row_col",
                   "blocks": {str(key): {"valid": mask.tolist()} for key, mask in sorted(self._masks.items())}}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
