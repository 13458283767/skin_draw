"""Independent per-row stable calibration, followed by frozen block baselines."""

import numpy as np
from copy import copy

from .models import Frame
from .sensor_masks import SensorMasks
from .calibration_row import CalibrationRow


class BaselineCalibration:
    def __init__(self, required_frames: int = 30, minimum_seconds: float = 1.):
        if required_frames < 1:
            raise ValueError("基线帧数必须大于零")
        self.required_frames = required_frames
        if not np.isfinite(minimum_seconds) or minimum_seconds < 0:
            raise ValueError("校准时长必须为非负有限数值")
        self.minimum_seconds = minimum_seconds
        self.direction = -1  # pressed ADC decreases by default
        self.sensor_masks = SensorMasks()
        self.reset()

    def reset(self):
        self.counts: dict[int, int] = {}
        self.baseline: dict[int, np.ndarray] = {}
        self.rows = {}
        self.expected_ids = ()
        self.shapes = {}
        self.last_sequences: dict[int, int] = {}
        self.message = "请保持无按压，等待实际连接的皮肤块数据"

    def snapshot(self):
        """Independent control state sharing only finalized, read-only baselines."""
        result = copy(self)
        for name in ("counts", "baseline", "shapes", "last_sequences"):
            setattr(result, name, getattr(self, name).copy())
        result.rows = {key: [row.snapshot() for row in rows] for key, rows in self.rows.items()}
        return result

    @property
    def count(self):
        """Overall progress is the least advanced connected block."""
        return min((self.counts.get(key, 0) for key in self.expected_ids), default=0)

    @property
    def ready(self):
        return bool(self.expected_ids) and all(key in self.baseline for key in self.expected_ids)

    @property
    def progress(self):
        return int(100 * min((min((row.progress for row in self.rows.get(key, [])), default=0.)
                             for key in self.expected_ids), default=0.))

    @property
    def pending_rows(self):
        return {key: [i + 1 for i, row in enumerate(rows) if row.result is None]
                for key, rows in self.rows.items() if key not in self.baseline}

    def set_direction(self, direction: int):
        if direction not in (-1, 1):
            raise ValueError("方向必须是 -1 或 +1")
        self.direction = direction

    def reset_block(self, block_id: int):
        for store in (self.counts, self.baseline, self.rows, self.shapes, self.last_sequences):
            store.pop(block_id, None)

    def consume(self, frame: Frame, expected_ids: tuple[int, ...]) -> dict[int, np.ndarray]:
        expected_ids = tuple(sorted(expected_ids))
        # Only an explicit change in the detected device set retires a block.
        # A missing reply in one scan does not erase anyone's calibration.
        stores = (self.counts, self.baseline, self.rows, self.shapes, self.last_sequences)
        if expected_ids != self.expected_ids:
            for store in stores:
                for key in set(store) - set(expected_ids):
                    del store[key]
        self.expected_ids = expected_ids
        for block in frame.blocks:
            key = block.block_id
            if key not in expected_ids:
                continue
            if key in self.shapes and self.shapes[key] != block.values.shape:
                self.reset_block(key)
            if key in self.baseline:
                continue
            if key in self.last_sequences and frame.sequence <= self.last_sequences[key]:
                continue
            self.last_sequences[key] = frame.sequence
            valid = self.sensor_masks.get(key, block.values.shape)
            if key not in self.rows:
                self.rows[key] = [CalibrationRow(int(mask.sum()), self.required_frames, self.minimum_seconds)
                                  for mask in valid]
                self.shapes[key] = block.values.shape
            for i, row in enumerate(self.rows[key]):
                row.consume(block.values[i, valid[i]], frame.received_at)
            self.counts[key] = min((min(row.count, self.required_frames) for row in self.rows[key]
                                    if row.total.size), default=self.required_frames)
            if all(row.result is not None for row in self.rows[key]):
                baseline = np.zeros(block.values.shape, dtype=float)
                for i, row in enumerate(self.rows[key]):
                    baseline[i, valid[i]] = row.result
                baseline.flags.writeable = False
                self.baseline[key] = baseline
        if self.ready:
            self.message = f"{len(expected_ids)} 块基线已就绪，可以按压传感器"
        else:
            self.message = (f"保持无按压 · 已完成 {len(self.baseline)}/{len(expected_ids)} 块；"
                            f"各行 ≥{self.minimum_seconds:g} 秒且 ≥{self.required_frames} 个有效样本")
        return self.project(frame)

    @property
    def progress_details(self):
        lines = []
        for key in self.expected_ids:
            if key in self.baseline:
                lines.append(f"BLOCK {key:02d}：已完成")
                continue
            lines.append(f"BLOCK {key:02d}：未完成原始行 " +
                         "、".join(map(str, self.pending_rows.get(key, []))))
            for i, row in enumerate(self.rows.get(key, [])):
                if row.result is None:
                    state = "建立多帧参考" if row.reference is None else "收集稳定样本"
                    lines.append(f"  行{i+1}：{state}，{row.count}/{self.required_frames}样本，"
                                 f"{row.seconds:.2f}/{self.minimum_seconds:g}秒")
        return "\n".join(lines)

    @property
    def pending_summary(self):
        lines = []
        for key, pending in self.pending_rows.items():
            rows = [self.rows[key][i - 1] for i in pending]
            lines.append(f"BLOCK {key:02d} 待完成行：" + "、".join(map(str, pending)) +
                         f"\n最少 {min(r.count for r in rows)} 样本 / {min(r.seconds for r in rows):.2f} 秒")
        return "\n".join(lines)

    def signed_delta(self, block_id: int, values: np.ndarray) -> np.ndarray:
        return np.where(self.sensor_masks.get(block_id, values.shape),
                        values.astype(np.float64) - self.baseline[block_id], 0)

    def project(self, frame: Frame) -> dict[int, np.ndarray]:
        if not self.baseline:
            return {}
        result = {}
        for block in frame.blocks:
            baseline = self.baseline.get(block.block_id)
            if baseline is not None and block.values.shape == baseline.shape:
                result[block.block_id] = np.maximum(0, self.direction * self.signed_delta(block.block_id, block.values))
        return result
