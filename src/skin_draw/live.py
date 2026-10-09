"""Delivery-pack preprocessing (M3 -> slow baseline -> normalization) on the worker thread.

The serial thread owns ingestion: every received block advances its
BlockPreprocessor. The GUI takes bounded snapshots; slow paints cannot drop
samples or create an ever-growing queue of old pictures.
"""

from collections import deque
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
import logging
import numpy as np

from .acquisition import Scan
from .diagnostics import event
from .models import Block, Frame
from .preprocessing import BlockPreprocessor, DEFAULT_FPS, NORM_MODES
from .reception_rate import ReceptionRate
from .sensor_masks import SensorMasks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessState:
    """Latest M3 + slow-baseline + normalization state, for the GUI."""
    ready: bool
    progress: int
    message: str
    direction: int                                    # -1 = drop, 1 = rise
    norm: str                                         # none / A / B
    target: int                                       # calibration samples per point (calib_n)
    baselines: dict[int, np.ndarray]                  # read-only baseline matrices, for the selected-point detail
    samples: dict[int, int]                           # per-block calibration samples, for the progress detail


@dataclass(frozen=True)
class LiveSnapshot:
    scan: Scan
    preprocess: PreprocessState
    layout_blocks: tuple[Block, ...]
    missing_counts: dict[int, int]
    rates: dict[int, float]
    received_blocks: int
    display_values: dict
    latest_received_at: float


def _direction_name(direction) -> str:
    return "drop" if int(direction) < 0 else "rise"


class LiveBuffer:
    def __init__(self, direction=-1, clock=perf_counter, *, calib_n=270, raw_adc=False, norm="B"):
        self.lock = Lock()
        self.clock = clock
        self.raw_adc = raw_adc
        self.norm = norm
        self.direction = -1 if int(direction) < 0 else 1
        self.calib_n = max(1, int(calib_n))
        self.sensor_masks = SensorMasks()
        self.preprocessors: dict[int, BlockPreprocessor] = {}
        self.blocks: dict[int, Block] = {}
        self.display_values = {}  # immutable responses, replaced only for updated blocks
        self.received_at: dict[int, float] = {}
        self.arrivals: dict[int, float] = {}
        self.intervals: dict[int, deque] = {}
        self.reception_rate = ReceptionRate()
        self.missing_counts: dict[int, int] = {}
        self.sequence = 0
        self.elapsed = 0.
        self.protocol_errors = 0
        self.dirty = False
        self.last_missing = {}

    def _measured_fps(self, key):
        """Measured per-block rate from recent intervals; delivery-pack fallback."""
        intervals = self.intervals.get(key)
        if not intervals:
            return DEFAULT_FPS
        total = sum(intervals)
        return len(intervals) / total if total > 0 else DEFAULT_FPS

    def _preprocessor(self, key, shape):
        preprocessor = self.preprocessors.get(key)
        if preprocessor is None or preprocessor.shape != tuple(shape):
            fps = self._measured_fps(key)
            preprocessor = BlockPreprocessor(shape, self.sensor_masks.get(key, shape),
                                             norm=self.norm, direction=_direction_name(self.direction),
                                             fps=fps, calib_n=self.calib_n)
            self.preprocessors[key] = preprocessor
            event(logger, logging.INFO, "皮肤块预处理已启动", block_id=key, shape=tuple(shape),
                  fps=round(fps, 1), upd_n=preprocessor.upd_n, norm=self.norm)
        return preprocessor

    def receive(self, block: Block):
        with self.lock:
            now = self.clock()
            key = block.block_id
            self.reception_rate.record(key, now)
            if key in self.received_at:
                interval = now - self.received_at[key]
                if interval >= 0:
                    self.intervals.setdefault(key, deque(maxlen=10)).append(interval)
            self.received_at[key] = now
            self.sequence += 1
            self.dirty = True
            previous = self.blocks.get(key)
            if ((previous is None or previous.values.shape != block.values.shape)
                    and self.sensor_masks.mismatched(key, block.values.shape)):
                event(logger, logging.WARNING, "有效点位尺寸不匹配，暂按全部有效处理，请重新编辑",
                      block_id=key, shape=block.values.shape)
            if previous is not None and previous.values.shape != block.values.shape:
                event(logger, logging.WARNING, "尺寸变化，重新自动标定该块", block_id=key,
                      previous=previous.values.shape, current=block.values.shape)
            self.blocks[key] = block
            self.arrivals[key] = now
            self.missing_counts[key] = 0
            if self.raw_adc:
                # Diagnostic display: preserve the exact parsed matrix, including
                # large jumps and zeros. Never filter or subtract a baseline.
                self.display_values[key] = block.values
                return
            self.display_values[key] = self._preprocessor(key, block.values.shape).update(block.values)

    def set_sensor_masks(self, masks, changed_id=None, *, recalibrate=True):
        """Serialize mask changes with ingestion; reset only the edited block."""
        with self.lock:
            self.sensor_masks = masks
            keys = tuple(self.blocks) if changed_id is None else (changed_id,)
            if not recalibrate or self.raw_adc:
                keys = ()
            for key in keys:
                self.display_values.pop(key, None)
                block = self.blocks.get(key)
                if block is None:
                    self.preprocessors.pop(key, None)
                    continue
                # Rebuild in place: the edited block must immediately report as
                # not ready instead of disappearing from the aggregate state.
                fps = self._measured_fps(key)
                self.preprocessors[key] = BlockPreprocessor(
                    block.values.shape, self.sensor_masks.get(key, block.values.shape),
                    norm=self.norm, direction=_direction_name(self.direction),
                    fps=fps, calib_n=self.calib_n)
            self.dirty = True
            if keys:
                event(logger, logging.INFO, "更新有效点位，重新自动标定对应皮肤块", blocks=keys)

    def finish_scan(self, scan: Scan, authoritative=True):
        with self.lock:
            self.elapsed = scan.elapsed
            self.protocol_errors = scan.protocol_errors
            if authoritative:
                received = {block.block_id for block in scan.frame.blocks}
                for key in scan.expected_ids:
                    if key not in received:
                        self.missing_counts[key] = self.missing_counts.get(key, 0) + 1
            self.dirty = True

    def reset_preprocessing(self):
        if self.raw_adc:
            return
        with self.lock:
            logger.info("重置滤波与基线，重新自动标定")
            for preprocessor in self.preprocessors.values():
                preprocessor.reset()
            self.display_values.clear()
            self.dirty = True

    def set_direction(self, direction):
        if self.raw_adc:
            return
        with self.lock:
            self.direction = -1 if int(direction) < 0 else 1
            logger.info("切换受压方向 direction=%s", _direction_name(self.direction))
            for key, preprocessor in self.preprocessors.items():
                preprocessor.set_direction(_direction_name(self.direction))
                self.display_values[key] = preprocessor.recompose()
            self.dirty = True

    def set_norm(self, norm):
        if norm not in NORM_MODES:
            raise ValueError("归一化模式必须为 none/A/B")
        if self.raw_adc:
            return
        with self.lock:
            self.norm = norm
            logger.info("切换归一化模式 norm=%s", norm)
            for key, preprocessor in self.preprocessors.items():
                preprocessor.set_norm(norm)
                self.display_values[key] = preprocessor.recompose()
            self.dirty = True

    def _message(self, preprocessors):
        if not preprocessors:
            return "等待数据"
        if self.norm == "none":
            return "归一化已关闭 · 显示 M3 滤波原值"
        if all(preprocessor.ready for preprocessor in preprocessors):
            return "自动标定完成 · 慢跟踪运行中（长按可能缓慢回落）"
        return "自动标定中 · 请保持无按压（约3秒）"

    def _preprocess_state(self):
        if self.raw_adc:
            return PreprocessState(True, 100, "原始 ADC 模式 · 不扣基线", self.direction, "none",
                                   self.calib_n, {}, {})
        keys = sorted(self.preprocessors)
        preprocessors = [self.preprocessors[key] for key in keys]
        ready = bool(preprocessors) and all(preprocessor.ready for preprocessor in preprocessors)
        progress = min((preprocessor.progress for preprocessor in preprocessors), default=0)
        return PreprocessState(ready, progress, self._message(preprocessors), self.direction, self.norm,
                               self.calib_n,
                               {key: self.preprocessors[key].baselines for key in keys},
                               {key: self.preprocessors[key].samples for key in keys})

    def reception_counts(self) -> dict[int, int]:
        """Read the last-second counts even when no new display snapshot exists."""
        with self.lock:
            return self.reception_rate.counts(self.clock())

    def take(self) -> LiveSnapshot | None:
        with self.lock:
            now = self.clock()
            rates = {key: len(values) / sum(values) for key, values in self.intervals.items() if sum(values) > 0}
            missing = dict(self.missing_counts)
            for key, arrived in self.arrivals.items():
                # Streaming has no request timeout. Detect silence using this
                # block's measured cadence, not the GUI's refresh count.
                timeout = max(1., 3. / rates[key]) if key in rates else 3.
                if now - arrived > timeout:
                    missing[key] = max(missing.get(key, 0), 3 if now - arrived > 2 * timeout else 1)
            if not self.dirty and missing == self.last_missing:
                return None
            self.dirty = False
            self.last_missing = missing
            fresh = tuple(self.blocks[key] for key in sorted(self.blocks) if not missing.get(key))
            frame = Frame(fresh, self.sequence, max(self.arrivals.values(), default=now))
            return LiveSnapshot(Scan(frame, tuple(sorted(self.blocks)), self.elapsed, self.protocol_errors),
                                self._preprocess_state(), tuple(self.blocks.values()),
                                missing, rates, self.sequence,
                                {b.block_id: self.display_values[b.block_id] for b in fresh
                                 if b.block_id in self.display_values},
                                max(self.received_at.values(), default=now))
