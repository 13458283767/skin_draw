"""Preprocessing (M3 -> fixed baseline -> normalization) on the worker thread.

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
    """Latest M3 + fixed-baseline + normalization state, for the GUI."""
    ready: bool
    progress: int
    message: str
    direction: int                                    # -1 = drop, 1 = rise
    norm: str                                         # none / A / B
    target: int                                       # calibration samples per point (calib_n)
    baselines: dict[int, np.ndarray]                  # read-only baseline matrices, for the selected-point detail
    samples: dict[int, int]                           # per-block calibration samples, for the progress detail
    filter_enabled: bool = True                       # snapshot 与显示值使用同一滤波状态
    baseline_issue: str = ""                          # 标定闸门拒绝原因；空串表示正常
    baseline_warning: str = ""                        # 不阻断的提醒，如重新贴附后的基线跳变


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
    def __init__(self, direction=-1, clock=perf_counter, *, calib_n=270, raw_adc=False, norm="B",
                 filter_enabled=True):
        self.lock = Lock()
        self.clock = clock
        self.raw_adc = raw_adc
        # 算法调用保留历史默认值；主界面显式传入按钮状态，启动时关闭。
        self.filter_enabled = bool(filter_enabled) and not raw_adc
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
        self.quality_intervals: dict[int, deque] = {}
        self.reception_rate = ReceptionRate()
        self.missing_counts: dict[int, int] = {}
        self.sequence = 0
        self.elapsed = 0.
        self.protocol_errors = 0
        self.dirty = False
        self.last_missing = {}
        # Optional raw-recording sink (block, receive_time). It runs on the serial
        # thread, before any filtering, and must therefore stay cheap and non-Qt.
        self.raw_sink = None

    def set_raw_sink(self, sink):
        """Attach/detach training recording; taking the lock serialises it with reception."""
        with self.lock:
            self.raw_sink = sink

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
                                             calib_n=self.calib_n,
                                             filter_enabled=self.filter_enabled)
            self.preprocessors[key] = preprocessor
            event(logger, logging.INFO, "皮肤块预处理已启动", block_id=key, shape=tuple(shape),
                  fps=round(fps, 1), baseline_mode="calib", norm=self.norm)
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
                    self.quality_intervals.setdefault(key, deque(maxlen=128)).append((now, interval))
            self.received_at[key] = now
            self.sequence += 1
            self.dirty = True
            previous = self.blocks.get(key)
            if ((previous is None or previous.values.shape != block.values.shape)
                    and self.sensor_masks.mismatched(key, block.values.shape)):
                event(logger, logging.WARNING, "有效点位尺寸不匹配，暂按全部有效处理，请重新编辑",
                      block_id=key, shape=block.values.shape)
            if previous is not None and previous.values.shape != block.values.shape:
                self.quality_intervals.pop(key, None)
                event(logger, logging.WARNING, "尺寸变化，重新自动标定该块", block_id=key,
                      previous=previous.values.shape, current=block.values.shape)
            self.blocks[key] = block
            self.arrivals[key] = now
            self.missing_counts[key] = 0
            # Hand the untouched block to the recorder first: training data must
            # never depend on the display preprocessing or on GUI refresh timing.
            if self.raw_sink is not None:
                self.raw_sink(block, now)
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
                self.preprocessors[key] = BlockPreprocessor(
                    block.values.shape, self.sensor_masks.get(key, block.values.shape),
                    norm=self.norm, direction=_direction_name(self.direction),
                    calib_n=self.calib_n, filter_enabled=self.filter_enabled)
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
                # 用户主动重置：保留上一代基线，下次标定才能发现“重新贴附”
                preprocessor.reset(keep_reference=True)
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

    def set_filter_enabled(self, enabled):
        """与串口接收互斥地切换滤波，清除旧显示和基线，不改写原始块。"""
        if self.raw_adc:
            return
        with self.lock:
            enabled = bool(enabled)
            if self.filter_enabled == enabled:
                return
            self.filter_enabled = enabled
            for key, preprocessor in self.preprocessors.items():
                preprocessor.set_filter_enabled(enabled)
                if not enabled:
                    # 关闭后可以立即显示最近的 ADC，但旧帧不能再计为标定样本。
                    np.copyto(preprocessor.filtered, np.where(
                        preprocessor.valid, self.blocks[key].values, np.nan))
                self.display_values[key] = preprocessor.recompose()
            self.dirty = True
            logger.info("切换 M3 滤波并重置基线 enabled=%s", enabled)

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
            return ("归一化已关闭 · 显示 M3 滤波原值" if self.filter_enabled else
                    "归一化已关闭 · 显示原始 ADC（滤波关闭）")
        # 闸门拒绝时优先说明原因：用户需要知道“标定还没完成”而不是“还在等数据”。
        blocker = next(((key, item) for key, item in sorted(self.preprocessors.items())
                        if item.baseline_issue), None)
        if blocker is not None:
            key, item = blocker
            tail = "；确认无接触后点“重置滤波与基线”" if item.baseline_blocked else "；请松开后等待重新标定"
            return f"标定未通过 · BLOCK {key:02d} {item.baseline_issue}{tail}"
        if all(preprocessor.ready for preprocessor in preprocessors):
            warning = next((item.baseline_warning for item in preprocessors if item.baseline_warning), "")
            return "自动标定完成 · 基线已固定" + (f" · {warning}" if warning else "")
        return "自动标定中 · 请保持无按压（约3秒）"

    def _preprocess_state(self):
        if self.raw_adc:
            return PreprocessState(True, 100, "原始 ADC 模式 · 不扣基线", self.direction, "none",
                                   self.calib_n, {}, {}, False, "", "")
        keys = sorted(self.preprocessors)
        preprocessors = [self.preprocessors[key] for key in keys]
        ready = bool(preprocessors) and all(preprocessor.ready for preprocessor in preprocessors)
        progress = min((preprocessor.progress for preprocessor in preprocessors), default=0)
        issue = next((f"BLOCK {key:02d} {self.preprocessors[key].baseline_issue}"
                      for key in keys if self.preprocessors[key].baseline_issue), "")
        warning = next((f"BLOCK {key:02d} {self.preprocessors[key].baseline_warning}"
                        for key in keys if self.preprocessors[key].baseline_warning), "")
        return PreprocessState(ready, progress, self._message(preprocessors), self.direction, self.norm,
                               self.calib_n,
                               {key: self.preprocessors[key].baselines for key in keys},
                               {key: self.preprocessors[key].samples for key in keys},
                               self.filter_enabled, issue, warning)

    def baseline_snapshot(self):
        """Copy the current raw-order baselines for a sample's calibration_id.

        Returns (``{block_id: matrix}``, minimum calibration samples). Matrices are
        copies, so later recalibration cannot rewrite an already saved
        sample; blocks whose baseline is still incomplete are omitted instead of
        being stored as fabricated zeros.
        """
        with self.lock:
            baselines, counts = {}, []
            for key, preprocessor in self.preprocessors.items():
                if preprocessor.norm == "none":
                    continue
                matrix = preprocessor.baselines
                if np.isfinite(matrix).any():
                    baselines[key] = np.array(matrix, dtype=float, copy=True)
                    counts.append(preprocessor.samples)
            return baselines, (min(counts) if counts else 0)

    def recording_cadence(self, ids, policy):
        """冻结录制前近期接收间隔；陈旧、突发成批或不稳定参考不用于放宽阈值。"""
        with self.lock:
            now = self.clock()
            references = []
            for key in ids:
                values = [interval for timestamp, interval in self.quality_intervals.get(key, ())
                          if now - policy.reference_window_s <= timestamp <= now
                          and np.isfinite(interval) and interval >= 0]
                median = float(np.median(values)) if values else 0.
                reliable = len(values) >= policy.min_reference_intervals and median >= .001
                if reliable:
                    low, high = np.percentile(values, [10, 90])
                    reliable = low >= .001 and high <= 5 * low
                    reliable = reliable and now - self.received_at.get(key, -float("inf")) <= max(1., 5 * median)
                references.append((key, median if reliable else None, len(values)))
            return tuple(references)

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
