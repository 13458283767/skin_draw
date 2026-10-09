"""Calibrate every sample, retain only the latest display state per block.

The serial thread owns ingestion. The GUI takes bounded snapshots; slow paints
cannot drop baseline samples or create an ever-growing queue of old pictures.
"""

from collections import deque
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
import logging
import numpy as np

from .acquisition import Scan
from .filtering import ADCOutlierFilter
from .models import Block, Frame
from .processing import BaselineCalibration
from .diagnostics import event
from .reception_rate import ReceptionRate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveSnapshot:
    scan: Scan
    calibration: BaselineCalibration
    layout_blocks: tuple[Block, ...]
    missing_counts: dict[int, int]
    rates: dict[int, float]
    received_blocks: int
    filtered_counts: dict[int, int]
    display_frame: Frame
    filtered_masks: dict
    display_values: dict
    latest_received_at: float


class LiveBuffer:
    def __init__(self, direction=-1, clock=perf_counter, *, calibration_seconds=1., raw_adc=False):
        self.lock = Lock()
        self.clock = clock
        self.raw_adc = raw_adc
        self.calibration = BaselineCalibration(30, minimum_seconds=calibration_seconds)
        self.calibration.set_direction(direction)
        self.outlier_filter = ADCOutlierFilter()
        self.blocks: dict[int, Block] = {}
        self.filtered_blocks: dict[int, Block] = {}
        self.filtered_masks = {}
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
        self.last_filter_log = float("-inf")

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
                    and self.calibration.sensor_masks.mismatched(key, block.values.shape)):
                event(logger, logging.WARNING, "有效点位尺寸不匹配，暂按全部有效处理，请重新编辑",
                      block_id=key, shape=block.values.shape)
            valid = self.calibration.sensor_masks.get(key, block.values.shape)
            self.blocks[key] = block
            if key in self.calibration.shapes and self.calibration.shapes[key] != block.values.shape:
                event(logger, logging.WARNING, "尺寸变化，重新校准该块", block_id=key,
                      previous=self.calibration.shapes[key], current=block.values.shape)
                self.calibration.reset_block(key)
                self.outlier_filter.previous.pop(key, None)
            if self.raw_adc:
                # Diagnostic display: preserve the exact parsed matrix, including
                # large jumps and zeros. Never establish or subtract a baseline.
                self.filtered_blocks[key] = block
                self.display_values[key] = block.values
                self.arrivals[key] = now
                self.missing_counts[key] = 0
                return
            was_ready = key in self.calibration.baseline
            # Calibration sees ORIGINAL rows independently of the runtime
            # last-good-value filter, so a corrupt first frame cannot lock it.
            if not was_ready:
                self.calibration.consume(Frame((block,), self.sequence, now), tuple(sorted(self.blocks)))
            if not was_ready and key in self.calibration.baseline:
                self.outlier_filter.previous[key] = Block(key, np.rint(self.calibration.baseline[key]))
                event(logger, logging.INFO, "皮肤块稳定基线校准完成", block_id=key,
                      shape=block.values.shape, received_blocks=self.sequence,
                      row_samples=[row.count for row in self.calibration.rows[key]],
                      row_seconds=[round(row.seconds, 3) for row in self.calibration.rows[key]])
            if key in self.calibration.baseline:
                filtered, invalid = self.outlier_filter.process(block, valid)
            else:
                filtered = block
                invalid = np.zeros(block.values.shape, dtype=bool)
                invalid.flags.writeable = False
            self.filtered_blocks[key] = filtered
            self.filtered_masks[key] = invalid
            if invalid.any() and now - self.last_filter_log >= 5:
                rows = [row for row in range(invalid.shape[0]) if invalid[row].any()]
                event(logger, logging.WARNING, "ADC异常行已过滤", block_id=key, rows_zero_based=rows,
                      max_jump=self.outlier_filter.max_jump,
                      total_rejected_points=dict(self.outlier_filter.rejected_counts),
                      raw_rows=block.values[rows[:2]].tolist(), held_rows=filtered.values[rows[:2]].tolist())
                self.last_filter_log = now
            if not valid.any() or np.any(valid & ~invalid):
                self.arrivals[key] = now
            else:
                self.arrivals.setdefault(key, now)
            self.missing_counts[key] = 0
            values = self.calibration.project(Frame((filtered,), self.sequence, now))
            response = values.get(key)
            if response is None:
                self.display_values.pop(key, None)
            else:
                response.flags.writeable = False
                self.display_values[key] = response
            self.dirty = True

    def set_sensor_masks(self, masks, changed_id=None, *, recalibrate=True):
        """Serialize mask changes with ingestion; reset only the edited block."""
        with self.lock:
            self.calibration.sensor_masks = masks
            keys = tuple(self.blocks) if changed_id is None else (changed_id,)
            if not recalibrate or self.raw_adc:
                keys = ()
            for key in keys:
                self.calibration.reset_block(key)
                self.outlier_filter.previous.pop(key, None)
                self.filtered_masks.pop(key, None)
                self.display_values.pop(key, None)
            self.dirty = True
            if keys:
                event(logger, logging.INFO, "更新有效点位，重新校准对应皮肤块", blocks=keys)

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

    def reset_calibration(self):
        if self.raw_adc:
            return
        with self.lock:
            logger.info("重置基线校准及异常过滤参考")
            self.calibration.reset()
            self.display_values.clear()
            self.outlier_filter.reset_reference()
            self.calibration.expected_ids = tuple(sorted(self.blocks))
            self.dirty = True

    def set_direction(self, direction):
        if self.raw_adc:
            return
        with self.lock:
            logger.info("切换受压方向 direction=%s", direction)
            self.calibration.set_direction(direction)
            self.display_values = self.calibration.project(Frame(tuple(self.filtered_blocks.values())))
            for response in self.display_values.values():
                response.flags.writeable = False
            self.dirty = True

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
            display_frame = Frame(tuple(self.filtered_blocks[b.block_id] for b in fresh),
                                  self.sequence, frame.received_at)
            return LiveSnapshot(Scan(frame, tuple(sorted(self.blocks)), self.elapsed, self.protocol_errors),
                                self.calibration.snapshot(), tuple(self.blocks.values()),
                                missing, rates, self.sequence, dict(self.outlier_filter.rejected_counts),
                                display_frame, dict(self.filtered_masks),
                                {b.block_id: self.display_values[b.block_id] for b in fresh
                                 if b.block_id in self.display_values},
                                max(self.received_at.values(), default=now))
