"""Transport observations and transition alerts, never a guess at hardware causes."""

from collections import Counter
import logging
from uuid import uuid4

from .diagnostics import event


logger = logging.getLogger(__name__)


class SerialDiagnostics:
    def __init__(self, port, clock):
        self.port, self.clock = port, clock
        self.session = uuid4().hex[:12]
        self.started = self.last_bytes = self.last_valid = self.last_summary = clock()
        self.byte_count = self.tx_bytes = 0
        self.block_counts = Counter()
        self.block_times = {}
        self.block_intervals = {}
        self.shapes = {}
        self.recent = b""
        self.silent = self.invalid_only = False
        self.missing = set()
        self.error_total = 0
        self.last_errors = float("-inf")
        self.previous_bytes = 0
        self.previous_counts = Counter()
        self.limited = {}

    def log(self, level, message, **fields):
        event(logger, level, message, session=self.session, port=self.port, **fields)

    def limited_log(self, key, message, **fields):
        count, last = self.limited.get(key, (0, float("-inf")))
        count += 1
        if self.clock() - last >= 10:
            self.log(logging.WARNING, message, occurrences=count, **fields)
            last = self.clock()
        self.limited[key] = count, last

    def received(self, data):
        if not data:
            return
        now = self.clock()
        if self.byte_count == 0:
            self.log(logging.INFO, "收到首批串口字节", bytes=len(data), wait_seconds=round(now - self.started, 3))
        if self.silent:
            self.log(logging.INFO, "串口字节接收恢复", silence_seconds=round(now - self.last_bytes, 3))
            self.silent = False
        self.byte_count += len(data)
        self.last_bytes = now
        self.recent = (self.recent + data[-96:])[-96:]

    def parsed(self, blocks, parser):
        now = self.clock()
        for block in blocks:
            key = block.block_id
            if self.invalid_only:
                self.log(logging.INFO, "有效数据帧恢复", gap_seconds=round(now - self.last_valid, 3))
                self.invalid_only = False
            if key in self.missing:
                self.log(logging.INFO, "皮肤块数据恢复", block_id=key, gap_seconds=round(now - self.block_times[key], 3))
                self.missing.remove(key)
            shape = block.values.shape
            if key not in self.shapes:
                self.log(logging.INFO, "识别到皮肤块首帧", block_id=key, shape=shape, formats=parser.counts)
            elif self.shapes[key] != shape:
                self.log(logging.WARNING, "皮肤块矩阵尺寸变化", block_id=key, previous=self.shapes[key], current=shape)
            if key in self.block_times:
                interval = now - self.block_times[key]
                if interval > 0:
                    previous = self.block_intervals.get(key, interval)
                    self.block_intervals[key] = .8 * previous + .2 * interval
            self.block_counts[key] += 1
            self.shapes[key] = shape
            self.block_times[key] = self.last_valid = now
        if parser.errors > self.error_total and now - self.last_errors >= 2:
            self.log(logging.WARNING, "串口解析异常", new_errors=parser.errors - self.error_total,
                     total_errors=parser.errors, reasons=parser.error_counts,
                     recent_rx_hex=self.recent.hex(" "), pending_binary_bytes=len(parser.buffer),
                     pending_text_bytes=len(parser.text.buffer))
            self.error_total, self.last_errors = parser.errors, now

    def tick(self, parser, mode):
        now = self.clock()
        if now - self.last_bytes >= 3 and not self.silent:
            self.log(logging.WARNING, "串口连续无字节", duration_seconds=round(now - self.last_bytes, 3),
                     received_any=bool(self.byte_count), mode=mode, total_rx_bytes=self.byte_count)
            self.silent = True
        if self.byte_count and now - self.last_bytes < 3 and now - self.last_valid >= 3 and not self.invalid_only:
            self.log(logging.WARNING, "串口有字节但无有效传感器帧", duration_seconds=round(now - self.last_valid, 3),
                     mode=mode, errors=parser.errors, reasons=parser.error_counts,
                     not_ready_ids=sorted(parser.not_ready_ids), recent_rx_hex=self.recent.hex(" "))
            self.invalid_only = True
        if mode == "stream":
            for key, last in self.block_times.items():
                threshold = max(3., 3 * self.block_intervals.get(key, 0.))
                if now - last >= threshold and key not in self.missing:
                    self.log(logging.WARNING, "皮肤块有效数据超时", block_id=key,
                             duration_seconds=round(now - last, 3), threshold_seconds=round(threshold, 3))
                    self.missing.add(key)
        if now - self.last_summary >= 5:
            self.summary(parser, mode)

    def summary(self, parser, mode):
        now = self.clock()
        seconds = max(now - self.last_summary, .001)
        self.log(logging.INFO, "串口收发统计", mode=mode, interval_seconds=round(seconds, 3),
                 total_rx_bytes=self.byte_count, total_tx_bytes=self.tx_bytes,
                 rx_bytes_per_second=round((self.byte_count - self.previous_bytes) / seconds, 1),
                 block_counts=dict(self.block_counts),
                 block_hz={k: round((v - self.previous_counts[k]) / seconds, 2) for k, v in self.block_counts.items()},
                 block_gap_seconds={k: round(now - t, 3) for k, t in self.block_times.items()},
                 formats=parser.counts, errors=parser.errors, reasons=parser.error_counts,
                 pending_binary_bytes=len(parser.buffer), pending_text_bytes=len(parser.text.buffer),
                 text_block_incomplete=parser.text.current)
        self.previous_bytes, self.previous_counts, self.last_summary = self.byte_count, self.block_counts.copy(), now
