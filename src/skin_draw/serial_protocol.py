"""Bounded incremental demultiplexing of text and binary serial replies."""

from time import monotonic
from collections import Counter

from .binary_protocol import binary_frame_length, decode_binary_frame
from .protocol import BlockParser


class SerialBlockParser:
    def __init__(self, mode="auto", frame_timeout=.25, clock=monotonic):
        if mode not in ("auto", "text", "binary"):
            raise ValueError("数据格式必须为 auto / text / binary")
        if frame_timeout <= 0:
            raise ValueError("残帧超时必须大于零")
        self.mode = mode
        self.frame_timeout, self.clock = frame_timeout, clock
        self.text = BlockParser()
        self.buffer = bytearray()
        self._binary_started = None
        self._errors = 0
        self._error_counts = Counter()
        self.binary_ends = 0
        self.not_ready_ids = set()
        self.counts = {"text": 0, "binary": 0}

    @property
    def errors(self):
        return self._errors + self.text.errors

    @property
    def error_counts(self):
        return dict(self._error_counts + self.text.error_counts)

    @property
    def ends(self):
        # A complete binary response has no optional text END to wait for.
        return self.text.ends + self.binary_ends

    def reset(self):
        self.buffer.clear()
        self.text.reset()
        self.not_ready_ids.clear()
        self._binary_started = None

    def _reject(self, reason):
        self._errors += 1
        self._error_counts[reason] += 1
        del self.buffer[0]  # Search again, including bytes from a damaged packet.
        self._binary_started = None

    def feed(self, data: bytes):
        completed = []
        if self.mode == "text":
            completed = self.text.feed(data)
            self.counts["text"] += len(completed)
            return completed
        # Check the old partial packet before appending new traffic. A 250 ms
        # deadline permits a full 517-byte frame at 115200 bps (~45 ms).
        if self._binary_started is not None and self.clock() - self._binary_started >= self.frame_timeout:
            self._reject("二进制残帧超时")
        self.buffer.extend(data)
        while self.buffer:
            if self.buffer[0] != 0xAA:
                end = self.buffer.find(b"\xaa")
                end = len(self.buffer) if end < 0 else end
                if self.mode == "auto":
                    blocks = self.text.feed(bytes(self.buffer[:end]))
                    self.counts["text"] += len(blocks)
                    completed.extend(blocks)
                del self.buffer[:end]
                continue
            if self._binary_started is None:
                self._binary_started = self.clock()
                if self.text.current is not None:
                    self._errors += 1
                    self._error_counts["明文未收齐即收到二进制头"] += 1
                self.text.reset()
            if len(self.buffer) < 4:
                break
            try:
                length = binary_frame_length(self.buffer[:4])
            except ValueError:
                self._reject("二进制ID或行列无效")
                continue
            # Detect an impossible ADC high byte before waiting for a damaged
            # length field. ADC low bytes may legitimately contain AA or LF;
            # only the byte at the declared frame end is the LF trailer.
            if any(byte > 0x0F for byte in self.buffer[4:min(len(self.buffer), length - 1):2]):
                self._reject("二进制ADC高字节越界")
                continue
            if len(self.buffer) < length:
                break
            packet = bytes(self.buffer[:length])
            try:
                block = decode_binary_frame(packet)
            except ValueError as error:
                self._reject(str(error))
                continue
            del self.buffer[:length]
            self._binary_started = None
            self.binary_ends += 1
            self.counts["binary"] += 1
            if block is None:
                self.not_ready_ids.add(packet[1])
            else:
                completed.append(block)
        return completed
