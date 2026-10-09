"""Incremental parser for BLOCK_id,rows,cols / CSV rows / optional END."""

import re
from collections import Counter

from .models import Block, MAX_SIDE

HEADER = re.compile(rb"BLOCK_(\d+),(\d+),(\d+):")


class BlockParser:
    def __init__(self):
        self.buffer = bytearray()
        self.current = None
        self.rows = []
        self.errors = 0
        self.error_counts = Counter()
        self.ends = 0
        self.discard_line = False

    def reset(self):
        self.buffer.clear()
        self.current = None
        self.rows = []
        self.discard_line = False

    def _bad_block(self, reason):
        self.errors += 1
        self.error_counts[reason] += 1
        self.current = None
        self.rows = []

    def feed(self, data: bytes) -> list[Block]:
        completed = []
        for byte in data:
            if byte == 10:
                if not self.discard_line:
                    block = self._line(bytes(self.buffer).rstrip(b"\r"))
                    if block is not None:
                        completed.append(block)
                self.buffer.clear()
                self.discard_line = False
            elif not self.discard_line:
                self.buffer.append(byte)
                if len(self.buffer) > 512:
                    self._bad_block("明文行超过512字节")
                    self.buffer.clear()
                    self.discard_line = True
        return completed

    def _line(self, line: bytes) -> Block | None:
        if line.startswith(b"BLOCK_"):
            if self.current is not None:
                self._bad_block("明文前一块未收齐即出现新帧头")
            match = HEADER.fullmatch(line)
            if match is None:
                self._bad_block("明文帧头格式错误")
                return None
            block_id, rows, cols = map(int, match.groups())
            if not (1 <= block_id <= 16 and 1 <= rows <= MAX_SIDE and 1 <= cols <= MAX_SIDE):
                self._bad_block("明文ID或行列越界")
                return None
            self.current = block_id, rows, cols
            self.rows = []
            return None
        if line == b"END":
            self.ends += 1
            if self.current is not None:  # END before all data rows is invalid.
                self._bad_block("明文数据未收齐即收到END")
            return None
        if self.current is None:
            return None
        block_id, rows, cols = self.current
        fields = line.split(b",")
        if len(fields) != cols or any(not field.isdigit() for field in fields):
            self._bad_block("明文数据列数或数字格式错误")
            return None
        values = list(map(int, fields))
        if any(value > 4095 for value in values):
            self._bad_block("明文ADC超过4095")
            return None
        self.rows.append(values)
        if len(self.rows) == rows:
            block = Block(block_id, self.rows)
            self.current = None
            self.rows = []
            return block
        return None
