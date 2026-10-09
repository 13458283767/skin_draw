"""neo_SKIN_01 v0.5: AA ID M N, row-major big-endian uint16 ADC, LF."""

import numpy as np

from .models import Block


PROTOCOL_VERSION = "0.5"
FRAME_TAIL = 0x0A


def binary_frame_length(header: bytes) -> int:
    if len(header) != 4 or header[0] != 0xAA:
        raise ValueError("二进制帧头无效")
    _, block_id, rows, cols = header
    if not 1 <= block_id <= 16 or not ((rows == cols == 0) or (1 <= rows <= 16 and 1 <= cols <= 16)):
        raise ValueError("二进制 ID 或行列数无效")
    return 5 + 2 * rows * cols


def decode_binary_frame(packet: bytes) -> Block | None:
    """None is a valid NOT_READY response, never a zero-valued sensor matrix."""
    length = binary_frame_length(packet[:4])
    if len(packet) != length:
        raise ValueError("二进制帧长度不匹配")
    if packet[-1] != FRAME_TAIL:
        raise ValueError("二进制帧尾错误（应为0x0A）")
    _, block_id, rows, cols = packet[:4]
    if rows == cols == 0:
        return None
    values = np.frombuffer(packet[4:-1], dtype=">u2").reshape(rows, cols)
    return Block(block_id, values)
