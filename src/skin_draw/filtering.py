"""Reject a row for this sample if any sensor exceeds the allowed ADC jump."""

import numpy as np

from .models import Block


DEFAULT_MAX_JUMP = 800


class ADCOutlierFilter:
    def __init__(self, max_jump: int = DEFAULT_MAX_JUMP):
        self.max_jump = max_jump
        self.previous: dict[int, Block] = {}
        self.rejected_counts: dict[int, int] = {}  # replaced sensor readings

    def reset_reference(self):
        self.previous.clear()

    def process(self, block: Block, valid_mask: np.ndarray | None = None) -> tuple[Block, np.ndarray]:
        if valid_mask is not None and (valid_mask.shape != block.values.shape or valid_mask.dtype != np.bool_):
            raise ValueError("有效点位必须匹配原始矩阵")
        previous = self.previous.get(block.block_id)
        if previous is None or previous.values.shape != block.values.shape:
            # No comparable sample yet; establish a separate reference for
            # each point. Never borrow another block's or another row's data.
            invalid = np.zeros(block.values.shape, dtype=bool)
        else:
            delta = block.values.astype(np.int32) - previous.values.astype(np.int32)
            jumps = np.abs(delta) > self.max_jump
            if valid_mask is not None:
                jumps &= valid_mask
            bad_rows = np.any(jumps, axis=1)
            invalid = np.broadcast_to(bad_rows[:, None], block.values.shape).copy()
            if valid_mask is not None:
                invalid &= valid_mask
        if invalid.any():
            key = block.block_id
            self.rejected_counts[key] = self.rejected_counts.get(key, 0) + int(invalid.sum())
            filtered = Block(key, np.where(invalid, previous.values, block.values))
        else:
            filtered = block
        # Only accepted rows advance their reference. Repeated corrupt
        # packets remain rejected; a normal next sample passes immediately.
        self.previous[block.block_id] = filtered
        invalid.flags.writeable = False
        return filtered, invalid
