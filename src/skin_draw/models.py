"""Validated data contracts; deliberately independent of Qt and data sources."""

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

ADC_MAX = 4095
MAX_SIDE = 16


@dataclass(frozen=True, eq=False)
class Block:
    block_id: int
    values: NDArray[np.uint16]

    def __post_init__(self) -> None:
        if type(self.block_id) is not int or not 1 <= self.block_id <= 16:
            raise ValueError("BLOCK ID 必须为 1–16 的整数")
        array = np.asarray(self.values)
        if array.ndim != 2 or any(not 1 <= n <= MAX_SIDE for n in array.shape):
            raise ValueError("数据必须为 1–16 行、1–16 列的二维矩阵")
        if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
            raise ValueError("ADC 必须为数值")
        if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > ADC_MAX):
            raise ValueError("ADC 必须为 0–4095 的有限数值")
        if np.any(array != np.floor(array)):
            raise ValueError("ADC 必须为整数")
        # Own a copy so callers cannot change a frame after it has been published.
        copied = np.array(array, dtype=np.uint16, copy=True, order="C")
        copied.flags.writeable = False
        object.__setattr__(self, "values", copied)


@dataclass(frozen=True)
class Frame:
    blocks: tuple[Block, ...]
    sequence: int = 0
    received_at: float = field(default_factory=perf_counter)

    def __post_init__(self) -> None:
        blocks = tuple(self.blocks)
        if any(not isinstance(block, Block) for block in blocks):
            raise ValueError("Frame 只能包含 Block")
        ids = [block.block_id for block in blocks]
        if len(set(ids)) != len(ids):
            raise ValueError("同一帧的 BLOCK ID 不得重复")
        object.__setattr__(self, "blocks", blocks)

