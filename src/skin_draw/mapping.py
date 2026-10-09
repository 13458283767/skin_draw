"""Physical layout and reversible array mapping, with no UI dependency."""

from dataclasses import dataclass, replace
import math

import numpy as np
from numpy.typing import NDArray

from .models import MAX_SIDE


@dataclass(frozen=True)
class BlockLayout:
    block_id: int
    rows: int
    cols: int
    x: float = 0
    y: float = 0
    rotation: int = 0
    flip_x: bool = False
    flip_y: bool = False
    row_map: tuple[int, ...] = ()
    col_map: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.block_id) is not int or not 1 <= self.block_id <= 16:
            raise ValueError("布局 BLOCK ID 必须为 1–16 的整数")
        for name, size in (("rows", self.rows), ("cols", self.cols)):
            if type(size) is not int or not 1 <= size <= MAX_SIDE:
                raise ValueError(f"{name} 必须为 1–16 的整数")
        if type(self.rotation) is not int or self.rotation not in (0, 90, 180, 270):
            raise ValueError("rotation 只能为 0、90、180、270（逆时针）")
        if type(self.flip_x) is not bool or type(self.flip_y) is not bool:
            raise ValueError("flip_x 和 flip_y 必须为布尔值")
        if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1000
               for v in (self.x, self.y)):
            raise ValueError("布局坐标必须为 -1000 到 1000 的有限数值")
        for name, size in (("row_map", self.rows), ("col_map", self.cols)):
            indices = tuple(getattr(self, name))
            if indices and (any(type(i) is not int for i in indices)
                            or sorted(indices) != list(range(size))):
                raise ValueError(f"{name} 必须是 0 到 {size - 1} 的完整索引排列")
            object.__setattr__(self, name, indices)

    @property
    def display_shape(self) -> tuple[int, int]:
        return (self.cols, self.rows) if self.rotation in (90, 270) else (self.rows, self.cols)

    def transform(self, values: NDArray) -> NDArray:
        if values.shape != (self.rows, self.cols):
            raise ValueError(f"BLOCK {self.block_id} 需要 {self.rows}×{self.cols} 数据")
        result = values
        if self.row_map or self.col_map:
            result = values[np.ix_(self.row_map or tuple(range(self.rows)),
                                   self.col_map or tuple(range(self.cols)))]
        if self.rotation:
            result = np.rot90(result, self.rotation // 90)
        if self.flip_x:
            result = np.fliplr(result)
        if self.flip_y:
            result = np.flipud(result)
        return result

    def index_map(self) -> NDArray:
        """Displayed cell -> original flattened row-major index."""
        return self.transform(np.arange(self.rows * self.cols).reshape(self.rows, self.cols))

    def raw_index(self, display_row: int, display_col: int) -> tuple[int, int]:
        index = int(self.index_map()[display_row, display_col])
        return divmod(index, self.cols)


@dataclass(frozen=True)
class BlockOrientation:
    """Display-only orientation, composed relative to the current screen axes."""

    rotation: int = 0
    flip_x: bool = False
    flip_y: bool = False

    def apply(self, spec: BlockLayout) -> BlockLayout:
        return replace(spec, rotation=self.rotation, flip_x=self.flip_x, flip_y=self.flip_y)

    def turned(self, quarter_turns: int) -> "BlockOrientation":
        # A reflection reverses rotation handedness: R F = F R^-1.
        direction = -1 if self.flip_x != self.flip_y else 1
        return replace(self, rotation=(self.rotation + direction * quarter_turns * 90) % 360)

    def mirrored(self, axis: str) -> "BlockOrientation":
        if axis == "horizontal":
            return replace(self, flip_x=not self.flip_x)
        if axis == "vertical":
            return replace(self, flip_y=not self.flip_y)
        raise ValueError("镜像方向必须为 horizontal 或 vertical")


def validate_layout(layout: tuple[BlockLayout, ...]) -> tuple[BlockLayout, ...]:
    layout = tuple(layout)
    if not 1 <= len(layout) <= 16:
        raise ValueError("布局需要 1–16 个皮肤块")
    if len({block.block_id for block in layout}) != len(layout):
        raise ValueError("布局中 BLOCK ID 不得重复")
    for i, a in enumerate(layout):
        ar, ac = a.display_shape
        for b in layout[i + 1:]:
            br, bc = b.display_shape
            if max(a.x, b.x) < min(a.x + ac, b.x + bc) and max(a.y, b.y) < min(a.y + ar, b.y + br):
                raise ValueError(f"BLOCK {a.block_id} 与 {b.block_id} 的传感器区域重叠")
    return layout
