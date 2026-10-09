"""Deterministic demo source, replaceable by a future acquisition adapter."""

import numpy as np

from .mapping import BlockLayout
from .models import Block, Frame

PATTERNS = ("移动热点", "双点接触", "水平渐变", "四角定位", "均匀数值")
DEMO_PRESETS = ("14 块 · 8 × 8", "14 块 · 混合尺寸", "14 块 · 16 × 16",
                "单块 · 5 × 4", "4 块 · 协议示例")


def demo_specs(index: int = 0) -> tuple[BlockLayout, ...]:
    """Only mock input dimensions. Their positions never control the view."""
    if index == 0:
        shapes = [(8, 8)] * 14
    elif index == 1:
        shapes = [(5, 3), (7, 3), (8, 8), (5, 4), (4, 6), (6, 4), (10, 5)] * 2
    elif index == 2:
        shapes = [(16, 16)] * 14
    elif index == 3:
        shapes = [(5, 4)]
    elif index == 4:
        shapes = [(5, 3), (5, 3), (7, 3), (7, 3)]
    else:
        raise ValueError("未知模拟数据规模")
    return tuple(BlockLayout(i + 1, rows, cols, x=i * 18) for i, (rows, cols) in enumerate(shapes))


def generate_frame(layout: tuple[BlockLayout, ...], sequence: int, pattern: str = PATTERNS[0]) -> Frame:
    blocks = []
    t = sequence / 20
    for spec in layout:
        y, x = np.mgrid[0:spec.rows, 0:spec.cols]
        nx = x / max(spec.cols - 1, 1)
        ny = y / max(spec.rows - 1, 1)
        phase = t * 0.7 + (spec.block_id - 1) * 0.8
        if pattern == "移动热点":
            cx, cy = 0.5 + 0.27 * np.sin(phase), 0.5 + 0.27 * np.cos(phase * 0.83)
            values = 260 + 3700 * np.exp(-((nx - cx) ** 2 + (ny - cy) ** 2) / 0.075)
        elif pattern == "双点接触":
            spread = 0.027 + 0.006 * np.sin(phase)
            values = (180 + 3550 * np.exp(-((nx - .28) ** 2 + (ny - .35) ** 2) / spread)
                      + 2950 * np.exp(-((nx - .76) ** 2 + (ny - .7) ** 2) / spread))
        elif pattern == "水平渐变":
            values = np.broadcast_to(nx * 4095, (spec.rows, spec.cols))
        elif pattern == "四角定位":
            values = np.zeros((spec.rows, spec.cols)) + 250
            values[0, 0], values[0, -1] = 1000, 2000
            values[-1, 0], values[-1, -1] = 3000, 4095
        elif pattern == "均匀数值":
            values = np.full((spec.rows, spec.cols), 2048)
        else:
            raise ValueError("未知模拟模式")
        blocks.append(Block(spec.block_id, np.rint(values).clip(0, 4095).astype(np.uint16)))
    return Frame(tuple(blocks), sequence=sequence)
