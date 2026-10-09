"""One shared continuous lookup table for cells and the legend."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

PALETTES = {
    "蓝青黄红": ("#203879", "#2565b6", "#1ca8c4", "#60cfbd", "#f3dd70", "#f49443", "#c9343b"),
    "蓝绿黄": ("#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"),
    "暖色": ("#231151", "#641a80", "#a52c74", "#e75b52", "#fca85d", "#fcf5b8"),
    "浅红深红": ("#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15", "#67000d"),
}
MISSING_COLOR = (203, 213, 225)


@lru_cache(maxsize=16)
def lookup_table(name: str, size: int = 256) -> np.ndarray:
    colors = np.array([tuple(int(code[i:i + 2], 16) for i in (1, 3, 5))
                       for code in PALETTES[name]])
    positions = np.linspace(0, 1, len(colors))
    samples = np.linspace(0, 1, size)
    table = np.rint(np.column_stack([np.interp(samples, positions, colors[:, i])
                                    for i in range(3)])).astype(np.uint8)
    table.flags.writeable = False
    return table


@dataclass(frozen=True)
class ColorScale:
    low: float = 0
    high: float = 4095
    palette: str = "蓝青黄红"

    def __post_init__(self) -> None:
        if not np.isfinite((self.low, self.high)).all() or self.low >= self.high:
            raise ValueError("色阶下限必须小于上限")
        if self.palette not in PALETTES:
            raise ValueError("未知配色")

    def map_values(self, values: np.ndarray) -> np.ndarray:
        # Clipping affects colors only, never the underlying ADC values.
        # NaN marks not-ready points and empty cells: paint them as missing.
        normalized = np.clip((values.astype(float) - self.low) / (self.high - self.low), 0, 1)
        missing = ~np.isfinite(normalized)
        colors = lookup_table(self.palette)[np.rint(np.where(missing, 0, normalized) * 255).astype(int)]
        if missing.any():
            colors = np.array(colors, copy=True)
            colors[missing] = MISSING_COLOR
        return colors
