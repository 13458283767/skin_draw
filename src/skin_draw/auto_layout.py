"""Viewport-aware packing of ADC blocks. Pure geometry, independent of Qt."""

from dataclasses import dataclass, replace
from math import ceil

from .mapping import BlockLayout, BlockOrientation
from .models import Frame


@dataclass(frozen=True)
class AutoLayout:
    blocks: tuple[BlockLayout, ...]
    columns: int
    rows: int
    bounds: tuple[float, float, float, float]
    cell_pixels: float
    title_offset: float


def arrange_frame(frame: Frame, width: float, height: float,
                  orientations: dict[int, BlockOrientation] | None = None) -> AutoLayout:
    """Maximize common sensor size while reserving screen space for captions.

    IDs are sorted for stable placement even when reception order changes.
    Empty slots in the last row are not rendered as imaginary skin blocks.
    """
    blocks = sorted(frame.blocks, key=lambda block: block.block_id)
    if not blocks:
        return AutoLayout((), 0, 0, (0, 0, 6, 6), 1, 1)
    width, height = max(200, width), max(160, height)
    oriented = [(orientations or {}).get(block.block_id, BlockOrientation()).apply(
        BlockLayout(block.block_id, *block.values.shape)) for block in blocks]
    max_rows = max(spec.display_shape[0] for spec in oriented)
    max_cols = max(spec.display_shape[1] for spec in oriented)
    margin, gutter, side, heading, foot = 12, 12, 20, 34, 8
    candidates = []
    for cols in range(1, len(blocks) + 1):
        rows = ceil(len(blocks) / cols)
        tile_w = (width - 2 * margin - (cols - 1) * gutter) / cols
        tile_h = (height - 2 * margin - (rows - 1) * gutter) / rows
        # Keep the full BLOCK ID + matrix shape label readable at fit scale.
        if tile_w < 108:
            continue
        cell = min((tile_w - side) / max_cols, (tile_h - heading - foot) / max_rows)
        if cell >= .5:
            candidates.append((cell, -(rows * cols - len(blocks)), cols, rows, tile_w, tile_h))
    if not candidates:
        # Extremely small embedded widget: fit geometry even if labels are dense.
        cols = max(1, min(len(blocks), round((len(blocks) * width / height) ** .5)))
        rows = ceil(len(blocks) / cols)
        tile_w, tile_h = width / cols, height / rows
        cell = max(.1, min(tile_w / (max_cols + 4), tile_h / (max_rows + 7)))
        heading, foot, side, margin, gutter = 3 * cell, cell, 2 * cell, 0, 0
    else:
        cell, _, cols, rows, tile_w, tile_h = max(candidates)
    specs = []
    for index, spec in enumerate(oriented):
        _, block_cols = spec.display_shape
        grid_row, grid_col = divmod(index, cols)
        x = margin + grid_col * (tile_w + gutter) + (tile_w - block_cols * cell) / 2
        y = margin + grid_row * (tile_h + gutter) + heading
        specs.append(replace(spec, x=x / cell, y=y / cell))
    return AutoLayout(tuple(specs), cols, rows, (0, 0, width / cell, height / cell), cell, min(24, heading * .7) / cell)
