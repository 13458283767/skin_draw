"""Anatomical placement only; never creates measurements for absent blocks."""

from dataclasses import dataclass, replace
from importlib.resources import files
import json

from .auto_layout import AutoLayout
from .mapping import BlockLayout, BlockOrientation
from .models import Frame


BODY_PRESET = tuple(json.loads(files("skin_draw").joinpath("presets/body_layout.json").read_text(encoding="utf-8"))["blocks"])
BODY_NAMES = {entry["id"]: entry["name"] for entry in BODY_PRESET}


@dataclass(frozen=True)
class BodyArrangement(AutoLayout):
    slots: tuple[BlockLayout, ...]
    headings: tuple[tuple[str, float, float], ...]
    divider_x: float


def arrange_body(frame: Frame, width: float, height: float,
                 orientations: dict[int, BlockOrientation] | None = None) -> BodyArrangement:
    """Reserve the full body when only some IDs are connected.

    Positions depend on physical slots and dimensions, never on arrival order.
    Unknown IDs 15/16 occupy a separate area beneath the back/waist.
    Matrix dimensions always come from data for observed blocks; placeholders
    use the diagram's illustrative sizes and never enter Frame/calibration.
    """
    shapes = {b.block_id: b.values.shape for b in frame.blocks}
    entries = list(BODY_PRESET)
    entries += [{"id": key, "name": "未分配",
                 "rows": shapes[key][0], "cols": shapes[key][1]}
                for key in sorted(shapes) if key not in BODY_NAMES]
    specs = {e["id"]: (orientations or {}).get(e["id"], BlockOrientation()).apply(
        BlockLayout(e["id"], *shapes.get(e["id"], (e["rows"], e["cols"])))) for e in entries}
    # Use actual oriented dimensions, rather than a rotation-sized square for
    # every block. Hands sit beside the thighs; each arm and the torso form
    # their own vertical chain, so a tall back does not stretch the whole body.
    def pack(cell):
        gap = max(.8, 10 / cell)
        vertical_gap = max(1.6, 23 / cell)
        margin = max(.8, 10 / cell)
        header = max(3.5, 38 / cell)
        title = max(1.2, 12 / cell)
        heights = {key: spec.display_shape[0] for key, spec in specs.items()}
        # Captions have fixed screen fonts. Reserve their width even for a
        # one-column matrix, without leaving big gaps for ordinary matrices.
        widths = {key: max(spec.display_shape[1], 56 / cell) for key, spec in specs.items()}
        left_width = max(widths[key] for key in (3, 5, 11))
        right_width = max(widths[key] for key in (4, 6, 12))
        leg_width = max(widths[9], widths[10])
        shoulder_width = max(widths[1], widths[2])
        center_width = max(widths[7], widths[8], 2 * leg_width + gap,
                           2 * shoulder_width + gap)
        cx = margin + left_width + gap + center_width / 2
        left_x = margin + left_width / 2
        right_x = cx + center_width / 2 + gap + right_width / 2
        front_right = right_x + right_width / 2
        back_width = max(widths[key] for key in specs if key >= 13)
        separation = max(2.5, 28 / cell)
        back_x = front_right + separation + back_width / 2
        chest_y = header + max(heights[1], heights[2]) + vertical_gap
        positions = {
            1: (cx - center_width / 2 + shoulder_width / 2, chest_y - vertical_gap - heights[1]),
            2: (cx + center_width / 2 - shoulder_width / 2, chest_y - vertical_gap - heights[2]),
            7: (cx, chest_y),
            8: (cx, chest_y + heights[7] + vertical_gap),
            13: (back_x, chest_y),
        }
        for upper, lower, hand, x in ((3, 5, 11, left_x), (4, 6, 12, right_x)):
            positions[upper] = (x, chest_y)
            positions[lower] = (x, chest_y + heights[upper] + vertical_gap)
            positions[hand] = (x, positions[lower][1] + heights[lower] + vertical_gap)
        legs_y = positions[8][1] + heights[8] + vertical_gap
        positions[9] = (cx - (leg_width + gap) / 2, legs_y)
        positions[10] = (cx + (leg_width + gap) / 2, legs_y)
        positions[14] = (back_x, max(positions[8][1], chest_y + heights[13] + vertical_gap))
        cursor = positions[14][1] + heights[14] + vertical_gap
        for key in sorted(specs):
            if key > 14:
                positions[key] = (back_x, cursor)
                cursor += heights[key] + vertical_gap
        slots = tuple(replace(spec, x=positions[key][0] - spec.display_shape[1] / 2,
                              y=positions[key][1]) for key, spec in sorted(specs.items()))
        bounds = (0, 0, back_x + back_width / 2 + margin,
                  max(s.y + s.display_shape[0] for s in slots) + margin)
        headings = (("正面 · 人体右侧在画面左侧", cx, max(1., 12 / cell)),
                    ("背面", back_x, max(1., 12 / cell)))
        return slots, bounds, headings, front_right + separation / 2, title

    # Converge screen-space caption spacing with the fit-to-window cell size.
    # This runs only on layout changes, never for normal incoming ADC frames.
    cell = max(.1, min(max(1, width) / 40, max(1, height) / 45))
    for _ in range(24):
        slots, bounds, headings, divider, title = pack(cell)
        fitted = max(.1, min(max(1, width) / bounds[2], max(1, height) / bounds[3]))
        if abs(fitted - cell) < .001:
            break
        cell = fitted
    cell = fitted
    return BodyArrangement(tuple(s for s in slots if s.block_id in shapes), 4, 5,
                           bounds, cell, title, slots, headings, divider)
