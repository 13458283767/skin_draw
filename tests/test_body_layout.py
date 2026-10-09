"""Preset placement must not alter measured data or create fake devices."""

from unittest.mock import patch

import numpy as np
import pytest
from PySide6.QtTest import QTest

from skin_draw.body_layout import BODY_PRESET, arrange_body
from skin_draw.mapping import BlockOrientation, validate_layout
from skin_draw.models import Block, Frame
from skin_draw.ui.main_window import MainWindow


def body_frame(ids=range(1, 15), shape=None):
    defaults = {e["id"]: (e["rows"], e["cols"]) for e in BODY_PRESET}
    return Frame(tuple(Block(key, np.full(shape or defaults.get(key, (5, 3)), 2500-key)) for key in ids))


def test_anatomical_relations_and_absent_slots_do_not_shift_present_blocks():
    frame = body_frame()
    full = arrange_body(frame, 600, 450)
    partial = arrange_body(body_frame((1, 4, 7, 14)), 600, 450)
    specs = {s.block_id: s for s in full.blocks}
    assert partial.blocks == tuple(s for s in full.blocks if s.block_id in (1, 4, 7, 14))
    assert len(partial.slots) == 14
    assert full.bounds == partial.bounds
    assert validate_layout(full.slots) == full.slots
    for right, left in ((1, 2), (3, 4), (5, 6), (9, 10), (11, 12)):
        assert specs[right].x < specs[left].x  # subject right = screen left
    assert specs[1].y < specs[3].y < specs[5].y < specs[11].y
    assert specs[7].y < specs[8].y < specs[9].y
    assert specs[3].x < specs[7].x < specs[4].x < specs[13].x
    assert specs[13].y < specs[14].y


def test_compact_body_uses_current_sizes_and_places_hands_beside_thighs():
    orientations = {key: BlockOrientation(90) for key in (7, 8, 13, 14)}
    plan = arrange_body(body_frame(), 800, 520, orientations)
    specs = {s.block_id: s for s in plan.blocks}
    validate_layout(plan.slots)
    # The former rotation-sized slots needed about 85 vertical cells. At
    # this viewport the compact body should give each sensor at least 12 px.
    assert plan.cell_pixels >= 12
    assert plan.bounds[3] < 43
    assert specs[11].y < specs[9].y + specs[9].display_shape[0]
    assert specs[12].y < specs[10].y + specs[10].display_shape[0]
    assert specs[10].x - (specs[9].x + specs[9].display_shape[1]) < 2
    assert specs[4].x - (specs[7].x + specs[7].display_shape[1]) < 4
    assert specs[13].x - (specs[4].x + specs[4].display_shape[1]) < 5


@pytest.mark.parametrize("shape", [(16, 16), (1, 16), (16, 1)])
def test_extreme_shapes_orientations_and_extra_ids_do_not_overlap(shape):
    frame = body_frame(range(1, 17), shape)
    plan = arrange_body(frame, 500, 400, {i: BlockOrientation((i%4)*90, bool(i%2)) for i in range(1, 17)})
    validate_layout(plan.blocks)
    for s in plan.blocks:
        rows, cols = s.display_shape
        assert 0 <= s.x and 0 <= s.y-plan.title_offset
        assert s.x+cols <= plan.bounds[2] and s.y+rows <= plan.bounds[3]


@pytest.fixture
def window(app):
    window = MainWindow()
    window.resize(1000, 720)
    window.show()
    QTest.qWait(100)
    yield window
    window.close()
    window.deleteLater()
    app.processEvents()


def test_idle_preset_only_shows_placeholders_and_can_return_to_grid(window):
    view = window.heatmap
    window.controls.body_positions.click()
    assert view.placement_mode == "body" and len(view.arrangement.slots) == 14
    assert not view.items and window.preprocess is None
    assert window.current_frame.blocks == ()
    assert "识别到 0 块" in window.controls.detected.text()
    assert window.controls.reset_positions.isEnabled()
    window.controls.reset_positions.click()
    assert view.placement_mode == "grid" and not view.body_decorations


def test_restore_preserves_arrays_orientation_selection_and_live_item_cache(window):
    frame = body_frame()
    view = window.heatmap
    window.present_frame(frame)
    view.set_orientation(7, BlockOrientation(90, True))
    view.select_sensor((7, 2, 1))
    before_reading = view.reading(view.selected)
    items = dict(view.items)
    pictures = {key: item.picture for key, item in items.items()}
    window.controls.body_positions.click()
    assert view.items == items
    assert view.reading(view.selected) == before_reading
    assert view.items[7].spec.rotation == 90 and view.items[7].spec.flip_x
    preset_positions = {key: item.pos() for key, item in items.items()}
    view.move_block(1, view.items[1].spec.x+2, view.items[1].spec.y+1)
    with patch.object(view, "_build_layout", wraps=view._build_layout) as build:
        for _ in range(30):
            window.present_frame(frame)
        assert build.call_count == 0
    window.controls.body_positions.click()
    assert not view.manual_positions
    assert all(view.items[key].pos() == pos for key, pos in preset_positions.items())
    assert all(view.items[key].picture is picture for key, picture in pictures.items())
    assert view.reading(view.selected) == before_reading
    assert window.current_frame is frame


def test_missing_and_reconnecting_preserve_mode_and_raw_sensor_values(window):
    view = window.heatmap
    frame = body_frame((1, 7, 13))
    view.set_frame(frame, layout_blocks=frame.blocks)
    window.controls.body_positions.click()
    pos = view.items[7].pos()
    for _ in range(3):
        view.set_frame(Frame(frame.blocks[:1]), layout_blocks=frame.blocks)
    assert view.items[7].stale and view.items[7].values is None
    assert view.items[7].pos() == pos
    window.controls.body_positions.click()
    assert view.items[7].missing_count == 3
    view.set_frame(Frame(()))
    assert view.placement_mode == "body" and not view.items
    view.set_frame(frame)
    assert view.items[7].pos() == pos
    assert view.reading((7, 0, 0)).value == 2493


@pytest.mark.parametrize("shape", [None, (16, 16), (16, 1), (1, 16)])
def test_all_grids_and_captions_fit_small_window_after_resize(window, app, shape):
    window.present_frame(body_frame(shape=shape))
    window.controls.body_positions.click()
    view = window.heatmap
    for size in ((1000, 720), (1250, 860)):
        window.resize(*size)
        QTest.qWait(150)
        window.grab()
        viewport = view.view.sceneBoundingRect().adjusted(-1, -1, 1, 1)
        assert all(viewport.contains(item.sceneBoundingRect()) for item in view.items.values())
        captions = list(view.labels.values())
        for i, caption in enumerate(captions):
            assert viewport.contains(caption.sceneBoundingRect())
            for other in captions[i+1:]:
                assert not caption.sceneBoundingRect().intersects(other.sceneBoundingRect())
        assert all(abs(abs(item.sceneTransform().m11())-abs(item.sceneTransform().m22())) < 1e-6
                   for item in view.items.values())

