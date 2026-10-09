"""User operations act on the displayed matrix, while ADC coordinates stay fixed."""

import itertools

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtTest import QTest

from skin_draw.auto_layout import arrange_frame
from skin_draw.mapping import BlockLayout, BlockOrientation, validate_layout
from skin_draw.models import Block, Frame
from skin_draw.ui.main_window import MainWindow


@pytest.mark.parametrize("rotation,flip_x,flip_y", list(itertools.product(range(0, 360, 90), (False, True), (False, True))))
def test_operations_follow_screen_axes(rotation, flip_x, flip_y):
    raw = np.arange(15).reshape(5, 3)
    spec = BlockLayout(1, 5, 3)
    orientation = BlockOrientation(rotation, flip_x, flip_y)
    displayed = orientation.apply(spec).transform(raw)
    for turns in (-1, 1):
        np.testing.assert_array_equal(orientation.turned(turns).apply(spec).transform(raw), np.rot90(displayed, turns))
    np.testing.assert_array_equal(orientation.mirrored("horizontal").apply(spec).transform(raw), np.fliplr(displayed))
    np.testing.assert_array_equal(orientation.mirrored("vertical").apply(spec).transform(raw), np.flipud(displayed))


@pytest.fixture
def window(app):
    window = MainWindow()
    window.show()
    app.processEvents()
    QTest.qWait(100)
    yield window
    window.close()
    window.deleteLater()
    app.processEvents()


def test_buttons_target_only_one_block_and_preserve_raw_selection(window, app):
    frame = Frame((Block(1, [[100, 200, 300], [400, 500, 600]]),
                   Block(2, [[700, 800, 900], [1000, 1100, 1200]])))
    window.present_frame(frame)
    controls, view = window.controls, window.heatmap
    view.select_sensor((2, 1, 2))
    assert controls.orientation_block.currentData() == 2
    controls.rotate_left.click()
    expected = np.rot90(frame.blocks[1].values)
    np.testing.assert_array_equal(view.items[2].values, expected)
    np.testing.assert_array_equal(view.items[1].values, frame.blocks[0].values)
    controls.mirror_horizontal.click()
    expected = np.fliplr(expected)
    controls.rotate_right.click()
    expected = np.rot90(expected, -1)
    controls.mirror_vertical.click()
    expected = np.flipud(expected)
    np.testing.assert_array_equal(view.items[2].values, expected)
    reading = view.reading(view.selected)
    assert (reading.block_id, reading.row, reading.col, reading.value) == (2, 1, 2, 1200)
    assert view.frame is frame
    assert "原始行 2 / 列 3" in window.selected_detail.text()
    app.processEvents()
    bitmap = view.canvas.grab().toImage()
    spec = view.items[2].spec
    for row, col in np.ndindex(expected.shape):
        point = view.canvas.mapFromScene(view.view.mapViewToScene(QPointF(spec.x + col + .5, spec.y + row + .5)))
        color = bitmap.pixelColor(point).getRgb()[:3]
        assert color == tuple(view.scale.map_values(expected)[row, col])
    items = dict(view.items)
    window.present_frame(frame)
    assert view.items == items  # Incoming frames reuse graphics after manual changes.
    np.testing.assert_array_equal(view.items[2].values, expected)
    controls.reset_orientation.click()
    np.testing.assert_array_equal(view.items[2].values, frame.blocks[1].values)
    controls.orientation_block.setCurrentIndex(0)
    assert view.selected[0] == 1


def test_direction_survives_calibration_gray_missing_blocks_and_reconnection(window):
    view = window.heatmap
    blocks = (Block(1, [[10, 20, 30], [40, 50, 60]]), Block(2, [[70, 80, 90]]))
    view.set_frame(Frame(blocks), {1: blocks[0].values, 2: blocks[1].values}, blocks)
    view.set_frame(Frame((blocks[1],)), {2: blocks[1].values}, blocks)
    assert view.items[1].missing_count == 1
    view.set_orientation(1, BlockOrientation(90))
    assert view.items[1].stale and view.items[1].missing_count == 1
    np.testing.assert_array_equal(view.items[1].values, np.rot90(blocks[0].values))
    view.set_frame(Frame((blocks[1],)), {2: blocks[1].values}, blocks)
    assert view.items[1].missing_count == 2
    view.set_frame(Frame((blocks[1],)), {2: blocks[1].values}, blocks)
    assert view.items[1].values is None
    view.set_orientation(1, BlockOrientation(270))
    assert view.items[1].values is None and view.items[1].missing_count == 3
    view.set_frame(Frame(blocks), {})
    view.set_orientation(1, BlockOrientation(90, True))
    assert view.items[1].values is None  # Calibration has not produced a response yet.
    view.set_frame(Frame(()))
    assert not window.controls.rotate_left.isEnabled()
    view.set_frame(Frame(blocks), {1: blocks[0].values / 2, 2: blocks[1].values})
    np.testing.assert_array_equal(view.items[1].values, np.fliplr(np.rot90(blocks[0].values / 2)))


@pytest.mark.parametrize("shape", [(5, 3), (16, 16), (1, 16)])
def test_fourteen_mixed_orientations_fit(window, app, shape):
    frame = Frame(tuple(Block(i, np.full(shape, i * 100)) for i in range(1, 15)))
    orientations = {i: BlockOrientation((i % 4) * 90, bool(i % 2)) for i in range(1, 15)}
    plan = arrange_frame(frame, 600, 440, orientations)
    validate_layout(plan.blocks)
    for spec in plan.blocks:
        rows, cols = spec.display_shape
        assert spec.x >= 0 and spec.y >= 0
        assert spec.x + cols <= plan.bounds[2]
        assert spec.y + rows <= plan.bounds[3]
    window.resize(1000, 720)
    window.present_frame(frame)
    for block_id, orientation in orientations.items():
        window.heatmap.set_orientation(block_id, orientation)
    app.processEvents()
    QTest.qWait(100)
    view = window.heatmap
    viewport = view.canvas.viewport().rect()
    for item in view.items.values():
        assert viewport.contains(view.canvas.mapFromScene(item.mapToScene(item.boundingRect())).boundingRect())
