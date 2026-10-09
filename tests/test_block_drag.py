import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest

from skin_draw.mapping import BlockOrientation
from skin_draw.models import Block, Frame
from skin_draw.ui.main_window import MainWindow


@pytest.fixture
def window(app):
    window = MainWindow()
    window.show()
    window.present_frame(Frame(tuple(Block(i, np.arange(15).reshape(5, 3) * 100)
                                    for i in range(1, 15))))
    app.processEvents()
    QTest.qWait(120)
    yield window
    window.close()
    window.deleteLater()
    app.processEvents()


def pixel_at(view, x, y):
    return view.canvas.mapFromScene(view.view.mapViewToScene(QPointF(x, y)))


def drag(view, start, end):
    viewport = view.canvas.viewport()
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(viewport, end, delay=20)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=end)


def test_drag_moves_grid_labels_and_selection_without_camera_or_pixel_rebuild(window, app):
    view = window.heatmap
    item = view.items[1]
    before = item.pos()
    title = QPointF(view.labels[1].pos())
    axis = QPointF(view.axis_labels[0].pos())
    picture = item.picture
    ranges = np.array(view.view.viewRange())
    others = {i: obj.pos() for i, obj in view.items.items() if i != 1}
    start = pixel_at(view, item.spec.x + .5, item.spec.y + .5)
    QTest.mouseClick(view.canvas.viewport(), Qt.MouseButton.LeftButton, pos=start)
    assert view.selected == (1, 0, 0)
    assert not view.manual_positions
    drag(view, start, start + QPoint(18, 24))
    app.processEvents()
    delta = item.pos() - before
    assert delta.x() > 0 and delta.y() > 0
    assert item.picture is picture
    assert view.labels[1].pos() - title == delta
    assert view.axis_labels[0].pos() - axis == delta
    assert view.selected == (1, 0, 0)
    assert view.selection_box.rect() == item.cell_rect(0, 0).translated(item.pos())
    np.testing.assert_allclose(view.view.viewRange(), ranges)
    assert all(view.items[i].pos() == pos for i, pos in others.items())
    assert "手动位置" in window.controls.detected.text()
    new_pixel = pixel_at(view, item.spec.x + 1.5, item.spec.y + 1.5)
    QTest.mouseClick(view.canvas.viewport(), Qt.MouseButton.LeftButton, pos=new_pixel)
    reading = view.reading(view.selected)
    assert (reading.block_id, reading.row, reading.col, reading.value) == (1, 1, 1, 400)


def test_positions_survive_frames_missing_rotation_resize_and_reset(window, app):
    view = window.heatmap
    frame = window.current_frame
    spec = view.items[1].spec
    view.move_block(1, spec.x + .6, spec.y + .4)
    positions = {i: item.pos() for i, item in view.items.items()}
    items = dict(view.items)
    for _ in range(30):
        window.present_frame(frame)
    assert view.items == items
    view.set_frame(Frame(frame.blocks[1:]), layout_blocks=frame.blocks)
    assert view.items[1].missing_count == 1
    view.set_orientation(1, BlockOrientation(90, True))
    assert view.items[1].missing_count == 1
    window.resize(1000, 720)
    app.processEvents()
    QTest.qWait(120)
    assert all(view.items[i].pos() == pos for i, pos in positions.items())
    view.set_frame(frame, {})
    assert view.items[1].values is None
    view.set_frame(Frame(()))
    view.set_frame(frame)
    assert all(view.items[i].pos() == pos for i, pos in positions.items())
    window.controls.reset_positions.click()
    assert not view.manual_positions
    assert view.items[1].spec.rotation == 90 and view.items[1].spec.flip_x
    assert view.items[1].pos() != positions[1]
    app.processEvents()
    viewport = view.canvas.viewport().rect()
    for item in view.items.values():
        assert viewport.contains(view.canvas.mapFromScene(item.mapToScene(item.boundingRect())).boundingRect())


def test_blank_drag_pans_camera_and_overlap_targets_visible_block(window, app):
    view = window.heatmap
    positions = {i: item.pos() for i, item in view.items.items()}
    ranges = np.array(view.view.viewRange())
    start = QPoint(10, view.canvas.viewport().height() - 12)
    drag(view, start, start + QPoint(30, -20))
    app.processEvents()
    assert not np.allclose(view.view.viewRange(), ranges)
    assert all(view.items[i].pos() == pos for i, pos in positions.items())
    assert not view.manual_positions
    spec = view.items[1].spec
    view.move_block(14, spec.x, spec.y)
    spec = view.items[14].spec
    assert view.sensor_at(spec.x + .5, spec.y + .5) == (14, 0, 0)
    QTest.mouseClick(view.canvas.viewport(), Qt.MouseButton.LeftButton,
                     pos=pixel_at(view, spec.x + .5, spec.y + .5))
    assert view.selected == (14, 0, 0)


def test_drag_clamps_at_visible_edges_and_handles_topology_change(window):
    view = window.heatmap
    spec = view.items[1].spec
    start = pixel_at(view, spec.x + .5, spec.y + .5)
    drag(view, start, QPoint(-200, -200))
    item = view.items[1]
    (left, right), (top, bottom) = view.view.viewRange()
    assert left <= item.spec.x and top <= item.spec.y - view.arrangement.title_offset
    assert item.spec.x + 3 <= right and item.spec.y + 5 <= bottom
    start = pixel_at(view, item.spec.x + .5, item.spec.y + .5)
    QTest.mousePress(view.canvas.viewport(), Qt.MouseButton.LeftButton, pos=start)
    window.present_frame(Frame(()))
    QTest.mouseRelease(view.canvas.viewport(), Qt.MouseButton.LeftButton, pos=start)
    assert view.block_drag.pending is None


def test_title_drag_moves_only_its_block_while_new_frames_arrive(window, app):
    view = window.heatmap
    block_id = 7
    positions = {i: QPointF(item.pos()) for i, item in view.items.items()}
    ranges = np.array(view.view.viewRange())
    caption = view.labels[block_id]
    start = view.canvas.mapFromScene(caption.mapToScene(caption.boundingRect().center()))
    viewport = view.canvas.viewport()
    QTest.mouseMove(viewport, start)
    assert viewport.cursor().shape() == Qt.CursorShape.OpenHandCursor
    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=start)
    assert view.selected[0] == block_id
    assert view.block_selection_box.isVisible()
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
    for step in range(1, 6):
        frame = Frame(tuple(Block(i, np.full((5, 3), 2000 + step)) for i in range(1, 15)))
        window.present_frame(frame)
        QTest.mouseMove(viewport, start + QPoint(step * 6, step * 4), delay=10)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=start + QPoint(30, 20))
    app.processEvents()
    assert view.items[block_id].pos() != positions[block_id]
    assert all(item.pos() == positions[i] for i, item in view.items.items() if i != block_id)
    np.testing.assert_allclose(view.view.viewRange(), ranges)
    assert view.reading(view.selected).value == 2005
    item = view.items[block_id]
    assert view.block_selection_box.rect().contains(item.boundingRect().translated(item.pos()))
