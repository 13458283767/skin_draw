import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtTest import QTest

from skin_draw.auto_layout import arrange_frame
from skin_draw.mapping import validate_layout
from skin_draw.models import Block, Frame
from skin_draw.simulator import demo_specs, generate_frame
from skin_draw.ui.main_window import MainWindow


@pytest.mark.parametrize("count", [1, 2, 3, 7, 14, 16])
@pytest.mark.parametrize("size", [(8, 8), (16, 16), (1, 1), (5, 3)])
@pytest.mark.parametrize("viewport", [(770, 525), (510, 385), (380, 700)])
def test_all_blocks_fit_without_overlap(count, size, viewport):
    frame = Frame(tuple(Block(i + 1, np.zeros(size)) for i in range(count)))
    plan = arrange_frame(frame, *viewport)
    assert len(plan.blocks) == count
    assert plan.rows * plan.columns >= count
    assert validate_layout(plan.blocks) == plan.blocks
    width, height = plan.bounds[2:]
    for block in plan.blocks:
        assert 0 <= block.x < block.x + block.cols <= width
        assert 0 <= block.y < block.y + block.rows <= height
        assert block.y - plan.title_offset >= 0


def test_noncontiguous_ids_and_mixed_shapes_are_taken_from_data():
    frame = Frame((Block(14, np.zeros((7, 3))), Block(2, np.zeros((5, 4))), Block(9, np.zeros((2, 8)))))
    plan = arrange_frame(frame, 700, 500)
    assert [(s.block_id, s.rows, s.cols) for s in plan.blocks] == [(2, 5, 4), (9, 2, 8), (14, 7, 3)]


def test_dynamic_structure_and_stable_items_and_zoom(app):
    window = MainWindow(demo=True)
    window.timer.stop()
    window.show()
    QTest.qWait(120)
    heatmap = window.heatmap
    assert len(heatmap.items) == 14
    frame = generate_frame(demo_specs(), 1)
    window.present_frame(frame)
    assert not window.timer.isActive()
    items = dict(heatmap.items)
    heatmap.view.scaleBy((.7, .7))
    zoom = heatmap.view.viewRange()
    window.present_frame(Frame(tuple(reversed(frame.blocks)), sequence=2))
    assert heatmap.items == items
    assert heatmap.view.viewRange() == zoom
    heatmap.select_sensor((14, 7, 7))
    window.present_frame(Frame((Block(14, np.full((3, 5), 1024)), Block(2, [[4095]]))))
    assert len(heatmap.items) == 2
    assert heatmap.items[14].values.shape == (3, 5)
    assert heatmap.selected is None
    assert "2 块 / 16 点" in window.layout_title.text()
    window.present_frame(Frame(()))
    assert heatmap.items == {}
    assert window.min_label.text() == "—"
    assert "0 块" in window.controls.detected.text()
    window.present_frame(frame)
    assert len(heatmap.items) == 14
    window.close()
    window.deleteLater()


@pytest.mark.parametrize("preset", [0, 1, 2])
@pytest.mark.parametrize("size", [(1250, 860), (1000, 720)])
def test_fourteen_blocks_visible_in_actual_widget(app, preset, size):
    window = MainWindow(demo=True)
    window.timer.stop()
    window.controls.preset.setCurrentIndex(preset)
    window.resize(*size)
    window.show()
    QTest.qWait(140)
    window.grab()  # force Qt/PyQtGraph's deferred paint transform before geometry QA
    heatmap = window.heatmap
    assert len(heatmap.items) == 14
    viewport = heatmap.view.sceneBoundingRect().adjusted(-1, -1, 1, 1)
    for block_id, item in heatmap.items.items():
        assert viewport.contains(item.sceneBoundingRect()), (preset, size, block_id)
        assert viewport.contains(heatmap.labels[block_id].sceneBoundingRect()), (preset, size, block_id, "caption")
        transform = item.sceneTransform()
        assert abs(abs(transform.m11()) - abs(transform.m22())) < 1e-6
    captions = list(heatmap.labels.values())
    for i, a in enumerate(captions):
        for b in captions[i + 1:]:
            assert not a.sceneBoundingRect().intersects(b.sceneBoundingRect())
    window.resize(1100, 780)
    QTest.qWait(140)
    window.grab()
    viewport = heatmap.view.sceneBoundingRect().adjusted(-1, -1, 1, 1)
    assert all(viewport.contains(item.sceneBoundingRect()) for item in heatmap.items.values())
    window.close()
    window.deleteLater()


def test_dense_fourteen_block_render_preserves_sensor_colors(app):
    window = MainWindow(demo=True)
    window.timer.stop()
    window.resize(1000, 720)
    window.controls.preset.setCurrentIndex(2)
    window.controls.pattern.setCurrentText("四角定位")
    window.show()
    QTest.qWait(140)
    heatmap = window.heatmap
    pixmap = heatmap.canvas.viewport().grab()
    image = pixmap.toImage()
    ratio = pixmap.devicePixelRatio()
    for item in heatmap.items.values():
        for row, col in [(0, 0), (0, 15), (15, 0), (15, 15)]:
            position = heatmap.canvas.mapFromScene(item.mapToScene(QPointF(col + .5, row + .5)))
            actual = image.pixelColor(int(position.x() * ratio), int(position.y() * ratio))
            expected = tuple(map(int, item.colors[row, col]))
            assert (actual.red(), actual.green(), actual.blue()) == expected
    window.close()
    window.deleteLater()
