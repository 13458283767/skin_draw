import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest

from skin_draw.colors import MISSING_COLOR
from skin_draw.mapping import BlockLayout
from skin_draw.models import Block, Frame
from skin_draw.ui.main_window import MainWindow


@pytest.fixture
def window(app):
    window = MainWindow(demo=True)
    window.timer.stop()
    window.show()
    app.processEvents()
    QTest.qWait(100)
    yield window
    window.close()
    window.deleteLater()
    app.processEvents()


def test_square_geometry_mouse_selection_and_pause(window, app):
    heatmap = window.heatmap
    heatmap.fit_view()
    app.processEvents()
    transform = heatmap.items[1].sceneTransform()
    assert abs(abs(transform.m11()) - abs(transform.m22())) < 1e-6
    spec = heatmap.items[1].spec
    assert heatmap.sensor_at(spec.x + .5, spec.y + .5) == (1, 0, 0)
    assert heatmap.sensor_at(spec.x + .01, spec.y + .5) is None
    scene = heatmap.view.mapViewToScene(QPointF(spec.x + .5, spec.y + .5))
    pixel = heatmap.canvas.mapFromScene(scene)
    QTest.mouseMove(heatmap.canvas.viewport(), pixel)
    QTest.mouseClick(heatmap.canvas.viewport(), Qt.MouseButton.LeftButton, pos=pixel)
    app.processEvents()
    assert heatmap.selected == (1, 0, 0)
    assert window.selected_label.text() == str(int(window.current_frame.blocks[0].values[0, 0]))
    window.controls.play.click()
    assert window.timer.isActive()
    window.controls.step.click()
    assert not window.timer.isActive()
    sequence = window.sequence
    QTest.qWait(110)
    assert window.sequence == sequence


def test_missing_frame_clears_old_values_and_bad_frame_is_atomic(window):
    from skin_draw.config import four_blocks
    window.heatmap.set_layout(four_blocks())
    frame = Frame((Block(1, np.full((5, 3), 2500)),))
    window.present_frame(frame)
    assert window.heatmap.items[2].values is None
    assert tuple(window.heatmap.items[2].colors[0, 0]) == MISSING_COLOR
    assert window.heatmap.reading((2, 0, 0)).value is None
    with pytest.raises(ValueError):
        window.present_frame(Frame((Block(1, [[5]]),)))
    assert window.current_frame is frame
    assert window.heatmap.frame is frame


def test_rotation_readout_and_selection_updates(window):
    window.heatmap.set_layout((BlockLayout(1, 2, 3, rotation=90, flip_x=True),))
    window.present_frame(Frame((Block(1, [[10, 20, 30], [40, 50, 60]]),)))
    window.heatmap.select_sensor((1, 0, 0))
    reading = window.heatmap.reading((1, 0, 0))
    assert (reading.row, reading.col, reading.value) == (1, 2, 60)
    assert "原始行 2 / 列 3" in window.selected_detail.text()
    window.present_frame(Frame((Block(1, [[10, 20, 30], [40, 50, 777]]),)))
    assert window.selected_label.text() == "777"


def test_controls_uniform_autoscale_export_and_layout_reset(window, app, tmp_path):
    window.controls.pattern.setCurrentText("均匀数值")
    window.controls.auto_levels.setChecked(True)
    assert window.heatmap.scale.low < 2048 < window.heatmap.scale.high
    window.controls.auto_levels.setChecked(False)
    assert (window.heatmap.scale.low, window.heatmap.scale.high) == (0, 4095)
    window.controls.low.setValue(4095)
    assert window.controls.low.value() < window.controls.high.value()
    window.controls.high.setValue(0)
    assert window.controls.low.value() < window.controls.high.value()
    window.controls.palette.setCurrentIndex(1)
    window.controls.show_values.setChecked(True)
    window.controls.gap.setValue(20)
    window.heatmap.select_sensor((1, 1, 1))
    window.controls.preset.setCurrentIndex(3)
    assert window.heatmap.selected == (1, 1, 1)  # surviving sensor remains selected
    assert window.heatmap.items[1].gap == .2
    assert window.heatmap.items[1].show_values
    app.processEvents()
    path = tmp_path / "heatmap.png"
    window.heatmap.export_png(path)
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
