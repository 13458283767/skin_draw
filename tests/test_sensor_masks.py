import json

import numpy as np
import pytest
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtTest import QTest

from skin_draw.colors import ColorScale
from skin_draw.filtering import ADCOutlierFilter
from skin_draw.live import LiveBuffer
from skin_draw.mapping import BlockLayout, BlockOrientation
from skin_draw.models import Block, Frame
from skin_draw.sensor_masks import SensorMasks
from skin_draw.ui.heatmap_view import HeatmapView, SensorGridItem
from skin_draw.ui.main_window import MainWindow
from skin_draw.ui.sensor_mask_dialog import SensorMaskDialog


def test_save_load_immutable_and_shape_mismatch(tmp_path):
    path = tmp_path / "configs" / "sensor_masks.json"
    array = np.array([[False, True, True], [True, True, False]])
    masks = SensorMasks().with_mask(7, array)
    array[:] = False
    masks.save(path)
    loaded = SensorMasks.load(path)
    assert loaded.get(7, (2, 3)).sum() == 4
    assert not loaded.get(7, (2, 3)).flags.writeable
    with pytest.raises(ValueError):
        loaded.get(7, (2, 3)).flags.writeable = True
    assert loaded.mismatched(7, (3, 2))
    assert loaded.get(7, (3, 2)).all()
    assert loaded.get(8, (2, 3)).all()
    assert not list(path.parent.glob("*.tmp"))
    assert json.loads(path.read_text())["coordinates"] == "raw_row_col"


@pytest.mark.parametrize("payload", ["garbled", '{"version":2}',
    json.dumps({"version":1, "coordinates":"display", "blocks":{}}),
    json.dumps({"version":1, "coordinates":"raw_row_col", "blocks":{"7":{"valid":[[0, 1]]}}}),
    json.dumps({"version":1, "coordinates":"raw_row_col", "blocks":{"7":{"valid":[[True], []]}}})])
def test_invalid_config_not_silently_accepted(tmp_path, payload):
    path = tmp_path / "mask.json"
    path.write_text(payload)
    with pytest.raises(ValueError):
        SensorMasks.load(path)


def test_atomic_save_failure_keeps_original(tmp_path, monkeypatch):
    path = tmp_path / "mask.json"
    SensorMasks().save(path)
    before = path.read_bytes()
    def fail(*args):
        raise PermissionError("read only")
    monkeypatch.setattr("skin_draw.sensor_masks.os.replace", fail)
    with pytest.raises(PermissionError):
        SensorMasks().with_mask(1, np.zeros((2, 3), bool)).save(path)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_hidden_jumps_do_not_reject_row_and_raw_values_remain_intact():
    mask = np.array([[False, True, True], [True, True, True]])
    filter_ = ADCOutlierFilter()
    filter_.process(Block(7, np.full((2, 3), 2500)), mask)
    raw = Block(7, [[50, 2000, 2400], [2500, 1000, 2500]])
    filtered, invalid = filter_.process(raw, mask)
    np.testing.assert_array_equal(invalid, [[False, False, False], [True, True, True]])
    np.testing.assert_array_equal(filtered.values, [[50, 2000, 2400], [2500, 2500, 2500]])
    assert raw.values[1, 1] == 1000
    assert filter_.rejected_counts == {7: 3}


def test_masked_points_in_rejected_row_are_not_held_or_counted():
    filter_ = ADCOutlierFilter()
    valid = np.array([[False, True]])
    filter_.process(Block(1, [[2500, 2500]]), valid)
    filtered, invalid = filter_.process(Block(1, [[20, 900]]), valid)
    np.testing.assert_array_equal(filtered.values, [[20, 2500]])
    np.testing.assert_array_equal(invalid, [[False, True]])
    assert filter_.rejected_counts[1] == 1


def test_calibration_ignores_arbitrary_empty_channels():
    masks = SensorMasks().with_mask(7, np.array([[False, True, True]]))
    buffer = LiveBuffer(calib_n=30, norm="A")
    buffer.set_sensor_masks(masks)
    for i in range(59):
        buffer.receive(Block(7, [[0 if i % 2 else 4095, 2500, 2450]]))
    snapshot = buffer.take()
    assert snapshot.preprocess.ready
    assert snapshot.preprocess.samples == {7: 30}
    np.testing.assert_array_equal(snapshot.preprocess.baselines[7], [[np.nan, 2500, 2450]])
    for _ in range(35):
        buffer.receive(Block(7, [[4095, 2000, 2300]]))
    pressed = buffer.take()
    np.testing.assert_array_equal(pressed.display_values[7], [[np.nan, 500, 150]])
    assert pressed.scan.frame.blocks[0].values[0, 0] == 4095


def test_edit_resets_only_target_and_old_snapshot_remains_unchanged():
    buffer = LiveBuffer(calib_n=30, norm="A")
    for _ in range(59):
        for key in (1, 2):
            buffer.receive(Block(key, [[2500, 2500]]))
    old = buffer.take()
    masks = SensorMasks().with_mask(1, np.array([[False, True]]))
    buffer.set_sensor_masks(masks, 1)
    new = buffer.take()
    assert not new.preprocess.ready
    assert new.preprocess.samples == {1: 0, 2: 30}
    assert new.display_values[2] is old.display_values[2]
    assert 1 not in new.display_values
    assert old.preprocess.baselines[1][0, 0] == 2500
    buffer.reset_preprocessing()
    assert buffer.sensor_masks is masks
    for _ in range(59):
        buffer.receive(Block(1, [[0, 2400]]))
    assert buffer.preprocessors[1].baselines[0, 1] == 2400


def test_hidden_positions_are_nan_and_block_stays_fresh():
    now = [0.]
    buffer = LiveBuffer(calib_n=30, norm="A", clock=lambda: now[0])
    buffer.set_sensor_masks(SensorMasks().with_mask(1, np.array([[False, True]])))
    for _ in range(59):
        buffer.receive(Block(1, [[0, 2500]]))
    for i in range(1, 36):
        now[0] = i / 10
        buffer.receive(Block(1, [[i * 100, 2000]]))
    snapshot = buffer.take()
    assert snapshot.missing_counts[1] == 0
    np.testing.assert_array_equal(snapshot.display_values[1], [[np.nan, 500]])


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("flip_x,flip_y", [(False,False), (True,False), (False,True), (True,True)])
def test_editor_uses_original_coordinates_for_every_orientation(app, rotation, flip_x, flip_y):
    spec = BlockLayout(7, 2, 3, rotation=rotation, flip_x=flip_x, flip_y=flip_y)
    original = np.ones((2, 3), bool)
    dialog = SensorMaskDialog(spec, original)
    dialog.toggle_cell(0, 0)
    raw = spec.raw_index(0, 0)
    assert not dialog.raw_mask()[raw]
    assert dialog.raw_mask().sum() == 5
    assert original.all()  # Cancel never mutates caller state.
    dialog.restore_button.click()
    assert dialog.raw_mask().all()
    dialog.close()


def test_hidden_pixels_are_transparent_even_in_gap(app):
    item = SensorGridItem(BlockLayout(1, 1, 2))
    item.set_data(np.array([[2500, 2000]]), ColorScale())
    item.set_valid_mask(np.array([[False, True]]), ColorScale())
    canvas = QImage(200, 100, QImage.Format.Format_ARGB32)
    canvas.fill(Qt.GlobalColor.transparent)
    painter = QPainter(canvas)
    painter.scale(100, 100)
    painter.drawPicture(0, 0, item.picture)
    painter.end()
    assert canvas.pixelColor(50, 50).alpha() == 0
    assert canvas.pixelColor(1, 50).alpha() == 0
    assert canvas.pixelColor(150, 50).alpha() == 255


def test_view_rotation_selection_and_all_hidden(app):
    view = HeatmapView()
    masks = SensorMasks().with_mask(7, np.array([[False, True, True], [True, True, False]]))
    view.set_sensor_masks(masks)
    view.set_frame(Frame((Block(7, np.full((2, 3), 2500)),)))
    view.select_sensor((7, 0, 0))
    assert view.selected is None
    spec = view.items[7].spec
    assert view.sensor_at(spec.x + .5, spec.y + .5) is None
    view.set_orientation(7, BlockOrientation(90, True))
    item = view.items[7]
    np.testing.assert_array_equal(item.valid_mask, item.spec.transform(masks.get(7, (2, 3))))
    view.select_sensor(view.first_sensor(7))
    view.set_sensor_masks(masks.with_mask(7, np.zeros((2, 3), bool)))
    assert view.selected is None
    assert view.first_sensor(7) is None
    assert not view.selection_box.isVisible()
    view.close()


def test_save_restart_stats_auto_levels_and_shape_warning(app, tmp_path):
    path = tmp_path / "sensor_masks.json"
    window = MainWindow(mask_path=path)
    frame = Frame((Block(7, [[2, 2400, 2500], [2450, 2500, 4095]]),))
    try:
        window.present_frame(frame)
        window.controls.auto_levels.setChecked(True)
        window.apply_sensor_mask(7, np.array([[False, True, True], [True, True, False]]))
        assert window.min_label.text() == "2400"
        assert window.max_label.text() == "2500"
        assert window.mean_label.text() == "2462.5"
        assert window.heatmap.scale.low == 2400
        assert window.heatmap.scale.high == 2500
        assert "4 点" in window.layout_title.text()
        assert window.current_frame is frame
    finally:
        window.close()
    restored = MainWindow(mask_path=path)
    try:
        restored.present_frame(frame)
        assert restored.heatmap.items[7].valid_mask.sum() == 4
        restored.present_frame(Frame((Block(7, np.full((3, 2), 2500)),)))
        assert restored.heatmap.items[7].valid_mask.all()
        assert "尺寸与点位配置不符" in restored.controls.mask_status.text()
        restored.apply_sensor_mask(7, np.zeros((3, 2), bool))
        assert restored.min_label.text() == "—"
        assert "0 点" in restored.layout_title.text()
        assert "尺寸与点位配置不符" not in restored.controls.mask_status.text()
    finally:
        restored.close()


def test_save_failure_does_not_change_view_or_baseline(app, tmp_path, monkeypatch):
    window = MainWindow(mask_path=tmp_path / "sensor_masks.json")
    try:
        window.present_frame(Frame((Block(1, [[2500, 2500]]),)))
        before = window.sensor_masks
        def fail(*args):
            raise PermissionError("write failure")
        monkeypatch.setattr(SensorMasks, "save", fail)
        with pytest.raises(PermissionError):
            window.apply_sensor_mask(1, np.array([[False, True]]))
        assert window.sensor_masks is before
        assert window.heatmap.items[1].valid_mask.all()
    finally:
        window.close()


def test_click_hidden_cell_does_not_select_first_sensor(app, tmp_path):
    window = MainWindow(mask_path=tmp_path / "mask.json")
    try:
        window.present_frame(Frame((Block(1, [[2500, 2500]]),)))
        window.apply_sensor_mask(1, np.array([[False, True]]))
        window.show()
        app.processEvents()
        QTest.qWait(100)
        spec = window.heatmap.items[1].spec
        scene = window.heatmap.view.mapViewToScene(QPointF(spec.x + .5, spec.y + .5))
        pixel = window.heatmap.canvas.mapFromScene(scene)
        QTest.mouseClick(window.heatmap.canvas.viewport(), Qt.MouseButton.LeftButton, pos=pixel)
        assert window.heatmap.selected is None
    finally:
        window.close()


def test_live_gui_edit_clears_only_target_and_recalibrates(app, tmp_path):
    from skin_draw.ui.serial_worker import SerialWorker
    window = MainWindow(mask_path=tmp_path / "mask.json")
    worker = SerialWorker("FAKE", window, norm="A")
    worker.buffer.calib_n = 30  # Mask reset, not duration, is under test.
    window.worker = worker
    window.accept_scans = True
    try:
        for _ in range(59):
            for key in (1, 2):
                worker.buffer.receive(Block(key, [[2500, 2500]]))
        window._flush_live()
        assert window.preprocess.ready
        other_values = worker.buffer.display_values[2]
        window.apply_sensor_mask(1, np.array([[False, True]]))
        window._flush_live()
        assert not window.preprocess.ready
        assert window.preprocess.samples == {1: 0, 2: 30}
        assert window.heatmap.items[1].values is None
        assert window.heatmap.items[2].values[0, 0] == 0
        assert window.preprocess.baselines[2][0, 0] == 2500
        assert worker.buffer.display_values[2] is other_values
        for i in range(59):
            worker.buffer.receive(Block(1, [[0 if i % 2 else 4095, 2400]]))
        for _ in range(35):
            worker.buffer.receive(Block(1, [[10, 2000]]))
        window._flush_live()
        assert window.preprocess.ready
        assert window.heatmap.reading((1, 0, 0)) is None
        assert window.heatmap.reading((1, 0, 1)).value == 400
        # Saving unchanged points must not discard the completed baseline.
        window.apply_sensor_mask(1, np.array([[False, True]]))
        window._flush_live()
        assert window.preprocess.ready
        assert window.heatmap.reading((1, 0, 1)).value == 400
    finally:
        window.worker = None
        window.accept_scans = False
        window.close()


def test_corrupt_saved_config_is_reported_without_overwriting(app, tmp_path):
    path = tmp_path / "mask.json"
    path.write_text("damaged")
    window = MainWindow(mask_path=path)
    try:
        window.present_frame(Frame((Block(1, [[2500]]),)))
        assert "加载失败" in window.controls.mask_status.text()
        with pytest.raises(ValueError, match="未覆盖"):
            window.apply_sensor_mask(1, np.array([[False]]))
        assert path.read_text() == "damaged"
    finally:
        window.close()
