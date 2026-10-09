from unittest.mock import patch

import numpy as np
import pytest

from skin_draw.live import LiveBuffer
from skin_draw.mapping import BlockOrientation
from skin_draw.models import Block
from skin_draw.ui.main_window import MainWindow
from test_live_buffer import Clock


def calibrated_buffer(ids=(1, 2), clock=None):
    buffer = LiveBuffer(calibration_seconds=0, **({"clock": clock} if clock else {}))
    for _ in range(30):
        for key in ids:
            buffer.receive(Block(key, [[2500, 2500, 2500], [2500, 2500, 2500]]))
    return buffer


def present(window, buffer):
    snapshot = buffer.take()
    assert snapshot is not None
    window._consume_scan(snapshot.scan, snapshot)
    return snapshot


@pytest.fixture
def live_window(app):
    window = MainWindow()
    yield window
    window.close()
    window.deleteLater()


def test_cached_responses_are_immutable_and_snapshots_keep_old_samples():
    buffer = calibrated_buffer()
    old = buffer.take()
    buffer.receive(Block(1, [[2000] * 3] * 2))
    new = buffer.take()
    assert new.display_values[2] is old.display_values[2]
    assert new.display_values[1] is not old.display_values[1]
    np.testing.assert_array_equal(old.display_values[1], np.zeros((2, 3)))
    np.testing.assert_array_equal(new.display_values[1], np.full((2, 3), 500))
    with pytest.raises(ValueError):
        new.display_values[1][0, 0] = 10
    new.display_values.clear()
    assert set(buffer.display_values) == {1, 2}


def test_only_updated_block_maps_colors_and_unchanged_selection_is_quiet(live_window):
    window = live_window
    buffer = calibrated_buffer(range(1, 15))
    present(window, buffer)
    window.heatmap.select_sensor((2, 0, 0))
    events = []
    window.heatmap.sensor_selected.connect(events.append)
    pictures = {key: item.picture for key, item in window.heatmap.items.items()}
    untouched = window.heatmap.items[2]
    with patch.object(untouched, "set_data", wraps=untouched.set_data) as redraw:
        buffer.receive(Block(1, [[2000] * 3] * 2))
        present(window, buffer)
    assert redraw.call_count == 0
    assert window.heatmap.items[1].picture is not pictures[1]
    assert all(window.heatmap.items[key].picture is pictures[key] for key in range(2, 15))
    assert window.heatmap.reading((1, 0, 0)).value == 500
    assert events == []


def test_selected_raw_outlier_updates_while_held_row_stays_unchanged(live_window):
    window = live_window
    buffer = calibrated_buffer()
    present(window, buffer)
    window.heatmap.select_sensor((1, 0, 0))
    buffer.receive(Block(1, [[500, 2500, 2500], [2000, 2000, 2000]]))
    snapshot = present(window, buffer)
    np.testing.assert_array_equal(snapshot.display_values[1], [[0, 0, 0], [500, 500, 500]])
    assert "500" in window.selected_detail.text() and "异常值已过滤" in window.selected_detail.text()
    buffer.receive(Block(1, [[2000] * 3] * 2))
    present(window, buffer)
    assert window.heatmap.reading((1, 0, 0)).value == 500
    assert "异常值已过滤" not in window.selected_detail.text()


def test_direction_and_recalibration_invalidate_responses_without_new_data(live_window):
    buffer = calibrated_buffer()
    present(live_window, buffer)
    buffer.receive(Block(1, [[2000] * 3] * 2))
    present(live_window, buffer)
    buffer.set_direction(1)
    present(live_window, buffer)
    assert live_window.heatmap.reading((1, 0, 0)).value == 0
    buffer.set_direction(-1)
    present(live_window, buffer)
    assert live_window.heatmap.reading((1, 0, 0)).value == 500
    buffer.reset_calibration()
    snapshot = present(live_window, buffer)
    assert snapshot.display_values == {}
    assert all(item.values is None for item in live_window.heatmap.items.values())
    for _ in range(30):
        buffer.receive(Block(1, [[2400] * 3] * 2))
    present(live_window, buffer)
    assert live_window.heatmap.reading((1, 0, 0)).value == 0
    assert live_window.heatmap.items[2].values is None


def test_shape_change_rebuilds_and_waits_for_new_baseline(live_window):
    buffer = calibrated_buffer()
    present(live_window, buffer)
    buffer.receive(Block(1, [[2000, 2000]]))
    snapshot = present(live_window, buffer)
    assert 1 not in snapshot.display_values and snapshot.calibration.counts[1] == 0
    assert len(snapshot.calibration.rows[1][0].history) == 1
    assert live_window.heatmap.items[1].values is None
    assert live_window.heatmap.items[1].spec.display_shape == (1, 2)
    assert live_window.heatmap.items[2].values.shape == (2, 3)


def test_rotation_and_mirror_keep_cached_other_blocks_correct(live_window):
    buffer = calibrated_buffer()
    present(live_window, buffer)
    buffer.receive(Block(2, [[2400, 2300, 2200], [2100, 2000, 1900]]))
    snapshot = present(live_window, buffer)
    view = live_window.heatmap
    view.set_orientation(2, BlockOrientation(rotation=90, flip_x=True))
    expected = np.fliplr(np.rot90(snapshot.display_values[2]))
    np.testing.assert_array_equal(view.items[2].values, expected)
    view.move_block(2, 20, 10)
    position = view.items[2].pos()  # movement is clamped to the visible canvas
    buffer.receive(Block(1, [[2300] * 3] * 2))
    present(live_window, buffer)
    np.testing.assert_array_equal(view.items[2].values, expected)
    assert view.items[2].pos() == position


def test_missing_gray_and_recovery_are_immediate_without_reflow(live_window):
    clock = Clock()
    buffer = calibrated_buffer(clock=clock)
    present(live_window, buffer)
    view = live_window.heatmap
    item = view.items[2]
    view.select_sensor((2, 0, 0))
    clock.now = 4
    buffer.receive(Block(1, [[2500] * 3] * 2))
    present(live_window, buffer)
    assert item.stale and item.values is not None
    assert live_window.selected_label.text() == "—"
    clock.now = 8
    buffer.receive(Block(1, [[2500] * 3] * 2))
    present(live_window, buffer)
    assert item.values is None
    buffer.receive(Block(2, [[2100] * 3] * 2))
    present(live_window, buffer)
    assert view.items[2] is item and not item.stale
    assert live_window.selected_label.text() == "400"


def test_heatmap_updates_before_slow_statistics_and_pending_stats_use_latest(live_window):
    buffer = calibrated_buffer()
    present(live_window, buffer)
    assert float(live_window.max_label.text()) == 0
    for value in (2400, 2300, 2000):
        buffer.receive(Block(1, [[value] * 3] * 2))
        present(live_window, buffer)
        assert live_window.heatmap.reading((1, 0, 0)).value == 2500 - value
    assert float(live_window.max_label.text()) == 0
    live_window._flush_live_info()
    assert live_window.max_label.text() == "500"
    assert live_window.pending_live_info is None


def test_pending_statistics_cannot_overwrite_serial_failure(live_window):
    buffer = calibrated_buffer()
    present(live_window, buffer)
    buffer.receive(Block(1, [[2000] * 3] * 2))
    present(live_window, buffer)
    live_window._serial_failed("串口采集失败：模拟断开")
    live_window._flush_live_info()
    assert live_window.controls.connection_status.text() == "串口采集失败：模拟断开"
