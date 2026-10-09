import numpy as np
import pytest

from skin_draw.live import LiveBuffer
from skin_draw.models import Block
from skin_draw.sensor_masks import SensorMasks


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now


def setup():
    clock = Clock()
    return clock, LiveBuffer(clock=clock)


@pytest.mark.parametrize("hz", [20, 45, 50, 90, 180])
def test_requires_duration_and_sample_count_at_each_rate(hz):
    clock, buffer = setup()
    for i in range(300):
        clock.now = 100 + i / hz
        buffer.receive(Block(1, [[2500, 2490]]))
        if buffer.calibration.ready:
            break
        assert buffer.calibration.progress < 100
    row = buffer.calibration.rows[1][0]
    assert row.count >= 30
    assert row.seconds >= 1 - 1e-9
    assert row.count == max(30, hz + 1)
    assert buffer.calibration.progress == 100
    np.testing.assert_array_equal(buffer.calibration.baseline[1], [[2500, 2490]])


def test_first_corrupt_frame_and_wrong_initial_cluster_both_recover():
    for bad_samples in (1, 10):
        clock, buffer = setup()
        for i in range(180):
            clock.now = 100 + i / 90
            buffer.receive(Block(6, [[500 if i < bad_samples else 2500, 2500]]))
        assert buffer.calibration.ready
        np.testing.assert_array_equal(buffer.calibration.baseline[6], [[2500, 2500]])
        np.testing.assert_array_equal(buffer.outlier_filter.previous[6].values, [[2500, 2500]])
        buffer.receive(Block(6, [[2000, 2100]]))
        np.testing.assert_array_equal(buffer.take().display_values[6], [[500, 400]])


def test_unstable_row_does_not_delay_other_rows_or_blocks():
    clock, buffer = setup()
    for i in range(110):
        clock.now += 1 / 90
        buffer.receive(Block(1, [[2500, 2500], [500 if i % 2 else 2500, 2500]]))
        buffer.receive(Block(2, [[2400, 2400]]))
    first = buffer.take()
    assert first.calibration.rows[1][0].result is not None
    assert first.calibration.rows[1][1].result is None
    assert first.calibration.pending_rows == {1: [2]}
    assert 2 in first.display_values and 1 not in first.display_values
    assert "行2" in first.calibration.progress_details
    saved = first.calibration.rows[1][0].result.copy()
    for i in range(110):
        clock.now += 1 / 90
        buffer.receive(Block(1, [[2450, 2450], [2500, 2500]]))
    assert buffer.calibration.ready
    np.testing.assert_array_equal(buffer.calibration.baseline[1][0], saved)
    # Previously delivered snapshots must not mutate as new samples arrive.
    assert first.calibration.rows[1][1].result is None


def test_intermittent_corruption_is_excluded_from_mean_and_runtime_rule_is_unchanged():
    clock, buffer = setup()
    for i in range(180):
        clock.now += 1 / 90
        buffer.receive(Block(1, [[1000 if i % 5 == 0 else 2500, 2490]]))
    assert buffer.calibration.ready
    np.testing.assert_array_equal(buffer.calibration.baseline[1], [[2500, 2490]])
    for _ in range(12):
        buffer.receive(Block(1, [[1000, 2490]]))
        snap = buffer.take()
        assert snap.filtered_masks[1].all()
        np.testing.assert_array_equal(snap.display_values[1], [[0, 0]])
    buffer.receive(Block(1, [[2000, 2000]]))
    np.testing.assert_array_equal(buffer.take().display_values[1], [[500, 490]])


def test_no_time_progress_from_burst_or_silence():
    clock, buffer = setup()
    for _ in range(100):
        buffer.receive(Block(1, [[2500]]))
    assert not buffer.calibration.ready
    clock.now += 10
    buffer.receive(Block(1, [[2500]]))
    assert not buffer.calibration.ready
    assert buffer.calibration.rows[1][0].seconds == 0
    for _ in range(31):
        clock.now += .04
        buffer.receive(Block(1, [[2500]]))
    assert buffer.calibration.ready


def test_masks_shape_changes_and_recalibration():
    clock, buffer = setup()
    mask = np.array([[False, True], [False, False]])
    buffer.set_sensor_masks(SensorMasks().with_mask(7, mask))
    for i in range(100):
        clock.now += 1 / 90
        buffer.receive(Block(7, [[4095 if i % 2 else 0, 2500], [0, 4095]]))
    np.testing.assert_array_equal(buffer.calibration.baseline[7], [[0, 2500], [0, 0]])
    buffer.receive(Block(7, [[2500]]))
    assert not buffer.calibration.ready
    for _ in range(100):
        clock.now += 1 / 90
        buffer.receive(Block(7, [[2500]]))
    assert buffer.calibration.ready
    buffer.reset_calibration()
    assert not buffer.calibration.ready and buffer.calibration.progress == 0
    assert not buffer.outlier_filter.previous


def test_seven_blocks_calibrate_independently_without_waiting_for_paint():
    clock, buffer = setup()
    for i in range(100):
        clock.now += 1 / 90
        for key in range(1, 8):
            buffer.receive(Block(key, np.full((10, 5), 2500 + key)))
    snap = buffer.take()
    assert len(snap.calibration.baseline) == 7
    assert snap.calibration.ready and snap.received_blocks == 700


def test_ui_shows_duration_and_pending_rows(app, tmp_path):
    from skin_draw.ui.main_window import MainWindow
    clock, buffer = setup()
    window = MainWindow(mask_path=tmp_path / "mask.json")
    try:
        for i in range(40):
            clock.now += 1 / 90
            buffer.receive(Block(1, [[2500], [500 if i % 2 else 2500]]))
        snap = buffer.take()
        window._consume_scan(snap.scan, snap)
        assert "秒" in window.controls.calibration_details.text()
        assert "样本" in window.controls.calibration_details.text()
        assert window.controls.calibration_progress.maximum() == 100
        assert window.controls.calibration_progress.value() < 100
    finally:
        window.close()
