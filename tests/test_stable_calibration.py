"""slow 标定语义：M3 暖机 + 中位数标定按样本计数，分块独立、与时间无关。"""

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
    return clock, LiveBuffer(clock=clock, calib_n=30, norm="A")


@pytest.mark.parametrize("hz", [20, 45, 90, 180])
def test_calibration_counts_samples_not_elapsed_time(hz):
    clock, buffer = setup()
    frames = 0
    for i in range(300):
        clock.now = 100 + i / hz
        buffer.receive(Block(1, [[2500, 2490]]))
        frames += 1
        if buffer.preprocessors[1].ready:
            break
    assert frames == 59                      # 29 帧暖机 + 30 个标定样本，与帧率无关
    np.testing.assert_allclose(buffer.preprocessors[1].baselines, [[2500, 2490]])
    assert buffer.preprocessors[1].samples == 30
    assert buffer.preprocessors[1].progress == 100


def test_startup_low_frames_are_filtered_by_m3():
    clock, buffer = setup()
    for i in range(120):
        clock.now += 1 / 90
        buffer.receive(Block(6, [[500 if i < 10 else 2500, 2500]]))
    state = buffer.preprocessors[6]
    assert state.ready
    np.testing.assert_array_equal(state.baselines, [[2500, 2500]])   # 启动低电平不进基线
    for _ in range(35):
        buffer.receive(Block(6, [[2000, 2100]]))                     # 按压超过窗口滞后
    snapshot = buffer.take()
    np.testing.assert_allclose(snapshot.display_values[6], [[500, 400]], atol=1e-6)


def test_seven_blocks_calibrate_independently():
    clock, buffer = setup()
    for i in range(59):
        clock.now += 1 / 90
        for key in range(1, 8):
            buffer.receive(Block(key, np.full((10, 5), 2500 + key)))
    snap = buffer.take()
    assert snap.preprocess.ready and snap.received_blocks == 413
    assert len(snap.preprocess.baselines) == 7
    for key in range(1, 8):
        assert np.all(snap.preprocess.baselines[key] == 2500 + key)


def test_slowest_block_does_not_block_others():
    clock, buffer = setup()
    for i in range(59):
        clock.now += 1 / 90
        buffer.receive(Block(1, [[2500]]))
        if i < 40:
            buffer.receive(Block(2, [[2400]]))
    snap = buffer.take()
    assert not snap.preprocess.ready
    assert snap.preprocess.samples == {1: 30, 2: 11}     # 40 帧 - 29 暖机
    assert np.isfinite(snap.preprocess.baselines[1]).all()
    assert np.isnan(snap.preprocess.baselines[2]).all()
    for _ in range(30):
        clock.now += 1 / 90
        buffer.receive(Block(2, [[2400]]))
    snap = buffer.take()
    assert snap.preprocess.ready
    assert snap.preprocess.baselines[2][0, 0] == 2400    # 自己的样本集独立标定


def test_intermittent_low_rail_is_excluded_from_baseline():
    clock, buffer = setup()
    for i in range(120):
        clock.now += 1 / 90
        buffer.receive(Block(1, [[1000 if i % 5 == 0 else 2500, 2490]]))
    state = buffer.preprocessors[1]
    assert state.ready
    np.testing.assert_allclose(state.baselines, [[2500, 2490]])


def test_progress_follows_samples_not_elapsed_time():
    clock, buffer = setup()
    for _ in range(29):
        buffer.receive(Block(1, [[2500]]))
    assert buffer.preprocessors[1].samples == 0
    clock.now += 10                                      # 安静 10 秒不推进进度
    buffer.receive(Block(1, [[2500]]))
    assert buffer.preprocessors[1].samples == 1
    for _ in range(29):
        buffer.receive(Block(1, [[2500]]))
    assert buffer.preprocessors[1].ready


def test_mask_change_rebuilds_only_target_block():
    clock, buffer = setup()
    buffer.set_sensor_masks(SensorMasks().with_mask(7, np.array([[False, True]])))
    for i in range(59):
        clock.now += 1 / 90
        buffer.receive(Block(7, [[0, 2500]]))
        buffer.receive(Block(2, [[2400, 2400]]))
    snap = buffer.take()
    assert snap.preprocess.ready
    assert np.isnan(snap.preprocess.baselines[7][0, 0])  # 空点位 NaN
    assert snap.preprocess.baselines[7][0, 1] == 2500
    buffer.set_sensor_masks(SensorMasks().with_mask(7, np.array([[True, True]])), 7)
    snap = buffer.take()
    assert not snap.preprocess.ready                    # 仅 BLOCK 7 重新标定
    assert np.isnan(snap.preprocess.baselines[7]).all()
    assert snap.preprocess.samples[2] == 30
    buffer.reset_preprocessing()
    assert not buffer.preprocessors[7].ready
    assert buffer.preprocessors[7].samples == 0
    assert not buffer.preprocessors[2].ready


def test_shape_change_rebuilds_and_waits_for_new_baseline():
    clock, buffer = setup()
    for i in range(59):
        clock.now += 1 / 90
        buffer.receive(Block(7, [[2500, 2500]]))
    assert buffer.preprocessors[7].ready
    buffer.receive(Block(7, [[2500]]))                   # 尺寸变化 -> 重建
    assert buffer.preprocessors[7].shape == (1, 1)
    assert not buffer.preprocessors[7].ready
    assert buffer.preprocessors[7].samples == 0


def test_ui_shows_sample_progress_and_block_details(app, tmp_path):
    from skin_draw.ui.main_window import MainWindow
    clock, buffer = setup()
    window = MainWindow(mask_path=tmp_path / "mask.json")
    try:
        for i in range(40):
            clock.now += 1 / 90
            buffer.receive(Block(1, [[2500], [500 if i % 2 else 2500]]))
        snap = buffer.take()
        window._consume_scan(snap.scan, snap)
        assert window.controls.calibration_progress.maximum() == 100
        assert window.controls.calibration_progress.value() == 36   # 11/30 样本
        assert "自动标定中" in window.controls.calibration_status.text()
        assert "BLOCK 01 11/30" in window.controls.calibration_details.text()
        assert not window.controls.calibration_details.isHidden()
    finally:
        window.close()
        window.deleteLater()
