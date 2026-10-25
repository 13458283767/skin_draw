"""滤波开关必须真正旁路 M3，同时保持原始录制与基线状态一致。"""

import numpy as np
import pytest

from skin_draw.live import LiveBuffer
from skin_draw.models import Block
from skin_draw.preprocessing import BlockPreprocessor
from skin_draw.sensor_masks import SensorMasks
from skin_draw.ui.main_window import MainWindow
from skin_draw.ui.serial_worker import SerialWorker


def test_disabled_filter_has_no_window_or_suppression():
    pre = BlockPreprocessor((1, 3), [[True, False, True]], norm="none", filter_enabled=False)
    for value in (2500, 0, 4095, 1800):
        output = pre.update([[value, 2500, value]])
        np.testing.assert_array_equal(output, [[value, np.nan, value]])
        assert not output.flags.writeable
    assert not pre.filters[0][0].buf
    assert pre.samples == 0  # 关闭归一化不应偷偷累计基线。


@pytest.mark.parametrize("norm, expected", [("A", 600), ("B", 1)])
def test_disabled_filter_calibrates_without_warmup_and_keeps_normalization(norm, expected):
    pre = BlockPreprocessor((1, 1), [[True]], norm=norm, calib_n=2, filter_enabled=False)
    assert np.isnan(pre.update([[2500]])[0, 0])
    assert pre.samples == 1
    assert pre.update([[2500]])[0, 0] == 0
    assert pre.ready
    assert pre.update([[1900]])[0, 0] == expected


def test_switch_recalibrates_and_never_changes_recorded_adc():
    buffer = LiveBuffer(norm="A", calib_n=2, filter_enabled=False)
    recorded = []
    buffer.set_raw_sink(lambda block, _: recorded.append(block))
    for _ in range(2):
        buffer.receive(Block(1, [[2500]]))
    before = buffer.take()
    assert before.preprocess.ready
    buffer.set_filter_enabled(True)
    pending = buffer.take()
    assert pending.preprocess.filter_enabled
    assert pending.preprocess.samples == {1: 0}
    assert np.isnan(pending.display_values[1][0, 0])
    assert before.preprocess.ready and not before.preprocess.filter_enabled
    for _ in range(31):
        buffer.receive(Block(1, [[2500]]))
    buffer.receive(Block(1, [[0]]))
    assert buffer.take().display_values[1][0, 0] == 0  # M3 抑制这个低值。
    assert recorded[-1].values[0, 0] == 0
    buffer.set_filter_enabled(False)
    assert buffer.take().preprocess.samples == {1: 0}
    buffer.receive(Block(1, [[2500]]))
    buffer.receive(Block(1, [[2500]]))
    buffer.receive(Block(1, [[0]]))
    assert buffer.take().display_values[1][0, 0] == 2500
    assert len(recorded) == 37  # 切换开关不能生成额外采集帧。
    buffer.set_filter_enabled(False)
    assert buffer.take() is None  # 相同设置不应再次清除标定。


def test_disabled_filter_applies_to_rebuilt_and_new_blocks():
    buffer = LiveBuffer(norm="none", filter_enabled=True)
    buffer.receive(Block(1, [[2500]]))
    buffer.set_filter_enabled(False)
    np.testing.assert_array_equal(buffer.take().display_values[1], [[2500]])
    buffer.set_sensor_masks(SensorMasks())
    for block in (Block(1, [[0, 4095]]), Block(2, [[100]])):
        buffer.receive(block)
    snapshot = buffer.take()
    np.testing.assert_array_equal(snapshot.display_values[1], [[0, 4095]])
    assert snapshot.display_values[2][0, 0] == 100
    assert all(not pre.filter_enabled for pre in buffer.preprocessors.values())
    assert "滤波关闭" in snapshot.preprocess.message


def test_button_defaults_off_and_controls_worker_on_connection(app, monkeypatch):
    # 不打开真实串口，通过连接入口验证按钮状态确实传给后台。
    monkeypatch.setattr(SerialWorker, "start", lambda self: None)
    window = MainWindow()
    try:
        assert not window.controls.filter_toggle.isChecked()
        window.controls.port.clear()
        window.controls.port.addItem("FAKE", "FAKE")
        window.controls.connect_button.click()
        buffer = window.worker.buffer
        assert not buffer.filter_enabled
        buffer.calib_n = 1
        buffer.receive(Block(1, [[2500]]))
        window._flush_live()
        assert window.preprocess.ready
        window.controls.filter_toggle.click()
        window._flush_live()
        assert buffer.filter_enabled
        assert not window.preprocess.ready
        assert window.controls.normalize.isChecked()
        assert "点击关闭" in window.controls.filter_toggle.text()
        window.controls.filter_toggle.click()
        buffer.receive(Block(1, [[2400]]))
        window._flush_live()
        assert window.preprocess.ready and not window.preprocess.filter_enabled
    finally:
        worker = window.worker
        window.worker = None
        window.accept_scans = False
        if worker is not None:
            worker.deleteLater()
        window.close()
        window.deleteLater()


def test_raw_diagnostic_mode_cannot_enable_filter(app):
    buffer = LiveBuffer(raw_adc=True)
    buffer.set_filter_enabled(True)
    buffer.receive(Block(1, [[0, 4095]]))
    snapshot = buffer.take()
    assert not snapshot.preprocess.filter_enabled
    np.testing.assert_array_equal(snapshot.display_values[1], [[0, 4095]])
    window = MainWindow(raw_adc=True)
    try:
        assert not window.controls.filter_toggle.isEnabled()
        assert not window.controls.filter_toggle.isChecked()
    finally:
        window.close()
        window.deleteLater()
