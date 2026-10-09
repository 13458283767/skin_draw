"""Raw diagnostics must show even the first sample and hardware spikes verbatim."""
import numpy as np
from PySide6.QtTest import QTest

from skin_draw.acquisition import Scan
from skin_draw.live import LiveBuffer
from skin_draw.models import Block, Frame
from skin_draw.sensor_masks import SensorMasks
from skin_draw.serial_protocol import SerialBlockParser
from skin_draw.ui.main_window import MainWindow


def test_raw_pipeline_skips_calibration_and_filter_for_text_and_binary(monkeypatch):
    buffer = LiveBuffer(raw_adc=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Raw mode must not calibrate, subtract, or filter")

    monkeypatch.setattr(buffer, "_preprocessor", forbidden)
    parser = SerialBlockParser()
    packets = [b"BLOCK_7,1,3:\r\n2500,2490,2510\r\n",
               bytes.fromhex("AA 07 01 03 02 30 00 00 0F FF 0A"),
               b"BLOCK_7,1,3:\r\n2501,2491,2511\r\n"]
    for sequence, packet in enumerate(packets, 1):
        decoded = []
        # Split across protocol boundaries, just as arbitrary serial reads do.
        for start in range(0, len(packet), 3):
            decoded.extend(parser.feed(packet[start:start+3]))
        assert len(decoded) == 1
        buffer.receive(decoded[0])
        snapshot = buffer.take()
        assert snapshot.received_blocks == sequence
        assert snapshot.display_values[7] is decoded[0].values
        assert snapshot.scan.frame.blocks[0] is decoded[0]
        assert not buffer.preprocessors
        assert snapshot.preprocess.message == "原始 ADC 模式 · 不扣基线"


def test_raw_mask_shape_and_freshness_keep_original_values():
    now = [0.]
    buffer = LiveBuffer(raw_adc=True, clock=lambda: now[0])
    first = Block(1, [[2500, 0]])
    buffer.receive(first)
    buffer.set_sensor_masks(SensorMasks().with_mask(1, [[True, False]]), 1)
    buffer.reset_preprocessing()
    buffer.set_direction(1)
    assert buffer.take().display_values[1] is first.values
    second = Block(1, [[500], [2600]])
    now[0] = .01
    buffer.receive(second)
    buffer.receive(Block(14, [[1900]]))
    snapshot = buffer.take()
    assert snapshot.display_values[1] is second.values
    assert set(snapshot.display_values) == {1, 14}
    assert not buffer.preprocessors
    now[0] = 5.
    stale = buffer.take()
    assert not stale.scan.frame.blocks
    assert not stale.display_values
    assert stale.missing_counts[1] >= 3


def test_raw_gui_first_frame_values_statistics_masks_and_orientation(app, tmp_path):
    window = MainWindow(raw_adc=True, mask_path=tmp_path / 'raw-mask.json')
    try:
        window.show()
        assert window.controls.show_values.isChecked()
        assert window.controls.calibration_progress.isHidden()
        assert not window.controls.recalibrate.isEnabled()
        assert not window.controls.direction.isEnabled()
        assert window.controls.connect_button.text() == "连接并显示"
        block = Block(1, [[2500, 0], [560, 4095]])
        window._consume_scan(Scan(Frame((block,), 1), (1,), 0.))
        assert window.unit == window.heatmap.bar.unit == "ADC"
        np.testing.assert_array_equal(window.heatmap.items[1].values, block.values)
        assert window.heatmap.items[1].show_values
        assert window.min_label.text() == "0" and window.max_label.text() == "4095"
        window.heatmap.select_sensor((1, 1, 0))
        assert window.selected_label.text() == "560"
        assert "基线" not in window.selected_detail.text()
        window._orient_block("left")
        reading = window.heatmap.reading((1, 1, 1))
        assert (reading.row, reading.col, reading.value) == (1, 0, 560)
        window.apply_sensor_mask(1, np.array([[True, False], [True, True]]))
        assert window.min_label.text() == "560"
        assert window.heatmap.reading((1, 1, 1)).value == 560
        window._recalibrate()
        window._toggle_direction()
        assert window.heatmap.reading((1, 1, 1)).value == 560
        assert window.preprocess is None
    finally:
        window.close()
        window.deleteLater()


def test_raw_worker_connect_disconnect_reconnect_never_waits_for_baseline(app, monkeypatch):
    from skin_draw.ui import serial_worker

    class Reader:
        def __init__(self, port, stop, on_scan, on_status, on_block, **kwargs):
            self.stop, self.on_block = stop, on_block
            self.active_mode = "stream"

        def run(self):
            self.on_block(Block(4, [[2500, 2510, 2490]]))
            self.on_block(Block(4, [[1196, 2037, 1516]]))
            self.stop.wait(5)

    monkeypatch.setattr(serial_worker, "SerialAcquisition", Reader)
    window = MainWindow(raw_adc=True)
    try:
        window.controls.port.clear()
        window.controls.port.addItem("FAKE", "FAKE")
        for _ in range(2):
            window.controls.connect_button.click()
            for _ in range(100):
                QTest.qWait(10)
                if window.current_frame.sequence == 2:
                    break
            assert window.worker.buffer.raw_adc
            assert window.preprocess is not None and window.preprocess.baselines == {}
            assert window.heatmap.reading((4, 0, 0)).value == 1196
            assert window.controls.high.value() == 4095
            assert not window.controls.auto_levels.isChecked()
            assert "校准" not in window.source_badge.text()
            assert not window.controls.recalibrate.isEnabled()
            window.controls.connect_button.click()
            for _ in range(100):
                QTest.qWait(10)
                if window.worker is None:
                    break
            assert window.worker is None
            assert window.controls.connect_button.text() == "连接并显示"
            assert window.heatmap.reading((4, 0, 0)).value == 1196
    finally:
        if window.worker is not None:
            window.worker.stop()
            for _ in range(100):
                QTest.qWait(10)
                if window.worker is None:
                    break
        window.close()
        window.deleteLater()
