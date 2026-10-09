import numpy as np
from PySide6.QtTest import QTest

from skin_draw.acquisition import Scan
from skin_draw.models import Block, Frame
from skin_draw.ui.main_window import MainWindow


def sample(sequence, value=2000, complete=True):
    blocks = tuple(Block(i, np.full((5, 4), value)) for i in range(1, 15 if complete else 14))
    return Scan(Frame(blocks, sequence, sequence * .04), tuple(range(1, 15)), .5)


def test_default_is_hardware_idle_and_calibration_then_direction_switch(app):
    window = MainWindow()
    window.show()
    QTest.qWait(100)
    assert not window.timer.isActive()
    assert window.input_source == "idle"
    assert not window.controls.demo_toggle.isChecked()
    assert not window.heatmap.items
    for sequence in range(1, 30):
        window._consume_scan(sample(sequence))
    assert window.controls.calibration_progress.value() == 96
    assert all(item.values is None for item in window.heatmap.items.values())
    window._consume_scan(sample(30, complete=False))
    assert window.controls.calibration_progress.value() == 96
    window._consume_scan(sample(31))
    assert window.calibration.ready and len(window.heatmap.items) == 14
    window._consume_scan(sample(32, 1700))
    window.heatmap.select_sensor((1, 0, 0))
    assert window.selected_label.text() == "300"
    assert "原始 ADC 1700" in window.selected_detail.text()
    assert "基线 2000.0" in window.selected_detail.text()
    assert window.heatmap.bar.unit == "ΔADC"
    window.controls.direction.click()
    assert window.selected_label.text() == "0"
    assert window.calibration.count == 30
    window._consume_scan(sample(33, 2300))
    assert window.selected_label.text() == "300"
    window.controls.recalibrate.click()
    assert window.calibration.count == 0
    assert not window.calibration.ready
    assert all(item.values is None for item in window.heatmap.items.values())
    window.close()
    window.deleteLater()


def test_demo_cannot_overwrite_a_connected_worker(app):
    window = MainWindow()
    class DummyWorker:
        def stop(self):
            pass
    window.worker = DummyWorker()
    window._consume_scan(sample(1))
    frame = window.current_frame
    window.toggle_play()
    window.step_demo()
    window._reset_demo()
    assert window.current_frame is frame
    window.worker = None
    window.close()
    window.deleteLater()


def test_invalid_port_worker_failure_restores_controls(app):
    window = MainWindow()
    window.controls.port.clear()
    window.controls.port.addItem("不存在的测试串口", "__SkinDraw_nonexistent_port__")
    window.controls.connect_button.click()
    assert window.worker is not None
    for _ in range(100):
        if window.worker is None:
            break
        QTest.qWait(20)
    assert window.worker is None
    assert window.controls.connect_button.isEnabled()
    assert window.controls.baudrate.isEnabled()
    assert window.controls.demo_toggle.isEnabled()
    assert not window.accept_scans
    assert "串口采集失败" in window.controls.connection_status.text()
    assert window.calibration.count == 0
    window.close()
    window.deleteLater()


def test_real_worker_buffer_drives_latest_gui_and_calibrates_before_paint(app, monkeypatch):
    from skin_draw.ui import serial_worker

    class FastReader:
        def __init__(self, port, stop, on_scan, on_status, on_block, mode, protocol, baudrate):
            assert protocol == "binary"
            assert baudrate == 460800
            self.stop, self.on_block = stop, on_block
            self.active_mode = "stream"

        def run(self):
            from time import perf_counter
            started = perf_counter()
            for sequence in range(100):
                window.worker.buffer.clock = lambda: started + sequence * .04
                for key in (2, 7):
                    self.on_block(Block(key, [[2000 if sequence < 30 else 1500]]))
            self.stop.wait(5)

    monkeypatch.setattr(serial_worker, "SerialAcquisition", FastReader)
    window = MainWindow()
    window.controls.port.clear()
    window.controls.port.addItem("FAKE", "FAKE")
    window.controls.protocol.setCurrentIndex(window.controls.protocol.findData("binary"))
    window.controls.baudrate.setCurrentIndex(window.controls.baudrate.findData(460800))
    assert not hasattr(window.controls, "verify_crc")
    window.controls.connect_button.click()
    assert not window.controls.protocol.isEnabled()
    assert not window.controls.baudrate.isEnabled()
    for _ in range(100):
        QTest.qWait(10)
        if window.current_frame and window.current_frame.sequence == 200:
            break
    assert window.calibration.counts == {2: 30, 7: 30}
    assert window.heatmap.reading((2, 0, 0)).value == 500
    assert window.current_frame.sequence == 200
    assert window.controls.palette.currentText() == "浅红深红"
    assert not window.controls.auto_levels.isChecked()
    window.controls.direction.click()
    QTest.qWait(30)
    assert window.heatmap.reading((2, 0, 0)).value == 0
    assert window.calibration.count == 30
    window.controls.recalibrate.click()
    QTest.qWait(30)
    assert window.calibration.count == 0
    assert window.heatmap.items[2].values is None
    window.controls.connect_button.click()
    for _ in range(100):
        QTest.qWait(10)
        if window.worker is None:
            break
    assert window.worker is None
    assert window.controls.protocol.isEnabled()
    assert window.controls.baudrate.isEnabled()
    assert window.controls.baudrate.currentData() == 460800
    assert not window.display_timer.isActive()
    window.close()
    window.deleteLater()
