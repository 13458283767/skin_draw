from types import SimpleNamespace

import pytest

from skin_draw.live import LiveBuffer
from skin_draw.models import Block
from skin_draw.reception_rate import ReceptionRate
from skin_draw.serial_protocol import SerialBlockParser
from skin_draw.ui.main_window import MainWindow
from test_live_buffer import Clock


def test_sliding_second_boundary_zero_after_silence_and_recovery():
    rate = ReceptionRate()
    assert rate.counts(0) == {}
    rate.record(1, 0)
    rate.record(1, .5)
    rate.record(2, .8)
    assert rate.counts(.9) == {1: 2, 2: 1}
    assert rate.counts(1) == {1: 1, 2: 1}  # (now - 1, now]
    assert rate.counts(1.5) == {1: 0, 2: 1}
    assert rate.counts(1.8) == {1: 0, 2: 0}
    rate.record(2, 3)
    assert rate.counts(3) == {1: 0, 2: 1}
    assert not rate.arrivals[1]


def test_fourteen_blocks_counts_actual_arrivals_without_gui_snapshots():
    clock = Clock()
    buffer = LiveBuffer(clock=clock)
    for frame in range(30):
        clock.now = frame / 30
        for key in range(1, 15):
            buffer.receive(Block(key, [[2500]]))
    clock.now = .999
    counts = buffer.reception_counts()
    assert counts == dict.fromkeys(range(1, 15), 30)
    assert sum(counts.values()) == 420
    buffer.take()
    for _ in range(20):
        assert buffer.take() is None
    assert buffer.reception_counts() == counts
    counts.clear()  # callers cannot change the counter
    assert len(buffer.reception_counts()) == 14
    clock.now = 2
    assert buffer.reception_counts() == dict.fromkeys(range(1, 15), 0)


def test_counts_include_filtered_frames_but_exclude_invalid_and_not_ready():
    clock = Clock()
    buffer = LiveBuffer(clock=clock)
    for _ in range(30):
        buffer.receive(Block(1, [[2500]]))
    clock.now = 2  # Calibration arrivals are outside this counter window.
    parser = SerialBlockParser()
    data = bytes.fromhex("AA 01 01 01 09 C4 0A AA 01 01 01 01 F4 0A "
                         "AA 01 01 01 09 C4 00 AA 02 00 00 0A")
    for block in parser.feed(data):
        buffer.receive(block)
    assert buffer.reception_counts() == {1: 2}
    assert buffer.preprocessors[1].samples == 3  # 帧 30 起收样本（29 帧暖机）
    buffer.reset_preprocessing()
    assert buffer.preprocessors[1].samples == 0
    buffer.set_direction(1)
    assert buffer.reception_counts() == {1: 2}
    assert LiveBuffer(clock=clock).reception_counts() == {}  # reconnect = new buffer


def test_counter_prunes_during_continuous_reception():
    rate = ReceptionRate()
    for second in range(100):
        for key in range(1, 15):
            rate.record(key, second)
    assert rate.counts(99) == dict.fromkeys(range(1, 15), 1)
    assert sum(map(len, rate.arrivals.values())) == 14


@pytest.fixture
def window(app):
    window = MainWindow()
    yield window
    window.worker = None
    window.close()
    window.deleteLater()


def test_rate_bar_total_details_and_silence_update_without_snapshot(window):
    clock = Clock()
    buffer = LiveBuffer(clock=clock)
    window.worker = SimpleNamespace(buffer=buffer)
    window.accept_scans = True
    for key in range(1, 15):
        for _ in range(key):
            buffer.receive(Block(key, [[2500]]))
    window._refresh_reception_rate()
    bar = window.reception_bar
    assert bar.total_label.text() == "接收 105 块帧/秒"
    assert bar.block_button.text() == "各块 1–14 帧/秒"
    assert len(bar.menu.actions()) == 14
    assert bar.menu.actions()[-1].text() == "BLOCK 14    14 帧/秒"
    clock.now = 1.1
    window._refresh_reception_rate()
    assert bar.total_label.text() == "接收 0 块帧/秒"
    assert bar.counts == dict.fromkeys(range(1, 15), 0)
    buffer.receive(Block(7, [[2400]]))
    window._refresh_reception_rate()
    assert bar.total_label.text() == "接收 1 块帧/秒"
    assert bar.counts[7] == 1


def test_idle_demo_disconnect_and_failure_never_show_old_rate(window):
    bar = window.reception_bar
    assert bar.counts == {} and not bar.block_button.isEnabled()
    window.controls.demo_toggle.setChecked(True)
    assert "模拟模式" in bar.note.text() and bar.counts == {}
    window.controls.demo_toggle.setChecked(False)
    bar.set_counts({1: 30})
    window.worker = SimpleNamespace(stop=lambda: None)
    window.accept_scans = True
    window.reception_timer.start()
    window._connect_hardware()
    assert not window.reception_timer.isActive()
    assert bar.total_label.text() == "接收 0 块帧/秒"
    assert bar.note.text() == "采集已停止"
    bar.set_counts({1: 30})
    window._serial_failed("test error")
    assert bar.counts == {} and bar.note.text() == "接收中断"
