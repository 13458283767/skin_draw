import logging
from pathlib import Path
from queue import Queue
from threading import Event

import pytest

from skin_draw.acquisition import SerialAcquisition
from skin_draw.diagnostics import BoundedQueueHandler, setup_logging, shutdown_logging
from skin_draw.live import LiveBuffer
from skin_draw.models import Block
from skin_draw.serial_diagnostics import SerialDiagnostics
from skin_draw.serial_protocol import SerialBlockParser
from skin_draw.ui.main_window import MainWindow
from test_acquisition import FakeClock, FakePort


def test_file_rotation_utf8_and_idempotent_setup(tmp_path):
    shutdown_logging()
    try:
        path = setup_logging(tmp_path, max_bytes=1024, backups=2)
        assert setup_logging(tmp_path) == path
        logger = logging.getLogger("skin_draw.test")
        for index in range(80):
            logger.info("轮换检查 index=%s %s", index, "中文日志" * 20)
        try:
            raise OSError("模拟串口拔出")
        except OSError:
            logger.exception("异常堆栈检查")
    finally:
        shutdown_logging()
    files = list(tmp_path.glob("*.log*"))
    assert 1 <= len(files) <= 3
    content = path.read_text(encoding="utf-8")
    assert "模拟串口拔出" in content and "Traceback" in content
    assert "INFO" in "".join(f.read_text(encoding="utf-8") for f in files)


def test_full_log_queue_is_nonblocking_and_reports_dropped_records():
    queue = Queue(maxsize=1)
    handler = BoundedQueueHandler(queue)
    def record():
        return logging.makeLogRecord({"msg": "test", "levelno": logging.INFO})
    handler.handle(record())
    handler.handle(record())
    assert handler.dropped == 1
    queue.get_nowait()
    handler.handle(record())
    assert "丢弃 1 条" in queue.get_nowait().getMessage()
    assert handler.dropped == 0
    handler.close()


def test_silence_invalid_data_single_block_and_recovery_are_distinct(caplog):
    caplog.set_level(logging.INFO, logger="skin_draw")
    clock = FakeClock()
    parser = SerialBlockParser(clock=clock)
    diag = SerialDiagnostics("FAKE", clock)
    def feed(data):
        diag.received(data)
        diag.parsed(parser.feed(data), parser)
        diag.tick(parser, "stream")
    feed(b"BLOCK_1,1,1:\r\n2500\r\nBLOCK_2,1,1:\r\n2400\r\n")
    clock.now = 4
    feed(b"")
    clock.now = 4.5
    feed(b"")
    assert caplog.text.count("串口连续无字节") == 1
    assert caplog.text.count("皮肤块有效数据超时") == 2
    clock.now = 5
    feed(b"garbage\r\n")
    assert "串口字节接收恢复" in caplog.text
    assert "串口有字节但无有效传感器帧" in caplog.text
    clock.now = 6
    feed(b"BLOCK_1,1,1:\r\n2450\r\n")
    assert "有效数据帧恢复" in caplog.text and "皮肤块数据恢复" in caplog.text
    assert diag.missing == {2}
    assert "串口收发统计" in caplog.text


def test_corrupt_tail_and_partial_timeout_have_specific_reasons(caplog):
    caplog.set_level(logging.INFO, logger="skin_draw")
    clock = FakeClock()
    parser = SerialBlockParser(clock=clock)
    diag = SerialDiagnostics("FAKE", clock)
    bad = bytes.fromhex("AA 01 01 01 09 C4 00")
    diag.received(bad)
    diag.parsed(parser.feed(bad), parser)
    assert "二进制帧尾错误（应为0x0A）" in caplog.text
    assert "aa 01 01 01 09 c4 00" in caplog.text
    parser.feed(b"\xaa\x01")
    clock.now = 3
    diag.parsed(parser.feed(b""), parser)
    assert "二进制残帧超时" in caplog.text
    parser.feed(b"\r\nBLOCK_1,1,1:\r\n9999\r\n")
    assert parser.error_counts["明文ADC超过4095"] == 1
    counts = parser.error_counts
    parser.reset()
    assert parser.error_counts == counts


def test_continuous_invalid_data_is_rate_limited(caplog):
    caplog.set_level(logging.INFO, logger="skin_draw")
    clock = FakeClock()
    parser = SerialBlockParser(clock=clock)
    diag = SerialDiagnostics("FAKE", clock)
    for _ in range(500):
        data = b"BLOCK_1,1,1:\r\n9999\r\n"
        diag.received(data)
        diag.parsed(parser.feed(data), parser)
    assert parser.errors == 500
    assert caplog.text.count("串口解析异常") == 1
    diag.summary(parser, "stream")
    assert '"errors": 500' in caplog.text


@pytest.mark.parametrize("failure", ["open", "read"])
def test_open_failure_and_unplug_preserve_exception_and_session(caplog, failure):
    caplog.set_level(logging.INFO, logger="skin_draw")
    clock, stop = FakeClock(), Event()
    port = FakePort(clock, fail=True)
    def factory(_):
        if failure == "open":
            raise OSError("port access denied")
        return port
    reader = SerialAcquisition("FAKE", stop, lambda _: None, lambda _: None, factory=factory, clock=clock)
    with pytest.raises(OSError):
        reader.run()
    assert "串口采集异常" in caplog.text and "Traceback" in caplog.text
    assert reader.diagnostics.session in caplog.text
    assert '"reason": "exception"' in caplog.text
    if failure == "read":
        assert port.closed


def test_poll_timeout_recovery_and_manual_stop_not_reported_as_disconnect(caplog):
    caplog.set_level(logging.INFO, logger="skin_draw")
    clock, stop = FakeClock(), Event()
    port = FakePort(clock, ids=())
    reader = SerialAcquisition("FAKE", stop, lambda _: None, lambda _: None,
                               clock=clock, response_timeout=.05)
    assert reader._request(port, 1) is None
    assert '"rx_bytes_during_request": 0' in caplog.text
    port.ids = (1,)
    assert reader._request(port, 1) is not None
    assert "轮询应答恢复" in caplog.text
    caplog.clear()
    stop.set()
    assert reader._request(port, 1) is None
    assert "轮询应答超时" not in caplog.text


def test_preprocess_events_do_not_log_every_frame(caplog):
    caplog.set_level(logging.INFO, logger="skin_draw")
    clock = FakeClock()
    buffer = LiveBuffer(calib_n=30, norm="A", clock=clock)
    for _ in range(130):
        buffer.receive(Block(1, [[2500, 2500]]))
    assert caplog.text.count("皮肤块预处理已启动") == 1
    assert '"block_id": 1' in caplog.text
    for _ in range(100):
        buffer.receive(Block(1, [[1000, 2300]]))
    assert caplog.text.count("皮肤块预处理已启动") == 1  # 启动后每帧不再记录
    assert buffer.preprocessors[1].samples == 30


def test_logs_button_opens_resolved_directory(app, tmp_path, monkeypatch):
    from skin_draw.ui import main_window
    opened = []
    monkeypatch.setattr(main_window, "log_directory", lambda: tmp_path)
    monkeypatch.setattr(main_window.QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()) or True)
    window = MainWindow()
    try:
        window.logs_button.click()
        assert [Path(path) for path in opened] == [tmp_path]
    finally:
        window.close()
        window.deleteLater()
