"""Qt adapter around the testable, UI-independent acquisition loop."""

from threading import Event, Lock
import logging

from PySide6.QtCore import QThread, Signal

from ..acquisition import SerialAcquisition
from ..live import LiveBuffer
from ..transport import DEFAULT_BAUDRATE


class SerialWorker(QThread):
    status_changed = Signal(str)
    failed = Signal(str)
    data_ready = Signal()

    def __init__(self, port: str, parent=None, mode="auto", direction=-1,
                 protocol="auto", baudrate=DEFAULT_BAUDRATE, *, raw_adc=False, norm="B"):
        super().__init__(parent)
        self.port = port
        self.mode = mode
        self.protocol = protocol
        self.baudrate = baudrate
        self.buffer = LiveBuffer(direction, raw_adc=raw_adc, norm=norm)
        self.stop_event = Event()
        self._notification_lock = Lock()
        self._notification_pending = False

    def _notify_data(self):
        with self._notification_lock:
            if self._notification_pending:
                return
            self._notification_pending = True
        self.data_ready.emit()

    def acknowledge_update(self):
        # The GUI acknowledges BEFORE taking a snapshot: a concurrent arrival
        # is either included in that snapshot or queues the next notification.
        with self._notification_lock:
            self._notification_pending = False

    def _receive_block(self, block):
        self.buffer.receive(block)
        self._notify_data()

    def _finish_scan(self, scan, authoritative):
        self.buffer.finish_scan(scan, authoritative)
        self._notify_data()

    def stop(self):
        logging.getLogger(__name__).info("请求停止串口采集 port=%s", self.port)
        self.stop_event.set()

    def run(self):
        try:
            reader = SerialAcquisition(self.port, self.stop_event,
                                       lambda scan: self._finish_scan(scan, reader.active_mode == "poll"),
                                       self.status_changed.emit, on_block=self._receive_block, mode=self.mode,
                                       protocol=self.protocol, baudrate=self.baudrate)
            reader.run()
        except Exception as error:
            self.failed.emit(f"串口采集失败：{error}")
