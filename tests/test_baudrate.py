import logging
from threading import Event
from unittest.mock import Mock

import pytest

from skin_draw.acquisition import SerialAcquisition
from skin_draw.transport import BAUDRATES
from skin_draw.ui.main_window import MainWindow


@pytest.mark.parametrize("baudrate", BAUDRATES)
def test_selected_rate_reaches_serial_driver_and_log(monkeypatch, caplog, baudrate):
    """Exercise the real acquisition -> transport path without opening hardware."""
    caplog.set_level(logging.INFO, logger="skin_draw")
    port = Mock()
    driver = Mock(return_value=port)
    monkeypatch.setattr("skin_draw.transport.serial.Serial", driver)
    stop = Event()
    stop.set()
    SerialAcquisition("FAKE", stop, lambda _: None, lambda _: None,
                      mode="stream", baudrate=baudrate).run()
    driver.assert_called_once_with(port="FAKE", baudrate=baudrate, bytesize=8, parity="N",
                                   stopbits=1, timeout=.002, write_timeout=.5,
                                   xonxoff=False, rtscts=False, dsrdtr=False)
    port.close.assert_called_once()
    assert f'"baud": {baudrate}' in caplog.text


def test_baudrate_options_and_default(app):
    window = MainWindow()
    try:
        combo = window.controls.baudrate
        assert [combo.itemData(i) for i in range(combo.count())] == [115200, 230400, 460800, 921600, 1843200]
        assert combo.currentData() == 115200
        assert combo.isEnabled()
    finally:
        window.close()
