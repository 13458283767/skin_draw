"""RS485 USB virtual serial port access; no application or Qt state."""

import serial
from serial.tools import list_ports

DEFAULT_BAUDRATE = 115200
BAUDRATES = tuple(DEFAULT_BAUDRATE * factor for factor in (1, 2, 4, 8, 16))


def available_ports() -> list[tuple[str, str]]:
    return [(port.device, port.description) for port in list_ports.comports()]


def open_port(name: str, baudrate: int = DEFAULT_BAUDRATE):
    # Expect an auto-direction USB-RS485 adapter. Software RTS timing is not
    # enabled implicitly; different adapter hardware may require configuration.
    return serial.Serial(port=name, baudrate=baudrate, bytesize=8, parity="N",
                         stopbits=1, timeout=.002, write_timeout=.5,
                         xonxoff=False, rtscts=False, dsrdtr=False)
