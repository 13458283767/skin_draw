from threading import Event

import numpy as np
import pytest

from skin_draw.acquisition import SerialAcquisition
from skin_draw.live import LiveBuffer

QUIET_ROUNDS = 90     # polling rounds at 2300 before the press
PRESS_ROUNDS = 35     # rounds at 1800 after the press (M3 latency is 29 frames)
TOTAL_SCANS = 1 + QUIET_ROUNDS + PRESS_ROUNDS


class FakeClock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now


class FakePort:
    def __init__(self, clock, ids=(1, 14), fail=False, quiet_requests=30):
        self.clock = clock
        self.ids = ids
        self.quiet_requests = quiet_requests
        self.pending = bytearray()
        self.commands = []
        self.counts = {}
        self.closed = False
        self.fail = fail

    def reset_input_buffer(self):
        self.pending.clear()

    @property
    def in_waiting(self):
        return len(self.pending)

    def write(self, command):
        assert not self.pending, "must finish/drain the previous reply before another request"
        self.commands.append(command)
        block_id = int(command.decode().split()[2])
        self.counts[block_id] = self.counts.get(block_id, 0) + 1
        if block_id in self.ids:
            value = 2300 if self.counts[block_id] <= self.quiet_requests else 1800
            # Mixed shapes and optional END in the same scan.
            if block_id % 2:
                text = f"BLOCK_{block_id},2,2:\r\n{value},{value}\r\n{value},{value}\r\nEND\r\n"
            else:
                text = f"BLOCK_{block_id},1,3:\r\n{value},{value},{value}\r\n"
            self.pending.extend(text.encode())
        return len(command)

    def read(self, count):
        if self.fail:
            raise OSError("USB removed")
        self.clock.now += .01
        chunk = bytes(self.pending[:min(count, 17)])
        del self.pending[:len(chunk)]
        return chunk

    def close(self):
        self.closed = True


@pytest.mark.parametrize("ids", [(1,), (2, 7), tuple(range(1, 15))])
def test_devices_discovery_to_auto_baseline_to_press(ids):
    stop, clock = Event(), FakeClock()
    port = FakePort(clock, ids, quiet_requests=1 + QUIET_ROUNDS)
    live = LiveBuffer(calib_n=30, norm="A")
    # Deterministic cadence: the real cadence would derive upd_n from the
    # measured rate; here the press window must stay below one slow-baseline
    # update so the frozen 2300 baseline yields exactly 500 on display.
    live._measured_fps = lambda key: 100.0
    scans = []

    def on_scan(scan):
        scans.append(scan)
        for block in scan.frame.blocks:
            live.receive(block)
        if len(scans) == TOTAL_SCANS:
            stop.set()

    SerialAcquisition("FAKE", stop, on_scan, lambda _: None, factory=lambda _: port,
                      response_timeout=.2, quiet_time=.01, clock=clock).run()
    assert len(scans) == TOTAL_SCANS and all(scan.complete for scan in scans)
    assert scans[0].expected_ids == ids
    assert len(port.commands) == 16 + (TOTAL_SCANS - 1) * len(ids)
    assert port.commands[0] == b"hello num 1\r\n"
    assert port.commands[15] == b"hello num 16\r\n"
    for key in ids:
        assert live.preprocessors[key].ready
        np.testing.assert_allclose(live.display_values[key], 500)
    assert port.closed


def test_unplug_closes_port_and_raises():
    clock, stop = FakeClock(), Event()
    port = FakePort(clock, fail=True)
    with pytest.raises(OSError, match="USB removed"):
        SerialAcquisition("FAKE", stop, lambda _: None, lambda _: None,
                          factory=lambda _: port, clock=clock).run()
    assert port.closed


def test_stop_during_discovery_publishes_no_partial_scan():
    clock, stop = FakeClock(), Event()
    port = FakePort(clock)
    scans = []
    def status(message):
        if "3/16" in message:
            stop.set()
    SerialAcquisition("FAKE", stop, scans.append, status, factory=lambda _: port,
                      response_timeout=.2, quiet_time=.01, clock=clock).run()
    assert scans == [] and port.closed


def test_block_is_delivered_before_other_device_discovery_or_settle():
    clock, stop = FakeClock(), Event()
    port = FakePort(clock, (1,))
    samples = []

    def receive(block):
        samples.append(block)
        assert port.commands == [b"hello num 1\r\n"]
        stop.set()

    SerialAcquisition("FAKE", stop, lambda _: None, lambda _: None,
                      factory=lambda _: port, clock=clock, on_block=receive).run()
    assert len(samples) == 1 and port.closed


@pytest.mark.parametrize("ending", [b"", b"END\r\n"])
def test_short_successful_reply_has_no_thirty_ms_idle_wait(ending):
    clock, stop = FakeClock(), Event()
    port = FakePort(clock, (1,))
    reader = SerialAcquisition("FAKE", stop, lambda _: None, lambda _: None,
                               factory=lambda _: port, clock=clock)

    def write(command):
        port.pending.extend(b"BLOCK_1,1,1:\r\n2000\r\n" + ending)
        return len(command)

    def read(count):
        if not port.pending:
            clock.now += .002
        result = bytes(port.pending[:count])
        del port.pending[:len(result)]
        return result

    port.write, port.read = write, read
    assert reader._request(port, 1).values.tolist() == [[2000]]
    assert not port.pending
    assert clock.now <= .0041
