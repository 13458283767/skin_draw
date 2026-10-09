from threading import Event

import numpy as np
import pytest

from skin_draw.acquisition import Scan, SerialAcquisition
from skin_draw.live import LiveBuffer
from skin_draw.models import Block, Frame


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class StreamPort:
    def __init__(self, data, clock, chunk=97):
        self.data = bytearray(data)
        self.clock, self.chunk = clock, chunk
        self.commands = []
        self.closed = False

    @property
    def in_waiting(self):
        return min(self.chunk, len(self.data))

    def read(self, size):
        self.clock.now += .0001 if self.data else .002
        result = bytes(self.data[:size])
        del self.data[:len(result)]
        return result

    def write(self, data):
        self.commands.append(data)
        raise AssertionError("continuous reception must never send polling commands")

    def reset_input_buffer(self):
        raise AssertionError("must not discard the initial calibration samples")

    def close(self):
        self.closed = True


@pytest.mark.parametrize("ids", [(1,), (2, 7), tuple(range(1, 15))])
@pytest.mark.parametrize("mode", ["stream", "auto"])
def test_stream_calibrates_all_first_thirty_samples_before_first_paint(ids, mode):
    data = b"".join(f"BLOCK_{key},1,2:\r\n{value},{value}\r\n".encode()
                    for sequence in range(1, 32)
                    for key in ids
                    for value in [2000 + sequence if sequence <= 30 else 1600])
    clock, stop = Clock(), Event()
    port = StreamPort(data, clock)
    buffer = LiveBuffer(calibration_seconds=0, clock=clock)

    def receive(block):
        buffer.receive(block)
        if buffer.sequence == 31 * len(ids):
            stop.set()

    reader = SerialAcquisition("FAKE", stop, lambda scan: buffer.finish_scan(scan, False),
                               lambda _: None, factory=lambda _: port, clock=clock,
                               mode=mode, on_block=receive)
    reader.run()
    snapshot = buffer.take()
    assert snapshot.received_blocks == 31 * len(ids)
    assert snapshot.calibration.counts == dict.fromkeys(ids, 30)
    assert len(snapshot.layout_blocks) == len(ids)
    for baseline in snapshot.calibration.baseline.values():
        np.testing.assert_array_equal(baseline, [[2015.5, 2015.5]])
    for response in snapshot.calibration.project(snapshot.scan.frame).values():
        np.testing.assert_array_equal(response, [[415.5, 415.5]])
    assert not port.commands and port.closed
    assert buffer.take() is None


def test_slow_gui_gets_latest_per_block_without_backlog_or_recalibrating():
    clock = Clock()
    buffer = LiveBuffer(calibration_seconds=0, clock=clock)
    for sequence in range(1, 1001):
        clock.now += .01
        for key in (2, 7):
            buffer.receive(Block(key, [[2000 + sequence if sequence <= 30 else 1600]]))
    snapshot = buffer.take()
    assert snapshot.received_blocks == 2000
    assert len(snapshot.scan.frame.blocks) == 2
    assert snapshot.calibration.counts == {2: 30, 7: 30}
    assert snapshot.rates[2] == pytest.approx(100)
    assert all(block.values[0, 0] == 1600 for block in snapshot.scan.frame.blocks)
    assert snapshot.calibration.baseline[2][0, 0] == 2015.5
    assert buffer.take() is None
    # Snapshot and worker baseline state must not share mutable dictionaries.
    snapshot.calibration.reset()
    assert buffer.calibration.ready


def test_recalibrate_uses_next_received_samples_and_direction_keeps_baseline():
    buffer = LiveBuffer(calibration_seconds=0)
    for _ in range(30):
        buffer.receive(Block(1, [[2000]]))
    buffer.reset_calibration()
    assert buffer.take().calibration.count == 0
    for _ in range(30):
        buffer.receive(Block(1, [[2500]]))
    buffer.set_direction(1)
    buffer.receive(Block(1, [[2700]]))
    snapshot = buffer.take()
    assert snapshot.calibration.count == 30
    assert snapshot.calibration.baseline[1][0, 0] == 2500
    assert snapshot.calibration.project(snapshot.scan.frame)[1][0, 0] == 200


def test_missing_counts_follow_acquisition_not_number_of_paints():
    clock = Clock()
    buffer = LiveBuffer(calibration_seconds=0, clock=clock)
    buffer.receive(Block(2, [[2000]]))
    buffer.receive(Block(7, [[2000]]))
    buffer.finish_scan(Scan(Frame((Block(2, [[2000]]),), 1), (2, 7), .01))
    for _ in range(20):
        buffer.receive(Block(2, [[1500]]))
        snapshot = buffer.take()
        assert snapshot.missing_counts[7] == 1
    buffer.receive(Block(7, [[1600]]))
    assert buffer.take().missing_counts[7] == 0
    clock.now += 4
    assert buffer.take().missing_counts == {2: 1, 7: 1}
    clock.now += 4
    snapshot = buffer.take()
    assert snapshot.missing_counts == {2: 3, 7: 3}
    assert not snapshot.scan.frame.blocks
    assert len(snapshot.layout_blocks) == 2
