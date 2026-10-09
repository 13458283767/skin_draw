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


TOTAL_FRAMES = 94  # 59 帧暖机+标定，随后 35 帧按压越过 M3 窗口滞后


def _quiet_press(sequence):
    return 2500 if sequence <= 59 else 2000


@pytest.mark.parametrize("ids", [(1,), (2, 7), tuple(range(1, 15))])
@pytest.mark.parametrize("mode", ["stream", "auto"])
def test_stream_calibrates_all_first_thirty_samples_before_first_paint(ids, mode):
    data = b"".join(f"BLOCK_{key},1,2:\r\n{value},{value}\r\n".encode()
                    for sequence in range(1, TOTAL_FRAMES + 1)
                    for key in ids
                    for value in [_quiet_press(sequence)])
    clock, stop = Clock(), Event()
    port = StreamPort(data, clock)
    buffer = LiveBuffer(clock=clock, calib_n=30, norm="A")

    def receive(block):
        buffer.receive(block)
        if buffer.sequence == TOTAL_FRAMES * len(ids):
            stop.set()

    reader = SerialAcquisition("FAKE", stop, lambda scan: buffer.finish_scan(scan, False),
                               lambda _: None, factory=lambda _: port, clock=clock,
                               mode=mode, on_block=receive)
    reader.run()
    snapshot = buffer.take()
    assert snapshot.received_blocks == TOTAL_FRAMES * len(ids)
    assert snapshot.preprocess.samples == dict.fromkeys(ids, 30)
    assert len(snapshot.layout_blocks) == len(ids)
    for baseline in snapshot.preprocess.baselines.values():
        np.testing.assert_array_equal(baseline, [[2500, 2500]])
    for response in snapshot.display_values.values():
        np.testing.assert_allclose(response, [[500, 500]])     # 2500 基线 - 2000 按压
    assert not port.commands and port.closed
    assert buffer.take() is None


def test_slow_gui_gets_latest_per_block_without_backlog_or_recalibrating():
    clock = Clock()
    buffer = LiveBuffer(clock=clock, calib_n=30, norm="A")
    for sequence in range(1, 181):
        clock.now += .01
        for key in (2, 7):
            buffer.receive(Block(key, [[2500 if sequence <= 90 else 1600]]))
    snapshot = buffer.take()
    assert snapshot.received_blocks == 360
    assert len(snapshot.scan.frame.blocks) == 2
    assert snapshot.preprocess.samples == {2: 30, 7: 30}
    assert snapshot.rates[2] == pytest.approx(100)
    assert all(block.values[0, 0] == 1600 for block in snapshot.scan.frame.blocks)
    assert snapshot.preprocess.baselines[2][0, 0] == 2500
    np.testing.assert_allclose(snapshot.display_values[2], [[900]])
    assert buffer.take() is None
    # Snapshots expose read-only baselines that survive a worker-side reset.
    with pytest.raises(ValueError):
        snapshot.preprocess.baselines[2][0, 0] = 0
    buffer.reset_preprocessing()
    assert not buffer.preprocessors[2].ready
    assert snapshot.preprocess.baselines[2][0, 0] == 2500


def test_recalibrate_uses_next_received_samples_and_direction_keeps_baseline():
    clock = Clock()
    buffer = LiveBuffer(clock=clock, calib_n=1, norm="A")
    for _ in range(30):
        buffer.receive(Block(1, [[2000]]))
    buffer.reset_preprocessing()
    snapshot = buffer.take()
    assert not snapshot.preprocess.ready
    assert snapshot.preprocess.samples == {1: 0}
    for _ in range(30):
        clock.now += .01
        buffer.receive(Block(1, [[2500]]))
    buffer.set_direction(1)
    for _ in range(35):
        clock.now += .01
        buffer.receive(Block(1, [[2700]]))
    snapshot = buffer.take()
    assert snapshot.preprocess.ready
    assert snapshot.preprocess.direction == 1
    assert snapshot.preprocess.baselines[1][0, 0] == 2500
    np.testing.assert_allclose(snapshot.display_values[1], [[200]])   # rise: 2700-2500


def test_missing_counts_follow_acquisition_not_number_of_paints():
    clock = Clock()
    buffer = LiveBuffer(clock=clock)
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
