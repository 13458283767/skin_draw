from threading import Event

import numpy as np
import pytest

from skin_draw.acquisition import SerialAcquisition
from skin_draw.binary_protocol import binary_frame_length, decode_binary_frame
from skin_draw.live import LiveBuffer
from skin_draw.serial_protocol import SerialBlockParser
from skin_draw.ui.main_window import MainWindow
from test_acquisition import FakeClock, FakePort
from test_live_buffer import StreamPort


EXAMPLE = bytes.fromhex("AA 01 03 03 08 00 04 00 02 00 0B B8 09 C4 07 D0 03 E8 05 DC 09 C3 0A")
NOT_READY = bytes.fromhex("AA 01 00 00 0A")


def binary(key, values=None):
    if values is None:
        head = bytes((0xAA, key, 0, 0))
    else:
        values = np.asarray(values, dtype=">u2")
        head = bytes((0xAA, key, *values.shape)) + values.tobytes()
    return head + b"\x0a"


def text(key, values):
    values = np.asarray(values)
    rows, cols = values.shape
    return (f"BLOCK_{key},{rows},{cols}:\r\n" +
            "".join(",".join(map(str, row)) + "\r\n" for row in values)).encode()


def test_v05_document_examples_length_endianness_and_not_ready():
    assert len(EXAMPLE) == binary_frame_length(EXAMPLE[:4]) == 23
    assert len(NOT_READY) == binary_frame_length(NOT_READY[:4]) == 5
    assert decode_binary_frame(EXAMPLE).values.tolist() == [
        [2048, 1024, 512], [3000, 2500, 2000], [1000, 1500, 2499]]
    assert decode_binary_frame(NOT_READY) is None


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 17, 517, 4096])
def test_mixed_formats_all_chunk_sizes_and_payload_delimiters(chunk):
    # AA, CR and LF in payload bytes must never be mistaken for framing.
    values = [[0xAAA, 0x10A, 0x20D], [0, 4095, 2500]]
    large = np.arange(256).reshape(16, 16)
    data = text(2, [[2500]]) + b"END\r\n" + binary(3, values) + NOT_READY + EXAMPLE + binary(16, large) + text(14, [[2000]])
    parser = SerialBlockParser()
    blocks = []
    for offset in range(0, len(data), chunk):
        blocks.extend(parser.feed(data[offset:offset + chunk]))
    assert [b.block_id for b in blocks] == [2, 3, 1, 16, 14]
    np.testing.assert_array_equal(blocks[1].values, values)
    np.testing.assert_array_equal(blocks[3].values, large)
    assert parser.not_ready_ids == {1}
    assert parser.counts == {"text": 2, "binary": 4}
    assert parser.errors == 0


@pytest.mark.parametrize("mode", ["auto", "binary"])
@pytest.mark.parametrize("tail", [b"\x00", b"\x0d", b"\xaa", b""])
def test_tail_check_is_mandatory_and_following_frame_survives(mode, tail):
    parser = SerialBlockParser(mode)
    assert len(parser.feed(EXAMPLE[:-1] + tail + EXAMPLE)) == 1
    assert parser.error_counts["二进制帧尾错误（应为0x0A）"] >= 1


@pytest.mark.parametrize("packet", [EXAMPLE, NOT_READY])
def test_missing_tail_never_publishes_until_complete(packet):
    parser = SerialBlockParser()
    assert parser.feed(packet[:-1]) == []
    assert parser.binary_ends == 0 and not parser.not_ready_ids
    blocks = parser.feed(packet[-1:])
    assert parser.errors == 0 and parser.binary_ends == 1
    assert len(blocks) == (1 if packet is EXAMPLE else 0)
    assert parser.not_ready_ids == (set() if packet is EXAMPLE else {1})


def test_invalid_not_ready_tail_does_not_detect_device():
    parser = SerialBlockParser()
    assert parser.feed(NOT_READY[:-1] + b"\x00") == []
    assert not parser.not_ready_ids and parser.binary_ends == 0
    parser.feed(NOT_READY)
    assert parser.not_ready_ids == {1}


@pytest.mark.parametrize("packet", [b"", EXAMPLE[:-1], EXAMPLE + b"\x0a",
                                   EXAMPLE[:-1] + b"\x00", NOT_READY[:-1] + b"\x00"])
def test_decoder_rejects_invalid_length_or_tail(packet):
    with pytest.raises(ValueError):
        decode_binary_frame(packet)


@pytest.mark.parametrize("bad", [
    b"\xaa\x00\x01\x01", b"\xaa\x11\x01\x01", b"\xaa\x01\x00\x01",
    b"\xaa\x01\x01\x00", b"\xaa\x01\x11\x01", binary(1, [[4096]]),
    EXAMPLE[:12] + EXAMPLE[13:], EXAMPLE[:12] + b"\xff" + EXAMPLE[12:],
])
def test_bad_headers_values_deleted_and_inserted_bytes_resynchronize(bad):
    parser = SerialBlockParser()
    blocks = parser.feed(bad + EXAMPLE)
    assert len(blocks) == 1 and blocks[0].values[0, 0] == 2048
    assert parser.errors >= 1


def test_partial_frame_timeout_idle_recovery_and_bounded_noise():
    clock = FakeClock()
    parser = SerialBlockParser(clock=clock)
    assert parser.feed(b"\xaa\x01\x10\x10\x01") == []
    clock.now = .3
    assert parser.feed(b"") == []
    assert parser.errors == 1
    assert len(parser.feed(EXAMPLE)) == 1
    parser.feed(b"x" * 100000 + b"\xaa\x01")
    assert len(parser.buffer) <= 517 and len(parser.text.buffer) <= 512
    parser.reset()
    assert len(parser.feed(EXAMPLE)) == 1


def test_max_frame_can_arrive_at_115200_without_timeout():
    clock = FakeClock()
    parser = SerialBlockParser(clock=clock)
    packet = binary(16, np.full((16, 16), 2500))
    blocks = []
    for byte in packet:
        clock.now += 10 / 115200
        blocks.extend(parser.feed(bytes([byte])))
    assert len(packet) == 517 and len(blocks) == 1 and parser.errors == 0


def test_every_split_preserves_mixed_shape_fourteen_block_scan():
    # Include absent-sensor zeros, ADC LF/AA bytes and all device IDs in order.
    shapes = [(5, 3), (10, 5), (13, 5), (3, 5), (16, 16)]
    expected = {key: (np.arange(np.prod(shapes[(key - 1) % 5])) * 10).reshape(shapes[(key - 1) % 5])
                for key in range(1, 15)}
    expected[1][0, 1] = 0x0AAA
    data = b"".join(binary(key, values) for key, values in expected.items())
    for split in range(len(data) + 1):
        parser = SerialBlockParser("binary")
        blocks = parser.feed(data[:split]) + parser.feed(data[split:])
        assert [b.block_id for b in blocks] == list(expected)
        assert parser.errors == 0 and not parser.buffer
        for block in blocks:
            np.testing.assert_array_equal(block.values, expected[block.block_id])


def test_polling_v05_fourteen_devices_preserves_order_and_drops_wrong_id(caplog):
    clock, stop = FakeClock(), Event()
    port = FakePort(clock, tuple(range(1, 15)))
    scans, samples = [], []

    def write(command):
        assert not port.pending
        port.commands.append(command)
        key = int(command.decode().split()[2])
        if key in port.ids:
            # A non-target reply must not be delivered, even with a valid tail.
            if key == 1:
                port.pending.extend(binary(16, [[2000]]))
            shape = (5, 3) if key % 2 else (10, 5)
            port.pending.extend(binary(key, np.full(shape, 2500)))
        return len(command)

    def scan(value):
        scans.append(value)
        if len(scans) == 2:
            stop.set()

    port.write = write
    reader = SerialAcquisition("FAKE", stop, scan, lambda _: None, factory=lambda _: port,
                               clock=clock, protocol="binary", response_timeout=.2,
                               on_block=samples.append)
    reader.run()
    assert all(s.complete and s.expected_ids == tuple(range(1, 15)) for s in scans)
    assert [b.block_id for b in samples] == list(range(1, 15)) * 2
    assert port.commands[16:] == [f"hello num {key}\r\n".encode() for key in range(1, 15)]
    assert "轮询收到非目标块应答" in caplog.text
    assert reader.parser.errors == 0 and port.closed


def test_lf_tail_cannot_detect_in_range_payload_corruption():
    # Document the intentional protocol limitation; this is not a CRC check.
    packet = bytearray(binary(1, [[2500]]))
    packet[5] ^= 1
    assert decode_binary_frame(packet).values.tolist() == [[2501]]


def test_explicit_format_modes():
    for mode, packet in (("text", text(2, [[2300]])), ("binary", binary(2, [[2300]]))):
        parser = SerialBlockParser(mode)
        assert parser.feed(packet)[0].values.tolist() == [[2300]]
    parser = SerialBlockParser("binary")
    assert parser.feed(text(2, [[2300]])) == []
    assert len(parser.feed(EXAMPLE)) == 1


@pytest.mark.parametrize("mode", ["auto", "stream"])
def test_mixed_stream_fourteen_blocks_calibrates_then_filters_and_draws(app, mode):
    clock, stop = FakeClock(), Event()
    chunks = [NOT_READY]
    for sequence in range(31):
        for key in range(1, 15):
            values = [[2500, 2500], [2500, 2500]] if sequence < 30 else [[2000, 1000], [2000, 2200]]
            chunks.append((binary if key % 2 else text)(key, values))
    port = StreamPort(b"".join(chunks), clock, chunk=37)
    buffer = LiveBuffer(calibration_seconds=0, clock=clock)

    def receive(block):
        buffer.receive(block)
        if buffer.sequence == 14 * 31:
            stop.set()

    reader = SerialAcquisition("FAKE", stop, lambda scan: buffer.finish_scan(scan, False),
                               lambda _: None, factory=lambda _: port, clock=clock,
                               mode=mode, on_block=receive)
    reader.run()
    snapshot = buffer.take()
    assert snapshot.calibration.counts == dict.fromkeys(range(1, 15), 30)
    assert not port.commands and port.closed
    assert reader.active_mode == "stream"
    window = MainWindow()
    try:
        window._consume_scan(snapshot.scan, snapshot)
        assert len(window.heatmap.items) == 14
        for item in window.heatmap.items.values():
            np.testing.assert_array_equal(item.values, [[0, 0], [500, 300]])
    finally:
        window.close()
        window.deleteLater()


def test_polling_keeps_not_ready_device_and_retries_with_unchanged_command():
    clock, stop = FakeClock(), Event()
    port = FakePort(clock, (1, 2))
    scans, samples = [], []

    def write(command):
        assert not port.pending
        port.commands.append(command)
        key = int(command.decode().split()[2])
        port.counts[key] = port.counts.get(key, 0) + 1
        if key == 1:
            port.pending.extend(NOT_READY if port.counts[key] == 1 else binary(1, [[2500]]))
        if key == 2:
            port.pending.extend(text(2, [[2300]]))
        return len(command)

    def scan(value):
        scans.append(value)
        if len(scans) == 2:
            stop.set()

    port.write = write
    reader = SerialAcquisition("FAKE", stop, scan, lambda _: None, factory=lambda _: port,
                               clock=clock, response_timeout=.2, on_block=samples.append)
    reader.run()
    assert scans[0].expected_ids == (1, 2) and not scans[0].complete
    assert scans[1].complete
    assert port.commands[-2:] == [b"hello num 1\r\n", b"hello num 2\r\n"]
    assert [b.block_id for b in samples] == [2, 1, 2]


def test_bad_tail_does_not_count_as_calibration_sample():
    parser = SerialBlockParser()
    buffer = LiveBuffer(calibration_seconds=0)
    packet = binary(1, [[2500]])
    for block in parser.feed(packet * 29 + packet[:-1] + b"\x00"):
        buffer.receive(block)
    assert buffer.calibration.counts == {1: 29}
    assert not buffer.calibration.ready
    for block in parser.feed(packet):
        buffer.receive(block)
    assert buffer.calibration.ready
