import numpy as np

from skin_draw.models import Block, Frame
from skin_draw.processing import BaselineCalibration


def frame(sequence, value, include_second=True, shape=(2, 3)):
    blocks = [Block(1, np.full(shape, value))]
    if include_second:
        blocks.append(Block(14, [[1000]]))
    return Frame(tuple(blocks), sequence)


def test_exact_thirty_frame_mean_and_direction_without_unsigned_underflow():
    calibration = BaselineCalibration(minimum_seconds=0)
    for sequence in range(1, 30):
        assert calibration.consume(frame(sequence, 2000 + sequence), (1, 14)) == {}
        assert not calibration.ready
    calibration.consume(frame(30, 2030), (1, 14))
    assert calibration.ready and calibration.count == 30
    np.testing.assert_allclose(calibration.baseline[1], 2015.5)
    raw = frame(31, 1700)
    response = calibration.consume(raw, (1, 14))
    np.testing.assert_allclose(response[1], 315.5)
    assert raw.blocks[0].values[0, 0] == 1700
    np.testing.assert_allclose(calibration.signed_delta(1, raw.blocks[0].values), -315.5)
    baseline = calibration.baseline[1].copy()
    calibration.set_direction(1)
    np.testing.assert_array_equal(calibration.project(raw)[1], np.zeros((2, 3)))
    np.testing.assert_allclose(calibration.project(frame(32, 2200))[1], 184.5)
    np.testing.assert_array_equal(calibration.baseline[1], baseline)
    assert calibration.count == 30


def test_missing_duplicate_and_empty_frames_do_not_count():
    calibration = BaselineCalibration(minimum_seconds=0)
    calibration.consume(frame(1, 2000), (1, 14))
    calibration.consume(frame(1, 2000), (1, 14))
    calibration.consume(frame(2, 2000, include_second=False), (1, 14))
    calibration.consume(Frame((), 3), (1, 14))
    assert calibration.count == 0 and not calibration.ready
    for sequence in range(4, 33):
        calibration.consume(frame(sequence, 2000), (1, 14))
    assert calibration.count == 30 and calibration.ready
    calibration.reset()
    assert calibration.count == 0 and not calibration.ready


def test_shape_and_id_changes_require_new_calibration():
    calibration = BaselineCalibration(2, minimum_seconds=0)
    calibration.consume(frame(1, 2000), (1, 14))
    calibration.consume(frame(2, 2000), (1, 14))
    assert calibration.ready
    response = calibration.consume(frame(3, 2000, shape=(3, 2)), (1, 14))
    assert set(response) == {14}  # unaffected blocks keep their completed baseline
    assert calibration.count == 0 and not calibration.ready
    calibration.consume(frame(4, 2000, shape=(3, 2)), (1, 14))
    assert calibration.ready
    calibration.consume(frame(5, 2000, include_second=False, shape=(3, 2)), (1,))
    assert calibration.ready and calibration.count == 2
    assert set(calibration.baseline) == {1}  # removing another block preserves this baseline


def test_each_connected_block_counts_its_own_valid_frames():
    calibration = BaselineCalibration(minimum_seconds=0)
    for sequence in range(1, 31):
        blocks = [Block(1, [[2000 + sequence]])]
        if sequence % 2 == 0:
            blocks.append(Block(7, [[1000]]))
        calibration.consume(Frame(tuple(blocks), sequence), (1, 7))
    assert calibration.counts == {1: 30, 7: 15}
    assert not calibration.ready
    assert calibration.count == 15
    np.testing.assert_allclose(calibration.baseline[1], 2015.5)
    for sequence in range(31, 46):
        calibration.consume(Frame((Block(7, [[1000]]),), sequence), (1, 7))
    assert calibration.ready and calibration.counts == {1: 30, 7: 30}
    assert set(calibration.baseline) == {1, 7}
    np.testing.assert_allclose(calibration.baseline[1], 2015.5)
