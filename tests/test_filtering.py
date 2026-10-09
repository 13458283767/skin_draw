import numpy as np
import pytest

from skin_draw.filtering import ADCOutlierFilter
from skin_draw.live import LiveBuffer
from skin_draw.models import Block
from skin_draw.ui.main_window import MainWindow


@pytest.mark.parametrize("value,invalid", [(1200, False), (2800, False), (1199, True),
                                          (2801, True), (0, True), (4095, True)])
def test_jump_threshold_is_strict_and_signed(value, invalid):
    filter_ = ADCOutlierFilter()
    filter_.process(Block(1, [[2000]]))
    result, mask = filter_.process(Block(1, [[value]]))
    assert bool(mask[0, 0]) is invalid
    assert result.values[0, 0] == (2000 if invalid else value)


def test_any_jumping_point_rejects_only_its_row():
    filter_ = ADCOutlierFilter()
    filter_.process(Block(2, [[2500, 2500, 2500], [2700, 2700, 2700], [2400, 2400, 2400]]))
    source = Block(2, [[1003, 2100, 2400], [2600, 3900, 2300], [2000, 2100, 2200]])
    result, mask = filter_.process(source)
    assert result.values.tolist() == [[2500, 2500, 2500], [2700, 2700, 2700], [2000, 2100, 2200]]
    assert mask.tolist() == [[True, True, True], [True, True, True], [False, False, False]]
    assert source.values.tolist() == [[1003, 2100, 2400], [2600, 3900, 2300], [2000, 2100, 2200]]
    assert filter_.rejected_counts == {2: 6}


def test_repeated_corrupt_values_never_become_reference_and_recovery_is_immediate():
    filter_ = ADCOutlierFilter()
    filter_.process(Block(2, [[2500, 2500]]))
    for second in range(2000, 2010):
        result, mask = filter_.process(Block(2, [[1003, second]]))
        assert result.values.tolist() == [[2500, 2500]]
        assert mask.tolist() == [[True, True]]
    result, mask = filter_.process(Block(2, [[2400, 2100]]))
    assert result.values.tolist() == [[2400, 2100]] and not mask.any()
    assert filter_.rejected_counts == {2: 20}


@pytest.mark.parametrize("jump", [900, 950, 1000])
def test_block_four_jump_is_rejected_without_affecting_normal_press(jump):
    buffer = LiveBuffer(calibration_seconds=0)
    for _ in range(30):
        buffer.receive(Block(4, [[2500, 2500, 2500], [2500, 2500, 2500]]))
    buffer.receive(Block(4, [[2500, 2500, 2500 - jump], [2000, 2100, 2200]]))
    snapshot = buffer.take()
    assert snapshot.display_values[4].tolist() == [[0, 0, 0], [500, 400, 300]]
    assert snapshot.filtered_masks[4].tolist() == [[True] * 3, [False] * 3]
    buffer.receive(Block(4, [[2500, 2500, 2510], [2000, 2100, 2200]]))
    snapshot = buffer.take()
    assert not snapshot.filtered_masks[4].any()
    assert snapshot.display_values[4].tolist() == [[0, 0, 0], [500, 400, 300]]


def test_no_fixed_low_value_or_row_amplitude_rule():
    filter_ = ADCOutlierFilter()
    for value in (2500, 1700, 900, 100):
        source = Block(1, [[value, value]])
        result, mask = filter_.process(source)
        assert result is source and not mask.any()


def test_first_sample_reset_shape_change_and_block_ids_have_separate_references():
    filter_ = ADCOutlierFilter()
    first = Block(1, [[300]])
    assert filter_.process(first)[0] is first
    other = Block(2, [[2700]])
    assert filter_.process(other)[0] is other
    resized = Block(1, [[2500, 2500]])
    assert filter_.process(resized)[0] is resized
    filter_.reset_reference()
    fresh = Block(1, [[100, 100]])
    assert filter_.process(fresh)[0] is fresh


def test_alternating_corruption_waits_for_consistent_calibration_samples():
    buffer = LiveBuffer(calibration_seconds=0)
    for count in range(1, 31):
        buffer.receive(Block(1, [[2500, 2500]]))
        buffer.receive(Block(1, [[1003, 2500]]))
        buffer.receive(Block(2, [[2000]]))
        snapshot = buffer.take()
        assert 1 not in snapshot.calibration.baseline
    assert snapshot.calibration.counts == {1: 0, 2: 30}
    for _ in range(40):
        buffer.receive(Block(1, [[2500, 2500]]))
    snapshot = buffer.take()
    np.testing.assert_array_equal(snapshot.calibration.baseline[1], [[2500, 2500]])
    # Calibration does not seed the runtime filter with uncertain samples.
    assert snapshot.filtered_counts == {}


def test_gui_updates_other_rows_and_recovers_next_frame(app):
    buffer, window = LiveBuffer(calibration_seconds=0), MainWindow()
    for _ in range(30):
        buffer.receive(Block(1, [[2500, 2500, 2500], [2500, 2500, 2500]]))
    buffer.receive(Block(1, [[1003, 2000, 2300], [2000, 2100, 2200]]))
    snapshot = buffer.take()
    window._consume_scan(snapshot.scan, snapshot)
    item = window.heatmap.items[1]
    assert item.values.tolist() == [[0, 0, 0], [500, 400, 300]]
    assert window.current_frame.blocks[0].values.tolist() == [[1003, 2000, 2300], [2000, 2100, 2200]]
    window.heatmap.select_sensor((1, 0, 0))
    assert "原始 ADC 1003" in window.selected_detail.text()
    assert "保持上一有效值" in window.selected_detail.text()
    buffer.receive(Block(1, [[2000, 2400, 2500], [2100, 2200, 2300]]))
    snapshot = buffer.take()
    window._consume_scan(snapshot.scan, snapshot)
    assert item is window.heatmap.items[1]
    assert item.values.tolist() == [[500, 100, 0], [400, 300, 200]]
    assert not snapshot.filtered_masks[1].any()
    window.close()
    window.deleteLater()
