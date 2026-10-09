from unittest.mock import patch

import numpy as np
from PySide6.QtTest import QTest

from skin_draw.acquisition import Scan
from skin_draw.models import Block, Frame
from skin_draw.ui.main_window import MainWindow


def packet(sequence, include_second=True, value=2000):
    blocks = [Block(2, np.full((5, 4), value))]
    if include_second:
        blocks.append(Block(7, np.full((3, 6), value)))
    return Scan(Frame(tuple(blocks), sequence, sequence * .04), (2, 7), .1)


def test_packet_loss_never_rebuilds_or_moves_the_whole_heatmap(app):
    window = MainWindow()
    window.show()
    QTest.qWait(150)
    with patch.object(window.heatmap, "_build_layout", wraps=window.heatmap._build_layout) as build:
        window._consume_scan(packet(1))
        window.grab()
        original_items = dict(window.heatmap.items)
        original_positions = {key: item.pos() for key, item in original_items.items()}
        window.heatmap.select_sensor((2, 1, 1))
        window.heatmap.view.scaleBy((.8, .8))
        zoom = window.heatmap.view.viewRange()
        for sequence in range(2, 62):
            window._consume_scan(packet(sequence, include_second=sequence % 2 == 1))
            app.processEvents()
        assert build.call_count == 1
        assert window.heatmap.items == original_items
        assert all(item.pos() == original_positions[key] for key, item in original_items.items())
        assert window.heatmap.view.viewRange() == zoom
        assert window.heatmap.selected == (2, 1, 1)
        assert window.preprocess is None  # direct injection bypasses the preprocessing chain
        # Spurious resize signals with unchanged dimensions must do nothing.
        for _ in range(5):
            window.heatmap._reflow()
        assert build.call_count == 1
        window.resize(1100, 780)
        QTest.qWait(120)
        window.grab()
        assert build.call_count == 1
        assert window.heatmap.items == original_items
    window.close()
    window.deleteLater()


def test_stale_values_are_not_reported_as_live_and_recovery_reuses_items(app):
    window = MainWindow()
    window.show()
    for sequence in range(1, 31):
        window._consume_scan(packet(sequence))
    window._consume_scan(packet(31, value=1600))
    item = window.heatmap.items[7]
    previous = item.values.copy()
    window._consume_scan(packet(32, include_second=False))
    assert item.stale
    np.testing.assert_array_equal(item.values, previous)
    assert window.heatmap.reading((7, 0, 0)).value is None
    assert "1/2" in window.status.text()
    window._consume_scan(packet(33, include_second=False))
    window._consume_scan(packet(34, include_second=False))
    assert item.values is None  # extended loss is gray; its slot still exists
    assert window.heatmap.items[7] is item
    window._consume_scan(packet(35, value=1500))
    assert not item.stale
    assert window.heatmap.reading((7, 0, 0)).value == 1500  # direct path shows raw ADC
    window.close()
    window.deleteLater()


def test_unchanged_gray_calibration_cells_reuse_their_picture(app):
    window = MainWindow()
    window._consume_scan(packet(1))
    pictures = {key: item.picture for key, item in window.heatmap.items.items()}
    for sequence in range(2, 20):
        window._consume_scan(packet(sequence))
    assert all(item.picture is pictures[key] for key, item in window.heatmap.items.items())
    window.close()
    window.deleteLater()


def test_batched_cell_picture_updates_gap_without_new_sensor_values(app):
    window = MainWindow()
    window._consume_scan(packet(1))
    item = window.heatmap.items[2]
    previous = item.picture
    window.heatmap.set_gap(.2)
    assert item.picture is not previous
    assert item.cell_rect(0, 0).width() == .8
    assert item._gap_path.contains(item.cell_rect(0, 0).topLeft()) is False
    window.close()
    window.deleteLater()
