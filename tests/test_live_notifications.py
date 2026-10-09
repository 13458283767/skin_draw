from dataclasses import FrozenInstanceError

import pytest
from PySide6.QtCore import Qt

from skin_draw.live import LiveBuffer
from skin_draw.models import Block
from skin_draw.ui.main_window import MainWindow
from skin_draw.ui.serial_worker import SerialWorker


def test_burst_coalesces_notifications_without_skipping_calibration():
    worker = SerialWorker("FAKE", norm="A")
    worker.buffer.calib_n = 30  # This test isolates notification delivery.
    notifications = []
    worker.data_ready.connect(lambda: notifications.append(True))
    for index in range(120):
        worker._receive_block(Block(4, [[2500 if index < 60 else 2000]]))
    assert len(notifications) == 1
    worker.acknowledge_update()
    snapshot = worker.buffer.take()
    assert snapshot.received_blocks == 120
    assert snapshot.preprocess.samples == {4: 30}
    assert snapshot.display_values[4][0, 0] == 500
    worker._receive_block(Block(4, [[2100]]))
    assert len(notifications) == 2


@pytest.mark.parametrize("arrival_before_take", [True, False])
def test_arrival_during_gui_handoff_is_not_lost(app, monkeypatch, arrival_before_take):
    window = MainWindow()
    worker = SerialWorker("FAKE", norm="A")
    worker.buffer.calib_n = 1  # This test isolates notification delivery.
    window.worker = worker
    window.accept_scans = True
    worker.data_ready.connect(window._request_live_update, Qt.ConnectionType.QueuedConnection)
    try:
        for _ in range(30):
            worker._receive_block(Block(4, [[2500]]))
        for _ in range(35):
            worker._receive_block(Block(4, [[2000]]))
        original_take = worker.buffer.take

        def take_with_arrival():
            if arrival_before_take:
                worker._receive_block(Block(4, [[2000]]))
            snapshot = original_take()
            if not arrival_before_take:
                worker._receive_block(Block(4, [[2000]]))
            return snapshot

        monkeypatch.setattr(worker.buffer, "take", take_with_arrival)
        window._flush_live()
        monkeypatch.setattr(worker.buffer, "take", original_take)
        app.processEvents()
        # A post-snapshot arrival must wake the GUI without periodic polling.
        assert window.current_frame.sequence == 66 or window.display_timer.isActive()
        window._flush_live()
        assert window.current_frame.sequence == 66
        assert window.heatmap.reading((4, 0, 0)).value == 500
        assert not window.display_timer.isActive()
    finally:
        window.accept_scans = False
        window.worker = None
        window.close()
        window.deleteLater()


def test_preprocess_snapshot_isolated_during_accumulation_and_reset():
    buffer = LiveBuffer(calib_n=30, norm="A")
    buffer.receive(Block(1, [[2500]]))
    pending = buffer.take()
    assert pending.preprocess.samples == {1: 0}  # Warm-up frames collect nothing.
    buffer.receive(Block(1, [[2400]]))
    assert pending.preprocess.samples == {1: 0}  # The old snapshot must not move.
    assert buffer.take().preprocess.samples == {1: 0}
    for _ in range(58):
        buffer.receive(Block(1, [[2500]]))
    ready = buffer.take()
    baseline = ready.preprocess.baselines[1]
    assert ready.preprocess.ready
    assert ready.preprocess.samples == {1: 30}
    assert baseline[0, 0] == 2500
    with pytest.raises(ValueError):
        baseline[0, 0] = 0
    buffer.reset_preprocessing()
    assert ready.preprocess.ready  # The snapshot keeps its calibrated state.
    assert ready.preprocess.samples == {1: 30}
    assert buffer.preprocessors[1].samples == 0
    assert baseline[0, 0] == 2500  # Baseline copy is independent of the worker.
    with pytest.raises(FrozenInstanceError):
        ready.preprocess.ready = False
