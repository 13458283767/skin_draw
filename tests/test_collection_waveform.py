import csv
import hashlib
import json
from queue import Full, Queue
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QTimer

from skin_draw.collection_layout import CollectionBlock, CollectionLayout
from skin_draw.collection_waveform import LiveWaveform, envelope, load_waveform
from skin_draw.dataset import DatasetStore, RawSampleRecorder
from skin_draw.live import LiveBuffer
from skin_draw.mapping import BlockLayout
from skin_draw.models import Block, Frame
from skin_draw.ui.main_window import MainWindow
from test_dataset import FakeClock, complete_ui_baseline, make_layout, make_plan, wait_collection


def ready_window(tmp_path):
    window = MainWindow(mask_path=tmp_path / "masks.json", dataset_dir=tmp_path / "dataset")
    raw = np.full((2, 3), 2500, dtype=np.uint16)
    window.present_frame(Frame((Block(2, raw), Block(7, raw))))
    window.save_collection_layout()
    clock = FakeClock()
    window.recorder.clock = clock
    buffer = LiveBuffer(clock=clock, filter_enabled=False, calib_n=3)
    window.worker = SimpleNamespace(buffer=buffer)
    window.attach_recording_sink(buffer)
    window.input_source = "hardware"
    for _ in range(3):
        for key in (2, 7):
            buffer.receive(Block(key, raw))
    return window, buffer, raw


def close_window(app, window):
    window.worker = None
    window.close()
    wait_collection(app, window)
    app.processEvents()
    window.close()


def test_full_display_queue_never_discards_raw_rows(tmp_path):
    clock = FakeClock()
    recorder = RawSampleRecorder(clock=clock)
    plan = make_plan(tmp_path, make_layout(BlockLayout(1, 1, 2)))
    recorder.start(plan)
    recorder.preview_queue = Queue(maxsize=1)  # No consumer: every later preview is dropped.
    for i in range(200):
        clock.value = i / 100
        recorder.submit(Block(1, [[2500, i]]), clock.value)
    result = recorder.stop()
    assert result.records == result.written_records == 200
    with (plan.sample_dir / "raw.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["adc_001"]) for row in rows] == list(range(200))
    assert recorder.preview_queue.qsize() == 1
    snapshot = load_waveform(plan, result.duration_s)
    assert snapshot.curves[1][0][-1] == pytest.approx(1.99)


def test_seal_fixes_stop_time_before_background_scheduling(tmp_path):
    clock = FakeClock()
    recorder = RawSampleRecorder(clock=clock)
    plan = make_plan(tmp_path, make_layout(BlockLayout(1, 1, 1)))
    recorder.start(plan)
    clock.value = .1
    recorder.submit(Block(1, [[2000]]), clock.value)
    clock.value = 1.
    recorder.seal()
    recorder.submit(Block(1, [[999]]), 2.)
    clock.value = 100.
    result = recorder.stop()
    assert result.duration_s == 1.
    assert result.records == result.written_records == 1


def test_replay_respects_mask_mapping_preserves_pulse_and_gap(tmp_path):
    spec = BlockLayout(7, 1, 2, flip_x=True)
    layout = CollectionLayout("skin_layout_v1", (CollectionBlock(spec, [[True, False]]),))
    plan = make_plan(tmp_path, layout)
    clock = FakeClock()
    recorder = RawSampleRecorder(clock=clock)
    recorder.start(plan)
    for i in range(200):
        if 90 <= i <= 110:
            continue
        clock.value = i / 100
        recorder.submit(Block(7, [[1000 if i == 42 else 2500, 0]]), clock.value)
    result = recorder.stop()
    snapshot = load_waveform(plan, result.duration_s, bins=20)
    x, y = snapshot.curves[7]
    assert np.nanmin(y) == 1000
    assert np.nanmax(y) == 2500  # Hidden point is excluded, even after reordering.
    assert np.isnan(y).any()
    assert len(x) <= 20 * 4
    assert x[0] == 0 and x[-1] == 1.99


def test_live_preview_is_bounded_and_stoppable(tmp_path):
    plan = make_plan(tmp_path, make_layout(BlockLayout(1, 1, 1)))
    preview = LiveWaveform(plan, capacity=2, max_points=20)
    try:
        for i in range(100):
            try:
                preview.queue.put_nowait([i, str(i / 100), 1, 2500])
            except Full:
                pass
        assert preview.queue.qsize() <= 2
    finally:
        preview.close()
        preview.thread.join(2)
    assert not preview.thread.is_alive()
    assert all(len(points) <= 20 for points in preview.points.values())


def test_annotation_is_atomic_validated_and_raw_unchanged(tmp_path):
    plan = make_plan(tmp_path, make_layout(BlockLayout(1, 1, 1)))
    clock = FakeClock()
    recorder = RawSampleRecorder(clock=clock)
    recorder.start(plan)
    clock.value = 2.
    recorder.submit(Block(1, [[2450]]), 2.)
    result = recorder.stop()
    store = DatasetStore(tmp_path)
    path = plan.sample_dir / "raw.csv"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    for start, end in [(1., .5), (-.1, 1), (0., 3), (float("nan"), 1), (True, 1.5)]:
        with pytest.raises(ValueError):
            store.update_annotation(result.meta_path, start, end)
    saved = store.update_annotation(result.meta_path, .2, 1.8)
    assert saved["annotation"]["status"] == "confirmed"
    assert saved["annotation"]["source"] == "manual"
    assert saved["accepted"] is None
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert {p.name for p in plan.sample_dir.iterdir()} == {"meta.json", "raw.csv"}


def test_stop_does_not_block_ui_and_replay_waits_for_writer(app, tmp_path, monkeypatch):
    window, buffer, raw = ready_window(tmp_path)
    release, entered = Event(), Event()
    try:
        plan = window.start_recording()
        complete_ui_baseline(window, buffer, raw, (2, 7))
        for key in (2, 7):
            buffer.receive(Block(key, raw))
        real_finish = window.recorder._writer.finish

        def stalled_finish():
            entered.set()
            assert release.wait(5)
            real_finish()

        monkeypatch.setattr(window.recorder._writer, "finish", stalled_finish)
        window.collection_bar.stop.click()
        assert entered.wait(1)
        assert window._collection_job[0] == "finalize"
        assert window.waveform.mode == "finalizing"
        assert not window.collection_bar.start.isEnabled()
        ticks = []
        QTimer.singleShot(0, lambda: ticks.append(True))
        app.processEvents()
        assert ticks  # UI callbacks still run while disk finalization is blocked.
        with pytest.raises(ValueError):
            window.start_recording()
        assert json.loads((plan.sample_dir / "meta.json").read_text(encoding="utf-8"))["annotation"]["action_start_s"] is None
        release.set()
        wait_collection(app, window)
        assert window.waveform.mode == "review"
        assert set(window.waveform.curves) == {2, 7}
        assert all(spin.value() == -1 for spin in window.waveform.spins)
        duration = window.waveform.duration
        window.waveform.spins[0].setValue(1.)
        window.waveform.spins[1].setValue(duration)
        assert window.waveform.save.isEnabled()
        raw_before = (plan.sample_dir / "raw.csv").read_bytes()
        window.waveform.save.click()
        wait_collection(app, window)
        assert not window._review_pending
        assert not window.waveform.curves
        meta = json.loads((plan.sample_dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["annotation"]["action_end_s"] == duration
        assert meta["annotation"]["status"] == "confirmed"
        assert (plan.sample_dir / "raw.csv").read_bytes() == raw_before
    finally:
        release.set()
        close_window(app, window)


def test_annotation_save_failure_retains_review(app, tmp_path, monkeypatch):
    window, buffer, raw = ready_window(tmp_path)
    try:
        window.start_recording()
        complete_ui_baseline(window, buffer, raw, (2, 7))
        for key in (2, 7):
            buffer.receive(Block(key, raw))
        window.finish_recording(background=True)
        wait_collection(app, window)

        def fail(*args):
            raise OSError("disk unavailable")

        monkeypatch.setattr(window.dataset_store, "update_annotation", fail)
        window._confirm_boundaries(0., window.waveform.duration)
        wait_collection(app, window)
        assert window._review_pending
        assert window.waveform.mode == "review"
        assert window.waveform.curves
        assert not window.collection_bar.start.isEnabled()
        window._defer_boundaries()
        assert not window._review_pending
    finally:
        close_window(app, window)


def test_envelope_preserves_short_peak():
    points = [(i * .001, 1000. if i == 412 else 2500.) for i in range(1000)]
    x, y = envelope(points, bins=10)
    assert x[np.argmin(y)] == pytest.approx(.412)
    assert min(y) == 1000.
    assert len(x) <= 44


def test_envelope_does_not_mistake_display_reduction_for_packet_gaps():
    points = [(i * .01, 2500. - (i % 50)) for i in range(1000)]
    _, y = envelope(points, bins=4)
    assert not np.isnan(y).any()
