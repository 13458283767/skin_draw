import json
from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from skin_draw.collection_baseline import TrialBaseline, TrialBaselinePolicy
from skin_draw.collection_layout import CollectionBlock, CollectionLayout
from skin_draw.dataset import CalibrationSnapshot, DatasetStore, RawSampleRecorder
from skin_draw.mapping import BlockLayout
from skin_draw.models import Block
from test_dataset import FakeClock, make_layout, make_plan, read_rows


def trial_plan(tmp_path, layout=None, policy=None):
    layout = layout or make_layout(BlockLayout(1, 1, 2))
    reference = CalibrationSnapshot.from_baselines("calib_001", "test", {
        key: np.full((block.spec.rows, block.spec.cols), 2500.) for key, block in layout.by_id.items()})
    return replace(make_plan(tmp_path, layout, calibration=reference),
                   trial_baseline_policy=policy or TrialBaselinePolicy())


def finish_baseline(recorder, clock):
    clock.value = 1.
    recorder.snapshot()
    recorder._trial_baseline.thread.join(2)
    assert not recorder._trial_baseline.thread.is_alive()
    return recorder.snapshot().baseline


def test_trial_raw_median_mapping_frozen_and_persisted(tmp_path):
    layout = CollectionLayout("skin_layout_v1", (
        CollectionBlock(BlockLayout(1, 1, 2, flip_x=True), [[True, False]]),))
    plan = trial_plan(tmp_path, layout)
    clock = FakeClock()
    recorder = RawSampleRecorder(clock=clock)
    recorder.start(plan)
    for i in range(100):
        clock.value = i / 100
        recorder.submit(Block(1, [[2490, 0]]), clock.value)
    assert finish_baseline(recorder, clock)["status"] == "ready"
    for i in range(100, 300):
        clock.value = i / 100
        recorder.submit(Block(1, [[2200, 4095]]), clock.value)
    result = recorder.stop()
    assert result.status == "complete"
    baseline = json.loads((plan.sample_dir / "baseline.json").read_text(encoding="utf-8"))
    assert baseline["baselines"] == {"1": [[2490., None]]}  # raw order, masked point null
    assert baseline["action_allowed_s"] == 1.
    assert baseline["blocks"]["1"]["samples"] == 100
    assert result.activation["max_delta_counts"]["1"] == 290.
    assert result.activation["frames_evaluated"] == 200
    rows = read_rows(plan.sample_dir / "raw.csv")
    assert len(rows) == 301
    assert rows[1][3:] == ["0", "2490"]  # CSV remains body-oriented raw values.
    assert rows[-1][3:] == ["4095", "2200"]
    meta = json.loads(result.meta_path.read_text(encoding="utf-8"))
    assert meta["schema_version"] == "1.4"
    assert meta["baseline"]["file"] == "baseline.json"
    assert meta["calibration_id"] == "calib_001"
    before = (plan.sample_dir / "raw.csv").read_bytes()
    with pytest.raises(ValueError, match="提示"):
        DatasetStore(tmp_path).update_annotation(result.meta_path, .5, 2.)
    DatasetStore(tmp_path).update_annotation(result.meta_path, 1.1, 2.8)
    assert (plan.sample_dir / "raw.csv").read_bytes() == before


@pytest.mark.parametrize("fault", ["early_contact", "unstable", "missing", "gap", "short", "overflow"])
def test_invalid_baseline_keeps_raw_and_cannot_be_accepted(tmp_path, fault):
    policy = TrialBaselinePolicy(max_samples_per_block=30) if fault == "overflow" else TrialBaselinePolicy()
    layout = make_layout(BlockLayout(1, 1, 2), BlockLayout(2, 1, 2))
    plan = trial_plan(tmp_path, layout, policy)
    clock = FakeClock()
    recorder = RawSampleRecorder(clock=clock)
    recorder.start(plan)
    count = 0
    for i in range(40 if fault == "short" else 100):
        if fault == "gap" and 30 < i < 60:
            continue
        clock.value = i / 100
        value = 2200 if fault == "early_contact" else (2500 + 50 * (i % 2) if fault == "unstable" else 2500)
        for key in ((1,) if fault == "missing" else (1, 2)):
            recorder.submit(Block(key, [[value, value]]), clock.value)
            count += 1
    if fault != "short":
        assert finish_baseline(recorder, clock)["status"] == "failed"
    result = recorder.stop()
    assert result.status == "incomplete"
    assert result.baseline["status"] == "failed"
    assert result.baseline["reasons"]
    assert result.integrity["status"] == "fail"
    assert len(read_rows(plan.sample_dir / "raw.csv")) == count + 1
    with pytest.raises(ValueError):
        DatasetStore(tmp_path).update_accepted(result.meta_path, True)


def test_baseline_computation_cannot_block_raw_writer(tmp_path, monkeypatch):
    entered, release = Event(), Event()
    original = TrialBaseline._evaluate

    def delayed(self):
        entered.set()
        assert release.wait(5)
        return original(self)

    monkeypatch.setattr(TrialBaseline, "_evaluate", delayed)
    clock = FakeClock()
    recorder = RawSampleRecorder(clock=clock)
    plan = trial_plan(tmp_path)
    recorder.start(plan)
    try:
        for i in range(100):
            clock.value = i / 100
            recorder.submit(Block(1, [[2500, 2500]]), clock.value)
        clock.value = 1.
        assert recorder.snapshot().baseline["status"] == "processing"
        assert entered.wait(2)
        for i in range(100, 200):
            clock.value = i / 100
            recorder.submit(Block(1, [[2500, 2500]]), clock.value)
        assert recorder.snapshot().records == 200
        assert recorder.snapshot().baseline["status"] == "processing"
        release.set()
        recorder._trial_baseline.thread.join(2)
        assert recorder.snapshot().baseline["action_allowed_s"] == 1.99
        result = recorder.stop()
        assert result.written_records == 200
        assert result.baseline["blocks"]["1"]["samples"] == 100
    finally:
        release.set()
        if recorder.active:
            recorder.stop()


def test_no_cue_before_stop_does_not_produce_valid_trial(tmp_path):
    clock = FakeClock()
    recorder = RawSampleRecorder(clock=clock)
    recorder.start(trial_plan(tmp_path))
    for i in range(100):
        clock.value = i / 100
        recorder.submit(Block(1, [[2500, 2500]]), clock.value)
    clock.value = 1.1
    result = recorder.stop()
    assert result.baseline["status"] == "failed"
    assert result.baseline["action_allowed_s"] is None


def test_start_requires_complete_connection_baseline(tmp_path):
    with pytest.raises(ValueError, match="标定"):
        replace(make_plan(tmp_path, make_layout(BlockLayout(1, 1, 1))),
                trial_baseline_policy=TrialBaselinePolicy())
    assert not list(tmp_path.iterdir())


def test_next_trial_recomputes_its_own_baseline(tmp_path):
    clock = FakeClock()
    recorder = RawSampleRecorder(clock=clock)
    base = trial_plan(tmp_path)
    for trial, level in (("pat001", 2490), ("pat002", 2510)):
        clock.value = 0.
        plan = replace(base, trial_id=trial, sample_dir=base.sample_dir.parent / trial)
        recorder.start(plan)
        for i in range(100):
            clock.value = i / 100
            recorder.submit(Block(1, [[level, level]]), clock.value)
        assert finish_baseline(recorder, clock)["status"] == "ready"
        clock.value = 1.01
        recorder.submit(Block(1, [[level - 200, level - 200]]), clock.value)
        result = recorder.stop()
        assert result.baseline["baselines"]["1"] == [[float(level), float(level)]]
        assert result.activation["max_delta_counts"]["1"] == 200.


def test_ui_waits_for_baseline_without_blocking_heatmap_events(app, tmp_path, monkeypatch):
    from PySide6.QtCore import QTimer
    from test_collection_waveform import ready_window, close_window

    window, buffer, raw = ready_window(tmp_path)
    release, entered = Event(), Event()
    original = TrialBaseline._evaluate

    def delayed(self):
        entered.set()
        assert release.wait(5)
        return original(self)

    monkeypatch.setattr(TrialBaseline, "_evaluate", delayed)
    try:
        window.start_recording()
        assert "请勿接触" in window.collection_bar.status.text()
        clock = window.recorder.clock
        for i in range(100):
            clock.value = i / 100
            for key in (2, 7):
                buffer.receive(Block(key, raw))
        clock.value = 1.
        window._refresh_recording()
        assert entered.wait(2)
        ticks = []
        QTimer.singleShot(0, lambda: ticks.append(True))
        app.processEvents()
        assert ticks
        assert "请勿接触" in window.collection_bar.status.text()
        assert window.recorder.snapshot().baseline["status"] == "processing"
        clock.value = 1.2
        release.set()
        window.recorder._trial_baseline.thread.join(2)
        window._refresh_recording()
        assert "可以开始动作" in window.waveform.message.text()
        assert window.recorder.snapshot().baseline["action_allowed_s"] == 1.2
        clock.value = 1.3
        window.finish_recording(background=True, notify=False)
    finally:
        release.set()
        close_window(app, window)


def test_ui_early_contact_automatically_finishes_and_preserves_trial(app, tmp_path):
    from test_collection_waveform import ready_window, close_window
    from test_dataset import wait_collection

    window, buffer, raw = ready_window(tmp_path)
    try:
        plan = window.start_recording()
        clock = window.recorder.clock
        for i in range(100):
            clock.value = i / 100
            for key in (2, 7):
                buffer.receive(Block(key, raw - 300))
        clock.value = 1.
        window._refresh_recording()
        worker = window.recorder._trial_baseline
        worker.thread.join(2)
        window._refresh_recording()
        wait_collection(app, window)
        meta = json.loads((plan.sample_dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["end_reason"] == "baseline_failed"
        assert meta["quality_status"] == "fail"
        assert meta["summary"]["records"] == 200
        assert not window.collection_bar.mark_valid.isEnabled()
        assert "基线不合格" in window.waveform.message.text()
    finally:
        close_window(app, window)
