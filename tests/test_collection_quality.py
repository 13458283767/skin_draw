"""H01 验收：真实时间边界、故障与文件内容，而非 GUI 刷新次数。"""

from dataclasses import replace
import json

import pytest

from skin_draw.collection_quality import IntegrityMonitor, QualityPolicy, load_quality_policy
from skin_draw.dataset import RawSampleRecorder, DatasetStore
from skin_draw.live import LiveBuffer
from skin_draw.models import Block
from skin_draw.mapping import BlockLayout
from skin_draw.ui.main_window import MainWindow
from test_dataset import FakeClock, make_layout, make_plan, read_rows


def monitor(ids=(1,), **kwargs):
    return IntegrityMonitor(ids, QualityPolicy(**kwargs), tuple((key, .02, 10) for key in ids))


@pytest.mark.parametrize("times,end,kind,level", [
    ([.3, .32], .34, "head_gap", "warn"),
    ([1.1, 1.12], 1.13, "head_gap", "fail"),
    ([.02, .32], .34, "interval_gap", "warn"),
    ([.02, 1.2], 1.22, "interval_gap", "fail"),
    ([.02, .04], .4, "tail_gap", "warn"),
    ([.02, .04], 1.2, "tail_gap", "fail"),
])
def test_head_middle_tail_gaps(times, end, kind, level):
    check = monitor()
    for time in times:
        check.receive(1, time)
    report = check.evaluate(end, final=True)
    assert report["status"] == level
    assert report["events"][0]["kind"] == kind
    assert report["events"][0]["duration_s"] > .2
    assert report["blocks"]["1"]["frames"] == len(times)


def test_periodic_silence_is_visible_without_next_frame_and_not_counted_twice():
    check = monitor((1, 2))
    check.receive(1, .02)
    check.receive(1, .04)
    first = check.evaluate(1.2)
    assert first["status"] == "fail"
    assert first["blocks"]["2"]["frames"] == 0
    assert first["events"][0]["kind"] == "ongoing_gap"
    assert first == check.evaluate(1.2)
    # 恢复接收后同一段空缺仅作为一条已完成事件计数，实时快照不污染累计值。
    check.receive(1, 1.22)
    final = check.evaluate(1.24, final=True)
    assert final["blocks"]["1"]["failure_count"] == 1
    assert first["blocks"]["1"]["frames"] == 2


def test_event_details_bounded_but_total_and_worst_severity_preserved():
    check = monitor(max_events=2)
    for time in (.02, .32, .62, 1.8):
        check.receive(1, time)
    report = check.evaluate(1.82, final=True)
    assert report["status"] == "fail"
    assert len(report["events"]) == 2
    assert report["event_count"] == 3 and report["events_truncated"]
    assert report["blocks"]["1"]["failure_count"] == 1
    report["events"][0]["duration_s"] = 999
    assert check.evaluate(1.82, final=True)["events"][0]["duration_s"] != 999


def test_threshold_boundary_and_frozen_per_block_reference():
    check = IntegrityMonitor((1, 2), QualityPolicy(), ((1, .02, 10), (2, .1, 10)))
    for key in (1, 2):
        check.receive(key, .2)
        check.receive(key, .4)
    assert check.evaluate(.6, final=True)["status"] == "pass"
    report = check.evaluate(.8)
    assert report["blocks"]["1"]["status"] == "warn"
    assert report["blocks"]["2"]["status"] == "pass"
    assert report["blocks"]["2"]["failure_threshold_s"] == 2.


def test_reference_insufficient_uses_fixed_threshold_and_requests_review():
    check = IntegrityMonitor((1,), QualityPolicy(), ((1, 10., 1),))
    check.receive(1, .02)
    check.receive(1, .04)
    report = check.evaluate(.06, final=True)
    assert report["status"] == "warn"
    assert report["blocks"]["1"]["failure_threshold_s"] == 1.
    assert "参考" in report["reasons"][0]


def test_cadence_snapshot_rejects_stale_bursts_and_remains_frozen():
    clock = FakeClock()
    buffer = LiveBuffer(clock=clock, raw_adc=True)
    policy = QualityPolicy()
    for i in range(11):
        clock.value = i * .02
        buffer.receive(Block(1, [[2500]]))
    reference = buffer.recording_cadence((1,), policy)
    assert reference[0][1] == pytest.approx(.02)
    assert reference[0][2] == 10
    clock.value = 2
    assert buffer.recording_cadence((1,), policy)[0][1] is None
    assert reference[0][1] == pytest.approx(.02)
    for _ in range(20):
        buffer.receive(Block(2, [[2500]]))
    assert buffer.recording_cadence((2,), policy)[0][1] is None


def run_sample(tmp_path, times, end, reason="user_stop"):
    clock = FakeClock()
    recorder = RawSampleRecorder(clock=clock)
    plan = replace(make_plan(tmp_path, make_layout(BlockLayout(1, 1, 2))),
                   cadence_references=((1, .02, 10),))
    recorder.start(plan)
    for time in times:
        clock.value = time
        recorder.submit(Block(1, [[0, 4095]]), time)
    clock.value = end
    result = recorder.stop(reason)
    return result, json.loads(result.meta_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("reason", ["disconnect", "app_exit", "write_error"])
def test_abnormal_end_never_complete_even_when_frames_are_continuous(tmp_path, reason):
    result, meta = run_sample(tmp_path, [.02, .04, .06], .08, reason)
    assert result.status == "incomplete" and meta["quality_status"] == "fail"
    assert meta["end_reason"] == reason
    assert meta["quality"]["integrity"]["final"]
    assert any(reason in text for text in meta["quality"]["integrity"]["reasons"])


@pytest.mark.parametrize("times,end", [([.1], 10.), ([.02, .04], 2.)])
def test_normally_saved_but_incomplete_contact_data_cannot_be_accepted(tmp_path, times, end):
    result, meta = run_sample(tmp_path, times, end)
    assert result.status == "complete"  # 文件收尾正常，数据质量单独判定。
    assert meta["quality_status"] == "fail"
    store = DatasetStore(tmp_path)
    with pytest.raises(ValueError, match="不能标记有效"):
        store.update_accepted(result.meta_path, True)
    store.rebuild_index()
    assert read_rows(store.index_path)[1][-2:] == ["complete", "fail"]
    assert all(row[-2:] == ["0", "4095"] for row in read_rows(result.sample_dir / "raw.csv")[1:])


def test_normal_data_passes_and_metadata_persists_frozen_rules(tmp_path):
    result, meta = run_sample(tmp_path, [.02, .04, .06], .08)
    assert result.integrity["status"] == meta["quality_status"] == "pass"
    report = meta["quality"]["integrity"]
    assert report["rule_version"] == "receive_integrity_v1"
    assert report["blocks"]["1"]["tail_gap_s"] == pytest.approx(.02)
    assert report["events"] == []
    DatasetStore(tmp_path).update_accepted(result.meta_path, True)


def test_write_drain_time_does_not_become_tail_gap(tmp_path, monkeypatch):
    clock = FakeClock()
    recorder = RawSampleRecorder(clock=clock)
    recorder.start(replace(make_plan(tmp_path, make_layout(BlockLayout(1, 1, 1))),
                           cadence_references=((1, .02, 10),)))
    for time in (.02, .04):
        clock.value = time
        recorder.submit(Block(1, [[0]]), time)
    finish = recorder._writer.finish

    def delayed_finish():
        clock.value = 20.
        finish()

    monkeypatch.setattr(recorder._writer, "finish", delayed_finish)
    clock.value = .06
    result = recorder.stop()
    assert result.duration_s == .06
    assert result.integrity["status"] == "pass"


@pytest.mark.parametrize("bad", [{"warning_floor_s": -1}, {"failure_floor_s": .1},
                                 {"max_events": 0}, {"min_reference_intervals": 129}, {"typo": 2}])
def test_invalid_policy_is_rejected(tmp_path, bad):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_quality_policy(path)


def test_ui_shows_silence_and_disables_accept_for_failed_quality(app, tmp_path):
    clock = FakeClock()
    window = MainWindow(dataset_dir=tmp_path)
    window.recorder = RawSampleRecorder(clock=clock)
    plan = replace(make_plan(tmp_path, make_layout(BlockLayout(1, 1, 1), BlockLayout(2, 1, 1))),
                   cadence_references=((1, .02, 10), (2, .02, 10)))
    try:
        window.recorder.start(plan)
        window.recording_plan = plan
        window.recorder.submit(Block(1, [[2500]]), 0.)
        clock.value = 2.
        window._refresh_recording()
        assert "不合格" in window.collection_bar.frame_button.text()
        assert "等待首帧" in window.collection_bar._frame_actions[2].text()
        assert "0.000～2.000" in window.collection_bar._frame_actions[2].toolTip()
        result = window.finish_recording(notify=False)
        assert result.integrity["status"] == "fail"
        assert not window.collection_bar.mark_valid.isEnabled()
        assert window.collection_bar.mark_void.isEnabled()
    finally:
        window.close()


def test_writer_failure_is_visible_even_when_no_more_frames_arrive(tmp_path):
    recorder = RawSampleRecorder(clock=FakeClock())
    recorder.start(make_plan(tmp_path, make_layout(BlockLayout(1, 1, 1))))
    try:
        recorder._writer.error = "disk full"
        snapshot = recorder.snapshot()
        assert snapshot.fatal_error == "disk full"
        assert snapshot.integrity["status"] == "fail"
    finally:
        result = recorder.stop("write_error")
    assert result.status == "incomplete"


def test_final_metadata_failure_cannot_report_complete(tmp_path, monkeypatch):
    recorder = RawSampleRecorder(clock=FakeClock())
    recorder.start(make_plan(tmp_path, make_layout(BlockLayout(1, 1, 1))))
    recorder.submit(Block(1, [[2000]]), 0.)
    recorder.submit(Block(1, [[2000]]), 0.)

    def fail(*args, **kwargs):
        raise OSError("metadata locked")

    monkeypatch.setattr(recorder, "_write_meta", fail)
    result = recorder.stop()
    assert result.status == "incomplete"
    assert result.end_reason == "metadata_error"
    assert result.integrity["status"] == "fail"
    assert result.meta_error
