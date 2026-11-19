"""接触区域摘要与语义检查：主接触块、区域集合、拥抱区域数、单区域换块。"""

import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from skin_draw.collection_activation import (ActivationPolicy, ActivationTracker,
                                             load_activation_policy, semantic_status)
from skin_draw.collection_layout import CollectionBlock, CollectionLayout
from skin_draw.dataset import (ActionClass, CalibrationSnapshot, DatasetStore, RawSampleRecorder,
                               RecordingPlan, auto_annotation, layout_adc_columns)
from skin_draw.mapping import BlockLayout
from skin_draw.models import Block

WALL = datetime(2026, 10, 22, 9, 30, 0).astimezone()


class FakeClock:
    def __init__(self):
        self.value = 0.

    def __call__(self):
        return self.value

    def wall(self):
        return WALL


def make_layout(*specs):
    blocks = tuple(CollectionBlock(spec, np.ones((spec.rows, spec.cols), bool)) for spec in specs)
    return CollectionLayout("skin_layout_v1", blocks, "grid", ())


def make_calibration(layout, values):
    baselines = {str(key): np.full((block.spec.rows, block.spec.cols), values[key], dtype=float)
                 for key, block in layout.by_id.items()}
    return CalibrationSnapshot.from_baselines("calib_001", WALL.isoformat(), baselines)


def make_plan(root, layout, calibration, *, action=("pat", 0)):
    name, class_id = action
    return RecordingPlan(
        sample_dir=root / "student01" / "session_20261022_01" / name / f"{name}001",
        subject_id="student01", session_id="session_20261022_01", trial_id=f"{name}001",
        action=ActionClass(class_id, name, name), layout=layout, labels_id="actions_v3",
        expected_block_ids=tuple(layout.by_id), max_adc_columns=layout_adc_columns(layout),
        calibration=calibration)


def test_tracker_reports_regions_above_threshold():
    layout = make_layout(BlockLayout(1, 1, 2), BlockLayout(2, 1, 2), BlockLayout(3, 1, 2))
    calibration = make_calibration(layout, {1: 2500., 2: 2500., 3: 2500.})
    tracker = ActivationTracker(layout.by_id, calibration, ActivationPolicy())
    assert tracker.ready
    tracker.observe(1, np.array([1500., 2400.]))   # 下降 1000，接触
    tracker.observe(2, np.array([2400., 2400.]))   # 下降 100，正好达阈值
    tracker.observe(3, np.array([2495., 2495.]))   # 下降 5，视为未接触
    summary = tracker.summary()
    assert summary["region_ids"] == [1, 2]
    assert summary["region_set"] == "12"
    assert summary["region_count"] == 2
    assert summary["primary_block"] == 1
    assert summary["max_delta_counts"]["1"] == 1000.0


def test_tracker_counts_block_switches_for_single_region_actions():
    layout = make_layout(BlockLayout(1, 1, 2), BlockLayout(2, 1, 2))
    calibration = make_calibration(layout, {1: 2500., 2: 2500.})
    tracker = ActivationTracker(layout.by_id, calibration, ActivationPolicy())
    # 1 号块先被接触（下降 300），2 号块仍是静置。
    tracker.observe(1, np.array([2200., 2200.]))
    tracker.observe(2, np.array([2500., 2500.]))
    # 主接触块换到 2 号（下降 1000，强于 1 号）。
    tracker.observe(2, np.array([1500., 1500.]))
    summary = tracker.summary()
    assert summary["region_count"] == 2
    assert summary["block_switches"] == 1
    assert summary["primary_block"] == 2


def test_missing_calibration_stays_unknown_instead_of_guessing():
    layout = make_layout(BlockLayout(1, 1, 2))
    tracker = ActivationTracker(layout.by_id, None, ActivationPolicy())
    assert not tracker.ready
    tracker.observe(1, np.array([1500., 1500.]))
    summary = tracker.summary()
    assert summary["region_count"] == 0 and summary["primary_block"] is None
    semantic = semantic_status(summary, "pat", ActivationPolicy())
    assert semantic["status"] == "unknown"


def test_semantic_rules_for_multi_region_and_single_region_actions():
    policy = ActivationPolicy()
    two = {"available": True, "region_count": 2, "block_switches": 0}
    three = {"available": True, "region_count": 3, "block_switches": 0}
    assert semantic_status(two, "hug", policy)["status"] == "warn"
    assert "少于 3" in semantic_status(two, "hug", policy)["reasons"][0]
    assert semantic_status(three, "hug", policy)["status"] == "pass"
    switched = {"available": True, "region_count": 2, "block_switches": 2}
    assert semantic_status(switched, "pat", policy)["status"] == "warn"
    empty = {"available": True, "region_count": 0, "block_switches": 0}
    assert semantic_status(empty, "pat", policy)["status"] == "warn"


def test_policy_rejects_bad_values_and_loads_from_file(tmp_path):
    with pytest.raises(ValueError):
        ActivationPolicy(threshold_counts=0)
    with pytest.raises(ValueError):
        ActivationPolicy(hug_min_regions=-1)
    path = tmp_path / "collection_activation.json"
    path.write_text(json.dumps({"threshold_counts": 80, "hug_min_regions": 2,
                                "max_block_switches": 1,
                                "multi_region_actions": ["hug", "hug_both"]}), encoding="utf-8")
    policy = load_activation_policy(path)
    assert policy.threshold_counts == 80 and policy.multi_region("hug_both")
    assert load_activation_policy(tmp_path / "missing.json") == ActivationPolicy()


def test_rotated_and_flipped_block_detects_contact(tmp_path):
    """真实布局里 1 号块是 180°+左右镜像；变换后的坐标不能让区域判定错位。"""
    spec = BlockLayout(1, 5, 3, rotation=180, flip_x=True)
    layout = make_layout(spec)
    calibration = make_calibration(layout, {1: 2500.})
    tracker = ActivationTracker(layout.by_id, calibration, ActivationPolicy())
    assert tracker.ready
    raw = np.full((5, 3), 2500., dtype=float)
    raw[0, 0] = 1500.                    # 原始矩阵左上角被按压
    tracker.observe(1, layout.by_id[1].transform(raw).ravel())
    summary = tracker.summary()
    assert summary["region_ids"] == [1] and summary["primary_block"] == 1
    assert summary["max_delta_counts"]["1"] == 1000.0


def test_recording_writes_primary_block_and_region_set(tmp_path):
    layout = make_layout(BlockLayout(1, 1, 2), BlockLayout(2, 1, 2))
    calibration = make_calibration(layout, {1: 2500., 2: 2500.})
    clock = FakeClock()
    recorder = RawSampleRecorder(clock=clock, wall_clock=cloud_clock)
    plan = make_plan(tmp_path, layout, calibration, action=("hug", 2))
    recorder.start(plan)
    for index in range(6):
        clock.value = .02 * (index + 1)
        recorder.submit(Block(1, [[1500, 1500]]), clock.value)
        recorder.submit(Block(2, [[2490, 1500]]), clock.value)
    clock.value = .2
    result = recorder.stop()
    meta = json.loads(result.meta_path.read_text(encoding="utf-8"))
    assert meta["schema_version"] == "1.3"
    assert meta["primary_block"] == 1
    assert meta["region_set"] == "12"
    assert meta["region_count"] == 2
    assert meta["activation"]["method"] == "contact_activation_v1"
    assert meta["activation"]["threshold_counts"] == 100
    assert meta["quality"]["semantic"]["status"] == "warn"      # 拥抱只有 2 个区域
    assert meta["quality_status"] == "warn"
    assert result.semantic["status"] == "warn"


def test_recording_without_calibration_keeps_quality_from_integrity(tmp_path):
    layout = make_layout(BlockLayout(1, 1, 2))
    clock = FakeClock()
    recorder = RawSampleRecorder(clock=clock, wall_clock=cloud_clock)
    recorder.start(make_plan(tmp_path, layout, None))
    for index in range(6):
        clock.value = .02 * (index + 1)
        recorder.submit(Block(1, [[1500, 1500]]), clock.value)
    clock.value = .2
    result = recorder.stop()
    meta = json.loads(result.meta_path.read_text(encoding="utf-8"))
    assert meta["primary_block"] is None and meta["region_set"] is None
    assert meta["quality"]["semantic"]["status"] == "unknown"
    assert meta["quality_status"] == result.integrity["status"]


def cloud_clock():
    return SimpleNamespace(isoformat=lambda timespec=None: WALL.isoformat())


def test_tracker_records_contact_span_from_frame_times():
    """首末个超阈值帧就是自动起止；静置帧不参与。"""
    layout = make_layout(BlockLayout(1, 1, 2))
    calibration = make_calibration(layout, {1: 2500.})
    tracker = ActivationTracker(layout.by_id, calibration, ActivationPolicy())
    tracker.observe(1, np.array([2500., 2500.]), .5)    # 未接触
    tracker.observe(1, np.array([2000., 2500.]), 1.0)   # 接触开始
    tracker.observe(1, np.array([1800., 2500.]), 1.2)   # 仍在接触
    tracker.observe(1, np.array([2500., 2500.]), 1.5)   # 已松开
    summary = tracker.summary()
    assert summary["contact_start_s"] == 1.0
    assert summary["contact_end_s"] == 1.2
    assert summary["contact_frames"] == 2
    assert summary["open_at_end"] is False
    assert summary["edge_counts"] == 50.0


def test_tracker_flags_unreleased_contact_at_end():
    """最后一帧仍高于起止阈值时，自动区间不可信，必须显式标出。"""
    layout = make_layout(BlockLayout(1, 1, 2))
    calibration = make_calibration(layout, {1: 2500.})
    tracker = ActivationTracker(layout.by_id, calibration, ActivationPolicy())
    tracker.observe(1, np.array([2000., 2000.]), 1.0)    # 下降 500
    tracker.observe(1, np.array([2440., 2440.]), 1.4)    # 下降 60：仍高于起止阈值 50
    summary = tracker.summary()
    assert summary["open_at_end"] is True
    assert summary["contact_end_s"] == 1.4
    tracker.observe(1, np.array([2470., 2470.]), 1.6)    # 下降 30：已释放
    summary = tracker.summary()
    assert summary["open_at_end"] is False
    assert summary["contact_end_s"] == 1.4


def test_policy_rejects_edge_above_threshold():
    with pytest.raises(ValueError, match="起点阈值"):
        ActivationPolicy(onset_counts=80, edge_counts=50)
    with pytest.raises(ValueError, match="终点阈值"):
        ActivationPolicy(threshold_counts=100, edge_counts=150)
    with pytest.raises(ValueError, match="外扩"):
        ActivationPolicy(edge_pad_s=-1)


def test_gradual_onset_is_caught_by_the_lower_threshold():
    """缓爬坡起步：起点取 20 ADC，终点仍按 50 ADC。"""
    layout = make_layout(BlockLayout(1, 1, 2))
    calibration = make_calibration(layout, {1: 2500.})
    tracker = ActivationTracker(layout.by_id, calibration, ActivationPolicy())
    tracker.observe(1, np.array([2480., 2480.]), 1.0)   # 下降 20：只达起点阈值
    tracker.observe(1, np.array([2455., 2455.]), 1.2)   # 下降 45：仍未达终点阈值
    tracker.observe(1, np.array([2000., 2000.]), 1.4)   # 下降 500
    summary = tracker.summary()
    assert summary["contact_start_s"] == 1.0
    assert summary["contact_end_s"] == 1.4
    assert summary["onset_counts"] == 20.0 and summary["edge_counts"] == 50.0


def test_untouched_block_never_extends_the_span():
    """余振块只有 30 ADC：超过起点阈值，但未达终点阈值，不得把终点拖长。"""
    layout = make_layout(BlockLayout(1, 1, 2), BlockLayout(2, 1, 2))
    calibration = make_calibration(layout, {1: 2500., 2: 2500.})
    tracker = ActivationTracker(layout.by_id, calibration, ActivationPolicy())
    tracker.observe(1, np.array([2000., 2000.]), 1.0)   # 真接触
    tracker.observe(1, np.array([2500., 2500.]), 2.0)   # 已松开
    tracker.observe(2, np.array([2470., 2470.]), 3.0)   # 下降 30：余振/漂移
    summary = tracker.summary()
    assert summary["contact_start_s"] == 1.0
    assert summary["contact_end_s"] == 1.0
    assert summary["region_ids"] == [1]
    assert summary["open_at_end"] is False


def test_auto_annotation_pads_without_crossing_the_cue():
    """外扩只能向空发，不得早于动作提示时刻。"""
    activation = {"available": True, "contact_start_s": 1.05, "contact_end_s": 1.4,
                  "edge_counts": 50., "edge_pad_s": .1, "contact_frames": 5,
                  "open_at_end": False}
    annotation = auto_annotation(activation, 2., "t", gate_s=1.1)
    assert annotation["action_start_s"] == 1.1
    assert annotation["action_end_s"] == 1.5
    assert annotation["auto"]["edge_pad_s"] == .1
    # 默认不外扩时保持首末帧原值。
    plain = {**activation, "edge_pad_s": 0.}
    assert auto_annotation(plain, 2.)["action_start_s"] == 1.05


def test_recording_writes_auto_annotation_and_manual_overrides(tmp_path):
    layout = make_layout(BlockLayout(1, 1, 2))
    calibration = make_calibration(layout, {1: 2500.})
    clock = FakeClock()
    recorder = RawSampleRecorder(clock=clock, wall_clock=cloud_clock)
    recorder.start(make_plan(tmp_path, layout, calibration))
    for moment, value in ((.5, 2500.), (1.0, 2000.), (1.2, 1800.), (1.5, 2500.)):
        clock.value = moment
        recorder.submit(Block(1, [[value, value]]), moment)
    clock.value = 2.
    result = recorder.stop()
    meta = json.loads(result.meta_path.read_text(encoding="utf-8"))
    annotation = meta["annotation"]
    assert annotation["status"] == "auto"
    assert annotation["source"] == "contact_activation_v1"
    assert annotation["action_start_s"] == 1.0 and annotation["action_end_s"] == 1.2
    assert annotation["auto"]["contact_frames"] == 2
    assert annotation["auto"]["open_at_end"] is False
    # 人工确认覆盖自动值，旧值进入历史而不是被丢掉。
    saved = DatasetStore(tmp_path).update_annotation(result.meta_path, 1.05, 1.25)
    assert saved["annotation"]["status"] == "confirmed"
    assert saved["annotation"]["source"] == "manual"
    assert saved["annotation_history"][0]["status"] == "auto"


def test_single_frame_contact_stays_unlabeled(tmp_path):
    """只超阈值一帧时无法界定区间，宁可不写也不刮造。"""
    layout = make_layout(BlockLayout(1, 1, 2))
    calibration = make_calibration(layout, {1: 2500.})
    clock = FakeClock()
    recorder = RawSampleRecorder(clock=clock, wall_clock=cloud_clock)
    recorder.start(make_plan(tmp_path, layout, calibration))
    for moment, value in ((.5, 2500.), (1.0, 1900.), (1.5, 2500.)):
        clock.value = moment
        recorder.submit(Block(1, [[value, value]]), moment)
    clock.value = 2.
    result = recorder.stop()
    meta = json.loads(result.meta_path.read_text(encoding="utf-8"))
    assert meta["annotation"]["status"] == "unlabeled"
    assert meta["annotation"]["action_start_s"] is None
    assert meta["activation"]["contact_frames"] == 1


def test_no_calibration_leaves_annotation_unlabeled(tmp_path):
    layout = make_layout(BlockLayout(1, 1, 2))
    clock = FakeClock()
    recorder = RawSampleRecorder(clock=clock, wall_clock=cloud_clock)
    recorder.start(make_plan(tmp_path, layout, None))
    clock.value = 1.
    recorder.submit(Block(1, [[1500., 1500.]]), 1.)
    clock.value = 2.
    result = recorder.stop()
    meta = json.loads(result.meta_path.read_text(encoding="utf-8"))
    assert meta["annotation"]["status"] == "unlabeled"
    assert meta["activation"]["contact_start_s"] is None
