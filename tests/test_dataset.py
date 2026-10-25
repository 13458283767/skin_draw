import csv
import json
from datetime import datetime
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest

from skin_draw.collection_layout import CollectionBlock, CollectionLayout
from skin_draw.dataset import (ActionClass, ActionLabels, CalibrationSnapshot, DatasetStore,
                               RawSampleRecorder, RecordingPlan, layout_adc_columns)
from skin_draw.live import LiveBuffer
from skin_draw.mapping import BlockLayout
from skin_draw.models import Block, Frame
from skin_draw.sensor_masks import SensorMasks
from skin_draw.ui.main_window import MainWindow

WALL = datetime(2026, 10, 20, 15, 30, 0).astimezone()


class FakeClock:
    """Deterministic monotonic clock so relative CSV times are exact in tests."""

    def __init__(self):
        self.value = 0.

    def __call__(self):
        return self.value


def make_layout(*specs):
    blocks = tuple(CollectionBlock(spec, np.ones((spec.rows, spec.cols), bool)) for spec in specs)
    return CollectionLayout("skin_layout_v1", blocks, "grid", ())


def make_plan(root, layout, *, trial="pat001", expected=None, calibration=None):
    return RecordingPlan(
        sample_dir=root / "student01" / "session_20261020_01" / "pat" / trial,
        subject_id="student01", session_id="session_20261020_01", trial_id=trial,
        action=ActionClass(0, "pat", "轻拍"), layout=layout, labels_id="actions_v1",
        expected_block_ids=tuple(layout.by_id) if expected is None else expected,
        max_adc_columns=layout_adc_columns(layout), calibration=calibration)


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def test_raw_csv_reorders_fixed_direction_and_keeps_every_value(tmp_path):
    raw = np.array([[2048, 2050, 2047], [2052, 0, 4095]], dtype=np.uint16)
    second = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    layout = make_layout(BlockLayout(1, 2, 3, flip_y=True))
    clock = FakeClock()
    recorder = RawSampleRecorder(clock=clock, wall_clock=lambda: WALL)
    plan = make_plan(tmp_path, layout)
    recorder.start(plan)
    clock.value = 0.003
    recorder.submit(Block(1, raw), clock.value)
    clock.value = 0.0586
    recorder.submit(Block(1, second), clock.value)
    result = recorder.stop()

    rows = read_rows(plan.sample_dir / "raw.csv")
    assert rows[0] == ["record_index", "time_s", "block_id",
                       "adc_000", "adc_001", "adc_002", "adc_003", "adc_004", "adc_005"]
    # Flip is applied before flattening: the bottom row comes first, values unchanged.
    assert rows[1] == ["0", "0.003000", "1", "2052", "0", "4095", "2048", "2050", "2047"]
    assert rows[2][0] == "1" and abs(float(rows[2][1]) - 0.0586) < 1e-9
    assert rows[2][3:] == ["4", "5", "6", "1", "2", "3"]
    assert result.status == "complete" and result.records == 2
    meta = json.loads((plan.sample_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["recording_status"] == "complete"
    assert meta["end_reason"] == "user_stop"
    assert meta["summary"] == {"records": 2, "frames_by_block": {"1": 2}}
    assert meta["calibration_id"] is None
    assert meta["quality"]["missing_block_ids"] == []
    assert meta["time_reference"] == "host_receive_seconds_since_recording_start"


def test_short_block_pads_columns_and_missing_block_marks_incomplete(tmp_path):
    layout = make_layout(BlockLayout(1, 2, 3), BlockLayout(2, 2, 2))
    clock = FakeClock()
    recorder = RawSampleRecorder(clock=clock, wall_clock=lambda: WALL)
    recorder.start(make_plan(tmp_path, layout))
    clock.value = 0.5
    recorder.submit(Block(2, [[2039, 2041], [2040, 2045]]), clock.value)
    result = recorder.stop()

    rows = read_rows(result.sample_dir / "raw.csv")
    # Six columns are decided by BLOCK 1; BLOCK 2 leaves the two extra cells empty.
    assert rows[1] == ["0", "0.500000", "2", "2039", "2041", "2040", "2045", "", ""]
    assert result.status == "incomplete"
    assert result.missing_block_ids == (1,)
    meta = json.loads((result.sample_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["recording_status"] == "incomplete"
    assert meta["end_reason"] == "missing_block"
    assert meta["quality"]["missing_block_ids"] == [1]
    assert meta["expected_block_ids"] == [1, 2]


def test_sample_directory_is_never_overwritten(tmp_path):
    layout = make_layout(BlockLayout(1, 2, 2))
    recorder = RawSampleRecorder(clock=FakeClock(), wall_clock=lambda: WALL)
    plan = make_plan(tmp_path, layout)
    recorder.start(plan)
    recorder.stop()
    before = (plan.sample_dir / "raw.csv").read_bytes()
    with pytest.raises(ValueError, match="已存在"):
        RawSampleRecorder(clock=FakeClock(), wall_clock=lambda: WALL).start(plan)
    assert (plan.sample_dir / "raw.csv").read_bytes() == before


def test_shape_mismatch_is_reported_and_never_silently_saved(tmp_path):
    layout = make_layout(BlockLayout(1, 2, 3))
    clock = FakeClock()
    recorder = RawSampleRecorder(clock=clock, wall_clock=lambda: WALL)
    result_plan = make_plan(tmp_path, layout)
    recorder.start(result_plan)
    clock.value = 0.1
    recorder.submit(Block(1, [[1, 2], [3, 4]]), clock.value)
    clock.value = 0.2
    recorder.submit(Block(2, [[1, 2, 3]]), clock.value)  # not in the fixed layout
    result = recorder.stop()

    rows = read_rows(result.sample_dir / "raw.csv")
    assert len(rows) == 1  # header only: neither frame could be described
    assert result.status == "incomplete"
    assert any("尺寸" in message for message in result.errors)
    assert any("不在固定采集布局" in message for message in result.errors)


def test_calibration_snapshot_uses_raw_order_and_null_for_unusable_cells():
    baseline = np.array([[np.nan, 2000.], [2100., np.nan]])
    snapshot = CalibrationSnapshot.from_baselines("calib_001", WALL.isoformat(), {3: baseline},
                                                  calib_samples=270)
    payload = snapshot.to_dict()
    assert payload["coordinates"] == "raw_row_col"
    assert payload["baselines"] == {"3": [[None, 2000.0], [2100.0, None]]}
    assert payload["calib_samples"] == 270
    assert snapshot.baselines[3][0][0] is None


def test_labels_sessions_and_index_roundtrip(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    labels = ActionLabels("actions_v1", (ActionClass(0, "pat", "轻拍"), ActionClass(1, "press", "按压")))
    store.ensure_labels(labels)
    layout = make_layout(BlockLayout(1, 2, 3, flip_y=True))
    store.ensure_layout(layout)
    store.open_session(subject_id="student01", session_id="session_20261020_01", layout=layout,
                       labels=labels, max_adc_columns=6, expected_block_ids=(1,),
                       transport={"baudrate": 115200})
    assert store.load_labels().to_dict() == labels.to_dict()
    assert (tmp_path / "dataset" / "labels.json").exists()
    assert (tmp_path / "dataset" / "skin_layout_v1.json").exists()
    session = json.loads(store.session_path("student01", "session_20261020_01").read_text(encoding="utf-8"))
    assert session["layout_id"] == "skin_layout_v1"
    assert session["protocol"] == {"baudrate": 115200}
    # A different ADC column count cannot join an existing session.
    with pytest.raises(ValueError, match="ADC 列"):
        store.open_session(subject_id="student01", session_id="session_20261020_01", layout=layout,
                           labels=labels, max_adc_columns=8, expected_block_ids=(1,))
    assert store.next_trial_number("student01", "session_20261020_01", labels.classes[0]) == 1
    # Reusing a layout id with different content is refused, not overwritten.
    with pytest.raises(ValueError, match="不一致"):
        store.ensure_layout(make_layout(BlockLayout(1, 2, 3, rotation=90)))


@pytest.mark.parametrize("value", ["../escape", "a b", "", "x/y", ".hidden"])
def test_invalid_dataset_names_are_rejected(tmp_path, value):
    store = DatasetStore(tmp_path / "dataset")
    with pytest.raises(ValueError):
        store.session_path(value, "session_20261020_01")
    with pytest.raises(ValueError):
        ActionClass(0, value, "轻拍")


def test_recording_end_to_end_with_ui_and_index(app, tmp_path):
    raw = np.array([[2048, 2050, 2047], [2052, 2500, 4095]], dtype=np.uint16)
    window = MainWindow(mask_path=tmp_path / "masks.json", dataset_dir=tmp_path / "dataset")
    try:
        window.present_frame(Frame((Block(7, raw),)))
        window.save_collection_layout()
        buffer = LiveBuffer()
        window.worker = SimpleNamespace(buffer=buffer)
        window.attach_recording_sink(buffer)
        window.input_source = "hardware"
        window._refresh_recording_availability()
        buffer.receive(Block(7, raw))  # first frame is needed before a sample may start
        assert window.collection_bar.start.isEnabled()

        window.collection_bar.subject.setCurrentText("student01")
        window.collection_bar.session.setCurrentText("session_20261020_01")
        plan = window.start_recording()
        assert plan.trial_id == "pat001"
        assert window.collection_bar.stop.isEnabled()
        assert not window.collection_bar.subject.isEnabled()
        assert not window.collection_bar.add_subject.isEnabled()
        assert not window.collection_bar.delete_session.isEnabled()
        assert not window.collection_bar.restore_identity.isEnabled()
        assert not window.controls.rotate_left.isEnabled()
        with pytest.raises(ValueError, match="正在采集"):
            window.save_collection_layout()

        buffer.receive(Block(7, raw))
        buffer.receive(Block(7, raw))
        result = window.finish_recording()
        assert result.status == "complete"
        assert result.records == 2
        assert window.collection_bar.subject.isEnabled()
        assert window.collection_bar.mark_valid.isEnabled()
        assert window.collection_bar.trial.text() == "pat002"

        window.mark_sample_accepted(True)
        index = read_rows(tmp_path / "dataset" / "index.csv")
        assert index[0] == ["subject_id", "session_id", "trial_id", "action_id",
                            "duration_s", "accepted", "sample_path", "recording_status", "quality_status"]
        assert index[1][:4] == ["student01", "session_20261020_01", "pat001", "0"]
        assert index[1][5] == "true"
        assert index[1][6] == "student01/session_20261020_01/pat/pat001"
        meta = json.loads((tmp_path / "dataset/student01/session_20261020_01/pat/pat001/meta.json")
                          .read_text(encoding="utf-8"))
        assert meta["accepted"] is True
        assert meta["layout_id"] == window.collection_layout.layout_id
    finally:
        window.worker = None
        window.close()


def test_recording_requires_hardware_fixed_layout_and_known_action(app, tmp_path):
    window = MainWindow(mask_path=tmp_path / "masks.json", dataset_dir=tmp_path / "dataset")
    try:
        with pytest.raises(ValueError, match="连接实际皮肤"):
            window.start_recording()
        window.present_frame(Frame((Block(7, [[1, 2, 3], [4, 5, 6]]),)))
        window.save_collection_layout()
        window.worker = SimpleNamespace(buffer=SimpleNamespace(blocks={}, lock=Lock()))
        window.input_source = "hardware"
        window._edit_collection_layout()  # layout no longer fixed
        with pytest.raises(ValueError, match="固定"):
            window.start_recording()
        window.collection_locked = True  # re-lock without touching the view
        with pytest.raises(ValueError, match="尚未收到数据"):
            window.start_recording()
        assert not (tmp_path / "dataset").exists()
    finally:
        window.worker = None
        window.close()


def test_closing_while_recording_finalises_meta(app, tmp_path):
    raw = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    window = MainWindow(mask_path=tmp_path / "masks.json", dataset_dir=tmp_path / "dataset")
    buffer = LiveBuffer()
    try:
        window.present_frame(Frame((Block(7, raw),)))
        window.save_collection_layout()
        window.worker = SimpleNamespace(buffer=buffer)
        window.attach_recording_sink(buffer)
        window.input_source = "hardware"
        buffer.receive(Block(7, raw))
        plan = window.start_recording()
        buffer.receive(Block(7, raw))
        window.worker = None
        window.close()
        meta = json.loads((plan.sample_dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["recording_status"] in ("complete", "incomplete")
        assert meta["end_reason"] == "app_exit"
        assert meta["duration_s"] > 0
    finally:
        window.worker = None


def test_raw_sink_receives_unprocessed_block_on_worker_thread():
    seen = []
    buffer = LiveBuffer(norm="B")
    buffer.set_raw_sink(lambda block, received_at: seen.append((block, received_at)))
    raw = np.array([[0, 4095], [2500, 1800]], dtype=np.uint16)
    buffer.receive(Block(3, raw))
    assert len(seen) == 1
    assert seen[0][0] is not None
    np.testing.assert_array_equal(seen[0][0].values, raw)
