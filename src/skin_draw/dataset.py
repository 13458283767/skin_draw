"""Training-sample recording for docs/训练数据采集与存储方案_CSV版.md.

The recording path is deliberately independent of Qt and of the display
preprocessing chain. Raw parsed blocks arrive on the serial thread, are
re-ordered by the fixed collection layout and appended to ``raw.csv`` by a
background writer. Nothing here filters, subtracts a baseline or discards
readings: whatever the hardware returned is what the file contains.

Concurrency contract
--------------------
``submit`` is called from the serial thread and never raises. ``start``,
``stop`` and ``snapshot`` are called from the GUI thread. ``stop`` flips the
accept flag under the same lock as ``submit`` and only then drains the queue,
so a late block can never leak into the next sample's file.

Data contract
-------------
Sample: ``<root>/<subject>/<session>/<action>/<trial>/raw.csv + meta.json``.
The CSV column set is fixed per session (``max_adc_columns``) and every row is
one block frame, not a synchronised whole-body frame.
"""

from dataclasses import dataclass, field
from datetime import datetime
import csv
import json
import logging
import math
import os
from pathlib import Path
from queue import Empty, Full, Queue
import re
import tempfile
import threading
from uuid import uuid4
from time import perf_counter

import numpy as np

from .collection_activation import ActivationPolicy, ActivationTracker, semantic_status
from .collection_layout import CollectionBlock, CollectionLayout
from .collection_quality import LEVELS, IntegrityMonitor, QualityPolicy

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
LABELS_SCHEMA_VERSION = "1.0"
# Same shape used by the dataset directory names; rejects path separators,
# dots and whitespace so an operator typo can never escape the dataset root.
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
TRIAL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
MAX_ERRORS_KEPT = 20
MAX_ADC_COLUMNS = 256  # hardest possible block is 16x16, see models.MAX_SIDE

DEFAULT_LABELS = (
    (0, "pat", "轻拍", "轻拍机器人手臂或肩部后离开", "短暂、局部接触"),
    (1, "grasp", "握持", "握住机器人的手或前臂，保持后松开", "持续、局部接触"),
    (2, "hug", "拥抱", "双臂环抱机器人，保持后松开", "多个区域的持续接触，取决于传感器覆盖"),
    (3, "press", "推压", "手掌贴住目标块持续前推，力由轻到重，不离开皮肤", "单区域、持续、力度递增"),
)


def _local_now() -> datetime:
    return datetime.now().astimezone()


def _atomic_write_text(path: Path, text: str):
    """Write through a temporary file, then replace, so meta JSON is never half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload):
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"无法读取 {Path(path).name}：{error}") from error


@dataclass(frozen=True)
class ActionClass:
    id: int
    name: str
    display_name: str
    # 采集定义随类别一起保存；旧类别表缺少这两个字段时仍可读取。
    definition: str = ""
    contact_features: str = ""

    def __post_init__(self):
        if type(self.id) is not int or self.id < 0:
            raise ValueError("动作编号必须为非负整数")
        if not isinstance(self.name, str) or not NAME_PATTERN.fullmatch(self.name):
            raise ValueError("动作名称只能包含字母、数字、下划线和连字符")
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError("动作需要中文显示名")
        if not isinstance(self.definition, str) or not isinstance(self.contact_features, str):
            raise ValueError("动作定义和接触特点必须为文本")

    def to_dict(self):
        result = {"id": self.id, "name": self.name, "display_name": self.display_name}
        # 不给历史类别表补写空字段，保持已有数据集的内容一致性检查。
        if self.definition:
            result["definition"] = self.definition
        if self.contact_features:
            result["contact_features"] = self.contact_features
        return result


@dataclass(frozen=True)
class ActionLabels:
    """Global, stable action ids; the GUI order must never renumber them."""

    labels_id: str
    classes: tuple[ActionClass, ...]

    def __post_init__(self):
        classes = tuple(self.classes)
        if not classes:
            raise ValueError("动作类别表不能为空")
        if len({c.id for c in classes}) != len(classes) or len({c.name for c in classes}) != len(classes):
            raise ValueError("动作编号和名称不得重复")
        object.__setattr__(self, "classes", classes)

    def by_id(self):
        return {item.id: item for item in self.classes}

    def to_dict(self):
        return {"schema_version": LABELS_SCHEMA_VERSION, "labels_id": self.labels_id,
                "classes": [item.to_dict() for item in self.classes]}

    @classmethod
    def from_dict(cls, data):
        try:
            if data["schema_version"] != LABELS_SCHEMA_VERSION:
                raise ValueError("不支持的动作类别表版本")
            return cls(data["labels_id"], tuple(ActionClass(**item) for item in data["classes"]))
        except (KeyError, TypeError) as error:
            raise ValueError(f"动作类别表无效：{error}") from error


def default_labels() -> ActionLabels:
    # 动作含义发生变化必须更新版本，防止旧场次的编号被解释成新动作。
    return ActionLabels("actions_v3", tuple(ActionClass(*item) for item in DEFAULT_LABELS))


@dataclass(frozen=True)
class CalibrationSnapshot:
    """One static baseline snapshot referenced by the sample's calibration_id.

    Baselines are stored in raw hardware row/column order. They are a snapshot,
    not a full history of the running slow tracker; a sample reproduces the
    baseline that was active when recording started, nothing more.
    """

    calibration_id: str
    captured_at: str
    baselines: dict[int, tuple[tuple[float | None, ...], ...]]
    method: str = "runtime_slow_tracking_snapshot"
    coordinates: str = "raw_row_col"
    baseline_mode: str = "slow"
    calib_samples: int = 0

    def to_dict(self):
        return {
            "calibration_id": self.calibration_id,
            "captured_at": self.captured_at,
            "method": self.method,
            "coordinates": self.coordinates,
            "baseline_mode": self.baseline_mode,
            "calib_samples": int(self.calib_samples),
            "confirmed_bad_points": [],
            # JSON has no NaN: unusable cells are null, never a fake number.
            "baselines": {str(key): [list(row) for row in matrix] for key, matrix in self.baselines.items()},
        }

    @classmethod
    def from_baselines(cls, calibration_id, captured_at, baselines, *, calib_samples=0):
        matrices = {}
        for key, array in sorted(baselines.items()):
            matrix = np.asarray(array, dtype=float)
            if matrix.ndim != 2:
                raise ValueError("基线必须是二维矩阵")
            rows = [[None if not math.isfinite(value) else round(float(value), 4) for value in row]
                    for row in matrix]
            if any(value is not None for row in rows for value in row):
                matrices[int(key)] = tuple(tuple(row) for row in rows)
        return cls(str(calibration_id), str(captured_at), matrices, calib_samples=calib_samples)


@dataclass(frozen=True)
class RecordingPlan:
    """Everything decided before the first row is accepted; identity is fixed."""

    sample_dir: Path
    subject_id: str
    session_id: str
    trial_id: str
    action: ActionClass
    layout: CollectionLayout
    labels_id: str
    expected_block_ids: tuple[int, ...]
    max_adc_columns: int
    calibration: CalibrationSnapshot | None = None
    blocks: dict[int, CollectionBlock] = field(default_factory=dict, compare=False, repr=False)
    quality_policy: QualityPolicy = field(default_factory=QualityPolicy)
    activation_policy: ActivationPolicy = field(default_factory=ActivationPolicy)
    cadence_references: tuple[tuple[int, float | None, int], ...] = ()

    def __post_init__(self):
        for name, value in (("subject_id", self.subject_id), ("session_id", self.session_id),
                            ("trial_id", self.trial_id)):
            if not isinstance(value, str) or not NAME_PATTERN.fullmatch(value):
                raise ValueError(f"{name} 只能包含字母、数字、下划线和连字符")
        if not TRIAL_PATTERN.fullmatch(self.trial_id):
            raise ValueError("次数编号格式无效")
        if not 1 <= self.max_adc_columns <= MAX_ADC_COLUMNS:
            raise ValueError("ADC 列数必须在 1～256 之间")
        ids = tuple(sorted(self.expected_block_ids))
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("预期皮肤块编号不能为空或重复")
        known = self.layout.by_id
        if any(key not in known for key in ids):
            raise ValueError("预期皮肤块必须存在于固定采集布局中")
        object.__setattr__(self, "sample_dir", Path(self.sample_dir))
        object.__setattr__(self, "expected_block_ids", ids)
        # Freeze the mapping used for this recording: a later layout edit in the
        # GUI must not change the column order of an already accepted sample.
        object.__setattr__(self, "blocks", {key: known[key] for key in ids})
        references = tuple(tuple(item) for item in self.cadence_references)
        if len({item[0] for item in references}) != len(references):
            raise ValueError("接收参考的块编号不得重复")
        for key, interval, count in references:
            if key not in ids or type(count) is not int or count < 0:
                raise ValueError("接收参考编号或样本数无效")
        object.__setattr__(self, "cadence_references", references)
        IntegrityMonitor(ids, self.quality_policy, references)  # 写文件前验证冻结规则。


@dataclass(frozen=True)
class RecordingStats:
    active: bool
    duration_s: float
    records: int
    frames_by_block: dict[int, int]
    missing_block_ids: tuple[int, ...]
    dropped: int
    errors: tuple[str, ...]
    integrity: dict = field(default_factory=dict)
    fatal_error: str = ""


@dataclass(frozen=True)
class SampleResult:
    sample_dir: Path
    status: str
    end_reason: str
    duration_s: float
    records: int
    written_records: int
    frames_by_block: dict[int, int]
    missing_block_ids: tuple[int, ...]
    errors: tuple[str, ...] = ()
    meta_path: Path | None = None
    meta_error: str = ""
    integrity: dict = field(default_factory=dict)
    activation: dict = field(default_factory=dict)
    semantic: dict = field(default_factory=dict)


class _SampleWriter:
    """Bounded queue + dedicated thread: disk stalls must not block reception.

    The CSV header is written synchronously in ``start`` so an unwritable
    dataset directory fails before recording is announced as started.
    """

    def __init__(self, path: Path, header, *, flush_interval=0.5, capacity=4096):
        self.path = Path(path)
        self.header = list(header)
        self.flush_interval = float(flush_interval)
        self.queue: Queue = Queue(maxsize=capacity)
        self.written = 0
        self.dropped = 0
        self.error = ""
        self._stream = None
        self._thread = None

    def start(self):
        self._stream = self.path.open("w", encoding="utf-8", newline="")
        try:
            csv.writer(self._stream, lineterminator="\n").writerow(self.header)
            self._stream.flush()
        except BaseException:
            self._stream.close()
            self._stream = None
            raise
        self._thread = threading.Thread(target=self._run, name="skin-sample-csv", daemon=True)
        self._thread.start()

    def submit(self, row) -> bool:
        if self.error:
            return False
        try:
            self.queue.put_nowait(row)
            return True
        except Full:
            self.dropped += 1
            return False

    def _flush(self):
        if self._stream is not None:
            self._stream.flush()

    def _run(self):
        writer = csv.writer(self._stream, lineterminator="\n")
        last_flush = perf_counter()
        try:
            while True:
                try:
                    row = self.queue.get(timeout=.25)
                except Empty:
                    if perf_counter() - last_flush >= self.flush_interval:
                        self._flush()
                        last_flush = perf_counter()
                    continue
                if row is None:
                    break
                writer.writerow(row)
                self.written += 1
                if perf_counter() - last_flush >= self.flush_interval:
                    self._flush()
                    last_flush = perf_counter()
        except (OSError, ValueError) as error:
            # Stop writing at the first failure instead of spinning on a dead disk.
            self.error = f"写盘失败：{error}"
        finally:
            try:
                self._stream.flush()
                os.fsync(self._stream.fileno())
            except OSError as error:
                self.error = self.error or f"刷新写盘失败：{error}"
            try:
                self._stream.close()
            except OSError as error:
                self.error = self.error or f"关闭文件失败：{error}"

    def finish(self):
        if self._thread is None:
            return
        try:
            self.queue.put(None, timeout=5)
        except Full:
            self.error = self.error or "写盘队列拥塞，未能排空"
        self._thread.join(15)
        if self._thread.is_alive():
            self.error = self.error or "写盘线程未在 15 秒内结束"
        self._thread = None


class RawSampleRecorder:
    """Owns the accept/stop boundary and per-sample counters for one sample."""

    def __init__(self, clock=perf_counter, wall_clock=_local_now):
        self.clock = clock
        self.wall_clock = wall_clock
        self._lock = threading.Lock()
        self._plan: RecordingPlan | None = None
        self._writer: _SampleWriter | None = None
        self._accepting = False
        self._started_clock = 0.
        self._started_at = ""
        self._ended_at = ""
        self._records = 0
        self._last_time = 0.
        self._frames: dict[int, int] = {}
        self._first_time: dict[int, float] = {}
        self._last_block_time: dict[int, float] = {}
        self._max_gap: dict[int, float] = {}
        self._errors: list[str] = []
        self._error_count = 0
        self.last_result: SampleResult | None = None
        self._integrity = None
        self._activation: ActivationTracker | None = None

    @property
    def active(self) -> bool:
        return self._accepting

    def _error_locked(self, message):
        self._error_count += 1
        if message not in self._errors and len(self._errors) < MAX_ERRORS_KEPT:
            self._errors.append(message)

    def start(self, plan: RecordingPlan):
        if not isinstance(plan, RecordingPlan):
            raise ValueError("录制计划无效")
        with self._lock:
            if self._accepting:
                raise ValueError("上一段样本尚未结束")
            if plan.sample_dir.exists():
                # Re-recording never overwrites; the caller must pick a new trial id.
                raise ValueError(f"样本目录已存在，未覆盖：{plan.sample_dir}")
        header = ["record_index", "time_s", "block_id",
                  *(f"adc_{index:03d}" for index in range(plan.max_adc_columns))]
        plan.sample_dir.mkdir(parents=True, exist_ok=False)
        writer = _SampleWriter(plan.sample_dir / "raw.csv", header)
        integrity = IntegrityMonitor(plan.expected_block_ids, plan.quality_policy, plan.cadence_references)
        activation = ActivationTracker(plan.blocks, plan.calibration, plan.activation_policy)
        try:
            writer.start()
            started_at = self.wall_clock().isoformat(timespec="milliseconds")
            self._write_meta(plan, "recording", "", started_at, started_at, 0.,
                             {}, {}, {}, {}, [], (), writer, 0,
                             integrity=integrity.evaluate(0.), activation={},
                             semantic=semantic_status({}, plan.action.name, plan.activation_policy))
        except BaseException:
            writer.finish()
            raise
        with self._lock:
            self._plan, self._writer = plan, writer
            self._started_clock = self.clock()
            self._started_at, self._ended_at = started_at, ""
            self._records, self._last_time = 0, 0.
            self._frames, self._first_time, self._last_block_time, self._max_gap = {}, {}, {}, {}
            self._errors, self._error_count = [], 0
            self._integrity = integrity
            self._activation = activation
            self._accepting = True
            self.last_result = None

    def submit(self, block, received_at):
        """Serial-thread entry point: record one raw block frame if accepted.

        Recording must never break reception, so unexpected errors only abort the
        current row and are reported through the sample's quality errors.
        """
        try:
            self._submit(block, received_at)
        except Exception:
            logger.exception("录制原始块失败 block_id=%s", getattr(block, "block_id", None))
            with self._lock:
                self._error_locked("录制内部错误，该帧未保存")

    def _submit(self, block, received_at):
        with self._lock:
            if not self._accepting:
                return
            plan, writer = self._plan, self._writer
            saved = plan.blocks.get(block.block_id)
            if saved is None:
                self._error_locked(f"BLOCK {block.block_id:02d} 不在固定采集布局中，已跳过")
                return
            raw = block.values
            if raw.shape != (saved.spec.rows, saved.spec.cols):
                self._error_locked(
                    f"BLOCK {block.block_id:02d} 尺寸 {raw.shape[0]}×{raw.shape[1]} 与固定布局 "
                    f"{saved.spec.rows}×{saved.spec.cols} 不符，已跳过")
                return
            # Reorder into the fixed body direction first, then flatten row-major;
            # cell values are untouched, including zero and out-of-range-looking ADC.
            ordered = np.asarray(saved.transform(raw)).ravel(order="C")
            if ordered.size > plan.max_adc_columns:
                self._error_locked(f"BLOCK {block.block_id:02d} 点数超出本场次固定列数，已跳过")
                return
            time_s = float(received_at) - self._started_clock
            if not math.isfinite(time_s):
                self._error_locked("接收时间不是有限数，该帧未保存")
                return
            if time_s < 0:
                self._error_locked("收到的时间早于录制起点，已按 0 秒记录")
                time_s = 0.
            if time_s < self._last_time:
                self._error_locked("相对时间出现倒退，请检查时钟源")
            row = [self._records, f"{time_s:.6f}", block.block_id]
            row.extend(int(value) for value in ordered)
            row.extend([""] * (plan.max_adc_columns - ordered.size))
            if not writer.submit(row):
                self._error_locked("写盘队列已满或已失败，该帧未保存")
                return
            key = block.block_id
            self._records += 1
            self._last_time = time_s
            self._frames[key] = self._frames.get(key, 0) + 1
            previous = self._last_block_time.get(key)
            if previous is not None:
                self._max_gap[key] = max(self._max_gap.get(key, 0.), time_s - previous)
            else:
                self._first_time[key] = time_s
            self._last_block_time[key] = time_s
            self._integrity.receive(key, time_s)
            if self._activation is not None:
                self._activation.observe(key, ordered)

    def snapshot(self) -> RecordingStats:
        with self._lock:
            plan = self._plan
            duration = max(0., self.clock() - self._started_clock) if self._accepting else 0.
            missing = ()
            if plan is not None:
                missing = tuple(key for key in plan.expected_block_ids if not self._frames.get(key))
            fatal = self._writer.error if self._writer else ""
            if self._writer and self._writer.dropped:
                fatal = fatal or "写盘队列溢出"
            failures = ([fatal] if fatal else []) + list(self._errors)
            integrity = self._integrity.evaluate(duration, failures=failures) if self._accepting else {}
            return RecordingStats(self._accepting, round(duration, 2), self._records,
                                  dict(self._frames), missing,
                                  self._writer.dropped if self._writer else 0, tuple(self._errors),
                                  integrity, fatal)

    def stop(self, reason="user_stop"):
        """Close the boundary, drain the writer, then finalise meta.json."""
        with self._lock:
            if not self._accepting:
                return None
            self._accepting = False
            plan, writer = self._plan, self._writer
            duration = max(0., self.clock() - self._started_clock)
            ended_at = self.wall_clock().isoformat(timespec="milliseconds")
            records = self._records
            frames = dict(self._frames)
            first_time = dict(self._first_time)
            last_time = dict(self._last_block_time)
            max_gap = dict(self._max_gap)
            errors = tuple(self._errors)
            integrity_monitor = self._integrity
            activation_tracker = self._activation
        try:
            writer.finish()
        except OSError as error:
            writer.error = writer.error or f"排空写盘队列失败：{error}"
        missing = tuple(key for key in plan.expected_block_ids if not frames.get(key))
        written = writer.written
        problems = []
        if writer.error or written != records:
            problems.append("write_error")
        if writer.dropped:
            problems.append("queue_overflow")
        if errors:
            problems.append("data_anomaly")
        if missing:
            problems.append("missing_block")
        if reason != "user_stop":
            problems.append("abnormal_end")
        status = "complete" if not problems else "incomplete"
        # A normal user stop keeps its reason; data problems become the reason only
        # when the operator did not stop for an abnormal connection event.
        end_reason = reason if reason != "user_stop" else (problems[0] if problems else "user_stop")
        failure_names = {"write_error": "文件写入或关闭失败", "queue_overflow": "写盘队列溢出",
                         "data_anomaly": "存在未正常保存的数据帧", "missing_block": "预期皮肤块整段缺失",
                         "abnormal_end": f"录制异常结束：{reason}"}
        # 用接纳边界的结束时刻检查尾部，磁盘排空耗时不计为传感器断流。
        integrity = integrity_monitor.evaluate(duration, final=True,
                                               failures=[failure_names[key] for key in problems])
        activation = activation_tracker.summary() if activation_tracker is not None else {}
        semantic = semantic_status(activation, plan.action.name, plan.activation_policy)
        meta_error = ""
        meta_path = plan.sample_dir / "meta.json"
        try:
            self._write_meta(plan, status, end_reason, self._started_at, ended_at, duration,
                             frames, first_time, last_time, max_gap, missing, errors, writer,
                             written, max(0, records - written), integrity=integrity,
                             activation=activation, semantic=semantic)
        except (OSError, ValueError) as error:
            meta_error = f"元数据写入失败：{error}"
            status = "incomplete"
            end_reason = "metadata_error"
            integrity["status"] = "fail"
            integrity["reasons"].append(meta_error)
        with self._lock:
            self._plan = None
            self._writer = None
            self._activation = None
        result = SampleResult(plan.sample_dir, status, end_reason, round(duration, 3), records, written,
                              frames, missing, errors, meta_path, meta_error, integrity,
                              activation, semantic)
        self.last_result = result
        return result

    def _write_meta(self, plan, status, end_reason, started_at, ended_at, duration,
                    frames, first_time, last_time, max_gap, missing, errors, writer,
                    written, unwritten=0, *, integrity, activation=None, semantic=None):
        rates = {}
        for key, count in frames.items():
            span = last_time.get(key, 0.) - first_time.get(key, 0.)
            rates[str(key)] = round((count - 1) / span, 3) if count >= 2 and span > 0 else None
        activation = dict(activation or {})
        semantic = dict(semantic or {})
        # 完整性检查与接触检查取较严重的一方；尚未评估的保持 pending。
        quality_status = integrity["status"] if integrity["final"] else "pending"
        semantic_status_name = semantic.get("status")
        if semantic_status_name in LEVELS and LEVELS[semantic_status_name] > LEVELS.get(quality_status, 0):
            quality_status = semantic_status_name
        payload = {
            "schema_version": "1.2",
            "subject_id": plan.subject_id,
            "session_id": plan.session_id,
            "trial_id": plan.trial_id,
            "layout_id": plan.layout.layout_id,
            "labels_id": plan.labels_id,
            "calibration_id": plan.calibration.calibration_id if plan.calibration else None,
            "action": plan.action.to_dict(),
            "data_file": "raw.csv",
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_s": round(float(duration), 6),
            "time_reference": "host_receive_seconds_since_recording_start",
            "adc_dtype": "uint16",
            "coordinates": "body_oriented_row_col",
            "flatten_order": "row_major",
            "max_adc_columns": plan.max_adc_columns,
            "expected_block_ids": list(plan.expected_block_ids),
            # Action boundaries are a human annotation, never the button times.
            "annotation": {"scope": "trial", "action_start_s": None, "action_end_s": None},
            "recording_status": status,
            "quality_status": quality_status,
            "end_reason": end_reason,
            "accepted": None,
            "reject_reason": "",
            # 接触区域摘要：这批数据要论证"动作识别与接触位置无关"，必须逐条记录。
            "primary_block": activation.get("primary_block"),
            "region_set": activation.get("region_set"),
            "region_count": activation.get("region_count"),
            "activation": activation,
            "summary": {"records": int(written),
                        "frames_by_block": {str(key): int(count) for key, count in sorted(frames.items())}},
            "quality": {
                "integrity": integrity,
                "semantic": semantic,
                "receive_rate_hz": rates,
                "max_receive_interval_s": {str(key): round(value, 6)
                                           for key, value in sorted(max_gap.items())},
                "missing_block_ids": list(missing),
                "dropped_records": int(writer.dropped),
                "unwritten_records": int(unwritten),
                "errors": list(errors),
            },
            "notes": "原始 ADC 未滤波、未扣基线；有效点位与通道映射见 layout_id 指向的布局文件",
        }
        atomic_write_json(plan.sample_dir / "meta.json", payload)


class DatasetStore:
    """Dataset root layout: shared labels/layout JSON, sessions, samples, index."""

    INDEX_HEADER = ["subject_id", "session_id", "trial_id", "action_id",
                    "duration_s", "accepted", "sample_path", "recording_status", "quality_status"]

    def __init__(self, root):
        self.root = Path(root)

    @property
    def labels_path(self) -> Path:
        return self.root / "labels.json"

    @property
    def index_path(self) -> Path:
        return self.root / "index.csv"

    def _identity_path(self, subject_id, session_id=None):
        """只管理数据集内的真实目录，拒绝路径穿越、符号链接和 Windows 联接。"""
        self._check_name(subject_id, "受试者编号")
        parts = [subject_id]
        if session_id is not None:
            self._check_name(session_id, "场次编号")
            parts.append(session_id)
        root = self.root.resolve()
        path = root
        for part in parts:
            path = path / part
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                raise ValueError("受试者和场次不能使用链接目录")
            if not path.resolve().is_relative_to(root):
                raise ValueError("目录超出数据集范围")
        return path

    @staticmethod
    def _directory_names(path):
        if not path.exists():
            return []
        # _trash 不符合编号规则，故不会作为受试者或场次出现在列表中。
        return sorted(item.name for item in path.iterdir()
                      if NAME_PATTERN.fullmatch(item.name) and item.is_dir()
                      and not item.is_symlink()
                      and not (hasattr(item, "is_junction") and item.is_junction()))

    def list_subjects(self):
        return self._directory_names(self.root)

    def list_sessions(self, subject_id):
        return self._directory_names(self._identity_path(subject_id))

    def create_subject(self, subject_id):
        path = self._identity_path(subject_id)
        self.root.mkdir(parents=True, exist_ok=True)
        path.mkdir()  # 已存在即报错，不能把“新增”悄悄变成选择旧数据。
        return path

    def create_session(self, subject_id, session_id):
        parent = self._identity_path(subject_id)
        if not parent.is_dir():
            raise ValueError("请先新增或选择已有受试者")
        path = self._identity_path(subject_id, session_id)
        # 新建空场次；首次采集时才写布局、类别版本等实验条件。
        path.mkdir()
        return path

    def next_identity(self, prefix, subject_id=None):
        """新增建议递增编号；回收中的编号也保留，避免恢复时发生名称冲突。"""
        names = set(self.list_subjects() if subject_id is None else self.list_sessions(subject_id))
        for manifest in (self.root / "_trash").glob("*/deleted.json"):
            try:
                info = read_json(manifest)
                parts = Path(info["original_path"]).parts
                if not parts or not (manifest.parent / "data").is_dir():
                    continue
            except (ValueError, KeyError, TypeError):
                continue
            if subject_id is None:
                names.add(parts[0])
            elif len(parts) == 2 and parts[0] == subject_id:
                names.add(parts[1])
        pattern = re.compile(re.escape(prefix) + r"([0-9]+)\Z")
        numbers = [int(match.group(1)) for name in names if (match := pattern.fullmatch(name))]
        return f"{prefix}{max(numbers, default=0) + 1:02d}"

    def archive_identity(self, subject_id, session_id=None):
        """把整棵目录移入本地回收区；索引写入失败则回滚移动，保留原列表。"""
        source = self._identity_path(subject_id, session_id)
        if not source.is_dir():
            raise ValueError("所选受试者或场次已不存在，请刷新列表")
        return self._archive_path(source, "subject" if session_id is None else "session")

    def _sample_path(self, parts):
        """样本只能位于四层正式目录，逐级拒绝链接与路径穿越。"""
        if len(parts) != 4:
            raise ValueError("样本路径必须包含受试者、场次、动作和样本编号")
        path = self._identity_path(*parts[:2])
        for part in parts[2:]:
            self._check_name(part, "样本路径")
            path = path / part
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                raise ValueError("样本目录不能使用链接")
            if not path.resolve().is_relative_to(self.root.resolve()):
                raise ValueError("样本路径超出数据集范围")
        return path

    def archive_sample(self, meta_path):
        """作废后移出正式数据集；移动或索引失败时恢复原元数据。"""
        meta_path = Path(meta_path).absolute()
        parts = meta_path.parent.relative_to(self.root.resolve()).parts
        source = self._sample_path(parts)
        if meta_path.name != "meta.json" or meta_path.is_symlink():
            raise ValueError("样本元数据路径无效")
        original = read_json(meta_path)
        self.update_accepted(meta_path, False)
        try:
            return self._archive_path(source, "sample")
        except Exception:
            atomic_write_json(meta_path, original)
            raise

    def _archive_path(self, source, kind):
        """共用同盘回收及索引回滚，成功后正式目录不再包含该数据。"""
        root = self.root.resolve()
        trash = root / "_trash"
        if trash.is_symlink() or (hasattr(trash, "is_junction") and trash.is_junction()):
            raise ValueError("回收目录不能是链接")
        trash.mkdir(exist_ok=True)
        archive = trash / uuid4().hex
        archive.mkdir()
        destination = archive / "data"
        # 在移动前检查最终绝对路径；只允许同一数据集内的可恢复重命名。
        if not destination.resolve().is_relative_to(root):
            raise ValueError("回收路径超出数据集范围")
        atomic_write_json(archive / "deleted.json", {
            "original_path": source.relative_to(root).as_posix(),
            "deleted_at": self._now_iso(),
            "kind": kind,
        })
        try:
            source.rename(destination)
            self.rebuild_index()
        except Exception:
            if destination.exists():
                destination.rename(source)
            (archive / "deleted.json").unlink()
            archive.rmdir()
            raise
        return archive

    def archived_identities(self):
        """只列出仍有数据的回收项；已恢复项不再显示。"""
        items = []
        for manifest in (self.root / "_trash").glob("*/deleted.json"):
            if (manifest.parent / "data").is_dir():
                info = read_json(manifest)
                items.append((manifest.parent.name, info))
        return sorted(items, key=lambda item: item[1]["deleted_at"], reverse=True)

    def restore_identity(self, archive_id):
        """按回收记录恢复原路径，不覆盖同名目录；恢复失败也回滚索引边界。"""
        if not re.fullmatch(r"[0-9a-f]{32}", archive_id):
            raise ValueError("回收记录编号无效")
        root = self.root.resolve()
        archive = root / "_trash" / archive_id
        source = archive / "data"
        if not source.resolve().is_relative_to(root / "_trash") or source.is_symlink():
            raise ValueError("回收数据路径无效")
        if not source.is_dir():
            raise ValueError("该回收项已恢复或不存在")
        info = read_json(archive / "deleted.json")
        parts = Path(info["original_path"]).parts
        if len(parts) not in (1, 2, 4):
            raise ValueError("原始目录记录无效")
        destination = self._sample_path(parts) if len(parts) == 4 else self._identity_path(*parts)
        if destination.exists():
            raise ValueError("原路径已有同名目录，不能覆盖；请先处理同名项")
        if not destination.parent.is_dir():
            raise ValueError("上级目录不存在，请先恢复所属受试者或场次")
        source.rename(destination)
        try:
            self.rebuild_index()
        except Exception:
            destination.rename(source)
            raise
        return parts

    def load_labels(self) -> ActionLabels:
        """Missing file means the built-in defaults; a damaged file is an error."""
        if not self.labels_path.exists():
            return default_labels()
        return ActionLabels.from_dict(read_json(self.labels_path))

    def ensure_labels(self, labels: ActionLabels):
        self.root.mkdir(parents=True, exist_ok=True)
        if self.labels_path.exists():
            existing = ActionLabels.from_dict(read_json(self.labels_path))
            if existing.to_dict() != labels.to_dict():
                raise ValueError("数据集内已有不同的 labels.json，请另建数据集目录")
            return existing
        atomic_write_json(self.labels_path, labels.to_dict())
        return labels

    def ensure_layout(self, layout: CollectionLayout):
        """Copy the referenced layout version once; a conflicting file is a hard error."""
        target = self.root / f"{layout.layout_id}.json"
        payload = layout.to_dict()
        if target.exists():
            existing = read_json(target)
            existing.pop("created_at", None)
            if existing != payload:
                raise ValueError(f"数据集内 {target.name} 的内容与当前布局不一致，请改用新版本布局")
            return target
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target, payload)
        return target

    def session_path(self, subject_id, session_id) -> Path:
        return self._identity_path(subject_id, session_id) / "session.json"

    @staticmethod
    def _check_name(value, name):
        if not isinstance(value, str) or not NAME_PATTERN.fullmatch(value):
            raise ValueError(f"{name}只能包含字母、数字、下划线和连字符")

    def open_session(self, *, subject_id, session_id, layout, labels, max_adc_columns,
                     expected_block_ids, transport=None):
        """Create the session manifest once, then verify later recordings match it."""
        path = self.session_path(subject_id, session_id)
        expected = sorted(int(key) for key in expected_block_ids)
        if path.exists():
            data = read_json(path)
            if data.get("layout_id") != layout.layout_id:
                raise ValueError(f"该场次已使用 {data.get('layout_id')}，与当前布局 {layout.layout_id} 不同；请新建场次")
            if data.get("labels_id") != labels.labels_id:
                raise ValueError("该场次的动作类别表与当前 labels.json 不同；请新建场次")
            if data.get("max_adc_columns") != int(max_adc_columns):
                raise ValueError(
                    f"该场次已固定 {data.get('max_adc_columns')} 个 ADC 列，与当前布局的 "
                    f"{max_adc_columns} 列不同；请新建场次")
            if sorted(data.get("expected_block_ids", [])) != expected:
                raise ValueError("该场次的预期皮肤块与当前固定布局不同；请新建场次")
            return data
        data = {
            "schema_version": SCHEMA_VERSION,
            "subject_id": subject_id,
            "session_id": session_id,
            "operator_id": "",
            "started_at": self._now_iso(),
            "notes": "",
            "software": {"name": "SkinDraw", "version": _app_version()},
            "protocol": dict(transport or {}),
            "layout_id": layout.layout_id,
            "labels_id": labels.labels_id,
            "max_adc_columns": int(max_adc_columns),
            "expected_block_ids": expected,
            "skin": {"block_ids_configured": sorted(layout.by_id), "installation": "", "conditions": ""},
            # Calibrations are appended per recording; ids are never reused.
            "calibrations": {},
        }
        atomic_write_json(path, data)
        return data

    @staticmethod
    def _now_iso():
        return _local_now().isoformat(timespec="milliseconds")

    def save_calibration(self, subject_id, session_id, calibration: CalibrationSnapshot):
        path = self.session_path(subject_id, session_id)
        data = read_json(path)
        calibrations = data.setdefault("calibrations", {})
        if calibration.calibration_id in calibrations:
            raise ValueError("标定编号重复，未覆盖已有基线")
        calibrations[calibration.calibration_id] = calibration.to_dict()
        atomic_write_json(path, data)
        return calibration.calibration_id

    def next_calibration_id(self, subject_id, session_id) -> str:
        path = self.session_path(subject_id, session_id)
        calibrations = read_json(path).get("calibrations", {}) if path.exists() else {}
        return f"calib_{len(calibrations) + 1:03d}"

    def next_trial_number(self, subject_id, session_id, action: ActionClass) -> int:
        """Next free trial number in this session/action; existing samples are never reused."""
        self.session_path(subject_id, session_id)
        self._check_name(action.name, "动作名称")
        folder = self.root / subject_id / session_id / action.name
        pattern = re.compile(re.escape(action.name) + r"([0-9]{3,})\Z")
        numbers = [int(match.group(1)) for path in folder.glob(f"{action.name}*")
                   if path.is_dir() and (match := pattern.fullmatch(path.name))]
        # 作废样本也占用原编号，避免重采复用后无法恢复或混淆审计记录。
        for manifest in (self.root / "_trash").glob("*/deleted.json"):
            try:
                parts = Path(read_json(manifest)["original_path"]).parts
            except (ValueError, KeyError, TypeError):
                continue
            if len(parts) == 4 and parts[:3] == (subject_id, session_id, action.name):
                match = pattern.fullmatch(parts[3])
                if match:
                    numbers.append(int(match.group(1)))
        return max(numbers, default=0) + 1

    def update_accepted(self, meta_path, accepted, reject_reason=""):
        meta_path = Path(meta_path)
        data = read_json(meta_path)
        if data.get("recording_status") == "recording":
            raise ValueError("样本仍在录制，不能标记")
        if accepted is True and (data.get("recording_status") != "complete"
                                 or data.get("quality_status") == "fail"):
            raise ValueError("录制未完成或完整性检查不合格，不能标记有效，请重新采集")
        if accepted is not None and type(accepted) is not bool:
            raise ValueError("accepted 只能为 true、false 或 null")
        data["accepted"] = accepted
        data["reject_reason"] = "" if accepted is not False else (reject_reason or "人工作废")
        data["accepted_at"] = self._now_iso() if accepted is not None else None
        atomic_write_json(meta_path, data)
        return data

    def rebuild_index(self):
        """Rebuild the manifest from sample meta.json files; never the reverse."""
        rows = []
        subjects = set(self.list_subjects())
        sessions = {key: set(self.list_sessions(key)) for key in subjects}
        for meta_path in self.root.glob("*/*/*/*/meta.json"):
            # 回收目录始终不参与训练索引；符号链接也不能引入外部数据。
            parts = meta_path.relative_to(self.root).parts
            if parts[0] not in subjects:
                continue
            if parts[1] not in sessions[parts[0]]:
                continue
            try:
                data = read_json(meta_path)
                relative = meta_path.parent.relative_to(self.root).as_posix()
                accepted = data.get("accepted")
                rows.append([data["subject_id"], data["session_id"], data["trial_id"],
                             data["action"]["id"], f"{float(data.get('duration_s', 0.)):.3f}",
                             "" if accepted is None else ("true" if accepted else "false"), relative,
                             data.get("recording_status", "unknown"), data.get("quality_status", "unknown")])
            except (KeyError, TypeError, ValueError):
                # A damaged meta.json is reported by its absence from the index,
                # not by rewriting or deleting the operator's original sample.
                continue
        rows.sort(key=lambda row: (row[0], row[1], int(row[3]), row[2]))
        lines = [",".join(self.INDEX_HEADER)]
        lines.extend(",".join(str(cell) for cell in row) for row in rows)
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self.index_path, "\n".join(lines) + "\n")
        return len(rows)


def _app_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version
        return version("skin-draw")
    except PackageNotFoundError:
        return "unknown"


def layout_adc_columns(layout: CollectionLayout) -> int:
    """Largest flattened point count of any configured block, fixed per session."""
    return max((rows * cols for rows, cols in (block.spec.display_shape for block in layout.blocks)),
               default=0)
