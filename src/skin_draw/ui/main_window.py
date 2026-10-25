"""Hardware samples calibrate in the worker; the GUI paints the latest state."""

from pathlib import Path
from collections import deque
from dataclasses import replace
from datetime import datetime
from math import ceil
import logging

import numpy as np
from time import perf_counter
from PySide6.QtCore import QSignalBlocker, QTimer, Qt, Slot, QUrl
from PySide6.QtGui import QAction, QKeySequence, QDesktopServices
from PySide6.QtWidgets import (QDialog, QFileDialog, QFrame, QHBoxLayout, QInputDialog, QLabel, QMainWindow,
                               QMessageBox, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from ..colors import ColorScale
from ..acquisition import Scan
from ..models import Frame
from ..mapping import BlockOrientation
from ..sensor_masks import SensorMasks, default_mask_path
from ..collection_activation import load_activation_policy
from ..collection_layout import CollectionLayoutStore
from ..collection_quality import LEVELS, QUALITY_NAMES, load_quality_policy
from ..dataset import (MAX_ADC_COLUMNS, CalibrationSnapshot, DatasetStore, RawSampleRecorder,
                       RecordingPlan, default_labels, layout_adc_columns)
from ..transport import available_ports
from ..simulator import demo_specs, generate_frame
from .control_panel import ControlPanel, label
from .collection_bar import CollectionBar
from .heatmap_view import HeatmapView, SensorReading
from .serial_worker import SerialWorker
from .reception_rate_bar import ReceptionRateBar
from .sensor_mask_dialog import SensorMaskDialog
from ..diagnostics import log_directory, event

logger = logging.getLogger(__name__)
# Minimum interval between live display commits. It only delays a commit while
# new data keeps arriving faster than this, so it bounds burst work instead of
# adding steady-state latency: after any quiet gap the next block commits at
# once. Every clock below the throttle must stay high resolution -- Windows'
# time.monotonic() is GetTickCount64 with ~15.6 ms granularity, which would
# quantize the remaining-budget arithmetic to whole 15.6 ms ticks and silently
# turn this value into a much coarser throttle.
DISPLAY_INTERVAL_MS = 4


def default_dataset_root() -> Path:
    """Dataset folder next to logs/, outside the package so it can be copied as a whole."""
    return Path(__file__).resolve().parents[3] / "dataset"


def _default_session_id() -> str:
    return f"session_{datetime.now():%Y%m%d}_01"


class MainWindow(QMainWindow):
    def __init__(self, demo: bool = False, *, mask_path=None, raw_adc=False, collection_layout_dir=None,
                 dataset_dir=None):
        super().__init__()
        self.raw_adc = raw_adc
        self.mask_path = Path(mask_path) if mask_path is not None else default_mask_path()
        self.mask_load_error = ""
        try:
            self.sensor_masks = SensorMasks.load(self.mask_path)
        except (OSError, ValueError) as error:
            self.sensor_masks = SensorMasks()
            self.mask_load_error = str(error)
            logger.exception("有效点位配置加载失败：%s", self.mask_path)
        directory = Path(collection_layout_dir) if collection_layout_dir is not None else self.mask_path.parent / "collection_layouts"
        self.collection_store = CollectionLayoutStore(directory)
        self.collection_layout = None
        self.collection_load_error = ""
        try:
            self.collection_layout = self.collection_store.load()
        except (OSError, ValueError) as error:
            self.collection_load_error = str(error)
            logger.exception("采集布局加载失败：%s", directory)
        self.collection_locked = self.collection_layout is not None
        if self.collection_layout is not None:
            self.sensor_masks = self.collection_layout.with_masks(self.sensor_masks)
        # Training dataset. Startup only READS labels.json: neither the dataset
        # folder nor any sample is created until the first recording starts.
        self.dataset_store = DatasetStore(Path(dataset_dir) if dataset_dir is not None
                                          else default_dataset_root())
        self.dataset_labels = default_labels()
        self.labels_error = ""
        try:
            self.dataset_labels = self.dataset_store.load_labels()
        except (OSError, ValueError) as error:
            self.labels_error = str(error)
            logger.exception("动作类别表加载失败：%s", self.dataset_store.labels_path)
        self.recorder = RawSampleRecorder()
        self.recording_plan: RecordingPlan | None = None
        self.last_sample = None
        self.setWindowTitle("SkinDraw · 原始 ADC 测试" if raw_adc else "SkinDraw · 电子皮肤热力图")
        self.resize(1250, 860)
        self.setMinimumSize(1000, 720)
        self.sequence = 0
        self.current_frame: Frame | None = None
        self.source_specs = demo_specs()
        self.input_source = "idle"
        self.preprocess = None  # latest PreprocessState from the worker
        self.worker: SerialWorker | None = None
        self.accept_scans = False
        self.close_when_stopped = False
        self.last_scan: Scan | None = None
        self.last_gui_tick = None
        self.last_gui_log = 0.
        self.gui_refreshes = 0
        self.gui_update_times = deque(maxlen=1024)
        self.gui_sample_ages = deque(maxlen=1024)
        self.display_values = None
        self.live_blocks = {}  # observed shapes for this connection, not fresh measurements
        self.live_missing_counts = None
        self.pending_live_info = None
        self.live_info_state = None
        self.unit = "ADC"
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(24, 18, 24, 16)
        outer.setSpacing(16)
        header = QHBoxLayout()
        brand = QVBoxLayout()
        brand.setSpacing(1)
        brand.addWidget(label("SkinDraw", "brand"))
        self.layout_title = label("电子皮肤 · 传感器阵列可视化")
        brand.addWidget(self.layout_title)
        header.addLayout(brand)
        header.addStretch()
        self.source_badge = label("等待连接硬件", "badge")
        header.addWidget(self.source_badge)
        self.logs_button = QPushButton("打开日志目录")
        self.logs_button.setToolTip("查看连接、超时、解析异常、数据中断与恢复记录。")
        self.logs_button.clicked.connect(self._open_logs)
        header.addWidget(self.logs_button)
        outer.addLayout(header)
        # Recording UI sits above the heatmap frame: identity, boundary buttons and
        # the counters required by the CSV plan.
        self.collection_bar = CollectionBar()
        self.collection_bar.set_actions(self.dataset_labels.classes)
        outer.addWidget(self.collection_bar)
        body = QHBoxLayout()
        body.setSpacing(16)
        self.controls = ControlPanel(raw_adc=raw_adc)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.controls)
        scroll.setFixedWidth(265)
        body.addWidget(scroll)
        right = QVBoxLayout()
        right.setSpacing(12)
        card = QFrame()
        card.setObjectName("card")
        card_box = QVBoxLayout(card)
        card_box.setContentsMargins(16, 12, 12, 12)
        toolbar = QHBoxLayout()
        self.reception_bar = ReceptionRateBar()
        toolbar.addWidget(self.reception_bar, 1)
        self.fit_button = QPushButton("适应窗口")
        self.export_button = QPushButton("导出 PNG")
        toolbar.addWidget(self.fit_button)
        toolbar.addWidget(self.export_button)
        card_box.addLayout(toolbar)
        self.heatmap = HeatmapView()
        self.heatmap.sensor_masks = self.sensor_masks
        if self.collection_layout is not None:
            self._apply_collection_view()
        card_box.addWidget(self.heatmap, 1)
        self.hover_label = label("移动鼠标查看传感器读数；单击锁定选点。")
        self.hover_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.hover_label.setFixedHeight(22)
        card_box.addWidget(self.hover_label)
        right.addWidget(card, 1)
        stats = QHBoxLayout()
        stats.setSpacing(12)
        self.min_title, self.min_label = self._stat(stats, "最小值 / ADC")
        self.max_title, self.max_label = self._stat(stats, "最大值 / ADC")
        self.mean_title, self.mean_label = self._stat(stats, "平均值 / ADC")
        self.selected_title, self.selected_label = self._stat(stats, "选中传感器 / ADC")
        right.addLayout(stats)
        self.selected_detail = label("尚未选点    ·    拖动皮肤块移动 / 拖动空白平移 / 滚轮缩放 / F 适应窗口")
        self.selected_detail.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.selected_detail.setFixedHeight(22)
        right.addWidget(self.selected_detail)
        body.addLayout(right, 1)
        outer.addLayout(body, 1)
        self.status = QLabel()
        self.statusBar().addWidget(self.status, 1)
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.advance_demo)
        self.display_timer = QTimer(self)
        self.display_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.display_timer.setInterval(DISPLAY_INTERVAL_MS)
        self.display_timer.setSingleShot(True)
        self.display_timer.timeout.connect(self._flush_live)
        self.info_timer = QTimer(self)
        self.info_timer.setInterval(200)
        self.info_timer.timeout.connect(self._flush_live_info)
        self.info_timer.start()
        self.reception_timer = QTimer(self)
        self.reception_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.reception_timer.setInterval(1000)
        self.reception_timer.timeout.connect(self._refresh_reception_rate)
        self.recording_timer = QTimer(self)
        self.recording_timer.setInterval(100)
        self.recording_timer.timeout.connect(self._refresh_recording)
        self._connect_controls()
        self._init_recording_identity()
        self.heatmap.set_show_values(self.controls.show_values.isChecked())
        self._refresh_ports()
        if demo:
            self.controls.demo_toggle.setChecked(True)
        else:
            self._display_frame(Frame(()))
        self._update_status()
        self.health_timer = QTimer(self)
        self.health_timer.setInterval(500)
        self.health_timer.timeout.connect(self._check_freshness)
        self.health_timer.start()
        QTimer.singleShot(0, self.heatmap.fit_view)

    @staticmethod
    def _stat(row: QHBoxLayout, title: str):
        card = QFrame()
        card.setObjectName("card")
        box = QVBoxLayout(card)
        box.setContentsMargins(16, 12, 16, 12)
        box.setSpacing(2)
        title_label = label(title)
        box.addWidget(title_label)
        value = label("—", "value")
        box.addWidget(value)
        row.addWidget(card, 1)
        return title_label, value

    def _connect_controls(self):
        controls = self.controls
        controls.refresh_ports.clicked.connect(self._refresh_ports)
        controls.connect_button.clicked.connect(self._connect_hardware)
        controls.recalibrate.clicked.connect(self._recalibrate)
        controls.filter_toggle.toggled.connect(self._filter_changed)
        controls.direction.clicked.connect(self._toggle_direction)
        controls.normalize.toggled.connect(self._norm_changed)
        controls.norm_mode.currentIndexChanged.connect(self._norm_mode_changed)
        controls.reset_positions.clicked.connect(self.heatmap.reset_positions)
        controls.body_positions.clicked.connect(self.heatmap.restore_body_positions)
        controls.orientation_block.currentIndexChanged.connect(self._orientation_target_changed)
        controls.rotate_left.clicked.connect(lambda: self._orient_block("left"))
        controls.rotate_right.clicked.connect(lambda: self._orient_block("right"))
        controls.mirror_horizontal.clicked.connect(lambda: self._orient_block("horizontal"))
        controls.mirror_vertical.clicked.connect(lambda: self._orient_block("vertical"))
        controls.reset_orientation.clicked.connect(lambda: self._orient_block("reset"))
        controls.edit_sensor_mask.clicked.connect(self._edit_sensor_mask)
        controls.save_collection_layout.clicked.connect(self._save_collection_layout)
        controls.edit_collection_layout.clicked.connect(self._edit_collection_layout)
        controls.restore_collection_layout.clicked.connect(self._restore_collection_layout)
        controls.demo_toggle.toggled.connect(self._toggle_demo)
        controls.preset.currentIndexChanged.connect(self._preset_changed)
        controls.palette.currentTextChanged.connect(self._update_scale)
        controls.low.valueChanged.connect(self._low_changed)
        controls.high.valueChanged.connect(self._high_changed)
        controls.auto_levels.toggled.connect(self._auto_changed)
        controls.show_values.toggled.connect(self.heatmap.set_show_values)
        controls.gap.valueChanged.connect(lambda value: self.heatmap.set_gap(value / 100))
        controls.pattern.currentTextChanged.connect(self._reset_demo)
        controls.play.clicked.connect(self.toggle_play)
        controls.step.clicked.connect(self.step_demo)
        self.fit_button.clicked.connect(self.heatmap.fit_view)
        self.export_button.clicked.connect(self._export)
        self.heatmap.sensor_hovered.connect(self._hovered)
        self.heatmap.sensor_selected.connect(self._selected)
        self.heatmap.layout_changed.connect(self._layout_changed)
        bar = self.collection_bar
        bar.start.clicked.connect(self._start_recording)
        bar.stop.clicked.connect(lambda: self.finish_recording("user_stop"))
        bar.mark_valid.clicked.connect(lambda: self._mark_sample(True))
        bar.mark_void.clicked.connect(lambda: self._mark_sample(False))
        bar.dataset_button.clicked.connect(self._open_dataset)
        # The trial number preview follows the identity fields, but is frozen while recording.
        bar.subject.currentTextChanged.connect(self._subject_changed)
        bar.session.currentTextChanged.connect(self._identity_changed)
        bar.add_subject.clicked.connect(lambda: self._add_identity(False))
        bar.add_session.clicked.connect(lambda: self._add_identity(True))
        bar.delete_subject.clicked.connect(lambda: self._delete_identity(False))
        bar.delete_session.clicked.connect(lambda: self._delete_identity(True))
        bar.restore_identity.clicked.connect(self._restore_identity)
        bar.action.currentIndexChanged.connect(self._refresh_trial_preview)
        for key, callback in (("Space", self.toggle_play), ("F", self.heatmap.fit_view)):
            action = QAction(self)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(callback)
            self.addAction(action)

    def _layout_changed(self):
        specs = self.heatmap.layout_specs
        count = sum(int(self.sensor_masks.get(s.block_id, (s.rows, s.cols)).sum()) for s in specs)
        self.layout_title.setText(f"传感器热力图   ·   {len(specs)} 块 / {count} 点")
        plan = self.heatmap.arrangement
        grid = f"{plan.rows} 行 × {plan.columns} 列排列" if plan and specs else "等待有效数据块"
        if self.heatmap.placement_mode == "body":
            grid = "人体预设 · 正面 / 背面"
        if specs and self.heatmap.manual_positions:
            grid = "手动位置 · 实时刷新保持位置"
        self.controls.reset_positions.setEnabled((bool(specs) or self.heatmap.placement_mode == "body") and self.heatmap.auto_layout)
        self.controls.body_positions.setEnabled(self.heatmap.auto_layout)
        self.controls.detected.setText(f"识别到 {len(specs)} 块 · {count} 个传感器\n{grid}")
        mismatches = [str(s.block_id) for s in specs if self.sensor_masks.mismatched(s.block_id, (s.rows, s.cols))]
        if self.mask_load_error:
            text = f"点位配置加载失败，暂显示全部点。请检查 {self.mask_path.name}：{self.mask_load_error}"
        elif mismatches:
            text = "BLOCK " + ", ".join(mismatches) + " 尺寸与点位配置不符；暂显示全部点，请重新编辑。"
        else:
            text = "有效点位按 BLOCK 编号保存，重启自动恢复。"
        self.controls.mask_status.setText(text)
        self.controls.mask_status.setToolTip(str(self.mask_path))
        self.controls.mask_status.setStyleSheet("color: #b45309" if self.mask_load_error or mismatches else "")
        combo = self.controls.orientation_block
        previous = combo.currentData()
        with QSignalBlocker(combo):
            combo.clear()
            for spec in specs:
                combo.addItem(f"BLOCK {spec.block_id:02d}", spec.block_id)
            index = combo.findData(previous)
            if index >= 0:
                combo.setCurrentIndex(index)
        enabled = bool(specs) and self.heatmap.auto_layout
        combo.setEnabled(enabled)
        self.controls.edit_sensor_mask.setEnabled(bool(specs) and not self.collection_locked
                                                   and not self._recording_active())
        for widget in (self.controls.rotate_left, self.controls.rotate_right,
                       self.controls.mirror_horizontal, self.controls.mirror_vertical,
                       self.controls.reset_orientation):
            widget.setEnabled(enabled and not self.collection_locked and not self._recording_active())
        self._update_collection_status()

    def _update_collection_status(self):
        specs = self.heatmap.layout_specs
        layout = self.collection_layout
        status_key = (layout.layout_id if layout else None, self.collection_locked,
                      self.input_source, self.heatmap.auto_layout, self.collection_load_error,
                      self.mask_load_error, self.heatmap.placement_mode, self._recording_active(),
                      tuple((s.block_id, s.rows, s.cols, s.x, s.y) for s in specs))
        if status_key == getattr(self, "_collection_status_key", None):
            return
        self._collection_status_key = status_key
        controls = self.controls
        is_demo = self.input_source == "demo"
        usable = (bool(specs) and self.heatmap.auto_layout
                  and self.input_source in ("hardware", "external", "stopped"))
        positions_changed = self._collection_positions_changed()
        controls.save_collection_layout.setEnabled(
            usable and (not self.collection_locked or positions_changed) and not self._recording_active()
            and not self.collection_load_error and not self.mask_load_error)
        controls.edit_collection_layout.setEnabled(layout is not None and self.collection_locked
                                                   and not is_demo and not self._recording_active())
        controls.restore_collection_layout.setEnabled(
            (layout is not None or bool(self.collection_load_error)) and not is_demo
            and not self._recording_active())
        problems = []
        if layout is not None:
            known = layout.by_id
            for spec in specs:
                block = known.get(spec.block_id)
                if block is None:
                    problems.append(f"{spec.block_id:02d} 未固定")
                elif (spec.rows, spec.cols) != (block.spec.rows, block.spec.cols):
                    problems.append(f"{spec.block_id:02d} 尺寸不符")
        if self.collection_load_error:
            text = f"采集布局加载失败，请修复配置后重启：{self.collection_load_error}"
        elif is_demo:
            text = "模拟演示不保存采集布局；请连接实际皮肤后保存。"
        elif layout is None:
            text = "尚未固定采集布局；调整方向和点位后点击保存。"
        else:
            revision = layout.layout_id.removeprefix("skin_layout_")
            state = "已固定" if self.collection_locked else "调整中 · 请保存新版本"
            if self.collection_locked and positions_changed:
                state = "位置待保存"
            text = f"{state} · 本地 {revision} · 已配置 {len(layout.blocks)}/14 块"
            if not layout.view_positions:
                text += "\n旧版未记录摆放位置，调整后保存一次即可记住。"
            missing = sorted(set(range(1, 15)) - set(layout.by_id))
            if missing:
                text += "\n待配置：" + "、".join(map(str, missing))
            if problems:
                text += "\n" + "；".join(problems) + "，请调整并重新保存。"
        controls.collection_layout_status.setText(text)
        controls.collection_layout_status.setToolTip(str(self.collection_store.directory))
        controls.collection_layout_status.setStyleSheet(
            "color: #b45309" if self.collection_load_error or problems else "")

    def _collection_positions_changed(self):
        if self.collection_layout is None:
            return False
        positions = self.collection_layout.positions
        return (self.heatmap.placement_mode != self.collection_layout.placement_mode
                or any(positions.get(spec.block_id) != (spec.x, spec.y) for spec in self.heatmap.layout_specs))

    def _init_recording_identity(self):
        bar = self.collection_bar
        if self.labels_error:
            bar.set_status(f"动作类别表不可用，采集功能已禁用：{self.labels_error}", "error")
        self._refresh_identity_choices(initial=True)
        self._refresh_trial_preview()
        self._refresh_recording_availability()

    @staticmethod
    def _fill_identity_combo(combo, names, selected):
        # 批量刷新时屏蔽中间空值信号，避免受试者切换过程中读到错误场次。
        with QSignalBlocker(combo):
            combo.clear()
            combo.addItems(names)
            combo.setCurrentText(selected)

    def _refresh_identity_choices(self, subject=None, session=None, *, initial=False):
        bar = self.collection_bar
        try:
            subjects = self.dataset_store.list_subjects()
            if subject is None:
                subject = subjects[0] if subjects else ("student01" if initial else "")
            sessions = self.dataset_store.list_sessions(subject) if subject else []
            if session is None:
                session = _default_session_id() if initial else (sessions[-1] if sessions else "")
            self._fill_identity_combo(bar.subject, subjects, subject)
            self._fill_identity_combo(bar.session, sessions, session)
            self._identity_changed()
        except (OSError, ValueError) as error:
            bar.set_status(f"读取受试者列表失败：{error}", "error")

    def _subject_changed(self, *_):
        if self._recording_active():
            return
        bar = self.collection_bar
        try:
            subject = bar.subject.currentText().strip()
            sessions = self.dataset_store.list_sessions(subject) if subject else []
        except (OSError, ValueError):
            sessions = []
        self._fill_identity_combo(bar.session, sessions, sessions[-1] if sessions else _default_session_id())
        self._identity_changed()

    def _identity_changed(self, *_):
        bar = self.collection_bar
        active = self._recording_active()
        try:
            subject = bar.subject.currentText().strip()
            subjects = self.dataset_store.list_subjects()
            sessions = self.dataset_store.list_sessions(subject) if subject in subjects else []
        except (OSError, ValueError):
            subjects, sessions = [], []
        bar.add_subject.setEnabled(not active)
        bar.delete_subject.setEnabled(not active and subject in subjects)
        bar.add_session.setEnabled(not active and subject in subjects)
        bar.delete_session.setEnabled(not active and bar.session.currentText().strip() in sessions)
        self._refresh_trial_preview()
        self._refresh_recording_availability()

    def _add_identity(self, is_session):
        if self._recording_active():
            return
        bar = self.collection_bar
        subject = bar.subject.currentText().strip()
        title = "场次" if is_session else "受试者"
        try:
            prefix = _default_session_id().rsplit("_", 1)[0] + "_" if is_session else "student"
            suggestion = self.dataset_store.next_identity(prefix, subject if is_session else None)
            name, accepted = QInputDialog.getText(self, f"新增{title}", f"{title}编号：", text=suggestion)
            if not accepted:
                return
            # 对话框期间仍可能收到界面事件，落盘前再次检查录制边界。
            if self._recording_active():
                raise ValueError("正在采集，不能新增受试者或场次")
            name = name.strip()
            if is_session:
                self.dataset_store.create_session(subject, name)
                self._refresh_identity_choices(subject, name)
            else:
                self.dataset_store.create_subject(name)
                self._refresh_identity_choices(name)
            bar.set_status(f"已新增{title}：{name}", "ok")
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, f"无法新增{title}", str(error))

    def delete_selected_identity(self, is_session=False):
        """删除当前选项并自动收缩列表，已保存的其他编号和标签保持不变。"""
        if self._recording_active():
            raise ValueError("正在采集，不能删除受试者或场次")
        bar = self.collection_bar
        subject = bar.subject.currentText().strip()
        session = bar.session.currentText().strip() if is_session else None
        source = self.dataset_store.root / subject
        if session is not None:
            source = source / session
        archive = self.dataset_store.archive_identity(subject, session)
        # 清掉指向已回收样本的操作，防止“标记有效”继续访问旧路径。
        if self.last_sample is not None and self.last_sample.sample_dir.resolve().is_relative_to(source.resolve()):
            self.last_sample = None
            self.recording_plan = None
            bar.set_sample_state("未判定")
            bar.set_stats(0., 0, {})
        self._refresh_identity_choices(subject if is_session else None)
        self._refresh_recording_availability()
        bar.set_status(f"已移入回收目录：{subject}" + (f" / {session}" if is_session else ""), "ok")
        bar.status.setToolTip(str(archive))
        return archive

    def _delete_identity(self, is_session):
        try:
            self.delete_selected_identity(is_session)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "无法删除", str(error))

    def _restore_identity(self):
        if self._recording_active():
            return
        try:
            entries = self.dataset_store.archived_identities()
            if not entries:
                self.collection_bar.set_status("回收目录为空，没有待恢复的数据")
                return
            choices = [f"{info['original_path']} · {info['deleted_at']} · {key[:8]}"
                       for key, info in entries]
            selected, accepted = QInputDialog.getItem(self, "恢复删除", "选择要恢复的受试者、场次或作废样本：",
                                                     choices, 0, False)
            if not accepted or self._recording_active():
                return
            parts = self.dataset_store.restore_identity(entries[choices.index(selected)][0])
            self._refresh_identity_choices(*parts[:2])
            self.collection_bar.set_status("已恢复：" + " / ".join(parts), "ok")
        except (OSError, ValueError, KeyError, TypeError) as error:
            QMessageBox.warning(self, "无法恢复", str(error))

    def _selected_action(self):
        actions = self.dataset_labels.by_id()
        action = actions.get(self.collection_bar.action_id())
        if action is None:
            raise ValueError("请选择动作类别")
        return action

    def _recording_active(self):
        return self.recorder.active

    def _refresh_trial_preview(self, *_):
        if self._recording_active():
            return
        bar = self.collection_bar
        try:
            subject = bar.subject.currentText().strip()
            session = bar.session.currentText().strip()
            action = self._selected_action()
            number = self.dataset_store.next_trial_number(subject, session, action)
            bar.set_trial(f"{action.name}{number:03d}",
                          f"当前场次「{action.display_name}」的下一条样本编号；已存在样本不会覆盖")
        except (OSError, ValueError):
            bar.set_trial("—", "请先填写受试者、场次并选择动作")

    def _refresh_recording_availability(self):
        bar = self.collection_bar
        if self._recording_active():
            bar.start.setEnabled(False)
            return
        ready = (self.worker is not None and self.input_source == "hardware"
                 and self.collection_layout is not None and self.collection_locked
                 and not self.labels_error and bool(bar.subject.currentText().strip())
                 and bool(bar.session.currentText().strip()))
        bar.start.setEnabled(ready)
        # Never replace the message of a just-finished sample with an availability hint.
        if ready or self.last_sample is not None:
            return
        if self.labels_error:
            bar.set_status(f"动作类别表不可用，采集功能已禁用：{self.labels_error}", "error")
        elif self.worker is None:
            bar.set_status("未开始 · 连接硬件后可采集训练样本")
        elif self.collection_layout is None:
            bar.set_status("未开始 · 尚未固定采集布局", "warn")

    def _check_recording_shapes(self, layout):
        """Reject a start that the fixed layout cannot describe."""
        if self.worker is None:
            return
        with self.worker.buffer.lock:
            blocks = dict(self.worker.buffer.blocks)
        for key, block in blocks.items():
            saved = layout.by_id.get(key)
            if saved is not None and block.values.shape != (saved.spec.rows, saved.spec.cols):
                raise ValueError(f"BLOCK {key:02d} 当前尺寸与固定布局不符，请重新保存采集布局")
        if not set(blocks) & set(layout.by_id):
            raise ValueError("固定布局中的皮肤块尚未收到数据，请等待首帧后再开始采集")

    def _capture_calibration(self, subject, session, layout):
        """Snapshot the running baseline; raw ADC mode has no baseline to reference."""
        if self.raw_adc or self.worker is None:
            return None
        baselines, samples = self.worker.buffer.baseline_snapshot()
        known = layout.by_id
        usable = {key: matrix for key, matrix in baselines.items()
                  if key in known and matrix.shape == (known[key].spec.rows, known[key].spec.cols)}
        if not usable:
            return None
        snapshot = CalibrationSnapshot.from_baselines(
            self.dataset_store.next_calibration_id(subject, session),
            datetime.now().astimezone().isoformat(timespec="milliseconds"), usable, calib_samples=samples)
        if not snapshot.baselines:
            return None
        self.dataset_store.save_calibration(subject, session, snapshot)
        return snapshot

    def start_recording(self):
        """Begin one action sample; identity is locked until ``finish_recording``."""
        if self._recording_active():
            raise ValueError("上一段样本尚未结束")
        if self.labels_error:
            raise ValueError(f"动作类别表不可用：{self.labels_error}")
        if self.worker is None or self.input_source != "hardware":
            raise ValueError("请先连接实际皮肤；模拟数据不能保存为训练样本")
        layout = self.collection_layout
        if layout is None or not self.collection_locked:
            raise ValueError("请先保存并固定采集布局，再开始采集")
        bar = self.collection_bar
        subject = bar.subject.currentText().strip()
        session = bar.session.currentText().strip()
        action = self._selected_action()
        self.dataset_store.session_path(subject, session)  # validates the two directory names
        self._check_recording_shapes(layout)
        quality_policy = load_quality_policy(self.mask_path.parent / "collection_quality.json")
        activation_policy = load_activation_policy(self.mask_path.parent / "collection_activation.json")
        cadence = self.worker.buffer.recording_cadence(tuple(layout.by_id), quality_policy)
        columns = layout_adc_columns(layout)
        if not 1 <= columns <= MAX_ADC_COLUMNS:
            raise ValueError("固定布局中没有可采集的点数")
        number = self.dataset_store.next_trial_number(subject, session, action)
        trial_id = f"{action.name}{number:03d}"
        # Global files first: labels, layout version and the session manifest must
        # exist before the sample references them.
        self.dataset_store.ensure_labels(self.dataset_labels)
        self.dataset_store.ensure_layout(layout)
        self.dataset_store.open_session(
            subject_id=subject, session_id=session, layout=layout, labels=self.dataset_labels,
            max_adc_columns=columns, expected_block_ids=tuple(layout.by_id),
            transport=self._session_transport())
        calibration = self._capture_calibration(subject, session, layout)
        plan = RecordingPlan(
            sample_dir=self.dataset_store.root / subject / session / action.name / trial_id,
            subject_id=subject, session_id=session, trial_id=trial_id, action=action,
            layout=layout, labels_id=self.dataset_labels.labels_id,
            expected_block_ids=tuple(layout.by_id), max_adc_columns=columns,
            calibration=calibration, quality_policy=quality_policy,
            activation_policy=activation_policy, cadence_references=cadence)
        self.recorder.start(plan)
        self.recording_plan = plan
        self.last_sample = None
        self.recording_timer.start()
        bar.set_recording(True)
        bar.set_trial(trial_id, "本条样本编号（编号冻结，结束后自动指向下一条）")
        bar.set_sample_state("录制中", "warn")
        bar.set_stats(0., 0, {})
        bar.set_integrity(self.recorder.snapshot().integrity)
        bar.set_status(f"录制中 · {subject} / {session} / {trial_id}；结束后先完成写盘再开始下一条", "warn")
        self._layout_changed()  # reapplies the editing locks for the recording state
        event(logger, logging.INFO, "开始采集训练样本", subject_id=subject, session_id=session,
              trial_id=trial_id, action_id=action.id, layout_id=layout.layout_id,
              max_adc_columns=columns, calibration_id=calibration.calibration_id if calibration else None)
        return plan

    def attach_recording_sink(self, buffer):
        """Offer this buffer's raw blocks to the recorder (also used by tests)."""
        buffer.set_raw_sink(self.recorder.submit)

    def _session_transport(self):
        controls = self.controls
        return {"transport": "serial", "port": controls.port.currentData(),
                "baudrate": controls.baudrate.currentData(),
                "receive_mode": controls.receive_mode.currentData(),
                "protocol": controls.protocol.currentData()}

    def finish_recording(self, reason="user_stop", *, notify=True):
        """Close the sample boundary, drain the writer, then refresh index.csv."""
        if not self._recording_active():
            return None
        self.recording_timer.stop()
        result = self.recorder.stop(reason)
        self.last_sample = result
        self.collection_bar.set_recording(False)
        index_error = ""
        if result is not None and not result.meta_error:
            try:
                self.dataset_store.rebuild_index()
            except OSError as error:
                index_error = str(error)
                logger.exception("样本清单刷新失败")
        self._show_sample_result(result, index_error)
        # 手动输入的新编号在首次录制后也需要加入下拉列表，重启可直接选择。
        self._refresh_identity_choices(self.collection_bar.subject.currentText(),
                                       self.collection_bar.session.currentText())
        self._layout_changed()
        self._refresh_recording_availability()
        event(logger, logging.INFO, "结束采集训练样本", reason=reason,
              status=getattr(result, "status", None), records=getattr(result, "records", None),
              written=getattr(result, "written_records", None),
              missing=list(getattr(result, "missing_block_ids", ())),
              sample=str(result.sample_dir) if result is not None else None)
        if notify and result is not None and result.status != "complete":
            QMessageBox.warning(self, "样本未完整保存",
                                "样本已保留但标记为未完成，不会自动进入正常训练集合。\n"
                                f"原因：{result.end_reason}\n目录：{result.sample_dir}")
        return result

    def _show_sample_result(self, result, index_error=""):
        bar = self.collection_bar
        if result is None:
            bar.set_status("结束采集时未能取得样本结果，请检查日志", "error")
            bar.set_sample_state("未知", "error")
            return
        bar.set_stats(result.duration_s, result.records, result.frames_by_block, result.missing_block_ids)
        bar.set_integrity(result.integrity)
        trial = result.sample_dir.name
        if result.status == "complete" and not result.meta_error:
            integrity = result.integrity or {}
            semantic = result.semantic or {}
            quality = integrity.get("status", "pending")
            if semantic.get("status") in LEVELS and LEVELS[semantic["status"]] > LEVELS.get(quality, 0):
                quality = semantic["status"]
            reasons = list(integrity.get("reasons", [])) + list(semantic.get("reasons", []))
            detail = "；".join(reasons[:2])
            activation = result.activation or {}
            region = f" · 区域 {activation.get('region_set') or '-'}（{activation.get('region_count', 0)} 块）"
            bar.set_status(f"已保存 {trial} · {result.duration_s:.1f} 秒 · {QUALITY_NAMES[quality]}"
                           + region + (f" · {detail}" if detail else " · 待人工确认动作"),
                           "error" if quality == "fail" else ("warn" if quality == "warn" else "ok"))
            bar.set_sample_state("不合格" if quality == "fail" else "未判定", "error" if quality == "fail" else "warn",
                                 can_accept=quality != "fail", can_void=True)
        else:
            reasons = []
            if result.meta_error:
                reasons.append(result.meta_error)
            if result.missing_block_ids:
                reasons.append("缺失 BLOCK " + "、".join(map(str, result.missing_block_ids)))
            if result.written_records != result.records:
                reasons.append(f"只写入 {result.written_records}/{result.records} 行")
            if result.errors:
                reasons.append(f"{len(result.errors)} 项数据异常")
            bar.set_status(f"未完成 {trial}（{result.end_reason}）：" + "；".join(reasons or ["请检查日志"]), "error")
            bar.set_sample_state("未完成", "error", can_void=result.meta_path is not None)
        bar.status.setToolTip(str(result.sample_dir) + "\n" + "\n".join(result.integrity.get("reasons", [])))
        if index_error:
            bar.set_status(bar.status.text() + f" · index.csv 未刷新：{index_error}", "error")

    def _refresh_recording(self):
        if not self._recording_active():
            self.recording_timer.stop()
            return
        stats = self.recorder.snapshot()
        self.collection_bar.set_stats(stats.duration_s, stats.records, stats.frames_by_block)
        self.collection_bar.set_integrity(stats.integrity)
        # 单块时间空缺继续保留其他块；磁盘已失败时及时结束本段。
        if stats.fatal_error:
            self.finish_recording("write_error", notify=False)
            return
        plan = self.recording_plan
        detail = (f"录制中 · {plan.subject_id} / {plan.session_id} / {plan.trial_id}"
                  if plan is not None else "录制中")
        if plan is not None:
            detail += f" · 已收 {len(stats.frames_by_block)}/{len(plan.expected_block_ids)} 块"
        if stats.dropped:
            detail += f" · 队列丢弃 {stats.dropped} 行"
        if stats.errors:
            detail += f" · {len(stats.errors)} 项异常（结束时标为未完成）"
        quality = stats.integrity.get("status", "pass")
        if stats.integrity.get("reasons"):
            detail += " · " + stats.integrity["reasons"][0]
        self.collection_bar.set_status(detail, "error" if quality == "fail" else "warn")

    def _start_recording(self):
        try:
            self.start_recording()
        except (OSError, ValueError) as error:
            logger.exception("未能开始采集")
            QMessageBox.warning(self, "无法开始采集", str(error))

    def mark_sample_accepted(self, accepted):
        """有效样本留在正式目录；作废样本整体回收，避免训练时误读。"""
        if self._recording_active():
            raise ValueError("正在录制，不能标记样本")
        result = self.last_sample
        if result is None or result.meta_path is None or not Path(result.meta_path).exists():
            raise ValueError("没有可标记的已结束样本")
        if accepted:
            self.dataset_store.update_accepted(result.meta_path, accepted)
            self.dataset_store.rebuild_index()
            self.collection_bar.set_sample_state("有效", "ok")
        else:
            archive = self.dataset_store.archive_sample(result.meta_path)
            # 已移动的样本不再作为“本次”操作目标，防止重复作废或标记有效。
            self.last_sample = None
            self.collection_bar.set_sample_state("已作废", "idle")
            self.collection_bar.set_status("已作废：本次样本已移出正式数据集，可通过“恢复删除”找回", "ok")
            self.collection_bar.setToolTip(f"本次样本回收位置：{archive}")
            self._refresh_trial_preview()

    def _mark_sample(self, accepted):
        try:
            self.mark_sample_accepted(accepted)
        except (OSError, ValueError) as error:
            logger.exception("样本标记失败")
            QMessageBox.warning(self, "样本未标记", str(error))

    def _open_dataset(self):
        try:
            self.dataset_store.root.mkdir(parents=True, exist_ok=True)
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.dataset_store.root))):
                raise OSError(f"无法打开数据集目录：{self.dataset_store.root}")
        except OSError as error:
            logger.exception("打开数据集目录失败")
            QMessageBox.warning(self, "数据集目录", str(error))

    def save_collection_layout(self):
        """Snapshot the actual UI permutation, never its preprocessed sample values."""
        if self._recording_active():
            raise ValueError("正在采集，不能修改采集布局；请先结束采集")
        if self.collection_load_error or self.mask_load_error:
            raise ValueError("配置加载失败，请先修复配置后重启，未覆盖已有文件")
        if self.collection_locked and not self._collection_positions_changed():
            raise ValueError("采集布局已固定；需要修改时先点击“调整采集布局”")
        if self.input_source not in ("hardware", "external", "stopped") or not self.heatmap.auto_layout:
            raise ValueError("请先连接实际皮肤并调整方向；模拟数据不能保存为采集布局")
        specs = self.heatmap.layout_specs
        if self.worker is not None:
            with self.worker.buffer.lock:
                for spec in specs:
                    block = self.worker.buffer.blocks.get(spec.block_id)
                    if block is None or block.values.shape != (spec.rows, spec.cols):
                        raise ValueError("硬件尺寸正在变化，请等待界面更新后重新保存")
        saved = self.collection_store.save(specs, self.sensor_masks, previous=self.collection_layout,
                                          placement_mode=self.heatmap.placement_mode)
        self.collection_layout = saved
        self.collection_locked = True
        self._apply_collection_view()
        self._layout_changed()
        self._refresh_recording_availability()
        event(logger, logging.INFO, "采集布局已保存并固定", layout_id=saved.layout_id,
              block_ids=list(saved.by_id), path=str(self.collection_store.directory))
        return saved

    def _save_collection_layout(self):
        try:
            self.save_collection_layout()
        except (OSError, ValueError) as error:
            logger.exception("采集布局未保存")
            QMessageBox.warning(self, "采集布局未保存", str(error))

    def _edit_collection_layout(self):
        if self._recording_active():
            return
        if self.collection_layout is None or self.input_source == "demo":
            return
        self.collection_locked = False
        self._layout_changed()

    def _restore_collection_layout(self):
        if self._recording_active() or self.input_source == "demo":
            return
        # Re-read disk, including any newer version saved since this window started.
        try:
            loaded = self.collection_store.load()
            if loaded is None:
                raise ValueError("本地还没有采集布局，请先调整并保存")
            if self.mask_load_error:
                raise ValueError("有效点位配置加载失败，请修复后重启")
            updated = loaded.with_masks(self.sensor_masks)
            changed_ids = {
                block.spec.block_id for block in loaded.blocks
                if not np.array_equal(self.sensor_masks.get(block.spec.block_id, (block.spec.rows, block.spec.cols)),
                                      np.asarray(block.raw_valid_mask))
            }
            updated.save(self.mask_path)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "采集布局未恢复", str(error))
            return
        self.collection_layout = loaded
        self.collection_load_error = ""
        self.sensor_masks = updated
        self.heatmap.set_sensor_masks(updated)
        if changed_ids:
            self.pending_live_info = None
            self.live_info_state = None
            if self.worker is not None:
                for block_id in changed_ids:
                    self.worker.buffer.set_sensor_masks(updated, block_id)
                self._request_live_update()
            if self.display_values is not None and not self.raw_adc:
                self.display_values = {key: value for key, value in self.display_values.items() if key not in changed_ids}
            if not self.raw_adc:
                self.controls.calibration_progress.setValue(0)
                self.controls.calibration_status.setText("已恢复有效点位；变更的皮肤块需重新标定")
                self.controls.calibration_details.hide()
        self.collection_locked = True
        self._apply_collection_view()
        frame = self.current_frame if self.current_frame is not None else Frame(())
        self._display_frame(frame, self.display_values)
        self._layout_changed()
        self._refresh_recording_availability()

    def _apply_collection_view(self):
        layout = self.collection_layout
        known = layout.by_id
        orientations = {key: block.orientation for key, block in known.items()}
        for spec in self.heatmap.layout_specs:
            saved = known.get(spec.block_id)
            if saved is None or (spec.rows, spec.cols) != (saved.spec.rows, saved.spec.cols):
                orientations[spec.block_id] = BlockOrientation()
        self.heatmap.apply_collection_preset(
            layout.placement_mode, orientations, layout.positions, tuple(block.spec for block in layout.blocks))

    def _prepare_collection_directions(self, blocks):
        if not self.collection_locked or self.input_source == "demo":
            return
        known = self.collection_layout.by_id
        for block in blocks:
            saved = known.get(block.block_id)
            if saved is not None and block.values.shape == (saved.spec.rows, saved.spec.cols):
                self.heatmap.orientations[block.block_id] = saved.orientation
            else:
                # A changed device must not be displayed as if its old mapping were verified.
                self.heatmap.orientations[block.block_id] = BlockOrientation()

    def _orientation_target_changed(self, *_):
        block_id = self.controls.orientation_block.currentData()
        if block_id in self.heatmap.items:
            if self.heatmap.selected is None or self.heatmap.selected[0] != block_id:
                self.heatmap.select_sensor(self.heatmap.first_sensor(block_id))

    def _edit_sensor_mask(self):
        if self.collection_locked:
            return
        block_id = self.controls.orientation_block.currentData()
        if block_id not in self.heatmap.items:
            return
        spec = self.heatmap.items[block_id].spec
        shape = (spec.rows, spec.cols)
        dialog = SensorMaskDialog(spec, self.sensor_masks.get(block_id, shape), self,
                                  mismatch=self.sensor_masks.mismatched(block_id, shape), raw_adc=self.raw_adc)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.apply_sensor_mask(block_id, dialog.raw_mask())
        except (OSError, ValueError) as error:
            logger.exception("保存有效点位失败 block_id=%s", block_id)
            QMessageBox.warning(self, "点位未保存", str(error))

    def apply_sensor_mask(self, block_id, mask):
        """Persist before committing to worker/view, on the GUI thread."""
        if self._recording_active():
            raise ValueError("正在采集，不能修改有效点位；请先结束采集")
        if self.collection_locked:
            raise ValueError("采集布局已固定，请先点击“调整采集布局”")
        if self.mask_load_error:
            raise ValueError(f"原点位配置加载失败，未覆盖原文件。请先修复配置后重启：\n{self.mask_path}")
        item = self.heatmap.items.get(block_id)
        if item is None or np.asarray(mask).shape != (item.spec.rows, item.spec.cols):
            raise ValueError("编辑期间皮肤块尺寸或连接状态已变化，请重新打开点位编辑。")
        # The editor's nested event loop keeps ingestion running. Check the
        # latest worker shape as well, even before its next GUI snapshot.
        if self.worker is not None:
            with self.worker.buffer.lock:
                block = self.worker.buffer.blocks.get(block_id)
                if block is not None and block.values.shape != np.asarray(mask).shape:
                    raise ValueError("硬件尺寸已变化，请稍后重新编辑点位。")
        updated = self.sensor_masks.with_mask(block_id, mask)
        changed = not np.array_equal(self.sensor_masks.get(block_id, np.asarray(mask).shape), mask)
        updated.save(self.mask_path)
        self.sensor_masks = updated
        if changed:
            self.pending_live_info = None
            self.live_info_state = None
            if self.worker is not None:
                self.worker.buffer.set_sensor_masks(updated, block_id)
                self._request_live_update()
            if self.display_values is not None and not self.raw_adc:
                self.display_values = {key: value for key, value in self.display_values.items() if key != block_id}
            if not self.raw_adc:
                self.controls.calibration_progress.setValue(0)
                self.controls.calibration_status.setText(
                    f"BLOCK {block_id:02d} 点位已更新，请保持无按压，该块重新自动标定（约3秒）")
                self.controls.calibration_details.hide()
        elif self.worker is not None:
            # Same effective mask requires no recalibration.
            self.worker.buffer.set_sensor_masks(updated, block_id, recalibrate=False)
        self.heatmap.set_sensor_masks(updated)
        if self.current_frame is not None:
            self._display_frame(self.current_frame, self.display_values)
        event(logger, logging.INFO, "有效点位已保存", block_id=block_id,
              shape=np.asarray(mask).shape, valid=int(np.asarray(mask).sum()), path=str(self.mask_path))

    def _orient_block(self, action):
        if self.collection_locked or self._recording_active():
            return
        block_id = self.controls.orientation_block.currentData()
        if block_id not in self.heatmap.items or not self.heatmap.auto_layout:
            return
        self._orientation_target_changed()
        orientation = self.heatmap.orientations.get(block_id, BlockOrientation())
        if action in ("left", "right"):
            orientation = orientation.turned(1 if action == "left" else -1)
        elif action == "reset":
            orientation = BlockOrientation()
        else:
            orientation = orientation.mirrored(action)
        self.heatmap.set_orientation(block_id, orientation)

    def present_frame(self, frame: Frame) -> None:
        """Feed a received ADC frame on the GUI thread; no layout setup needed.

        Stop the demo before accepting external data so mock frames cannot
        overwrite it. Re-enter the demo explicitly with its play/preset controls.
        """
        self.timer.stop()
        self.input_source = "external"
        self.source_badge.setText("ADC 数据输入")
        self.controls.play.setText("返回演示")
        self._display_frame(frame)

    def _display_frame(self, frame: Frame, display_values=None, *, changed_ids=None, refresh_statistics=True):
        layout_blocks = tuple(self.live_blocks.values()) if self.input_source in ("hardware", "stopped") else None
        self._prepare_collection_directions(layout_blocks if layout_blocks is not None else frame.blocks)
        # Cache before set_frame: the heatmap re-emits the selected reading
        # while refreshing, and the detail panel reads this cache.
        self.display_values = display_values
        self.heatmap.set_frame(frame, display_values, layout_blocks,
                               self.live_missing_counts if layout_blocks is not None else None,
                               changed_ids=changed_ids)
        self.current_frame = frame
        self._update_collection_status()
        if refresh_statistics:
            self._refresh_statistics()
        if self.controls.auto_levels.isChecked():
            self._update_scale()

    def _refresh_statistics(self):
        # Not-ready points and empty cells are NaN: skip them in every statistic.
        arrays = self._active_arrays()
        values = np.concatenate([array.ravel() for array in arrays]) if arrays else np.array([])
        finite = values[np.isfinite(values)] if values.size else values
        for widget, value in ((self.min_label, f"{finite.min():.0f}" if finite.size else "—"),
                              (self.max_label, f"{finite.max():.0f}" if finite.size else "—"),
                              (self.mean_label, f"{finite.mean():.1f}" if finite.size else "—")):
            widget.setText(value)
        self._update_status()

    def _active_arrays(self):
        arrays = ({b.block_id: b.values for b in self.current_frame.blocks} if self.current_frame else {}) if self.display_values is None else self.display_values
        result = []
        for key, array in arrays.items():
            active = array[self.sensor_masks.get(key, array.shape)]
            if active.size:
                result.append(active)
        return result

    def advance_demo(self):
        self.sequence += 1
        self._display_frame(generate_frame(self.source_specs, self.sequence, self.controls.pattern.currentText()))

    def _reset_demo(self, *_):
        if self.worker is not None:
            return
        self.input_source = "demo"
        self._set_unit("ADC")
        self.source_badge.setText("模拟预览")
        self.controls.play.setText("暂停演示" if self.timer.isActive() else "继续演示")
        self.sequence = 0
        self._display_frame(generate_frame(self.source_specs, 0, self.controls.pattern.currentText()))

    def toggle_play(self):
        if self.worker is not None or not self.controls.demo_toggle.isChecked():
            return
        if self.input_source != "demo":
            self._reset_demo()
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start()
        self.controls.play.setText("暂停演示" if self.timer.isActive() else "继续演示")
        self._update_status()

    def step_demo(self):
        if self.worker is not None or not self.controls.demo_toggle.isChecked():
            return
        if self.input_source != "demo":
            self._reset_demo()
        if self.timer.isActive():
            self.toggle_play()
        self.advance_demo()

    def _preset_changed(self, index):
        self.source_specs = demo_specs(index)
        self._reset_demo()

    def _low_changed(self, value):
        if value >= self.controls.high.value():
            self.controls.low.setValue(self.controls.high.value() - 1)
        self._update_scale()

    def _high_changed(self, value):
        if value <= self.controls.low.value():
            self.controls.high.setValue(self.controls.low.value() + 1)
        self._update_scale()

    def _auto_changed(self, enabled):
        self.controls.low.setEnabled(not enabled)
        self.controls.high.setEnabled(not enabled)
        self._update_scale()

    def _update_scale(self, *_):
        low, high = self.controls.low.value(), self.controls.high.value()
        arrays = self._active_arrays()
        if self.controls.auto_levels.isChecked() and arrays:
            parts = [part for part in (array[np.isfinite(array)] for array in arrays) if part.size]
            if parts:
                low = min(float(part.min()) for part in parts)
                high = max(float(part.max()) for part in parts)
                if low == high:
                    low, high = max(0, low - 1), min(4095, high + 1)
        self.heatmap.set_scale(ColorScale(low, high, self.controls.palette.currentText()))

    def _update_status(self):
        count = len(self.current_frame.blocks) if self.current_frame else 0
        sequence = self.current_frame.sequence if self.current_frame else 0
        state = "演示运行中 · 目标 20 Hz" if self.timer.isActive() else "演示已暂停"
        if self.input_source == "external":
            state = "ADC 数据输入"
        elif self.input_source == "idle":
            state = "未连接硬件"
        elif self.input_source == "hardware":
            if self.raw_adc:
                state = "实时原始 ADC · 不校准 / 不过滤"
            elif self.preprocess is None:
                state = "实时采集"
            elif self.preprocess.ready:
                state = "实时采集"
            else:
                state = f"自动标定 {self.preprocess.progress}%"
            if self.last_scan and not self.last_scan.complete:
                state += " · 本轮缺块"
        elif self.input_source == "stopped":
            state = "采集已停止 · 当前画面非实时"
        coverage = f"本轮更新 {count}/{len(self.live_blocks)} 块" if self.input_source == "hardware" else f"显示 {count} 块"
        self.status.setText(f"  {state}     |     帧 {sequence:06d}     |     {coverage}     |     {self.unit}")

    def _hovered(self, reading: SensorReading | None):
        if reading is None:
            self.hover_label.setText("单击选点，按住皮肤块拖动；橙色虚线表示本块暂未更新。")
        else:
            value = "自动标定中或无显示数据" if reading.value is None else f"{reading.value:g} {self.unit}"
            if self.heatmap.items[reading.block_id].stale:
                value = "本块暂未更新（非实时）"
            self.hover_label.setText(f"BLOCK {reading.block_id:02d}   /   原始行 {reading.row + 1} · 列 {reading.col + 1}   /   {value}")

    def _selected(self, reading: SensorReading | None):
        if reading is not None:
            combo = self.controls.orientation_block
            index = combo.findData(reading.block_id)
            if index >= 0 and index != combo.currentIndex():
                with QSignalBlocker(combo):
                    combo.setCurrentIndex(index)
        self.selected_label.setText("—" if reading is None or reading.value is None else f"{reading.value:g}")
        if reading is None:
            self.selected_detail.setText("尚未选点    ·    拖动皮肤块移动 / 拖动空白平移 / 滚轮缩放 / F 适应窗口")
        else:
            self.selected_detail.setText(f"已锁定 BLOCK {reading.block_id:02d} · 原始行 {reading.row + 1} / 列 {reading.col + 1}    ·    单击其他方块切换，单击空白取消")
            self._show_selected_detail(reading)

    def _show_selected_detail(self, reading: SensorReading):
        # 原始 ADC 与基线来自点击时刻的快照；显示值按当前归一化模式裁剪。
        if self.heatmap.frame is None or not self._at_sensor(reading):
            return
        raw = next((b for b in self.heatmap.frame.blocks if b.block_id == reading.block_id), None)
        state = self.preprocess
        if raw is None or state is None:
            return
        parts = [f"BLOCK {reading.block_id:02d} · 行 {reading.row + 1}/列 {reading.col + 1}",
                 f"原始 ADC {int(raw.values[reading.row, reading.col])}"]
        shown = (self.display_values or {}).get(reading.block_id)
        shown = None if shown is None else float(shown[reading.row, reading.col])
        baseline = state.baselines.get(reading.block_id)
        base = None if baseline is None else float(baseline[reading.row, reading.col])
        if state.norm == "none":
            if shown is not None and np.isfinite(shown):
                parts.append(f"{'滤波' if state.filter_enabled else 'ADC'} {shown:.1f}")
        else:
            if base is not None and np.isfinite(base):
                parts.append(f"基线 {base:.1f}")
            if shown is not None and np.isfinite(shown):
                parts.append(f"偏差 {shown:.1f}" if state.norm == "A" else f"归一 {shown:.2f}")
        if len(parts) > 2:
            self.selected_detail.setText("   ".join(parts))

    def _at_sensor(self, reading):
        block = next((b for b in self.heatmap.frame.blocks if b.block_id == reading.block_id), None)
        return block is not None and reading.row < block.values.shape[0] and reading.col < block.values.shape[1]

    def _set_unit(self, unit):
        if self.unit == unit:
            return
        self.unit = unit
        self.heatmap.set_unit(unit)
        for widget, text in ((self.min_title, "最小值"), (self.max_title, "最大值"),
                             (self.mean_title, "平均值"), (self.selected_title, "选中传感器")):
            widget.setText(f"{text} / {unit}")

    def _refresh_ports(self):
        previous = self.controls.port.currentData()
        self.controls.port.clear()
        try:
            for name, description in available_ports():
                self.controls.port.addItem(f"{name} · {description}", name)
            event(logger, logging.INFO, "刷新系统串口列表",
                  ports=[self.controls.port.itemText(i) for i in range(self.controls.port.count())])
        except Exception as error:
            logger.exception("串口枚举失败")
            self.controls.connection_status.setText(f"串口枚举失败：{error}")
        if previous:
            index = self.controls.port.findData(previous)
            if index >= 0:
                self.controls.port.setCurrentIndex(index)
        if not self.controls.port.count():
            self.controls.connection_status.setText("未发现串口，请接入 USB-RS485 后刷新")

    def _toggle_demo(self, enabled):
        if self.worker is not None or self._recording_active():
            return
        self.reception_bar.set_idle("模拟模式 · 不统计接收" if enabled else "未连接硬件")
        self.pending_live_info = None
        self.live_info_state = None
        self.controls.demo_panel.setVisible(enabled)
        self.controls.filter_toggle.setEnabled(not enabled and not self.raw_adc)
        self.controls.recalibrate.setEnabled(False)
        self.preprocess = None
        self.live_blocks.clear()
        self.live_missing_counts = None
        self.controls.calibration_progress.setValue(0)
        self.controls.calibration_details.hide()
        if enabled:
            self.controls.calibration_status.setText("模拟绘图测试，不使用硬件基线")
            self._reset_demo()
            self.timer.start()
        else:
            self.timer.stop()
            self.input_source = "idle"
            self.source_badge.setText("等待连接硬件")
            self.controls.calibration_status.setText("收到首帧立即绘图，无需校准。" if self.raw_adc else
                                                     "连接后保持无按压，约3秒自动标定完成")
            self._display_frame(Frame(()))
        self._refresh_recording_availability()

    def _connect_hardware(self):
        if self.worker is not None:
            logger.info("用户点击断开硬件")
            # A disconnect ends the open sample first: the CSV boundary must not
            # outlive the data source, and the writer has to drain before restart.
            if self._recording_active():
                self.finish_recording("disconnect", notify=False)
            self.collection_bar.start.setEnabled(False)
            self.pending_live_info = None
            self.accept_scans = False
            self.worker.stop()
            self.reception_timer.stop()
            self.reception_bar.set_idle("采集已停止")
            self.controls.connect_button.setEnabled(False)
            self.controls.connection_status.setText("正在停止采集并释放串口…")
            self.source_badge.setText("停止中 · 画面非实时")
            return
        port = self.controls.port.currentData()
        if not port:
            logger.warning("连接请求未选择有效串口")
            self.controls.connection_status.setText("请先连接 USB-RS485、刷新并选择串口")
            return
        event(logger, logging.INFO, "用户点击连接并显示原始ADC" if self.raw_adc else "用户点击连接并校准", port=port,
              device=self.controls.port.currentText(),
              baud=self.controls.baudrate.currentData(),
              mode=self.controls.receive_mode.currentData(), protocol=self.controls.protocol.currentData())
        self.last_gui_tick = None
        self.last_gui_log = perf_counter()
        self.gui_refreshes = 0
        self.gui_update_times.clear()
        self.gui_sample_ages.clear()
        self.pending_live_info = None
        self.live_info_state = None
        self.controls.demo_toggle.setChecked(False)
        self.timer.stop()
        direction = self.preprocess.direction if self.preprocess is not None else -1
        self.preprocess = None
        self.live_blocks.clear()
        self.live_missing_counts = None
        self.last_scan = None
        self.input_source = "hardware"
        self.controls.auto_levels.setChecked(False)
        self.controls.palette.setCurrentText("蓝青黄红" if self.raw_adc else "浅红深红")
        if self.raw_adc:
            self._set_unit("ADC")
            self.controls.low.setValue(0)
            self.controls.high.setValue(4095)
        else:
            self.controls.normalize.setEnabled(True)
            self.controls.norm_mode.setEnabled(self.controls.normalize.isChecked())
            self._norm_applied()
        self.controls.calibration_progress.setValue(0)
        self.controls.calibration_status.setText("等待原始 ADC，收到首帧立即绘图" if self.raw_adc else "检测设备中，请保持无按压，约3秒自动标定")
        self.controls.calibration_details.hide()
        self.source_badge.setText("连接并检测硬件")
        self._display_frame(Frame(()), {})
        self.worker = SerialWorker(port, self, mode=self.controls.receive_mode.currentData(),
                                   direction=direction,
                                   protocol=self.controls.protocol.currentData(),
                                   baudrate=self.controls.baudrate.currentData(),
                                   raw_adc=self.raw_adc, norm=self._active_norm(),
                                   filter_enabled=self.controls.filter_toggle.isChecked())
        self.worker.buffer.set_sensor_masks(self.sensor_masks)
        # Raw blocks are offered to the recorder before any display preprocessing.
        self.attach_recording_sink(self.worker.buffer)
        self.worker.status_changed.connect(self._serial_status)
        self.worker.failed.connect(self._serial_failed)
        self.worker.data_ready.connect(self._request_live_update, Qt.ConnectionType.QueuedConnection)
        self.worker.finished.connect(self._worker_finished)
        self.accept_scans = True
        self.reception_bar.set_counts({})
        self.reception_timer.start()
        self.controls.connect_button.setText("断开硬件")
        for widget in (self.controls.port, self.controls.refresh_ports, self.controls.demo_toggle,
                       self.controls.receive_mode, self.controls.protocol, self.controls.baudrate):
            widget.setEnabled(False)
        self.worker.start()
        self._refresh_recording_availability()

    @Slot()
    def _request_live_update(self):
        sender = self.sender()
        if isinstance(sender, SerialWorker) and sender is not self.worker:
            return
        if not self.accept_scans or self.worker is None or self.display_timer.isActive():
            return
        # perf_counter, not monotonic: this budget is sub-10 ms and would be
        # quantized to the ~15.6 ms Windows system tick by the coarse clock.
        remaining = 0 if self.last_gui_tick is None else DISPLAY_INTERVAL_MS - (perf_counter() - self.last_gui_tick) * 1000
        self.display_timer.start(max(0, ceil(remaining)))

    def _refresh_reception_rate(self):
        if self.accept_scans and self.worker is not None:
            self.reception_bar.set_counts(self.worker.buffer.reception_counts())

    def _flush_live(self):
        self.display_timer.stop()
        if self.accept_scans and self.worker is not None:
            now = perf_counter()
            self.worker.acknowledge_update()
            snapshot = self.worker.buffer.take()
            if snapshot is not None:
                previous_sequence = self.current_frame.sequence if self.current_frame else 0
                self._consume_scan(snapshot.scan, snapshot)
                finished = perf_counter()
                self.last_gui_tick = now
                self.gui_update_times.append((finished - now) * 1000)
                if snapshot.received_blocks > previous_sequence:
                    self.gui_sample_ages.append(max(0., finished - snapshot.latest_received_at) * 1000)
                self.gui_refreshes += 1
            if now - self.last_gui_log >= 5:
                state = self.preprocess
                event(logger, logging.INFO, "界面刷新统计", rendered_snapshots=self.gui_refreshes,
                      display_mode="raw_adc" if self.raw_adc else "preprocessed",
                      sequence=self.current_frame.sequence if self.current_frame else 0,
                      preprocess_ready=state.ready if state else None,
                      preprocess_progress=state.progress if state else None,
                      preprocess_norm=state.norm if state else None,
                      calibration_samples=dict(state.samples) if state else None,
                      update_ms_p95=round(float(np.percentile(self.gui_update_times, 95)), 3) if self.gui_update_times else None,
                      update_ms_max=round(max(self.gui_update_times), 3) if self.gui_update_times else None,
                      latest_sample_age_ms_p95=round(float(np.percentile(self.gui_sample_ages, 95)), 3) if self.gui_sample_ages else None)
                self.last_gui_log = now
                self.gui_update_times.clear()
                self.gui_sample_ages.clear()

    @Slot(object)
    def _receive_scan(self, scan):
        if self.accept_scans:
            self._consume_scan(scan)

    def _consume_scan(self, scan: Scan, snapshot=None):
        self.last_scan = scan
        self.input_source = "hardware"
        changed_ids = None
        if snapshot is None:
            # Direct injection without a worker: no preprocessing chain exists,
            # so the raw frame is displayed as-is.
            self.live_missing_counts = None
            self.live_blocks = {key: block for key, block in self.live_blocks.items() if key in scan.expected_ids}
            self.live_blocks.update({block.block_id: block for block in scan.frame.blocks})
            self._set_unit("ADC")
            values = None
            state = (scan.expected_ids, tuple(b.block_id for b in scan.frame.blocks), bool(scan.protocol_errors))
        else:
            self.live_blocks = {block.block_id: block for block in snapshot.layout_blocks}
            self.live_missing_counts = snapshot.missing_counts
            self.preprocess = snapshot.preprocess
            values = snapshot.display_values
            previous = self.display_values or {}
            changed_ids = {key for key in previous.keys() | values.keys()
                           if previous.get(key) is not values.get(key)}
            # Preprocessing/device/missing state transitions remain immediate;
            # continuously changing rates and statistics can update at 5 Hz.
            state = (scan.expected_ids,
                     tuple(b.block_id for b in scan.frame.blocks),
                     tuple(sorted(snapshot.missing_counts.items())),
                     snapshot.preprocess.ready, snapshot.preprocess.progress,
                     snapshot.preprocess.norm, snapshot.preprocess.direction,
                     snapshot.preprocess.filter_enabled,
                     tuple(sorted(snapshot.preprocess.samples.items())),
                     bool(scan.protocol_errors))
        refresh_info = snapshot is None or state != self.live_info_state
        self.live_info_state = state
        self.pending_live_info = (scan, snapshot)
        if refresh_info:
            self._update_live_info(scan, snapshot)
            self.pending_live_info = None
        self._display_frame(scan.frame, values, changed_ids=changed_ids, refresh_statistics=refresh_info)

    def _flush_live_info(self):
        if self.pending_live_info is not None and self.input_source == "hardware":
            scan, snapshot = self.pending_live_info
            self.pending_live_info = None
            self._update_live_info(scan, snapshot)
            self._refresh_statistics()

    def _update_live_info(self, scan, snapshot):
        state = snapshot.preprocess if snapshot is not None else None
        if self.raw_adc:
            self.controls.calibration_status.setText("直接显示原始 ADC · 不扣基线 · 不过滤跳变")
            self.source_badge.setText("实时原始 ADC")
        elif state is not None:
            self._set_unit({"B": "归一化", "A": "ΔADC", "none": "ADC"}[state.norm])
            self.controls.calibration_progress.setValue(state.progress)
            pending = {key: count for key, count in sorted(state.samples.items()) if count < state.target}
            self.controls.calibration_status.setText(state.message)
            if state.ready or not pending:
                self.controls.calibration_progress.setToolTip("")
                self.controls.calibration_details.hide()
                self.controls.calibration_details.setToolTip("")
            else:
                self.controls.calibration_progress.setToolTip(
                    f"标定样本最少 {min(state.samples.values())}/{state.target}（慢跟踪基线）")
                self.controls.calibration_details.setText(" · ".join(
                    f"BLOCK {key:02d} {count}/{state.target}" for key, count in pending.items()))
                self.controls.calibration_details.setToolTip("标定完成后基线仍会约每秒用中位数微调（慢跟踪）")
                self.controls.calibration_details.setVisible(True)
            self.controls.recalibrate.setEnabled(True)
            self.source_badge.setText("实时按压热力图" if state.ready else f"自动标定 {state.progress}%")
        missing = sorted(set(scan.expected_ids) - {b.block_id for b in scan.frame.blocks})
        if not scan.frame.blocks:
            self.source_badge.setText("等待数据 · 画面非实时")
        detail = f"本轮 {scan.elapsed:.2f} 秒 · {len(scan.frame.blocks)}/{len(scan.expected_ids)} 块"
        if snapshot is not None:
            detail = f"累计 {snapshot.received_blocks} 个 BLOCK · 帧率见热力图上方"
        if missing:
            detail += " · 缺失 " + ",".join(map(str, missing))
        if scan.protocol_errors:
            detail += f" · 累计协议错误 {scan.protocol_errors}"
        self.controls.connection_status.setText(detail)

    @Slot(str)
    def _serial_status(self, message):
        logger.info("采集状态：%s", message)
        if self.accept_scans:
            self.controls.connection_status.setText(message)

    @Slot(str)
    def _serial_failed(self, message):
        if self._recording_active():
            self.finish_recording("disconnect", notify=False)
        self.accept_scans = False
        self.display_timer.stop()
        self.reception_timer.stop()
        self.reception_bar.set_idle("接收中断")
        self.pending_live_info = None
        logger.error("采集失败通知：%s", message)
        self.controls.connection_status.setText(message)
        self.source_badge.setText("采集失败 · 画面非实时")
        self._refresh_recording_availability()

    @Slot()
    def _worker_finished(self):
        # The worker is gone: close the sample so the writer drains and meta.json
        # records why the recording ended instead of leaving it "recording".
        if self._recording_active():
            self.finish_recording("disconnect", notify=False)
        self.reception_timer.stop()
        self.reception_bar.set_idle("采集已停止")
        self._flush_live()
        self._flush_live_info()
        worker = self.worker
        self.worker = None
        self.accept_scans = False
        self.display_timer.stop()
        self.input_source = "stopped"
        self.source_badge.setText("已断开 · 画面非实时")
        self.controls.connect_button.setEnabled(True)
        self.controls.connect_button.setText("连接并显示" if self.raw_adc else "连接并校准")
        for widget in (self.controls.port, self.controls.refresh_ports, self.controls.demo_toggle,
                       self.controls.receive_mode, self.controls.protocol, self.controls.baudrate):
            widget.setEnabled(True)
        self.controls.recalibrate.setEnabled(False)
        if not self.raw_adc:
            self.controls.normalize.setEnabled(False)
            self.controls.norm_mode.setEnabled(False)
        self.controls.calibration_status.setText("已断开；下次连接直接显示原始 ADC" if self.raw_adc else "已断开；下次连接将重新自动标定")
        self.controls.calibration_details.hide()
        self._update_status()
        self._refresh_recording_availability()
        if worker:
            worker.deleteLater()
        if self.close_when_stopped:
            self.close()

    def _recalibrate(self):
        if self.raw_adc:
            return
        self.pending_live_info = None
        self.live_info_state = None
        logger.info("用户点击重置滤波与基线")
        if self.worker is not None:
            self.worker.buffer.reset_preprocessing()
            self._request_live_update()
        self.preprocess = None
        self.controls.calibration_progress.setValue(0)
        self.controls.calibration_progress.setToolTip("")
        self.controls.calibration_details.hide()
        self.controls.calibration_status.setText("请松开传感器，重新采集约3秒无按压样本")
        self.source_badge.setText("自动标定 0%")
        if self.current_frame:
            self._display_frame(self.current_frame, {})

    def _toggle_direction(self):
        if self.raw_adc:
            return
        self.pending_live_info = None
        self.live_info_state = None
        current = self.preprocess.direction if self.preprocess is not None else -1
        direction = -current
        logger.info("用户切换受压方向，目标 direction=%s", direction)
        if self.preprocess is not None:
            self.preprocess = replace(self.preprocess, direction=direction)
        if self.worker is not None:
            self.worker.buffer.set_direction(direction)
            self._request_live_update()
        self.controls.direction.setText("受压变小 ↓  · 点击切换" if direction == -1 else "受压变大 ↑  · 点击切换")

    def _filter_changed(self, enabled):
        """按钮只控制 M3；归一化选项和训练录制的原始数据保持独立。"""
        if self.raw_adc:
            return
        self.controls.filter_toggle.setText(
            "M3 滤波：开启 · 点击关闭" if enabled else "M3 滤波：关闭 · 点击开启")
        # 丢弃待显示的旧状态；后台切换会清空基线，并从下一帧重新标定。
        self.pending_live_info = None
        self.live_info_state = None
        if self.worker is not None:
            self.worker.buffer.set_filter_enabled(enabled)
            self._request_live_update()

    def _active_norm(self):
        """Current normalization mode from the two controls (B 为默认推荐)."""
        if self.raw_adc or not self.controls.normalize.isChecked():
            return "none"
        return self.controls.norm_mode.currentData()

    def _norm_applied(self):
        """Unit, scale bounds and widget states for the active normalization mode."""
        norm = self._active_norm()
        self.controls.auto_levels.setChecked(False)
        if norm == "B":
            self._set_unit("归一化")
            self.controls.low.setValue(0)
            self.controls.high.setValue(1)
        elif norm == "A":
            self._set_unit("ΔADC")
            self.controls.low.setValue(0)
            self.controls.high.setValue(600)
        else:
            self._set_unit("ADC")
            self.controls.low.setValue(0)
            self.controls.high.setValue(4095)
        for widget in (self.controls.low, self.controls.high, self.controls.auto_levels):
            widget.setEnabled(norm != "B")
        self._update_scale()

    def _norm_changed(self, enabled):
        if self.raw_adc:
            return
        self.controls.norm_mode.setEnabled(enabled)
        self.pending_live_info = None
        self.live_info_state = None
        norm = self._active_norm()
        logger.info("用户切换归一化开关 enabled=%s norm=%s", enabled, norm)
        if self.worker is not None:
            self.worker.buffer.set_norm(norm)
            self._request_live_update()
        self._norm_applied()

    def _norm_mode_changed(self, *_):
        if self.raw_adc or not self.controls.normalize.isChecked():
            return
        self.pending_live_info = None
        self.live_info_state = None
        norm = self._active_norm()
        logger.info("用户切换归一化策略 norm=%s", norm)
        if self.worker is not None:
            self.worker.buffer.set_norm(norm)
            self._request_live_update()
        self._norm_applied()

    def _check_freshness(self):
        # No incoming-data signal exists during silence. This low-rate check
        # keeps stale/gray transitions and diagnostics alive without 125 Hz polling.
        if self.accept_scans and self.worker is not None and not self.display_timer.isActive():
            self._flush_live()
        if self.accept_scans and self.last_scan:
            timeout = max(3, len(self.last_scan.expected_ids) * 1.0 + 1)
            if perf_counter() - self.last_scan.frame.received_at > timeout:
                self.source_badge.setText("等待数据 · 画面已过期")

    def _open_logs(self):
        try:
            folder = log_directory()
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
                raise OSError(f"无法打开日志目录：{folder}")
        except OSError as error:
            logger.exception("打开日志目录失败")
            QMessageBox.warning(self, "日志目录", str(error))

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出当前热力图", "SkinDraw_heatmap.png", "PNG 图片 (*.png)")
        if path:
            if not Path(path).suffix:
                path += ".png"
            try:
                self.heatmap.export_png(path)
                self.statusBar().showMessage(f"已导出：{path}", 6000)
            except OSError as error:
                QMessageBox.warning(self, "导出失败", str(error))

    def closeEvent(self, event):
        logger.info("关闭窗口 requested worker_active=%s", self.worker is not None)
        if self._recording_active():
            # Close the boundary and drain the CSV before the process can exit.
            self.finish_recording("app_exit", notify=False)
        self.timer.stop()
        self.display_timer.stop()
        self.health_timer.stop()
        self.info_timer.stop()
        self.reception_timer.stop()
        if self.worker is not None:
            self.close_when_stopped = True
            self.accept_scans = False
            self.worker.stop()
            event.ignore()
            return
        super().closeEvent(event)
