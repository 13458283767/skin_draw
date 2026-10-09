"""Hardware samples calibrate in the worker; the GUI paints the latest state."""

from pathlib import Path
from collections import deque
from dataclasses import replace
from math import ceil
import logging

import numpy as np
from time import perf_counter
from PySide6.QtCore import QSignalBlocker, QTimer, Qt, Slot, QUrl
from PySide6.QtGui import QAction, QKeySequence, QDesktopServices
from PySide6.QtWidgets import (QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from ..colors import ColorScale
from ..acquisition import Scan
from ..models import Frame
from ..mapping import BlockOrientation
from ..sensor_masks import SensorMasks, default_mask_path
from ..transport import available_ports
from ..simulator import demo_specs, generate_frame
from .control_panel import ControlPanel, label
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


class MainWindow(QMainWindow):
    def __init__(self, demo: bool = False, *, mask_path=None, raw_adc=False):
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
        self._connect_controls()
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
        self.controls.edit_sensor_mask.setEnabled(bool(specs))
        for widget in (combo, self.controls.rotate_left, self.controls.rotate_right,
                       self.controls.mirror_horizontal, self.controls.mirror_vertical,
                       self.controls.reset_orientation):
            widget.setEnabled(enabled)

    def _orientation_target_changed(self, *_):
        block_id = self.controls.orientation_block.currentData()
        if block_id in self.heatmap.items:
            if self.heatmap.selected is None or self.heatmap.selected[0] != block_id:
                self.heatmap.select_sensor(self.heatmap.first_sensor(block_id))

    def _edit_sensor_mask(self):
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
        # Cache before set_frame: the heatmap re-emits the selected reading
        # while refreshing, and the detail panel reads this cache.
        self.display_values = display_values
        self.heatmap.set_frame(frame, display_values, layout_blocks,
                               self.live_missing_counts if layout_blocks is not None else None,
                               changed_ids=changed_ids)
        self.current_frame = frame
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
                parts.append(f"滤波 {shown:.1f}")
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
        if self.worker is not None:
            return
        self.reception_bar.set_idle("模拟模式 · 不统计接收" if enabled else "未连接硬件")
        self.pending_live_info = None
        self.live_info_state = None
        self.controls.demo_panel.setVisible(enabled)
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

    def _connect_hardware(self):
        if self.worker is not None:
            logger.info("用户点击断开硬件")
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
                                   raw_adc=self.raw_adc, norm=self._active_norm())
        self.worker.buffer.set_sensor_masks(self.sensor_masks)
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
        self.accept_scans = False
        self.display_timer.stop()
        self.reception_timer.stop()
        self.reception_bar.set_idle("接收中断")
        self.pending_live_info = None
        logger.error("采集失败通知：%s", message)
        self.controls.connection_status.setText(message)
        self.source_badge.setText("采集失败 · 画面非实时")

    @Slot()
    def _worker_finished(self):
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
