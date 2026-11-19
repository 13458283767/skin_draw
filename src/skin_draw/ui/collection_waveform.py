"""Reusable low-rate plot and explicit human boundary selection (no file I/O)."""

from time import perf_counter

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal, QSignalBlocker
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QDoubleSpinBox, QCheckBox)

from ..body_layout import BODY_NAMES


class CollectionWaveform(QWidget):
    confirmed = Signal(float, float)
    deferred = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setMaximumHeight(300)
        self.mode = "idle"
        self.curves = {}
        self.duration = 0.
        self.minimum_start_s = 0.
        self.pick = None
        self.last_draw_ms = 0.
        self._x_range = None
        self._y_range = None
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.message = QLabel("采集波形")
        self.message.setWordWrap(True)
        row.addWidget(self.message, 1)
        self.live_enabled = QCheckBox("显示实时波形")
        self.live_enabled.setChecked(True)
        self.live_enabled.setToolTip("关闭仅暂停绘图，原始数据写入与热力图继续运行")
        row.addWidget(self.live_enabled)
        box.addLayout(row)
        self.plot = pg.PlotWidget(background="w")
        self.plot.setLabel("left", "有效点平均原始 ADC")
        self.plot.setLabel("bottom", "相对录制开始", units="s")
        self.plot.showGrid(x=True, y=True, alpha=.15)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=True, y=True)
        self.legend = self.plot.addLegend(offset=(10, 5), colCount=7,
                                         brush=pg.mkBrush(255, 255, 255, 230), labelTextColor="#334155")
        box.addWidget(self.plot, 1)
        self.baseline_region = pg.LinearRegionItem((0., 1.), movable=False,
                                                   brush=pg.mkBrush(245, 158, 11, 35),
                                                   pen=pg.mkPen(245, 158, 11, 90))
        self.baseline_region.setZValue(-5)
        self.plot.addItem(self.baseline_region)
        self.baseline_region.hide()
        self.lines = [pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen(c, width=2))
                      for c in ("#16a34a", "#dc2626")]
        for i, line in enumerate(self.lines):
            self.plot.addItem(line)
            line.hide()
            line.sigPositionChanged.connect(lambda _, index=i: self._line_changed(index))
        bottom = QHBoxLayout()
        self.spins = []
        self.pick_buttons = []
        for i, name in enumerate(("接触开始", "接触结束")):
            button = QPushButton("图上选" + ("起点" if i == 0 else "终点"))
            button.clicked.connect(lambda _, index=i: self._select(index))
            self.pick_buttons.append(button)
            bottom.addWidget(button)
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(-1., 0.)
            spin.setSpecialValueText("未设置")
            spin.setValue(-1.)
            spin.setSuffix(" s")
            spin.setToolTip(name + "，相对于录制开始；可输入或拖动竖线")
            spin.valueChanged.connect(lambda _, index=i: self._spin_changed(index))
            self.spins.append(spin)
            bottom.addWidget(spin)
        self.save = QPushButton("确认起止并保存")
        self.save.clicked.connect(self._confirm)
        self.defer = QPushButton("暂不标注")
        self.defer.setToolTip("保留原始文件和未标注状态，清空显示后可继续采集")
        self.defer.clicked.connect(self.deferred.emit)
        bottom.addWidget(self.save)
        bottom.addWidget(self.defer)
        box.addLayout(bottom)
        self.plot.scene().sigMouseClicked.connect(self._clicked)
        self.set_mode("idle", "")
        self.hide()

    def begin(self, block_ids, baseline_seconds=0.):
        self.clear()
        if baseline_seconds:
            self.baseline_region.setRegion((0., baseline_seconds))
            self.baseline_region.show()
        colors = ("#2563eb", "#dc2626", "#15803d", "#9333ea", "#c2410c", "#0e7490", "#be185d",
                  "#4d7c0f", "#4338ca", "#92400e", "#0f766e", "#a21caf", "#475569", "#0369a1")
        for i, key in enumerate(sorted(block_ids)):
            self.curves[key] = self.plot.plot(pen=pg.mkPen(colors[i % len(colors)], width=1),
                                              name=BODY_NAMES.get(key, f"未分配部位（{key}）"),
                                              antialias=False, connect="finite")
        # Recomputing autorange on every curve update can delay the heatmap.
        self.plot.enableAutoRange(x=False, y=False)
        self._x_range = self._y_range = None
        self.set_mode("live", "请勿接触 · 正在采集本条基线" if baseline_seconds else
                      "实时滚动 · 原始 ADC 摘要；动作起止在停止后确认")
        self.show()

    def set_baseline_review(self, report, activation=None):
        report = report or {}
        if report.get("status") == "ready":
            self.minimum_start_s = report["action_allowed_s"]
            self.message.setText(f"橙色区为基线段；动作提示于 {self.minimum_start_s:.3f} 秒发出，请人工选择之后的接触起止")
            self.save.setToolTip("接触开始不得早于可以开始动作的提示时间")
        elif report:
            reasons = report.get("reasons", [])
            self.message.setText("本条基线不合格，原始数据已保留，请重采：" + "；".join(reasons[:2]))
            self.message.setToolTip("\n".join(reasons))
        self._validate()
        # 自动检测结果只作为初值；没有 baseline 报告的历史样本也能显示。
        self._apply_auto_bounds(activation)

    def _apply_auto_bounds(self, activation):
        """用接触检测的首末帧预填两条竖线；已有手动选点时不动。"""
        if not activation or self.spins[0].value() >= 0:
            return
        start, end = activation.get("contact_start_s"), activation.get("contact_end_s")
        if start is None or end is None or self.duration <= 0:
            return
        end = min(float(end), self.duration)
        if not 0 <= start < end:
            return
        self.spins[0].setValue(start)
        self.spins[1].setValue(end)
        if activation.get("open_at_end"):
            self.message.setText(f"自动检测 {start:.3f}–{end:.3f} 秒，但结束时仍在接触，终点不可信，请确认")
        else:
            self.message.setText(f"已自动标出响应区间 {start:.3f}–{end:.3f} 秒，可直接保存或微调")
        self.defer.setToolTip("保留自动检测的起止，不人工修改；清空显示后可继续采集")

    def set_mode(self, mode, message):
        self.mode = mode
        self.message.setText(message)
        for widget in self.spins + self.pick_buttons + [self.defer]:
            widget.setEnabled(mode == "review")
        self.live_enabled.setEnabled(mode == "live")
        for line in self.lines:
            line.setMovable(mode == "review")
        self._validate()

    def draw(self, snapshot, *, full=False):
        begin = perf_counter()
        low, high = float("inf"), float("-inf")
        for key, curve in self.curves.items():
            x, y = snapshot.curves.get(key, ([], []))
            curve.setData(x, y, connect="finite")
            finite = np.asarray(y)[np.isfinite(y)]
            if finite.size:
                low, high = min(low, float(finite.min())), max(high, float(finite.max()))
        if low <= high and (full or self._y_range is None or low < self._y_range[0] or high > self._y_range[1]):
            margin = max(10., (high - low) * .1)
            self._y_range = (low - margin, high + margin)
            self.plot.setYRange(*self._y_range, padding=0)
        if full:
            self.duration = snapshot.latest_s
            for spin, line in zip(self.spins, self.lines):
                spin.setMaximum(self.duration)
                line.setBounds((0., self.duration))
            self.plot.setXRange(0., max(.01, self.duration), padding=.01)
            self.set_mode("review", snapshot.status + " · 请人工选择起止")
        else:
            end = max(10., snapshot.latest_s)
            limits = (max(0., end - 10.), end)
            if limits != self._x_range:
                self.plot.setXRange(*limits, padding=0)
                self._x_range = limits
        self.last_draw_ms = (perf_counter() - begin) * 1000

    def _select(self, index):
        self.pick = index
        self.message.setText("点击波形选择" + ("接触开始" if index == 0 else "接触结束") + "，随后可拖动竖线微调")

    def _clicked(self, event):
        if self.mode != "review" or self.pick is None or event.button() != Qt.MouseButton.LeftButton:
            return
        view = self.plot.getViewBox()
        if view.sceneBoundingRect().contains(event.scenePos()):
            value = view.mapSceneToView(event.scenePos()).x()
            self.spins[self.pick].setValue(max(0., min(self.duration, value)))
            self.pick = None

    def _spin_changed(self, index):
        value = self.spins[index].value()
        with QSignalBlocker(self.lines[index]):
            self.lines[index].setValue(max(0., value))
        self.lines[index].setVisible(value >= 0)
        self._validate()

    def _line_changed(self, index):
        with QSignalBlocker(self.spins[index]):
            self.spins[index].setValue(self.lines[index].value())
        self._validate()

    def _validate(self):
        a, b = (spin.value() for spin in self.spins)
        self.save.setEnabled(self.mode == "review" and self.minimum_start_s <= a < b <= self.duration)

    def _confirm(self):
        if self.save.isEnabled():
            self.confirmed.emit(*(spin.value() for spin in self.spins))

    def clear(self):
        self.mode = "idle"
        self.pick = None
        self.minimum_start_s = 0.
        self.baseline_region.hide()
        self.save.setToolTip("")
        self.message.setToolTip("")
        for curve in self.curves.values():
            self.plot.removeItem(curve)
        self.curves.clear()
        self.legend.clear()
        for spin, line in zip(self.spins, self.lines):
            with QSignalBlocker(spin):
                spin.setValue(-1.)
            line.hide()
        self.set_mode("idle", "")
        self.hide()
