"""Compact reception totals with a per-block dropdown, refreshed at 1 Hz."""

from PySide6.QtWidgets import QHBoxLayout, QMenu, QSizePolicy, QToolButton, QWidget

from .control_panel import label


class ReceptionRateBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        self.total_label = label("接收 0 块帧/秒", "badge")
        self.total_label.setToolTip("一块皮肤的一个完整有效矩阵记为1块帧。统计最近1秒解析成功并交付采集处理的数据，包含标定帧和随后经滤波处理的帧；不含解析失败、未就绪回复或模拟数据。")
        row.addWidget(self.total_label)
        self.block_button = QToolButton()
        self.block_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.block_button.setToolTip("展开查看各块最近1秒的实收帧数。总块帧率不等于全身更新频率，也不等于屏幕帧率。")
        self.menu = QMenu(self.block_button)
        self.block_button.setMenu(self.menu)
        row.addWidget(self.block_button)
        row.addStretch()
        self.note = label("未连接硬件")
        self.note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        row.addWidget(self.note, 1)
        self.counts = {}
        self._actions = {}
        self.set_idle("未连接硬件")

    def set_counts(self, counts: dict[int, int]):
        self.counts = dict(sorted(counts.items()))
        self.total_label.setText(f"接收 {sum(counts.values())} 块帧/秒")
        self.note.setText("最近1秒 · 每秒更新")
        if set(self._actions) != set(self.counts):
            self.menu.clear()
            self._actions = {key: self.menu.addAction("") for key in self.counts}
        for key, count in self.counts.items():
            self._actions[key].setText(f"BLOCK {key:02d}    {count} 帧/秒")
        if counts:
            low, high = min(counts.values()), max(counts.values())
            rate = str(low) if low == high else f"{low}–{high}"
            self.block_button.setText(f"各块 {rate} 帧/秒")
        else:
            self.block_button.setText("各块帧率 · 等待数据")
        self.block_button.setEnabled(bool(counts))

    def set_idle(self, reason: str):
        self.set_counts({})
        self.block_button.setText("各块帧率")
        self.note.setText(reason)
