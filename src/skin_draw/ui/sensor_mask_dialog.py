"""Edit in screen orientation, return validity in raw hardware coordinates."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox, QHBoxLayout,
                               QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)


class SensorMaskDialog(QDialog):
    def __init__(self, spec, valid_mask, parent=None, *, mismatch=False, raw_adc=False):
        super().__init__(parent)
        self.spec = spec
        self._raw_mask = np.array(valid_mask, dtype=bool, copy=True)
        self.indices = spec.index_map()
        self.setWindowTitle(f"编辑有效点位 · BLOCK {spec.block_id:02d}")
        rows, cols = spec.display_shape
        self.resize(min(920, max(480, cols * 48 + 65)), min(800, max(440, rows * 44 + 240)))
        box = QVBoxLayout(self)
        note = QLabel("单击格子切换：蓝色“有”为有效传感器，灰色 × 为空位。\n"
                      "排列与当前画面方向一致；悬停可查看原始行、列。")
        note.setWordWrap(True)
        box.addWidget(note)
        if mismatch:
            warning = QLabel("该块尺寸与已保存配置不同，旧配置未应用。请按当前尺寸重新标记。")
            warning.setWordWrap(True)
            warning.setStyleSheet("color: #b45309")
            box.addWidget(warning)
        self.grid = QTableWidget(rows, cols)
        self.grid.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.grid.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.grid.horizontalHeader().setDefaultSectionSize(44)
        self.grid.verticalHeader().setDefaultSectionSize(40)
        self.grid.setHorizontalHeaderLabels([str(c + 1) for c in range(cols)])
        self.grid.setVerticalHeaderLabels([str(r + 1) for r in range(rows)])
        for row in range(rows):
            for col in range(cols):
                raw_row, raw_col = divmod(int(self.indices[row, col]), spec.cols)
                item = QTableWidgetItem()
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setToolTip(f"原始行 {raw_row + 1} / 列 {raw_col + 1}")
                self.grid.setItem(row, col, item)
        self.grid.cellClicked.connect(self.toggle_cell)
        box.addWidget(self.grid, 1)
        row = QHBoxLayout()
        self.restore_button = QPushButton("全部恢复有效")
        self.restore_button.clicked.connect(self.restore_all)
        row.addWidget(self.restore_button)
        self.count_label = QLabel()
        row.addWidget(self.count_label, 1)
        box.addLayout(row)
        note = QLabel(("保存后自动记住点位，立即更新显示，无需校准。\n" if raw_adc else
                       "保存后自动记住点位；连接中修改会让该块重新采集约1秒且至少30个有效样本，请保持无按压。\n")
                      + "其他块继续工作。原始数据不删除；全部隐藏时该块不显示传感器。")
        note.setWordWrap(True)
        box.addWidget(note)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存点位")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        box.addWidget(self.buttons)
        self.refresh()

    def raw_mask(self):
        return self._raw_mask.copy()

    def toggle_cell(self, row, col):
        index = self.indices[row, col]
        self._raw_mask.flat[index] = not self._raw_mask.flat[index]
        self.refresh()

    def restore_all(self):
        self._raw_mask[:] = True
        self.refresh()

    def refresh(self):
        display = self.spec.transform(self._raw_mask)
        for row in range(display.shape[0]):
            for col in range(display.shape[1]):
                valid = display[row, col]
                item = self.grid.item(row, col)
                item.setText("有" if valid else "×")
                item.setBackground(QColor("#c6e7f5" if valid else "#e5e7eb"))
                item.setForeground(QColor("#075985" if valid else "#8993a1"))
        self.count_label.setText(f"有效 {int(display.sum())} / {display.size} · 隐藏 {int((~display).sum())}")
