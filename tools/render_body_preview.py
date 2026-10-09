"""Render anatomical placement using labeled simulation only; no serial access."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QScrollArea

from skin_draw.body_layout import BODY_PRESET
from skin_draw.mapping import BlockOrientation
from skin_draw.models import Block, Frame
from skin_draw.ui.main_window import MainWindow
from skin_draw.ui.theme import apply_theme


app = QApplication([])
apply_theme(app)
window = MainWindow()
blocks, response = [], {}
for entry in BODY_PRESET:
    key, rows, cols = entry["id"], entry["rows"], entry["cols"]
    yy, xx = np.mgrid[:rows, :cols]
    values = np.rint(580*np.exp(-((yy-(rows-1)/2)**2+(xx-(cols-1)/2)**2)/4))
    response[key] = values
    blocks.append(Block(key, 2500-values))
frame = Frame(tuple(blocks))
window.present_frame(frame)
window._set_unit("ΔADC")
window._display_frame(frame, response)
window.controls.palette.setCurrentText("浅红深红")
window.controls.high.setValue(1000)
window.source_badge.setText("人体预设 · 模拟数据")
window.reception_bar.set_idle("模拟预览 · 未连接硬件")
# Preview demonstrates existing rotation controls; preset itself never rotates.
for key in (7, 8, 13, 14):
    window.heatmap.set_orientation(key, BlockOrientation(90))
window.controls.body_positions.click()
window.show()
window.findChild(QScrollArea).ensureWidgetVisible(window.controls.body_positions)
output = Path(__file__).resolve().parents[1] / "artifacts" / "body-layout-preview"
output.mkdir(parents=True, exist_ok=True)
for size, name in (((1250, 860), "full.png"), ((1000, 720), "small.png")):
    window.resize(*size)
    app.processEvents()
    window.findChild(QScrollArea).ensureWidgetVisible(window.controls.body_positions)
    QTest.qWait(160)
    window.heatmap.fit_view()
    app.processEvents()
    assert window.grab().save(str(output / name))
partial = Frame(tuple(b for b in blocks if b.block_id in (1, 3, 7, 14)))
window._display_frame(partial, {k: response[k] for k in (1, 3, 7, 14)})
window.source_badge.setText("4块连接示意 · 模拟数据")
QTest.qWait(160)
assert window.grab().save(str(output / "partial.png"))
window.close()
print(output)
