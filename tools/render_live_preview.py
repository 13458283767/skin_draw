"""UI-only synthetic verification of connection and calibration states."""

import os
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from skin_draw.acquisition import Scan
from skin_draw.models import Block, Frame
from skin_draw.simulator import demo_specs, generate_frame
from skin_draw.ui.main_window import MainWindow
from skin_draw.ui.theme import apply_theme

output = Path(__file__).resolve().parents[1] / "artifacts"
output.mkdir(exist_ok=True)
app = QApplication([])
apply_theme(app)
window = MainWindow()
window.controls.palette.setCurrentText("浅红深红")
window.show()
QTest.qWait(120)
window.grab().save(str(output / "preview-connect.png"))
specs = demo_specs()
for sequence in range(1, 31):
    blocks = tuple(Block(s.block_id, np.full((s.rows, s.cols), 2000)) for s in specs)
    window._consume_scan(Scan(Frame(blocks, sequence), tuple(range(1, 15)), 1.1))
    if sequence == 15:
        window.source_badge.setText("界面验证 · 校准测试数据")
        QTest.qWait(120)
        window.grab().save(str(output / "preview-calibrating.png"))
hotspots = generate_frame(specs, 40)
pressed = tuple(Block(b.block_id, np.rint(2000 - b.values.astype(float) * .28)) for b in hotspots.blocks)
window.controls.high.setValue(1000)
window._consume_scan(Scan(Frame(pressed, 31), tuple(range(1, 15)), 1.1))
window.source_badge.setText("界面验证 · 按压测试数据")
window.heatmap.select_sensor((1, 4, 4))
QTest.qWait(120)
window.grab().save(str(output / "preview-press.png"))
window.close()
print(output)
