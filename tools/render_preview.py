"""Render the app's own Qt widgets offscreen for visual QA (no desktop capture)."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from skin_draw.ui.main_window import MainWindow
from skin_draw.ui.theme import apply_theme

root = Path(__file__).resolve().parents[1]
output = root / "artifacts"
output.mkdir(exist_ok=True)
app = QApplication([])
apply_theme(app)
window = MainWindow(demo=True)
window.timer.stop()
window.controls.play.setText("继续演示")
window._update_status()
window.show()
app.processEvents()
from PySide6.QtTest import QTest
QTest.qWait(120)
window.heatmap.fit_view()
window.heatmap.select_sensor((1, 4, 3))
app.processEvents()
assert window.grab().save(str(output / "preview.png"))
window.controls.preset.setCurrentIndex(1)
app.processEvents()
window.heatmap.fit_view()
app.processEvents()
assert window.grab().save(str(output / "preview-multi.png"))
window.controls.preset.setCurrentIndex(3)
window.controls.show_values.setChecked(True)
window.controls.pattern.setCurrentText("四角定位")
app.processEvents()
window.heatmap.fit_view()
app.processEvents()
assert window.grab().save(str(output / "preview-values.png"))
window.resize(1000, 720)
window.controls.preset.setCurrentIndex(2)
app.processEvents()
QTest.qWait(120)
window.heatmap.fit_view()
app.processEvents()
assert window.grab().save(str(output / "preview-small.png"))
window.close()
print(output)
