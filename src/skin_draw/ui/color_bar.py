from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import QWidget

from ..colors import ColorScale, lookup_table


class ColorBar(QWidget):
    """Legend uses the exact same lookup table as the sensor cells."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scale = ColorScale()
        self.unit = "ADC"
        self.setFixedWidth(92)
        self.setMinimumHeight(220)

    def set_scale(self, scale: ColorScale) -> None:
        self.scale = scale
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        top, bottom = 48, self.height() - 46
        rect = QRectF(10, top, 18, bottom - top)
        gradient = QLinearGradient(0, bottom, 0, top)
        for i, rgb in enumerate(lookup_table(self.scale.palette)):
            gradient.setColorAt(i / 255, QColor(*map(int, rgb)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(rect, 4, 4)
        painter.setPen(QColor("#64748b"))
        painter.drawText(QRectF(9, 13, 80, 22), self.unit)
        for i in range(5):
            value = self.scale.high - (self.scale.high - self.scale.low) * i / 4
            y = top + (bottom - top) * i / 4
            painter.drawLine(30, round(y), 34, round(y))
            painter.drawText(QRectF(39, y - 10, 52, 20), Qt.AlignmentFlag.AlignVCenter, f"{value:.0f}")
        painter.end()
