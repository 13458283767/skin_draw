"""Viewport mouse gestures for moving blocks without panning the camera."""

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QApplication


class BlockDrag:
    def __init__(self, heatmap):
        self.heatmap = heatmap
        self.pending = None
        self.moving = False

    def cancel(self):
        self.pending = None
        self.moving = False
        self.heatmap.canvas.viewport().unsetCursor()

    def handle(self, event):
        view = self.heatmap
        kind = event.type()
        if kind in (QEvent.Type.Hide, QEvent.Type.UngrabMouse, QEvent.Type.WindowDeactivate):
            self.cancel()
        if kind == QEvent.Type.Wheel and self.pending is not None:
            return True  # Keep the gesture's coordinate system fixed.
        if kind not in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove, QEvent.Type.MouseButtonRelease):
            return False
        pixel = event.position().toPoint()
        point = view.view.mapSceneToView(view.canvas.mapToScene(pixel))
        if kind == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            block_id = view.draggable_block_at(point.x(), point.y())
            if block_id is None:
                return False
            item = view.items[block_id]
            self.pending = (block_id, pixel, point, item.pos())
            self.moving = False
            return True
        if self.pending is None:
            if kind == QEvent.Type.MouseMove:
                if view.draggable_block_at(point.x(), point.y()) is not None:
                    view.canvas.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
                else:
                    view.canvas.viewport().unsetCursor()
            return False
        block_id, start_pixel, start_point, origin = self.pending
        if kind == QEvent.Type.MouseMove or (kind == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton):
            if not self.moving and (pixel - start_pixel).manhattanLength() >= QApplication.startDragDistance():
                self.moving = True
                view.canvas.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                if view.selected is None or view.selected[0] != block_id:
                    view.select_sensor(view.first_sensor(block_id))
            if self.moving:
                target = origin + point - start_point
                view.move_block(block_id, target.x(), target.y())
            if kind == QEvent.Type.MouseButtonRelease:
                moved = self.moving
                self.cancel()
                if moved:
                    view.layout_changed.emit()
                else:
                    key = view.sensor_at(point.x(), point.y())
                    # A click on the BLOCK caption selects that whole block too.
                    if key is None and view.block_at(point.x(), point.y()) is None:
                        key = view.first_sensor(block_id)
                    view.select_sensor(key)
            return True
        return False
