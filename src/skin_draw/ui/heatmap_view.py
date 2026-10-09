"""Reusable heatmap: set_layout + set_frame; no simulator or serial knowledge."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPicture
from PySide6.QtWidgets import QApplication, QGraphicsRectItem, QHBoxLayout, QWidget

from ..colors import ColorScale, MISSING_COLOR
from ..auto_layout import arrange_frame
from ..body_layout import BODY_NAMES, arrange_body
from ..mapping import BlockLayout, BlockOrientation, validate_layout
from ..models import Block, Frame
from ..sensor_masks import SensorMasks
from .color_bar import ColorBar
from .block_drag import BlockDrag


@dataclass(frozen=True)
class SensorReading:
    block_id: int
    row: int  # zero-based original hardware coordinates
    col: int
    value: int | float | None


class SensorGridItem(pg.GraphicsObject):
    """One graphics item per block. QPicture caches the colored square geometry.

    Squares are solid-filled; the color gradient is across ADC values, never
    spatial interpolation between sensors. ViewBox locks the x/y aspect ratio.
    """

    def __init__(self, spec: BlockLayout):
        super().__init__()
        self.spec = spec
        self.values: np.ndarray | None = None
        self.colors: np.ndarray | None = None
        self.valid_mask = np.ones(spec.display_shape, dtype=bool)
        self._valid_path = QPainterPath()
        self._valid_path.addRect(self.boundingRect())
        self.gap = .10
        self.show_values = False
        self.stale = False
        self.missing_count = 0
        self.picture = QPicture()
        self._picture_gap = None
        self._gap_path = QPainterPath()
        self.setPos(spec.x, spec.y)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.set_data(None, ColorScale())

    def boundingRect(self):
        rows, cols = self.spec.display_shape
        return QRectF(0, 0, cols, rows)

    def cell_rect(self, row: int, col: int) -> QRectF:
        inset = self.gap / 2
        return QRectF(col + inset, row + inset, 1 - self.gap, 1 - self.gap)

    def set_data(self, values: np.ndarray | None, scale: ColorScale) -> None:
        previous_values = self.values
        self.values = values
        rows, cols = self.spec.display_shape
        colors = (scale.map_values(values) if values is not None
                  else np.broadcast_to(MISSING_COLOR, (rows, cols, 3)))
        if self.colors is not None and np.array_equal(colors, self.colors) and self._picture_gap == self.gap:
            if self.show_values and not np.array_equal(previous_values, values):
                self.update()
            return
        self.colors = colors
        if self._picture_gap != self.gap:
            self._gap_path = QPainterPath()
            self._gap_path.setFillRule(Qt.FillRule.WindingFill)
            # A cached mask makes exact-width gaps without recording one draw
            # command per sensor. Its color matches the canvas background.
            inset = self.gap / 2
            for col in range(cols + 1):
                left, right = max(0, col - inset), min(cols, col + inset)
                self._gap_path.addRect(QRectF(left, 0, right - left, rows))
            for row in range(rows + 1):
                top, bottom = max(0, row - inset), min(rows, row + inset)
                self._gap_path.addRect(QRectF(0, top, cols, bottom - top))
            self._picture_gap = self.gap
        pixels = np.ascontiguousarray(colors, dtype=np.uint8)
        bitmap = QImage(pixels.data, cols, rows, pixels.strides[0], QImage.Format.Format_RGB888).copy()
        self.picture = QPicture()
        painter = QPainter(self.picture)
        painter.setClipPath(self._valid_path)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(self.boundingRect(), bitmap)
        painter.fillPath(self._gap_path, QColor("#f8fafc"))
        painter.end()
        self.update()

    def set_valid_mask(self, mask, scale):
        if np.array_equal(self.valid_mask, mask):
            return
        self.valid_mask = np.array(mask, dtype=bool, copy=True)
        self._valid_path = QPainterPath()
        self._valid_path.setFillRule(Qt.FillRule.WindingFill)
        for row, col in np.argwhere(self.valid_mask):
            self._valid_path.addRect(QRectF(int(col), int(row), 1, 1))
        self.colors = None
        self.set_data(self.values, scale)

    def set_stale(self, stale: bool):
        if self.stale != stale:
            self.stale = stale
            self.update()

    def paint(self, painter, option, widget=None):
        painter.drawPicture(0, 0, self.picture)
        if self.stale:
            painter.save()
            painter.setPen(pg.mkPen("#d28b24", width=2, cosmetic=True, style=Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.boundingRect().adjusted(.02, .02, -.02, -.02))
            painter.restore()
        # Keep numbers legible. Hover always works even when cells are small.
        if self.stale or not self.show_values or self.values is None or abs(painter.worldTransform().m11()) < 38:
            return
        painter.save()
        painter.scale(.01, .01)
        font = QFont("Segoe UI")
        font.setPixelSize(23)
        painter.setFont(font)
        for row in range(self.values.shape[0]):
            for col in range(self.values.shape[1]):
                if not self.valid_mask[row, col] or not np.isfinite(self.values[row, col]):
                    continue
                red, green, blue = self.colors[row, col]
                lightness = .2126 * red + .7152 * green + .0722 * blue
                painter.setPen(QColor("#ffffff" if lightness < 140 else "#163044"))
                painter.drawText(QRectF(col * 100, row * 100, 100, 100),
                                 Qt.AlignmentFlag.AlignCenter, str(int(self.values[row, col])))
        painter.restore()


class HeatmapView(QWidget):
    sensor_hovered = Signal(object)   # SensorReading | None
    sensor_selected = Signal(object)
    layout_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout_specs: tuple[BlockLayout, ...] = ()
        self.frame: Frame | None = None
        self.display_values: dict[int, np.ndarray] | None = None
        self.geometry_frame: Frame | None = None
        self.retain_missing = False
        self.missing_counts = None
        self._last_reflow_key = None
        self._layout_signature = ()
        self._label_text = {}
        self.scale = ColorScale()
        self.items: dict[int, SensorGridItem] = {}
        self.labels: dict[int, pg.TextItem] = {}
        self.selected: tuple[int, int, int] | None = None
        self.hovered: tuple[int, int, int] | None = None
        self.gap = .10
        self.show_values = False
        self.auto_layout = True
        self.placement_mode = "grid"
        self.body_decorations = []
        self.orientations: dict[int, BlockOrientation] = {}
        self.sensor_masks = SensorMasks()
        self.manual_positions: dict[int, tuple[float, float]] = {}
        self.arrangement = None
        self.axis_labels = []
        box = QHBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        self.canvas = pg.GraphicsLayoutWidget()
        self.canvas.setBackground("#f8fafc")
        self.canvas.setMinimumSize(350, 320)
        self.canvas.setMouseTracking(True)
        self.view = self.canvas.addViewBox(lockAspect=True, invertY=True, enableMenu=False)
        self.view.setDefaultPadding(.04)
        self.view.setMouseMode(pg.ViewBox.PanMode)
        self.view.setLimits(minXRange=2, minYRange=2, maxXRange=2200, maxYRange=2200)
        self.bar = ColorBar()
        box.addWidget(self.canvas, 1)
        box.addWidget(self.bar)
        self.canvas.scene().sigMouseMoved.connect(self._mouse_moved)
        self.canvas.scene().sigMouseClicked.connect(self._mouse_clicked)
        self.canvas.viewport().installEventFilter(self)
        self.block_drag = BlockDrag(self)
        self.selection_box = self._outline("#142f44", 3)
        self.block_selection_box = self._outline("#0b8187", 2)
        self.block_selection_box.setPen(pg.mkPen("#0b8187", width=2, cosmetic=True,
                                                 style=Qt.PenStyle.DashLine))
        self.hover_box = self._outline("#ffffff", 2)
        self.resize_timer = QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.setInterval(80)
        self.resize_timer.timeout.connect(self._reflow)
        self.view.sigResized.connect(lambda *_: self.resize_timer.start())
        self.view.sigRangeChanged.connect(self._update_annotations)

    @staticmethod
    def _outline(color: str, width: float) -> QGraphicsRectItem:
        rect = QGraphicsRectItem()
        rect.setPen(pg.mkPen(color, width=width, cosmetic=True))
        rect.setBrush(Qt.BrushStyle.NoBrush)
        rect.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        rect.setZValue(20)
        rect.hide()
        return rect

    def set_layout(self, specs: tuple[BlockLayout, ...]) -> None:
        """Optional explicit physical layout; normal use only needs set_frame()."""
        specs = validate_layout(specs)
        self.auto_layout = False
        self.manual_positions.clear()
        self.arrangement = None
        self._last_reflow_key = None
        self._build_layout(specs)

    def _build_layout(self, specs: tuple[BlockLayout, ...]) -> None:
        self.block_drag.cancel()
        self.view.clear()
        self.items.clear()
        self.labels.clear()
        self._label_text.clear()
        self.body_decorations.clear()
        self.axis_labels.clear()
        self.layout_specs = specs
        self.frame = None
        self.selected = self.hovered = None
        for spec in specs:
            item = SensorGridItem(spec)
            item.set_valid_mask(spec.transform(self.sensor_masks.get(spec.block_id, (spec.rows, spec.cols))), self.scale)
            item.gap, item.show_values = self.gap, self.show_values
            item.set_data(None, self.scale)
            self.items[spec.block_id] = item
            self.view.addItem(item)
            offset = self.arrangement.title_offset if self.arrangement else 1.0
            self.labels[spec.block_id] = self._text(self._caption(spec),
                                                   spec.x + spec.display_shape[1] / 2,
                                                   spec.y - offset, (.5, .5), True)
            self._label_text[spec.block_id] = self._caption(spec)
            self.labels[spec.block_id].setToolTip("按住标题或彩色区域，拖动这一块皮肤的位置")
            rows, cols = spec.display_shape
            axis_top = 10 / self.arrangement.cell_pixels if self.arrangement else .32
            axis_left = 7 / self.arrangement.cell_pixels if self.arrangement else .30
            for col in range(cols):
                self.axis_labels.append(self._text(str(col + 1), spec.x + col + .5, spec.y - axis_top, (.5, .5)))
            for row in range(rows):
                self.axis_labels.append(self._text(str(row + 1), spec.x - axis_left, spec.y + row + .5, (1, .5)))
        # clear() removes these from the scene, but their Python owners persist.
        for rect in (self.selection_box, self.block_selection_box, self.hover_box):
            rect.hide()
            self.view.addItem(rect, ignoreBounds=True)
        self._update_body_decorations()
        self.sensor_selected.emit(None)
        self.sensor_hovered.emit(None)
        self.fit_view()
        self.layout_changed.emit()

    def _signature(self, frame: Frame):
        return tuple(sorted((b.block_id, *b.values.shape,
                             self.orientations.get(b.block_id, BlockOrientation())) for b in frame.blocks))

    def set_orientation(self, block_id: int, orientation: BlockOrientation) -> None:
        if not self.auto_layout or block_id not in self.items:
            raise ValueError("请先在自动布局中选择已识别的皮肤块")
        orientation.apply(self.items[block_id].spec)  # Validate before changing state.
        if orientation == self.orientations.get(block_id, BlockOrientation()):
            return
        self.orientations[block_id] = orientation
        self._reflow(count_missing=False)

    def _reflow(self, *, count_missing=True):
        if not self.auto_layout or self.frame is None or self.geometry_frame is None:
            return
        signature = self._signature(self.geometry_frame)
        key = (self.placement_mode, signature, round(self.view.width()), round(self.view.height()))
        if key == self._last_reflow_key:
            return
        self.block_drag.cancel()
        self._last_reflow_key = key
        frame, selected = self.frame, self.reading(self.selected)
        arrange = arrange_body if self.placement_mode == "body" else arrange_frame
        self.arrangement = arrange(self.geometry_frame, self.view.width(), self.view.height(), self.orientations)
        if self.manual_positions:
            self.arrangement = replace(self.arrangement, blocks=tuple(
                replace(spec, x=self.manual_positions[spec.block_id][0], y=self.manual_positions[spec.block_id][1])
                if spec.block_id in self.manual_positions else spec for spec in self.arrangement.blocks))
        if signature == self._layout_signature and self.items:
            # Resize moves existing items; it never clears the visible scene.
            self._position_items(self.arrangement.blocks)
            self._update_body_decorations()
            self.fit_view()
            self._refresh_readings()
            self.layout_changed.emit()
            return
        self._layout_signature = signature
        previous = {}
        for block_id, item in self.items.items():
            raw = None
            if item.values is not None:
                raw = np.empty((item.spec.rows, item.spec.cols), dtype=item.values.dtype)
                raw.ravel()[item.spec.index_map().ravel()] = item.values.ravel()
            previous[block_id] = (item.spec, raw, item.missing_count, item.stale)
        self._build_layout(self.arrangement.blocks)
        for block_id, item in self.items.items():
            if block_id in previous:
                old_spec, raw, missing_count, stale = previous[block_id]
                if (old_spec.rows, old_spec.cols) == (item.spec.rows, item.spec.cols):
                    item.set_data(item.spec.transform(raw) if raw is not None else None, self.scale)
                    item.missing_count = missing_count
                    item.set_stale(stale)
                    if stale:
                        self.labels[block_id].setColor("#b7791f")
        self._apply_frame(frame, count_missing=count_missing)
        if selected is not None and selected.block_id in self.items:
            spec = self.items[selected.block_id].spec
            if selected.row < spec.rows and selected.col < spec.cols:
                row, col = np.argwhere(spec.index_map() == selected.row * spec.cols + selected.col)[0]
                self.select_sensor((selected.block_id, int(row), int(col)))

    def _position_items(self, specs):
        """Move cached grids and their labels together; never rebuild pixels."""
        self.layout_specs = tuple(specs)
        axis_index = 0
        for spec in self.layout_specs:
            item = self.items[spec.block_id]
            item.spec = spec
            item.setPos(spec.x, spec.y)
            rows, cols = spec.display_shape
            title = self.arrangement.title_offset if self.arrangement else 1.0
            top = 10 / self.arrangement.cell_pixels if self.arrangement else .32
            left = 7 / self.arrangement.cell_pixels if self.arrangement else .30
            self.labels[spec.block_id].setPos(spec.x + cols / 2, spec.y - title)
            caption = self._caption(spec)
            if self._label_text.get(spec.block_id) != caption:
                self.labels[spec.block_id].setText(caption)
                self._label_text[spec.block_id] = caption
            for col in range(cols):
                self.axis_labels[axis_index].setPos(spec.x + col + .5, spec.y - top)
                axis_index += 1
            for row in range(rows):
                self.axis_labels[axis_index].setPos(spec.x - left, spec.y + row + .5)
                axis_index += 1

    def block_at(self, x, y):
        # Match painting order when manually positioned blocks overlap.
        for block_id in reversed(self.items):
            item = self.items[block_id]
            if item.boundingRect().contains(x - item.spec.x, y - item.spec.y):
                return block_id
        return None

    def draggable_block_at(self, x, y):
        block_id = self.block_at(x, y)
        if block_id is not None:
            return block_id
        scene_pos = self.view.mapViewToScene(QPointF(x, y))
        for block_id in reversed(self.labels):
            caption = self.labels[block_id]
            if caption.boundingRect().contains(caption.mapFromScene(scene_pos)):
                return block_id
        return None

    def move_block(self, block_id: int, x: float, y: float):
        item = self.items[block_id]
        if not np.isfinite([x, y]).all():
            raise ValueError("皮肤块坐标必须为有限数值")
        rows, cols = item.spec.display_shape
        (left, right), (top, bottom) = self.view.viewRange()
        title = self.arrangement.title_offset if self.arrangement else 1.0
        # Keep the matrix and caption inside the visible canvas when they fit.
        if right - left >= cols + 1:
            x = min(max(x, left + .5), right - cols - .5)
        if bottom - top >= rows + title + 1:
            y = min(max(y, top + title + .5), bottom - rows - .5)
        x, y = float(np.clip(x, -1000, 1000)), float(np.clip(y, -1000, 1000))
        if (x, y) == (item.spec.x, item.spec.y):
            return
        # Freeze all current positions, so resize/rotation cannot undo the arrangement.
        for spec in self.layout_specs:
            self.manual_positions.setdefault(spec.block_id, (spec.x, spec.y))
        self.manual_positions[block_id] = (x, y)
        self._position_items(replace(spec, x=x, y=y) if spec.block_id == block_id else spec
                             for spec in self.layout_specs)
        if self.arrangement:
            self.arrangement = replace(self.arrangement, blocks=self.layout_specs)
        self.hovered = None
        self._refresh_readings()

    def reset_positions(self):
        """Resume automatic packing, retaining each block's orientation."""
        if not self.auto_layout:
            return
        self.block_drag.cancel()
        self.placement_mode = "grid"
        self.manual_positions.clear()
        self._last_reflow_key = None
        self._reflow(count_missing=False)

    def restore_body_positions(self):
        """Restore only placement; measurements and orientation stay intact."""
        if not self.auto_layout:
            return
        self.block_drag.cancel()
        self.placement_mode = "body"
        self.manual_positions.clear()
        self._last_reflow_key = None
        self._reflow(count_missing=False)

    def _caption(self, spec):
        if self.auto_layout and self.placement_mode == "body":
            return f"{spec.block_id:02d} {BODY_NAMES.get(spec.block_id, '未分配')}"
        return f"BLOCK {spec.block_id:02d} · {spec.rows}×{spec.cols}"

    def _update_body_decorations(self):
        """Static placeholders only; absent devices are never sensor items."""
        for obj in self.body_decorations:
            self.view.removeItem(obj)
        self.body_decorations.clear()
        if not self.auto_layout or self.placement_mode != "body" or self.arrangement is None:
            return
        plan = self.arrangement
        for text, x, y in plan.headings:
            obj = self._text(text, x, y, (.5, .5), True)
            obj.setColor("#0b8187")
            self.body_decorations.append(obj)
        divider = QGraphicsRectItem(QRectF(plan.divider_x, 5, .02, plan.bounds[3]-8))
        divider.setPen(pg.mkPen(None))
        divider.setBrush(QColor("#dbe5ed"))
        divider.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.view.addItem(divider)
        self.body_decorations.append(divider)
        for spec in plan.slots:
            if spec.block_id in self.items:
                continue
            rows, cols = spec.display_shape
            rect = QGraphicsRectItem(QRectF(spec.x, spec.y, cols, rows))
            rect.setPen(pg.mkPen("#c9d4df", width=1, cosmetic=True, style=Qt.PenStyle.DashLine))
            rect.setBrush(QColor("#f0f4f8"))
            rect.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            rect.setZValue(-5)
            self.view.addItem(rect)
            caption = self._text(self._caption(spec), spec.x+cols/2,
                                 spec.y-plan.title_offset, (.5, .5), True)
            note = self._text("未连接", spec.x+cols/2, spec.y+rows/2, (.5, .5))
            self.body_decorations.extend((rect, caption, note))

    def _update_annotations(self, *_):
        # ViewBox applies its transform lazily at paint time. Use the requested
        # range here so annotation visibility never depends on a stale matrix.
        x_range = self.view.viewRange()[0]
        size = self.view.width() / max(x_range[1] - x_range[0], .001)
        for item in self.axis_labels:
            item.setVisible(size >= 22)

    def _text(self, text, x, y, anchor, bold=False):
        item = pg.TextItem(text, color="#52657c" if bold else "#8a98a9", anchor=anchor)
        item.setFont(QFont(QApplication.font().family(), 8 if self.auto_layout else 9,
                           QFont.Weight.DemiBold if bold else QFont.Weight.Normal))
        item.setPos(x, y)
        self.view.addItem(item, ignoreBounds=True)
        return item

    def set_frame(self, frame: Frame, display_values: dict[int, np.ndarray] | None = None,
                  layout_blocks: tuple[Block, ...] | None = None,
                  missing_counts: dict[int, int] | None = None, *,
                  changed_ids: set[int] | None = None) -> None:
        """Automatically derive count, shapes and positions from received blocks.

        Same structure reuses items and preserves zoom. Empty frames clear the
        automatic layout. Explicit set_layout() retains strict physical mapping.
        Call on the Qt main thread; the live GUI pulls the latest worker state.
        changed_ids is an optional hint from the immutable live response cache;
        callers must include every changed response. Omit it for mutable input.
        """
        geometry = frame if layout_blocks is None else Frame(layout_blocks)
        geometry_shapes = {block.block_id: block.values.shape for block in geometry.blocks}
        if any(geometry_shapes.get(block.block_id) != block.values.shape for block in frame.blocks):
            raise ValueError("实时数据必须匹配已识别的布局尺寸")
        reflow = self.auto_layout and (self.frame is None or self._signature(geometry) != self._layout_signature)
        if reflow or self.frame is None:
            changed_ids = None
        prepared = None
        if display_values is not None:
            shapes = {block.block_id: block.values.shape for block in frame.blocks}
            prepared = {}
            for block_id, values in display_values.items():
                if (changed_ids is not None and block_id not in changed_ids and self.display_values is not None
                        and block_id in self.display_values
                        and self.display_values[block_id].shape == shapes.get(block_id)):
                    prepared[block_id] = self.display_values[block_id]
                    continue
                array = np.array(values, dtype=float, copy=True)
                if (block_id not in shapes or array.shape != shapes[block_id]
                        or np.isinf(array).any() or np.any(array < 0) or np.any(array > 4095)):
                    raise ValueError("显示值必须与原始块形状一致，且位于 0–4095（NaN 表示未就绪）")
                array.flags.writeable = False
                prepared[block_id] = array
        # In physical-layout mode validate raw frames before replacing display state.
        if not self.auto_layout:
            for block in frame.blocks:
                if block.block_id not in self.items:
                    raise ValueError(f"布局中没有 BLOCK {block.block_id}")
                self.items[block.block_id].spec.transform(block.values)
        self.display_values = prepared
        self.geometry_frame = geometry
        self.retain_missing = layout_blocks is not None
        self.missing_counts = missing_counts
        if reflow:
            self.frame = frame
            self._reflow()
            return
        self._apply_frame(frame, changed_ids=changed_ids)

    def _apply_frame(self, frame: Frame, *, count_missing=True, changed_ids=None):
        readings_changed = set(changed_ids or ())
        transformed = {}
        for block in frame.blocks:
            if block.block_id not in self.items:
                raise ValueError(f"布局中没有 BLOCK {block.block_id}")
            if changed_ids is not None and block.block_id not in changed_ids and not self.items[block.block_id].stale:
                continue
            values = block.values if self.display_values is None else self.display_values.get(block.block_id)
            if values is not None:
                transformed[block.block_id] = self.items[block.block_id].spec.transform(values)
        self.frame = frame
        actual_ids = {block.block_id for block in frame.blocks}
        for block_id, item in self.items.items():
            was_stale = item.stale
            missing = block_id not in actual_ids
            if missing and self.retain_missing:
                if count_missing:
                    item.missing_count = (self.missing_counts.get(block_id, 1) if self.missing_counts is not None
                                          else item.missing_count + 1)
                item.set_stale(True)
                # Brief packet loss keeps the old pattern, explicitly marked
                # stale. Longer loss turns gray without changing its position.
                if item.missing_count >= 3 and item.values is not None:
                    item.set_data(None, self.scale)
            else:
                item.missing_count = 0
                item.set_stale(False)
                if changed_ids is None or block_id in changed_ids or was_stale:
                    item.set_data(transformed.get(block_id), self.scale)
            suffix = f"{item.spec.rows}×{item.spec.cols}" if block_id in actual_ids or self.retain_missing else "缺失数据"
            text = self._caption(item.spec) if self.placement_mode == "body" and self.auto_layout else f"BLOCK {block_id:02d} · {suffix}"
            if self._label_text.get(block_id) != text:
                self.labels[block_id].setText(text)
                self._label_text[block_id] = text
            if item.stale != was_stale:
                readings_changed.add(block_id)
                self.labels[block_id].setColor("#b7791f" if item.stale else "#52657c")
        if changed_ids is None:
            self._refresh_readings()
        else:
            # Data-only updates do not move outlines or change selection.
            if self.selected is not None and self.selected[0] in readings_changed:
                self.sensor_selected.emit(self.reading(self.selected))
            if self.hovered is not None and self.hovered[0] in readings_changed:
                self.sensor_hovered.emit(self.reading(self.hovered))

    def set_scale(self, scale: ColorScale) -> None:
        if scale == self.scale:
            return
        self.scale = scale
        self.bar.set_scale(scale)
        for item in self.items.values():
            item.set_data(item.values, scale)

    def set_unit(self, unit: str):
        if self.bar.unit == unit:
            return
        self.bar.unit = unit
        self.bar.update()

    def set_gap(self, gap: float) -> None:
        if not 0 <= gap <= .3:
            raise ValueError("方块间距比例必须为 0–0.3")
        self.gap = gap
        for item in self.items.values():
            item.gap = gap
            item.set_data(item.values, self.scale)
        self._refresh_readings()

    def set_show_values(self, enabled: bool) -> None:
        self.show_values = enabled
        for item in self.items.values():
            item.show_values = enabled
            item.update()

    def fit_view(self) -> None:
        if self.auto_layout and self.arrangement is not None and not self.manual_positions:
            self.view.setRange(QRectF(*self.arrangement.bounds), padding=0)
            self._update_annotations()
            return
        if not self.layout_specs:
            return
        left = min(s.x for s in self.layout_specs) - .8
        title = self.arrangement.title_offset if self.arrangement else 1.0
        top = min(s.y for s in self.layout_specs) - title - .6
        right = max(s.x + s.display_shape[1] for s in self.layout_specs) + .5
        bottom = max(s.y + s.display_shape[0] for s in self.layout_specs) + .5
        if self.auto_layout and self.placement_mode == "body" and self.arrangement is not None:
            bx, by, bw, bh = self.arrangement.bounds
            left, top = min(left, bx), min(top, by)
            right, bottom = max(right, bx+bw), max(bottom, by+bh)
        rect = QRectF(left, top, max(right - left, 6), max(bottom - top, 6))
        self.view.setRange(rect, padding=.035)

    def set_sensor_masks(self, masks):
        self.sensor_masks = masks
        for item in self.items.values():
            spec = item.spec
            item.set_valid_mask(spec.transform(masks.get(spec.block_id, (spec.rows, spec.cols))), self.scale)
        if self.reading(self.selected) is None:
            self.selected = None
        if self.reading(self.hovered) is None:
            self.hovered = None
        self._refresh_readings()
        self.layout_changed.emit()

    def first_sensor(self, block_id):
        cells = np.argwhere(self.items[block_id].valid_mask)
        return (block_id, int(cells[0, 0]), int(cells[0, 1])) if len(cells) else None

    def sensor_at(self, x: float, y: float) -> tuple[int, int, int] | None:
        """Hit-test a square in view coordinates, excluding the gaps."""
        block_id = self.block_at(x, y)
        if block_id is not None:
            item = self.items[block_id]
            local_x, local_y = x - item.spec.x, y - item.spec.y
            rows, cols = item.spec.display_shape
            if 0 <= local_x < cols and 0 <= local_y < rows:
                row, col = int(local_y), int(local_x)
                if item.valid_mask[row, col] and item.cell_rect(row, col).contains(local_x, local_y):
                    return block_id, row, col
        return None

    def reading(self, key: tuple[int, int, int] | None) -> SensorReading | None:
        if key is None:
            return None
        block_id, row, col = key
        item = self.items[block_id]
        raw_row, raw_col = item.spec.raw_index(row, col)
        if not item.valid_mask[row, col]:
            return None
        value = None if item.stale or item.values is None else item.values[row, col].item()
        if value is not None and not np.isfinite(value):
            value = None  # not-ready / empty points report no value
        return SensorReading(block_id, raw_row, raw_col, value)

    def select_sensor(self, key: tuple[int, int, int] | None) -> None:
        if key is not None:
            if self.reading(key) is None:
                key = None
        self.selected = key
        self._refresh_readings()

    def _place_outline(self, box, key):
        if key is None:
            box.hide()
            return
        block_id, row, col = key
        item = self.items[block_id]
        box.setRect(item.cell_rect(row, col).translated(item.spec.x, item.spec.y))
        box.show()

    def _refresh_readings(self):
        self._place_outline(self.selection_box, self.selected)
        if self.selected is None:
            self.block_selection_box.hide()
        else:
            item = self.items[self.selected[0]]
            self.block_selection_box.setRect(item.boundingRect().translated(item.pos()).adjusted(-.08, -.08, .08, .08))
            self.block_selection_box.show()
        self._place_outline(self.hover_box, self.hovered)
        self.sensor_selected.emit(self.reading(self.selected))
        self.sensor_hovered.emit(self.reading(self.hovered))

    def _mouse_moved(self, scene_pos):
        pos = self.view.mapSceneToView(scene_pos)
        self.hovered = self.sensor_at(pos.x(), pos.y())
        self._place_outline(self.hover_box, self.hovered)
        self.sensor_hovered.emit(self.reading(self.hovered))

    def _mouse_clicked(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.view.mapSceneToView(event.scenePos())
            self.select_sensor(self.sensor_at(pos.x(), pos.y()))

    def eventFilter(self, watched, event):
        from PySide6.QtCore import QEvent
        if watched is self.canvas.viewport() and self.block_drag.handle(event):
            return True
        if event.type() == QEvent.Type.Leave:
            if not self.block_drag.moving:
                self.canvas.viewport().unsetCursor()
            self.hovered = None
            self.hover_box.hide()
            self.sensor_hovered.emit(None)
        return super().eventFilter(watched, event)

    def export_png(self, path: str | Path) -> None:
        """Export the current viewport, including its ADC legend."""
        if not self.grab().save(str(path), "PNG"):
            raise OSError(f"无法保存图片：{path}")
