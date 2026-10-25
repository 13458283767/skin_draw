"""Versioned, reversible acquisition coordinates; never stores display values."""

from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from .mapping import BlockLayout, BlockOrientation
from .sensor_masks import SensorMasks


@dataclass(frozen=True)
class CollectionBlock:
    spec: BlockLayout
    raw_valid_mask: tuple[tuple[bool, ...], ...]

    def __post_init__(self):
        if not 1 <= self.spec.block_id <= 14:
            raise ValueError("采集布局只接受 1～14 号皮肤块")
        if self.spec.row_map or self.spec.col_map:
            raise ValueError("请使用界面的旋转和镜像确定采集方向")
        mask = np.asarray(self.raw_valid_mask)
        if mask.dtype != np.bool_ or mask.shape != (self.spec.rows, self.spec.cols):
            raise ValueError("采集点位掩码必须与硬件原始矩阵尺寸一致")
        object.__setattr__(self, "raw_valid_mask", tuple(tuple(bool(v) for v in row) for row in mask))

    @property
    def orientation(self):
        return BlockOrientation(self.spec.rotation, self.spec.flip_x, self.spec.flip_y)

    def transform(self, raw):
        """Reorder ALL returned ADC cells, including hidden cells, without filtering."""
        return self.spec.transform(raw)

    def restore_raw(self, saved):
        saved = np.asarray(saved)
        if saved.shape != self.spec.display_shape:
            raise ValueError("数据尺寸与保存后的采集布局不一致")
        raw = np.empty((self.spec.rows, self.spec.cols), dtype=saved.dtype)
        raw.ravel()[self.spec.index_map().ravel()] = saved.ravel()
        return raw

    def to_dict(self):
        rows, cols = self.spec.display_shape
        return {
            "raw_rows": self.spec.rows, "raw_cols": self.spec.cols,
            "rows": rows, "cols": cols,
            "orientation": {"rotation": self.spec.rotation,
                            "flip_x": self.spec.flip_x, "flip_y": self.spec.flip_y},
            "saved_to_raw_index": self.spec.index_map().tolist(),
            "raw_valid_mask": [list(row) for row in self.raw_valid_mask],
            "valid_mask": self.spec.transform(np.asarray(self.raw_valid_mask)).tolist(),
        }

    @classmethod
    def from_dict(cls, block_id, data):
        spec = BlockLayout(block_id, data["raw_rows"], data["raw_cols"], **data["orientation"])
        result = cls(spec, data["raw_valid_mask"])
        # Validate redundant fields too: a damaged mapping must not be silently regenerated.
        if json.dumps(result.to_dict(), sort_keys=True) != json.dumps(data, sort_keys=True):
            raise ValueError(f"BLOCK {block_id:02d} 的尺寸、方向、索引或掩码不一致")
        return result


@dataclass(frozen=True)
class CollectionLayout:
    layout_id: str
    blocks: tuple[CollectionBlock, ...]
    placement_mode: str = "grid"
    view_positions: tuple[tuple[int, float, float], ...] = ()

    def __post_init__(self):
        if not re.fullmatch(r"skin_layout_v[1-9][0-9]*", self.layout_id):
            raise ValueError("采集布局版本编号无效")
        ids = [block.spec.block_id for block in self.blocks]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("采集布局不能为空或包含重复编号")
        if self.placement_mode not in ("grid", "body"):
            raise ValueError("采集布局的显示方式无效")
        position_ids = []
        for key, x, y in self.view_positions:
            if type(key) is not int or key not in ids:
                raise ValueError("布局位置必须对应已保存的皮肤块")
            if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1000 for v in (x, y)):
                raise ValueError("布局位置必须为 -1000～1000 的有限坐标")
            position_ids.append(key)
        if len(position_ids) != len(set(position_ids)):
            raise ValueError("布局位置包含重复的皮肤块")

    @property
    def by_id(self):
        return {block.spec.block_id: block for block in self.blocks}

    @property
    def positions(self):
        """ViewBox grid coordinates, not physical sensor coordinates or ADC indices."""
        return {key: (x, y) for key, x, y in self.view_positions}

    def with_masks(self, masks):
        for block in self.blocks:
            masks = masks.with_mask(block.spec.block_id, np.asarray(block.raw_valid_mask))
        return masks

    def to_dict(self):
        return {
            "schema_version": "1.1", "layout_id": self.layout_id,
            "coordinates": "body_oriented_row_col", "flatten_order": "row_major",
            "placement_mode": self.placement_mode,
            "view_positions": {str(key): {"x": x, "y": y} for key, x, y in self.view_positions},
            "blocks": {str(block.spec.block_id): block.to_dict() for block in self.blocks},
        }

    @classmethod
    def from_dict(cls, data):
        try:
            if (data["schema_version"] not in ("1.0", "1.1")
                    or data["coordinates"] != "body_oriented_row_col"
                    or data["flatten_order"] != "row_major"
                    or not isinstance(data["blocks"], dict)):
                raise ValueError("不支持的采集布局格式")
            blocks = []
            for key, value in data["blocks"].items():
                block_id = int(key)
                if str(block_id) != key:
                    raise ValueError("皮肤块编号格式无效")
                blocks.append(CollectionBlock.from_dict(block_id, value))
            raw_positions = data["view_positions"] if data["schema_version"] == "1.1" else {}
            positions = []
            for key, position in raw_positions.items():
                block_id = int(key)
                if str(block_id) != key:
                    raise ValueError("布局位置的皮肤块编号格式无效")
                positions.append((block_id, position["x"], position["y"]))
            return cls(data["layout_id"], tuple(sorted(blocks, key=lambda b: b.spec.block_id)),
                       data["placement_mode"], tuple(sorted(positions)))
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            raise ValueError(f"采集布局无效：{error}") from error


class CollectionLayoutStore:
    def __init__(self, directory):
        self.directory = Path(directory)

    def load(self):
        pointer = self.directory / "active.json"
        if not pointer.exists():
            return None
        try:
            active = json.loads(pointer.read_text(encoding="utf-8"))
            layout_id = active["layout_id"]
            if not isinstance(layout_id, str) or not re.fullmatch(r"skin_layout_v[1-9][0-9]*", layout_id):
                raise ValueError("活动布局编号无效")
            data = json.loads((self.directory / f"{layout_id}.json").read_text(encoding="utf-8"))
            layout = CollectionLayout.from_dict(data)
            if layout.layout_id != layout_id:
                raise ValueError("活动布局编号与文件内容不一致")
            return layout
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"采集布局加载失败：{error}") from error

    def save(self, specs, masks: SensorMasks, *, previous=None, placement_mode="grid"):
        specs = tuple(specs)
        if not specs:
            raise ValueError("请先连接并识别皮肤块，再保存采集布局")
        if len({s.block_id for s in specs}) != len(specs):
            raise ValueError("皮肤块编号重复")
        blocks = previous.by_id.copy() if previous is not None else {}
        positions = previous.positions.copy() if previous is not None else {}
        for spec in specs:
            if masks.mismatched(spec.block_id, (spec.rows, spec.cols)):
                raise ValueError(f"BLOCK {spec.block_id:02d} 的点位尺寸不符，请先编辑有效点位")
            # Persist display placement separately; it is not part of the ADC permutation.
            positions[spec.block_id] = (float(spec.x), float(spec.y))
            canonical = BlockLayout(spec.block_id, spec.rows, spec.cols, rotation=spec.rotation,
                                    flip_x=spec.flip_x, flip_y=spec.flip_y,
                                    row_map=spec.row_map, col_map=spec.col_map)
            blocks[spec.block_id] = CollectionBlock(canonical, masks.get(spec.block_id, (spec.rows, spec.cols)))
        self.directory.mkdir(parents=True, exist_ok=True)
        revisions = [int(m.group(1)) for p in self.directory.glob("skin_layout_v*.json")
                     if (m := re.fullmatch(r"skin_layout_v([1-9][0-9]*)\.json", p.name))]
        revision = max(revisions, default=0) + 1
        layout = CollectionLayout(f"skin_layout_v{revision}",
                                  tuple(blocks[key] for key in sorted(blocks)), placement_mode,
                                  tuple((key, *positions[key]) for key in sorted(positions)))
        payload = layout.to_dict()
        payload["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        target = self.directory / f"{layout.layout_id}.json"
        # Exclusive creation preserves old versions, even if another process saves concurrently.
        with target.open("x", encoding="utf-8") as stream:
            try:
                json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                stream.close()
                target.unlink(missing_ok=True)
                raise
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory,
                                             prefix="active.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump({"layout_id": layout.layout_id}, stream)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.directory / "active.json")
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return layout
