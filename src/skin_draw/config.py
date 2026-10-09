"""Small JSON layout format and demo presets."""

import json
from pathlib import Path

from .mapping import BlockLayout, validate_layout


def single_block(rows: int = 8, cols: int = 8) -> tuple[BlockLayout, ...]:
    return (BlockLayout(1, rows, cols),)


def four_blocks() -> tuple[BlockLayout, ...]:
    return (BlockLayout(1, 5, 3), BlockLayout(2, 5, 3, x=5),
            BlockLayout(3, 7, 3, y=7), BlockLayout(4, 7, 3, x=5, y=7))


def load_layout(path: str | Path) -> tuple[BlockLayout, ...]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("布局文件需要 version: 1")
        blocks = data.get("blocks")
        if not isinstance(blocks, list):
            raise ValueError("布局文件需要 blocks 数组")
        return validate_layout(tuple(BlockLayout(**entry) for entry in blocks))
    except (TypeError, KeyError) as error:
        raise ValueError(f"布局字段不正确：{error}") from error

