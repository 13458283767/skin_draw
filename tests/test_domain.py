import json

import numpy as np
import pytest

from skin_draw.colors import ColorScale, lookup_table
from skin_draw.config import four_blocks, load_layout, single_block
from skin_draw.mapping import BlockLayout, validate_layout
from skin_draw.models import Block, Frame
from skin_draw.simulator import PATTERNS, generate_frame


@pytest.mark.parametrize("values", [ [[-1]], [[4096]], [[float('nan')]], [[float('inf')]],
                                     [[1.2]], [["1"]], [1, 2], [[]] ])
def test_invalid_adc_is_rejected_before_cast(values):
    with pytest.raises(ValueError):
        Block(1, values)


def test_frame_owns_data_and_rejects_duplicate_ids():
    source = np.array([[0, 4095]], dtype=np.uint16)
    block = Block(1, source)
    source[0, 0] = 888
    assert block.values.tolist() == [[0, 4095]]
    with pytest.raises(ValueError):
        block.values[0, 0] = 1
    with pytest.raises(ValueError):
        Frame((block, block))


def test_mapping_known_asymmetric_example():
    raw = np.array([[10, 20, 30], [40, 50, 60]])
    spec = BlockLayout(1, 2, 3, rotation=90, flip_x=True)
    assert spec.transform(raw).tolist() == [[60, 30], [50, 20], [40, 10]]
    assert spec.raw_index(0, 0) == (1, 2)
    assert spec.raw_index(2, 1) == (0, 0)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("flip_x,flip_y", [(False, False), (True, False), (False, True), (True, True)])
def test_index_inverse_after_reorder_rotation_and_flip(rotation, flip_x, flip_y):
    raw = np.arange(6).reshape(2, 3) * 137
    spec = BlockLayout(1, 2, 3, row_map=(1, 0), col_map=(2, 0, 1),
                       rotation=rotation, flip_x=flip_x, flip_y=flip_y)
    displayed = spec.transform(raw)
    seen = set()
    for row, col in np.ndindex(displayed.shape):
        original = spec.raw_index(row, col)
        seen.add(original)
        assert raw[original] == displayed[row, col]
    assert len(seen) == 6


def test_layout_validation_and_json(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text(json.dumps({"version": 1, "blocks": [{"block_id": 1, "rows": 2, "cols": 3, "rotation": 90}]}))
    assert load_layout(path)[0].display_shape == (3, 2)
    with pytest.raises(ValueError, match="重叠"):
        validate_layout((BlockLayout(1, 2, 3), BlockLayout(2, 2, 3, x=2)))
    with pytest.raises(ValueError):
        BlockLayout(1, 2, 3, row_map=(0, 0))
    with pytest.raises(ValueError):
        BlockLayout(1, 2, 3, x=float("nan"))


def test_fixed_color_scale_preserves_data_and_clips_colors_only():
    values = np.array([[0, 100, 200, 4095]], dtype=np.uint16)
    scale = ColorScale(100, 200)
    colors = scale.map_values(values)
    lut = lookup_table(scale.palette)
    np.testing.assert_array_equal(colors[0, :2], [lut[0], lut[0]])
    np.testing.assert_array_equal(colors[0, 2:], [lut[-1], lut[-1]])
    assert values.tolist() == [[0, 100, 200, 4095]]
    with pytest.raises(ValueError):
        ColorScale(20, 20)


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("layout", [single_block(1, 1), single_block(16, 16), four_blocks()])
def test_demo_contract_for_all_sizes(pattern, layout):
    frame = generate_frame(layout, 23, pattern)
    assert len(frame.blocks) == len(layout)
    for spec, block in zip(layout, frame.blocks):
        assert block.values.shape == (spec.rows, spec.cols)
        assert block.values.min() >= 0 and block.values.max() <= 4095

