import json
from types import SimpleNamespace

import numpy as np
import pytest

from skin_draw.collection_layout import CollectionBlock, CollectionLayoutStore
from skin_draw.mapping import BlockLayout, BlockOrientation
from skin_draw.live import LiveBuffer
from skin_draw.models import Block, Frame
from skin_draw.sensor_masks import SensorMasks
from skin_draw.ui.main_window import MainWindow


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("flip_x,flip_y", [(False, False), (True, False), (False, True), (True, True)])
def test_every_orientation_roundtrips_all_adc_and_mask(rotation, flip_x, flip_y):
    raw = np.array([[0, 2001, 2102], [2203, 2304, 4095]], dtype=np.uint16)
    mask = np.array([[False, True, True], [True, True, False]])
    block = CollectionBlock(BlockLayout(7, 2, 3, rotation=rotation, flip_x=flip_x, flip_y=flip_y), mask)
    expected = np.rot90(raw, rotation // 90)
    expected_mask = np.rot90(mask, rotation // 90)
    if flip_x:
        expected, expected_mask = np.fliplr(expected), np.fliplr(expected_mask)
    if flip_y:
        expected, expected_mask = np.flipud(expected), np.flipud(expected_mask)
    record = block.to_dict()
    np.testing.assert_array_equal(block.transform(raw), expected)
    np.testing.assert_array_equal(record["valid_mask"], expected_mask)
    np.testing.assert_array_equal(block.restore_raw(expected), raw)
    assert block.transform(raw).dtype == np.uint16
    assert sorted(np.array(record["saved_to_raw_index"]).ravel()) == list(range(6))
    assert (record["rows"], record["cols"]) == expected.shape
    assert raw[0, 0] == 0 and raw[1, 2] == 4095
    np.testing.assert_array_equal(CollectionBlock.from_dict(7, record).transform(raw), expected)


def test_versions_merge_absent_blocks_and_preserve_old_file(tmp_path):
    store = CollectionLayoutStore(tmp_path)
    first = store.save([BlockLayout(7, 2, 3, x=8.5, y=9, flip_y=True), BlockLayout(14, 3, 2, x=25, y=18)],
                       SensorMasks(), placement_mode="body")
    before = (tmp_path / "skin_layout_v1.json").read_bytes()
    second = store.save([BlockLayout(7, 2, 3, rotation=90)], SensorMasks(),
                        previous=first, placement_mode="body")
    assert second.layout_id == "skin_layout_v2"
    assert set(second.by_id) == {7, 14}
    assert second.positions[14] == (25, 18)
    assert (tmp_path / "skin_layout_v1.json").read_bytes() == before
    assert store.load() == second
    assert store.load().by_id[7].spec.display_shape == (3, 2)


def test_failed_activation_preserves_previous_version(tmp_path, monkeypatch):
    store = CollectionLayoutStore(tmp_path)
    old = store.save([BlockLayout(7, 2, 3)], SensorMasks())

    def fail(*args):
        raise PermissionError("read only")

    monkeypatch.setattr("skin_draw.collection_layout.os.replace", fail)
    with pytest.raises(PermissionError):
        store.save([BlockLayout(7, 2, 3, flip_y=True)], SensorMasks(), previous=old)
    assert store.load() == old
    assert not (tmp_path / "skin_layout_v2.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("field,value", [
    ("saved_to_raw_index", [[0, 0, 2], [3, 4, 5]]),
    ("valid_mask", [[True, False, True], [True, True, True]]),
    ("rows", 3),
])
def test_corrupted_derived_fields_are_not_silently_rebuilt(tmp_path, field, value):
    store = CollectionLayoutStore(tmp_path)
    store.save([BlockLayout(7, 2, 3)], SensorMasks())
    path = tmp_path / "skin_layout_v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["blocks"]["7"][field] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="不一致"):
        store.load()


def test_incompatible_masks_and_non_collection_ids_rejected(tmp_path):
    store = CollectionLayoutStore(tmp_path)
    with pytest.raises(ValueError, match="点位尺寸"):
        store.save([BlockLayout(7, 2, 3)], SensorMasks().with_mask(7, np.ones((3, 2), bool)))
    with pytest.raises(ValueError, match="1～14"):
        store.save([BlockLayout(15, 2, 3)], SensorMasks())
    assert not (tmp_path / "active.json").exists()


def test_ui_save_restart_restore_and_new_version(app, tmp_path):
    path = tmp_path / "sensor_masks.json"
    raw = np.array([[0, 2200, 2300], [2400, 2500, 4095]], dtype=np.uint16)
    frame = Frame((Block(7, raw),))
    window = MainWindow(mask_path=path)
    try:
        window.present_frame(frame)
        window.apply_sensor_mask(7, np.array([[False, True, True], [True, True, False]]))
        window.controls.body_positions.click()
        window.controls.rotate_left.click()
        window.controls.mirror_horizontal.click()
        expected = np.fliplr(np.rot90(raw))
        np.testing.assert_array_equal(window.heatmap.items[7].values, expected)
        window.controls.save_collection_layout.click()
        first = window.collection_layout
        assert first is not None
        assert window.collection_locked
        assert not window.controls.rotate_left.isEnabled()
        assert not window.controls.edit_sensor_mask.isEnabled()
        assert window.controls.orientation_block.isEnabled()  # Inspection still works.
        assert "1/14" in window.controls.collection_layout_status.text()
        with pytest.raises(ValueError, match="已固定"):
            window.apply_sensor_mask(7, np.ones((2, 3), bool))
        # Direct UI action handler cannot change a locked orientation either.
        window._orient_block("reset")
        np.testing.assert_array_equal(window.heatmap.items[7].values, expected)
        window.present_frame(Frame(()))
        window.present_frame(frame)
        np.testing.assert_array_equal(window.heatmap.items[7].values, expected)
        assert window.current_frame is frame
        np.testing.assert_array_equal(frame.blocks[0].values, raw)
    finally:
        window.close()
    restored = MainWindow(mask_path=path)
    try:
        restored.present_frame(frame)
        assert restored.collection_layout == first
        assert restored.heatmap.placement_mode == "body"
        np.testing.assert_array_equal(restored.heatmap.items[7].values, expected)
        np.testing.assert_array_equal(restored.heatmap.items[7].valid_mask, first.by_id[7].to_dict()["valid_mask"])
        restored.controls.edit_collection_layout.click()
        assert restored.controls.rotate_left.isEnabled()
        restored.controls.reset_orientation.click()
        restored.apply_sensor_mask(7, np.ones((2, 3), bool))
        restored.controls.restore_collection_layout.click()
        assert restored.collection_locked
        np.testing.assert_array_equal(restored.heatmap.items[7].values, expected)
        np.testing.assert_array_equal(restored.heatmap.items[7].valid_mask, first.by_id[7].to_dict()["valid_mask"])
        restored.controls.edit_collection_layout.click()
        restored.controls.reset_orientation.click()
        restored.controls.save_collection_layout.click()
        assert restored.collection_layout.layout_id == "skin_layout_v2"
        assert (tmp_path / "collection_layouts/skin_layout_v1.json").exists()
        assert restored.collection_layout.by_id[7].orientation == BlockOrientation()
    finally:
        restored.close()


def test_ui_shape_change_warns_without_applying_unverified_orientation(app, tmp_path):
    window = MainWindow(mask_path=tmp_path / "masks.json")
    try:
        window.present_frame(Frame((Block(7, [[1, 2, 3], [4, 5, 6]]),)))
        window.controls.rotate_left.click()
        saved = window.save_collection_layout()
        changed = np.array([[10, 20], [30, 40], [50, 60]])
        window.present_frame(Frame((Block(7, changed), Block(14, [[70, 80]]))))
        text = window.controls.collection_layout_status.text()
        assert "07 尺寸不符" in text and "14 未固定" in text
        np.testing.assert_array_equal(window.heatmap.items[7].values, changed)
        assert window.collection_layout is saved
        with pytest.raises(ValueError):
            saved.by_id[7].transform(changed)
    finally:
        window.close()


def test_demo_cannot_save_and_bad_active_file_is_preserved(app, tmp_path):
    window = MainWindow(mask_path=tmp_path / "masks.json", demo=True)
    try:
        assert not window.controls.save_collection_layout.isEnabled()
        with pytest.raises(ValueError, match="模拟"):
            window.save_collection_layout()
        assert not window.collection_store.directory.exists()
    finally:
        window.close()
    folder = tmp_path / "collection_layouts"
    folder.mkdir()
    (folder / "active.json").write_text('{"layout_id": "../../bad"}')
    broken = MainWindow(mask_path=tmp_path / "masks.json")
    try:
        broken.present_frame(Frame((Block(7, [[1, 2]]),)))
        assert broken.collection_load_error
        assert "加载失败" in broken.controls.collection_layout_status.text()
        assert not broken.controls.save_collection_layout.isEnabled()
        with pytest.raises(ValueError, match="配置加载失败"):
            broken.save_collection_layout()
        assert (folder / "active.json").read_text() == '{"layout_id": "../../bad"}'
    finally:
        broken.close()


def test_restore_only_recalibrates_blocks_with_changed_masks(app, tmp_path, monkeypatch):
    window = MainWindow(mask_path=tmp_path / "masks.json")
    buffer = LiveBuffer(calib_n=30, norm="A")
    frame = Frame((Block(7, [[2500, 2500]]), Block(14, [[2400, 2400]])))
    try:
        window.present_frame(frame)
        window.save_collection_layout()
        for _ in range(59):
            for block in frame.blocks:
                buffer.receive(block)
        old_processors = buffer.preprocessors.copy()
        assert buffer.take().preprocess.ready
        window.worker = SimpleNamespace(buffer=buffer)
        monkeypatch.setattr(window, "_request_live_update", lambda: None)
        window.controls.edit_collection_layout.click()
        window.controls.rotate_left.click()
        window.controls.restore_collection_layout.click()
        assert buffer.preprocessors == old_processors
        window.controls.edit_collection_layout.click()
        window.apply_sensor_mask(7, np.array([[False, True]]))
        window.controls.restore_collection_layout.click()
        assert buffer.preprocessors[14] is old_processors[14]
        assert buffer.sensor_masks.get(7, (1, 2)).all()
        assert buffer.take().preprocess.samples[7] == 0
    finally:
        window.worker = None
        window.close()


def test_local_positions_survive_restart_partial_arrivals_and_explicit_reload(app, tmp_path):
    path = tmp_path / "masks.json"
    frame = Frame((Block(7, [[1, 2, 3], [4, 5, 6]]), Block(8, [[7, 8], [9, 10]])))
    window = MainWindow(mask_path=path)
    try:
        window.present_frame(frame)
        window.controls.body_positions.click()
        window.controls.rotate_left.click()
        window.heatmap.move_block(7, 4.25, 12.5)
        window.heatmap.move_block(8, 21.5, 19.25)
        positions = {s.block_id: (s.x, s.y) for s in window.heatmap.layout_specs}
        saved = window.save_collection_layout()
        assert saved.positions == positions
    finally:
        window.close()
    restarted = MainWindow(mask_path=path)
    try:
        restarted.resize(1000, 720)
        app.processEvents()
        # Offline startup shows saved geometry, never fake ADC or sensor items.
        assert restarted.current_frame.blocks == ()
        assert not restarted.heatmap.items
        slots = {s.block_id: s for s in restarted.heatmap.arrangement.slots}
        assert (slots[7].rows, slots[7].cols) == (2, 3)
        assert slots[7].display_shape == (3, 2)
        assert (slots[7].x, slots[7].y) == positions[7]
        assert restarted.controls.restore_collection_layout.isEnabled()
        restarted.present_frame(Frame((frame.blocks[1],)))
        assert restarted.heatmap.items[8].pos().toTuple() == positions[8]
        restarted.present_frame(frame)
        assert {s.block_id: (s.x, s.y) for s in restarted.heatmap.layout_specs} == positions
        np.testing.assert_array_equal(restarted.heatmap.items[7].values, np.rot90(frame.blocks[0].values))
        # Even while directions are locked, moving blocks can be saved or undone.
        restarted.heatmap.move_block(7, positions[7][0] + 1, positions[7][1] + 1)
        restarted.heatmap.layout_changed.emit()
        assert restarted.controls.save_collection_layout.isEnabled()
        assert "位置待保存" in restarted.controls.collection_layout_status.text()
        restarted.controls.restore_collection_layout.click()
        assert restarted.heatmap.items[7].pos().toTuple() == positions[7]
        restarted.controls.reset_positions.click()
        assert restarted.heatmap.placement_mode == "grid"
        restarted.controls.restore_collection_layout.click()
        assert restarted.heatmap.placement_mode == "body"
        assert {s.block_id: (s.x, s.y) for s in restarted.heatmap.layout_specs} == positions
        restarted.heatmap.move_block(8, positions[8][0] + .5, positions[8][1] + .5)
        restarted.heatmap.layout_changed.emit()
        moved = restarted.heatmap.items[8].pos().toTuple()
        restarted.controls.save_collection_layout.click()
        assert restarted.collection_layout.layout_id == "skin_layout_v2"
        assert restarted.collection_layout.positions[8] == moved
    finally:
        restarted.close()


def test_old_local_layout_without_positions_remains_readable(tmp_path):
    store = CollectionLayoutStore(tmp_path)
    saved = store.save([BlockLayout(7, 2, 3, rotation=90, flip_y=True)], SensorMasks(),
                       placement_mode="body")
    path = tmp_path / "skin_layout_v1.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    old["schema_version"] = "1.0"
    del old["view_positions"]
    path.write_text(json.dumps(old), encoding="utf-8")
    before = path.read_bytes()
    loaded = store.load()
    assert loaded.by_id == saved.by_id
    assert loaded.positions == {}
    assert path.read_bytes() == before  # No automatic rewriting of the operator's old file.


@pytest.mark.parametrize("positions", [
    {"7": {"x": float("nan"), "y": 1}},
    {"7": {"x": 1, "y": 1001}},
    {"7": {"x": True, "y": 1}},
    {"14": {"x": 1, "y": 1}},  # Block 14 does not exist in this saved layout.
    {"07": {"x": 1, "y": 1}},
    [],
])
def test_corrupted_view_positions_rejected(tmp_path, positions):
    store = CollectionLayoutStore(tmp_path)
    store.save([BlockLayout(7, 2, 3)], SensorMasks())
    path = tmp_path / "skin_layout_v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["view_positions"] = positions
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        store.load()


def test_call_local_layout_reloads_disk_instead_of_cached_snapshot(app, tmp_path):
    window = MainWindow(mask_path=tmp_path / "masks.json")
    try:
        window.present_frame(Frame((Block(7, [[1, 2, 3], [4, 5, 6]]),)))
        first = window.save_collection_layout()
        second = window.collection_store.save(
            [BlockLayout(7, 2, 3, x=9.5, y=12, flip_y=True)], SensorMasks(),
            previous=first, placement_mode="body")
        window.controls.restore_collection_layout.click()
        assert window.collection_layout == second
        assert window.heatmap.items[7].pos().toTuple() == (9.5, 12)
        np.testing.assert_array_equal(window.heatmap.items[7].values, [[4, 5, 6], [1, 2, 3]])
    finally:
        window.close()
