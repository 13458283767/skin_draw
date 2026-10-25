"""受试者/场次管理：关联文件、索引、回收恢复和界面选择必须一起变化。"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from skin_draw.dataset import DatasetStore, atomic_write_json, read_json
from skin_draw.ui.main_window import MainWindow


def make_sample(store, subject, session):
    folder = store.root / subject / session / "pat" / "pat001"
    folder.mkdir(parents=True)
    (folder / "raw.csv").write_bytes(b"frame,adc\n1,2048\n")
    atomic_write_json(folder / "meta.json", {
        "subject_id": subject, "session_id": session, "trial_id": "pat001",
        "action": {"id": 0}, "labels_id": "actions_v1", "duration_s": 1, "accepted": True,
    })
    atomic_write_json(folder.parents[1] / "session.json", {"labels_id": "actions_v1"})
    return folder


@pytest.mark.parametrize("session_only", [False, True])
def test_archive_restore_preserves_bytes_other_ids_and_index(tmp_path, session_only):
    store = DatasetStore(tmp_path / "dataset")
    sample = make_sample(store, "student02", "session_01")
    make_sample(store, "student03", "session_02")
    store.rebuild_index()
    original = {path.relative_to(sample): path.read_bytes() for path in sample.iterdir()}
    archive = store.archive_identity("student02", "session_01" if session_only else None)
    assert not sample.exists()
    assert "student02" not in store.index_path.read_text()
    assert "student03" in store.index_path.read_text()
    assert store.list_subjects() == (["student02", "student03"] if session_only else ["student03"])
    if session_only:
        assert store.list_sessions("student02") == []
        assert store.next_identity("session_", "student02") == "session_02"
    assert store.next_identity("student") == "student04"
    assert store.archived_identities()[0][0] == archive.name
    store.restore_identity(archive.name)
    assert store.archived_identities() == []
    assert {path.relative_to(sample): path.read_bytes() for path in sample.iterdir()} == original
    assert "student02" in store.index_path.read_text()
    assert read_json(sample.parents[1] / "session.json")["labels_id"] == "actions_v1"


def test_failed_index_rebuild_rolls_back_archive_and_restore(tmp_path, monkeypatch):
    store = DatasetStore(tmp_path)
    sample = make_sample(store, "student01", "session_01")
    store.rebuild_index()
    original_index = store.index_path.read_bytes()

    def fail():
        raise OSError("disk write failed")

    with monkeypatch.context() as patch:
        patch.setattr(store, "rebuild_index", fail)
        with pytest.raises(OSError):
            store.archive_identity("student01")
    assert sample.exists()
    assert store.index_path.read_bytes() == original_index
    assert not store.archived_identities()
    archive = store.archive_identity("student01")
    archived_index = store.index_path.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(store, "rebuild_index", fail)
        with pytest.raises(OSError):
            store.restore_identity(archive.name)
    assert not sample.exists()
    assert (archive / "data").exists()
    assert store.index_path.read_bytes() == archived_index


def test_create_rejects_duplicates_invalid_paths_and_restore_collisions(tmp_path):
    store = DatasetStore(tmp_path)
    store.create_subject("student01")
    store.create_session("student01", "session_01")
    assert not store.session_path("student01", "session_01").exists()
    with pytest.raises(FileExistsError):
        store.create_subject("student01")
    for invalid in ("../outside", "_trash", "", "C:\\outside"):
        with pytest.raises(ValueError):
            store.archive_identity(invalid)
    archive = store.archive_identity("student01", "session_01")
    store.create_session("student01", "session_01")
    with pytest.raises(ValueError, match="同名"):
        store.restore_identity(archive.name)
    assert (archive / "data").is_dir()
    assert store.next_identity("session_", "student01") == "session_02"
    # 回收清单也是外部文件，不能信任其中的路径字段。
    info = read_json(archive / "deleted.json")
    info["original_path"] = "../outside"
    atomic_write_json(archive / "deleted.json", info)
    with pytest.raises(ValueError):
        store.restore_identity(archive.name)


def test_linked_subject_cannot_be_moved(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    store.root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (store.root / "student01").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("This Windows account cannot create symbolic links")
    assert store.list_subjects() == []
    with pytest.raises(ValueError):
        store.archive_identity("student01")
    assert outside.exists()


def test_void_sample_removes_index_preserves_raw_and_number(tmp_path):
    store = DatasetStore(tmp_path)
    sample = make_sample(store, "student01", "session_01")
    raw = (sample / "raw.csv").read_bytes()
    store.rebuild_index()
    archive = store.archive_sample(sample / "meta.json")
    assert not sample.exists()
    assert "pat001" not in store.index_path.read_text()
    assert (archive / "data/raw.csv").read_bytes() == raw
    assert read_json(archive / "data/meta.json")["accepted"] is False
    action = store.load_labels().classes[0]
    assert store.next_trial_number("student01", "session_01", action) == 2
    assert store.restore_identity(archive.name) == ("student01", "session_01", "pat", "pat001")
    assert (sample / "raw.csv").read_bytes() == raw
    # 恢复不会偷偷将人工判废样本改为有效。
    assert read_json(sample / "meta.json")["accepted"] is False


def test_void_sample_index_failure_restores_original_metadata(tmp_path, monkeypatch):
    store = DatasetStore(tmp_path)
    sample = make_sample(store, "student01", "session_01")
    store.rebuild_index()
    original = (sample / "meta.json").read_bytes()
    index = store.index_path.read_bytes()

    def fail():
        raise OSError("disk write failed")

    monkeypatch.setattr(store, "rebuild_index", fail)
    with pytest.raises(OSError):
        store.archive_sample(sample / "meta.json")
    assert (sample / "meta.json").read_bytes() == original
    assert store.index_path.read_bytes() == index
    assert not store.archived_identities()


def test_ui_void_moves_sample_and_restore_button_recovers_it(app, tmp_path, monkeypatch):
    store = DatasetStore(tmp_path)
    sample = make_sample(store, "student01", "session_01")
    window = MainWindow(dataset_dir=tmp_path)
    try:
        window._refresh_identity_choices("student01", "session_01")
        window.last_sample = SimpleNamespace(sample_dir=sample, meta_path=sample / "meta.json")
        window.collection_bar.set_sample_state("未判定", can_accept=True, can_void=True)
        window.collection_bar.mark_void.click()
        assert not sample.exists()
        assert "pat001" not in store.index_path.read_text()
        assert window.last_sample is None
        assert not window.collection_bar.mark_valid.isEnabled()
        assert not window.collection_bar.mark_void.isEnabled()
        assert window.collection_bar.trial.text() == "pat002"
        monkeypatch.setattr("skin_draw.ui.main_window.QInputDialog.getItem",
                            lambda parent, title, prompt, choices, *args: (choices[0], True))
        window.collection_bar.restore_identity.click()
        assert sample.exists()
        assert window.collection_bar.subject.currentText() == "student01"
        assert window.collection_bar.session.currentText() == "session_01"
    finally:
        window.close()


def test_ui_add_switch_delete_and_restore(app, tmp_path, monkeypatch):
    window = MainWindow(dataset_dir=tmp_path)
    bar = window.collection_bar
    try:
        monkeypatch.setattr("skin_draw.ui.main_window.QInputDialog.getText",
                            lambda *args, **kwargs: ("student01", True))
        bar.add_subject.click()
        assert bar.subject.currentText() == "student01"
        assert bar.add_session.isEnabled()
        monkeypatch.setattr("skin_draw.ui.main_window.QInputDialog.getText",
                            lambda *args, **kwargs: ("session_01", True))
        bar.add_session.click()
        assert bar.session.currentText() == "session_01"
        window.dataset_store.create_subject("student03")
        window.dataset_store.create_session("student03", "session_03")
        window._refresh_identity_choices("student01", "session_01")
        bar.subject.setCurrentText("student03")
        assert bar.session.currentText() == "session_03"
        bar.subject.setCurrentText("student01")
        assert bar.session.currentText() == "session_01"
        bar.delete_subject.click()
        assert bar.subject.count() == 1
        assert bar.subject.currentText() == "student03"
        assert bar.session.currentText() == "session_03"
        monkeypatch.setattr("skin_draw.ui.main_window.QInputDialog.getItem",
                            lambda parent, title, prompt, choices, *args: (choices[0], True))
        bar.restore_identity.click()
        assert bar.subject.count() == 2
        assert bar.subject.currentText() == "student01"
        assert bar.session.currentText() == "session_01"
        bar.delete_session.click()
        assert bar.session.count() == 0 and bar.session.currentText() == ""
        assert not bar.delete_session.isEnabled()
        assert not bar.start.isEnabled()
        bar.delete_subject.click()
        bar.delete_subject.click()
        assert bar.subject.count() == 0 and bar.subject.currentText() == ""
        assert not bar.delete_subject.isEnabled()
        assert not bar.add_session.isEnabled()
    finally:
        window.close()


def test_ui_deletion_clears_last_sample_and_blocks_during_recording(app, tmp_path, monkeypatch):
    store = DatasetStore(tmp_path)
    sample = make_sample(store, "student01", "session_01")
    window = MainWindow(dataset_dir=tmp_path)
    try:
        window._refresh_identity_choices("student01", "session_01")
        window.last_sample = SimpleNamespace(sample_dir=sample)
        window.collection_bar.set_sample_state("有效", can_accept=True, can_void=True)
        with monkeypatch.context() as patch:
            patch.setattr(window, "_recording_active", lambda: True)
            with pytest.raises(ValueError, match="正在采集"):
                window.delete_selected_identity()
        assert sample.exists()
        window.delete_selected_identity(True)
        assert window.last_sample is None
        assert not window.collection_bar.mark_valid.isEnabled()
        assert not window.collection_bar.mark_void.isEnabled()
    finally:
        window.close()
