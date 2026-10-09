import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from skin_draw.ui.theme import apply_theme


@pytest.fixture(autouse=True)
def isolated_sensor_config(tmp_path, monkeypatch):
    # Tests never load or overwrite the operator's saved sensor mapping.
    monkeypatch.setattr("skin_draw.ui.main_window.default_mask_path", lambda: tmp_path / "sensor_masks.json")


@pytest.fixture(scope="session")
def app():
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    yield app

