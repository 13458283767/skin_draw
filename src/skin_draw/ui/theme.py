import os
from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    # Windows' offscreen Qt backend does not enumerate system fonts. Register
    # the existing OS fonts for deterministic QA; do not ship font binaries.
    if app.platformName() == "offscreen" and os.name == "nt":
        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for name in ("msyh.ttc", "segoeui.ttf"):
            path = fonts / name
            if path.exists():
                QFontDatabase.addApplicationFont(str(path))
    families = QFontDatabase.families()
    family = next((name for name in ("Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans CJK SC")
                   if name in families), app.font().family())
    app.setFont(QFont(family, 10))
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#f0f3f7"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#233348"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#233348"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#233348"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#147d83"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    app.setStyleSheet("""
        QMainWindow { background: #f0f3f7; }
        QLabel { background: transparent; }
        QLabel#brand { font-size: 25px; font-weight: 700; color: #183547; }
        QLabel#muted { color: #748397; font-size: 12px; }
        QLabel#section { font-size: 13px; font-weight: 600; color: #35465c; }
        QLabel#badge { color: #147d83; background: #e6f3f2; border-radius: 5px; padding: 5px 10px; }
        QLabel#value { font-size: 27px; font-weight: 600; color: #183547; }
        QFrame#card { background: white; border: 1px solid #dee5ed; border-radius: 9px; }
        QPushButton { background: #ffffff; border: 1px solid #d4dce5; border-radius: 5px;
                      padding: 7px 12px; color: #35465c; min-height: 19px; }
        QPushButton:hover { border-color: #147d83; background: #f1f8f8; }
        QPushButton:pressed { background: #e0eeed; }
        QPushButton#primary { background: #147d83; color: white; border-color: #147d83; }
        QPushButton#primary:hover { background: #116b70; }
        QPushButton:disabled { color: #a2adba; background: #f3f5f7; border-color: #e6eaf0; }
        QComboBox, QSpinBox { background: #f9fbfd; border: 1px solid #d8e0e9;
                             border-radius: 5px; padding: 5px 8px; min-height: 21px; }
        QComboBox:focus, QSpinBox:focus { border-color: #147d83; }
        QComboBox QAbstractItemView { background: white; selection-background-color: #e1f0ef;
                                     selection-color: #183547; }
        QCheckBox { spacing: 8px; }
        QStatusBar { color: #63748a; background: #e9eef4; font-size: 12px; }
        QScrollArea { background: transparent; border: none; }
        QToolTip { background: #233348; color: white; border: none; padding: 5px; }
    """)
