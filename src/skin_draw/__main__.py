import sys
import logging
import platform
from importlib.metadata import version, PackageNotFoundError

from PySide6.QtWidgets import QApplication

from .ui.main_window import MainWindow
from .ui.theme import apply_theme
from .diagnostics import setup_logging, shutdown_logging
from .timer_resolution import enable_high_resolution_timer, restore_timer_resolution


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("SkinDraw")
    app.setOrganizationName("SkinDraw")
    logger = logging.getLogger(__name__)
    log_error = None
    try:
        path = setup_logging()
        logger.info("程序启动 Python=%s platform=%s log=%s", sys.version, platform.platform(), path)
        for package in ("skin-draw", "PySide6", "pyqtgraph", "numpy", "pyserial"):
            try:
                logger.info("依赖版本 %s=%s", package, version(package))
            except PackageNotFoundError:
                logger.info("依赖版本 %s=unknown", package)
    except OSError as error:
        log_error = str(error)
    # Qt's PreciseTimer cannot beat the Windows default ~15.6 ms system timer
    # granularity; request 1 ms before any timer starts.
    enable_high_resolution_timer()
    previous_hook = sys.excepthook

    def exception_hook(kind, value, traceback):
        logger.critical("未处理异常", exc_info=(kind, value, traceback))
        previous_hook(kind, value, traceback)

    sys.excepthook = exception_hook
    try:
        apply_theme(app)
        # main build: calibrate a stable baseline before displaying pressure.
        window = MainWindow(raw_adc=False)
        window.show()
        if log_error:
            window.statusBar().showMessage(log_error)
        return app.exec()
    except Exception:
        logger.exception("程序运行失败")
        raise
    finally:
        logger.info("程序退出")
        sys.excepthook = previous_hook
        restore_timer_resolution()
        shutdown_logging()


if __name__ == "__main__":
    raise SystemExit(main())
