"""Double-click launcher. Prefer the project's isolated Python environment."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"


def run() -> None:
    local_python = VENV / "Scripts" / "pythonw.exe"
    if local_python.exists() and Path(sys.prefix).resolve() != VENV.resolve():
        subprocess.Popen([str(local_python), str(Path(__file__).resolve())], cwd=ROOT)
        return
    sys.path.insert(0, str(ROOT / "src"))
    missing = [name for name in ("PySide6", "pyqtgraph", "numpy", "serial") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("缺少依赖：" + ", ".join(missing) + "\n请先运行 install.ps1，详见 README.md。")
    from skin_draw.__main__ import main
    raise SystemExit(main())


if __name__ == "__main__":
    try:
        run()
    except Exception:
        details = traceback.format_exc()
        error_file = ROOT / "startup-error.log"
        error_file.write_text(details, encoding="utf-8")
        if os.name == "nt":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, f"启动失败，详情已写入：\n{error_file}\n\n{details[-1600:]}", "SkinDraw", 0x10)
        else:
            print(details, file=sys.stderr)
