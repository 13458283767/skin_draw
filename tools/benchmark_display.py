"""Bounded offscreen display benchmark; never opens a hardware serial port."""

import argparse
import json
import logging
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from skin_draw.live import LiveBuffer
from skin_draw.models import Block
from skin_draw.ui.main_window import DISPLAY_INTERVAL_MS, MainWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    app = QApplication([])
    window = MainWindow()
    window.show()
    window.timer.stop()
    window.display_timer.stop()
    window.health_timer.stop()
    app.processEvents()
    buffer = LiveBuffer(calib_n=30)
    for _ in range(30):
        for key in range(1, 15):
            buffer.receive(Block(key, np.full((13, 5), 2500)))
    snapshot = buffer.take()
    window._consume_scan(snapshot.scan, snapshot)
    app.processEvents()
    samples = []
    for index in range(1050):
        block = Block(index % 14 + 1, np.full((13, 5), 2000 + index % 400))
        a = perf_counter()
        buffer.receive(block)
        b = perf_counter()
        snapshot = buffer.take()
        c = perf_counter()
        window._consume_scan(snapshot.scan, snapshot)
        d = perf_counter()
        app.processEvents()
        e = perf_counter()
        if index >= 50:
            samples.append([(b-a)*1000, (c-b)*1000, (d-c)*1000, (e-d)*1000, (e-a)*1000])
    arr = np.asarray(samples)
    report = {
        "conditions": "offscreen simulation; 14 blocks x 65 cells; one new block per update; 1000 samples",
        "limitations": "excludes hardware sampling, serial transport, display timer wait and physical screen presentation",
        "minimum_update_interval_ms": DISPLAY_INTERVAL_MS,
        "milliseconds": {
            name: {"median": round(float(np.median(arr[:, i])), 3),
                   "p95": round(float(np.percentile(arr[:, i], 95)), 3)}
            for i, name in enumerate(("ingest", "snapshot", "gui_update", "event_processing", "total"))
        },
    }
    window.close()
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
