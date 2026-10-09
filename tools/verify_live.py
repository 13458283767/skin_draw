"""Exercise the real worker -> 30-sample baseline -> GUI path offscreen.

Use --port COM6 for bounded real reception, or omit it for a 14 x 16 x 16
synthetic rendering benchmark. Prints JSON; no firmware configuration is sent.
"""

import argparse
import json
import os
from time import monotonic, perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from skin_draw.live import LiveBuffer
from skin_draw.models import Block
from skin_draw.ui.main_window import MainWindow
from skin_draw.ui.theme import apply_theme


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port")
    parser.add_argument("--timeout", type=float, default=75)
    args = parser.parse_args()
    app = QApplication([])
    apply_theme(app)
    window = MainWindow()
    window.show()
    app.processEvents()
    if not args.port:
        buffer = LiveBuffer()
        timings = []
        for sequence in range(90):
            started = perf_counter()
            for key in range(1, 15):
                values = np.full((16, 16), 2500) if sequence < 30 else np.random.default_rng(sequence + key).integers(1500, 2500, (16, 16))
                buffer.receive(Block(key, values))
            snapshot = buffer.take()
            window._consume_scan(snapshot.scan, snapshot)
            window.grab()  # force a complete Qt paint, not just event scheduling
            if sequence >= 30:
                timings.append((perf_counter() - started) * 1000)
        print(json.dumps({"synthetic": True, "blocks": 14, "sensors": 3584,
                          "mean_ms": round(float(np.mean(timings)), 2),
                          "p95_ms": round(float(np.percentile(timings, 95)), 2),
                          "calibration_counts": window.calibration.counts}), flush=True)
        window.close()
        return
    window.controls.port.clear()
    window.controls.port.addItem(args.port, args.port)
    started = monotonic()
    last_progress = -1
    rendered = []
    original = window._consume_scan

    def consume(scan, snapshot=None):
        nonlocal last_progress
        paint_start = perf_counter()
        original(scan, snapshot)
        window.grab()
        rendered.append((perf_counter() - paint_start) * 1000)
        count = window.calibration.count
        if count != last_progress and (count == 1 or count % 5 == 0):
            last_progress = count
            print(json.dumps({"elapsed_seconds": round(monotonic() - started, 3),
                              "counts": window.calibration.counts,
                              "rates": snapshot.rates if snapshot else {},
                              "ready": window.calibration.ready,
                              "paint_ms": round(rendered[-1], 2)}), flush=True)
        if window.calibration.ready and snapshot and snapshot.received_blocks >= 31 * len(scan.expected_ids):
            print(json.dumps({"verified": True, "received_blocks": snapshot.received_blocks,
                              "mean_paint_ms": round(float(np.mean(rendered[1:])), 2),
                              "response_blocks": len(window.display_values)}), flush=True)
            window.close()

    window._consume_scan = consume
    window.controls.connect_button.click()
    QTimer.singleShot(round(args.timeout * 1000), window.close)
    app.exec()
    if not window.calibration.ready:
        raise SystemExit("Baseline did not complete before the bounded verification ended")


if __name__ == "__main__":
    main()
