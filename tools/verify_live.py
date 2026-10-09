"""Exercise the real worker -> M3 filter + slow baseline -> GUI path offscreen.

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
        buffer = LiveBuffer(calib_n=30)
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
        state = window.preprocess
        print(json.dumps({"synthetic": True, "blocks": 14, "sensors": 3584,
                          "mean_ms": round(float(np.mean(timings)), 2),
                          "p95_ms": round(float(np.percentile(timings, 95)), 2),
                          "preprocess_ready": state.ready if state else None,
                          "preprocess_samples": dict(state.samples) if state else None}), flush=True)
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
        state = window.preprocess
        count = min(state.samples.values(), default=0) if state is not None else 0
        if count != last_progress and (count == 1 or count % 30 == 0):
            last_progress = count
            print(json.dumps({"elapsed_seconds": round(monotonic() - started, 3),
                              "samples": dict(state.samples) if state else {},
                              "rates": snapshot.rates if snapshot else {},
                              "ready": bool(state is not None and state.ready),
                              "paint_ms": round(rendered[-1], 2)}), flush=True)
        if state is not None and state.ready and snapshot:
            print(json.dumps({"verified": True, "received_blocks": snapshot.received_blocks,
                              "mean_paint_ms": round(float(np.mean(rendered[1:])), 2),
                              "response_blocks": len(window.display_values)}), flush=True)
            window.close()

    window._consume_scan = consume
    window.controls.connect_button.click()
    QTimer.singleShot(round(args.timeout * 1000), window.close)
    app.exec()
    if window.preprocess is None or not window.preprocess.ready:
        raise SystemExit("Baseline did not complete before the bounded verification ended")


if __name__ == "__main__":
    main()
