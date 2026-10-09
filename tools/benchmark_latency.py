"""Measure the live display throttle: arrival -> committed-to-heatmap latency.

Offscreen and hardware-free. Blocks are pushed through the same Qt path the
serial worker uses (data_ready -> queued _request_live_update -> display_timer
-> _flush_live), and two things are recorded per arrival:

  throttle_wait_ms  time from scheduling a commit to _flush_live actually
                    running; this isolates the DISPLAY_INTERVAL_MS budget
  latency_ms        arrival -> the value being visible in the heatmap, which
                    adds the snapshot, draw and commit work on top

DISPLAY_INTERVAL_MS only delays a commit while data keeps arriving faster than
it, so arrivals are driven well under any tested interval to keep the budget
spent. Compare candidate values with --interval-ms:

    .venv/Scripts/python.exe tools/benchmark_latency.py --interval-ms 8
    .venv/Scripts/python.exe tools/benchmark_latency.py --interval-ms 4
"""

import argparse
import json
import logging
import os
from time import monotonic, perf_counter, sleep

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from skin_draw.models import Block
from skin_draw.timer_resolution import enable_high_resolution_timer, restore_timer_resolution
from skin_draw.ui import main_window as mw
from skin_draw.ui.serial_worker import SerialWorker

BLOCK_ID = 1
SIDE = 16
BASELINE_RAW = 2500
CALIBRATION_FRAMES = 30


def build_window(app, interval_ms, clock=perf_counter):
    # The app resolves its clock from the module global, so swapping it here
    # reproduces the pre-fix behaviour without touching the source.
    mw.perf_counter = clock
    mw.DISPLAY_INTERVAL_MS = interval_ms
    window = mw.MainWindow()
    window.show()
    window.timer.stop()
    window.health_timer.stop()   # its 500 ms poll would flush outside the throttle
    window.info_timer.stop()
    app.processEvents()
    worker = SerialWorker("FAKE")
    window.worker = worker
    window.accept_scans = True
    return window, worker


def instrument(window, worker, waits):
    """Record how long each scheduled commit actually had to wait."""
    pending = {}
    original_request = window._request_live_update
    original_flush = window._flush_live

    def request():
        original_request()
        if window.display_timer.isActive():
            pending.setdefault("at", perf_counter())

    def flush():
        started = pending.pop("at", None)
        if started is not None:
            waits.append((perf_counter() - started) * 1000)
        original_flush()

    window._request_live_update = request
    window.display_timer.timeout.disconnect()
    window.display_timer.timeout.connect(flush)
    worker.data_ready.connect(request, Qt.ConnectionType.QueuedConnection)
    return pending


def shown_value(window):
    """Latest value committed to the heatmap, or None before the layout exists."""
    item = window.heatmap.items.get(BLOCK_ID)
    return None if item is None or item.values is None else float(item.values[0, 0])


def pump_until(app, done, deadline):
    # Drain the queue at least once before checking: returning without pumping
    # would leave the queued _request_live_update unserviced, so the next
    # iteration would inherit its scheduling timestamp and mispair the wait.
    # Sleeping instead of busy-draining would add ~1 ms of measurement noise.
    while perf_counter() < deadline:
        app.processEvents()
        if done():
            return


def stats(values):
    arr = np.asarray(values) if values else np.zeros(1)
    return {
        "median": round(float(np.median(arr)), 3),
        "mean": round(float(arr.mean()), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "max": round(float(arr.max()), 3),
    }


def measure(app, args):
    app_clock = {"perf_counter": perf_counter, "monotonic": monotonic}[args.clock]
    window, worker = build_window(app, args.interval_ms, app_clock)
    waits, latencies = [], []
    pending = instrument(window, worker, waits)
    try:
        flat = np.full((SIDE, SIDE), BASELINE_RAW)
        for _ in range(CALIBRATION_FRAMES):
            worker._receive_block(Block(BLOCK_ID, flat))
        pump_until(app, lambda: shown_value(window) is not None, perf_counter() + 2)
        if shown_value(window) != 0:
            raise RuntimeError(f"基线未归零，无法测量增量（当前 {shown_value(window)}）")

        committed = 0
        started = perf_counter()
        for step in range(args.samples):
            response = step % 400 + 1          # consecutive jumps stay under 800
            arrival = perf_counter()
            pending.clear()                    # pair each wait with its own arrival
            worker._receive_block(Block(BLOCK_ID, np.full((SIDE, SIDE), BASELINE_RAW - response)))
            expected = float(response)
            pump_until(app, lambda: shown_value(window) == expected, arrival + 1.0)
            if shown_value(window) == expected:
                latencies.append((perf_counter() - arrival) * 1000)
                committed += 1
            if args.cadence_ms:
                sleep(args.cadence_ms / 1000)
        elapsed = perf_counter() - started
    finally:
        window.accept_scans = False
        window.worker = None
        window.close()

    return {
        "interval_ms": args.interval_ms,
        "app_clock": args.clock,
        "arrivals": args.samples,
        "committed": committed,
        "elapsed_s": round(elapsed, 3),
        "committed_rate_hz": round(committed / elapsed, 1) if elapsed else 0.0,
        "cadence_ms": args.cadence_ms,
        "throttle_wait_ms": stats(waits),
        "latency_ms": stats(latencies),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-ms", type=float, default=None,
                        help="throttle to test; default keeps the shipped value")
    parser.add_argument("--clock", choices=("perf_counter", "monotonic"),
                        default="perf_counter",
                        help="app clock; 'monotonic' simulates the pre-fix 15.6 ms clock")
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--cadence-ms", type=float, default=1.0,
                        help="delay between arrivals; must be under the interval")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.interval_ms is None:
        args.interval_ms = mw.DISPLAY_INTERVAL_MS

    logging.disable(logging.CRITICAL)
    # The application raises the timer resolution at startup; match it, or the
    # measured numbers describe a configuration the app never runs in.
    enable_high_resolution_timer()
    app = QApplication([])
    try:
        report = measure(app, args)
    finally:
        restore_timer_resolution()
    report["conditions"] = (f"offscreen; one {SIDE}x{SIDE} block; arrivals every "
                            f"~{args.cadence_ms} ms; excluded: hardware sampling, "
                            f"serial transport, screen presentation")
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
