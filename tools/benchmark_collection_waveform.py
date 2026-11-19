"""Synthetic A/B check: heatmap + CSV, with/without live waveform.

Uses temporary datasets and an offscreen Qt window; never opens a serial port.
Run from the repository: .venv/Scripts/python tools/benchmark_collection_waveform.py
The result is not a substitute for a physical-device/visible-display test.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
from threading import Thread
from time import perf_counter, sleep

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from skin_draw.models import Block, Frame
from skin_draw.ui.main_window import MainWindow
from skin_draw.ui.serial_worker import SerialWorker


def wait_until(app, condition, timeout=15.):
    deadline = perf_counter() + timeout
    while not condition():
        if perf_counter() > deadline:
            raise TimeoutError("benchmark timed out")
        app.processEvents()
        sleep(.001)


def run(app, directory, enabled, seconds=3.):
    window = MainWindow(mask_path=directory / "masks.json", dataset_dir=directory / "dataset")
    shapes = [(5, 3), (5, 3), (7, 3), (7, 3), (7, 4), (7, 4), (8, 5), (10, 5)]
    arrays = [np.full(shape, 2500, dtype=np.uint16) for shape in shapes]
    blocks = [Block(i + 1, a) for i, a in enumerate(arrays)]
    for key in (5, 6):
        mask = np.ones(shapes[key - 1], dtype=bool)
        mask.ravel()[-6:] = False
        window.sensor_masks = window.sensor_masks.with_mask(key, mask)
    window.heatmap.sensor_masks = window.sensor_masks
    window.present_frame(Frame(tuple(blocks)))
    window.save_collection_layout()
    worker = SerialWorker("BENCHMARK_NO_PORT", filter_enabled=False)
    worker.buffer.set_sensor_masks(window.sensor_masks)
    window.worker = worker
    window.attach_recording_sink(worker.buffer)
    window.input_source = "hardware"
    window.accept_scans = True
    worker.data_ready.connect(window._request_live_update, Qt.ConnectionType.QueuedConnection)
    for _ in range(270):
        for block in blocks:
            worker._receive_block(block)
    window.show()
    app.processEvents()
    window.waveform.live_enabled.setChecked(enabled)
    window.start_recording()
    if not enabled:
        window._preview.close()
        window.recorder.preview_queue = None
        window.waveform_timer.stop()
        window.waveform.hide()
    receive_ms, commits, waveform_ms, backlog = [], [], [], []
    original = window._consume_scan

    def measured(*args, **kwargs):
        start = perf_counter()
        original(*args, **kwargs)
        commits.append((start, (perf_counter() - start) * 1000))

    window._consume_scan = measured
    original_draw = window.waveform.draw

    def draw(*args, **kwargs):
        original_draw(*args, **kwargs)
        waveform_ms.append(window.waveform.last_draw_ms)

    window.waveform.draw = draw
    cycles = int(seconds * 100)

    def produce():
        begin = perf_counter()
        for i in range(cycles):
            for key, array in enumerate(arrays, 1):
                value = array.copy()
                # First second is the required no-contact trial baseline.
                if i >= 130:
                    value.ravel()[0] -= int(400 * (1 + np.sin(i / 10)) / 2)
                before = perf_counter()
                worker._receive_block(Block(key, value))
                receive_ms.append((perf_counter() - before) * 1000)
            backlog.append(window.recorder._writer.queue.qsize())
            sleep(max(0., begin + (i + 1) / 100 - perf_counter()))

    thread = Thread(target=produce, daemon=True)
    thread.start()
    wait_until(app, lambda: not thread.is_alive())
    stop_start = perf_counter()
    window.finish_recording(background=True, notify=False)
    stop_return_ms = (perf_counter() - stop_start) * 1000
    wait_until(app, lambda: window._collection_job is None)
    result = window.last_sample
    metadata = json.loads(result.meta_path.read_text(encoding="utf-8"))
    with (result.sample_dir / "raw.csv").open(encoding="utf-8") as stream:
        record_count = sum(1 for _ in stream) - 1
    p95 = lambda values: round(float(np.percentile(values, 95)), 3) if len(values) else None
    answer = {
        "live_waveform": enabled, "expected_records": cycles * 8,
        "written_records": result.written_records, "csv_records": record_count,
        "dropped_records": metadata["quality"]["dropped_records"],
        "baseline_status": metadata["baseline"]["status"],
        "action_allowed_s": metadata["baseline"]["action_allowed_s"],
        "max_writer_queue": max(backlog), "receive_callback_ms_p95": p95(receive_ms),
        "heatmap_commit_ms_p95": p95([entry[1] for entry in commits]),
        "heatmap_commit_interval_ms_p95": p95(np.diff([entry[0] for entry in commits]) * 1000),
        "waveform_updates_including_replay": len(waveform_ms),
        "waveform_update_ms_p95": p95(waveform_ms), "stop_handler_ms": round(stop_return_ms, 3),
    }
    assert result.written_records == record_count == cycles * 8
    assert metadata["quality"]["dropped_records"] == 0
    assert metadata["baseline"]["status"] == "ready"
    if enabled:
        assert len(waveform_ms) > 2, "No live plots rendered; A/B result would be invalid"
    window.worker = None
    window.accept_scans = False
    window.close()
    worker.deleteLater()
    app.processEvents()
    return answer


if __name__ == "__main__":
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    with tempfile.TemporaryDirectory(prefix="skin-waveform-bench-") as name:
        results = [run(app, Path(name) / str(i), enabled)
                   for i, enabled in enumerate((False, True, True, False))]
    output = Path(__file__).resolve().parents[1] / "artifacts" / "collection-waveform-benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"mode": "synthetic_offscreen", "results": results}, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
