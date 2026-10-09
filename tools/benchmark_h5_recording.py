"""Offline HDF5 append benchmark; never opens a serial port or application UI."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
try:
    import h5py
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT / "artifacts" / "_h5_demo_deps"))
    import h5py


def run_case(directory, hz, compression):
    # Ten virtual seconds, written as fast as possible; no real-time sleeping.
    seconds, blocks, rows, cols, batches = 10, 14, 13, 5, 50
    batch_frames = hz // 5
    frames = hz * seconds
    path = directory / f"14blocks_{hz}hz_{compression or 'none'}.h5"
    rng = np.random.default_rng(20260926)
    # Generate inputs outside the I/O timing, then check every saved value.
    source = rng.integers(1900, 2531, size=(blocks, frames, rows, cols), dtype=np.uint16)
    timestamps = np.rint(np.arange(frames) * (1e9 / hz)).astype(np.int64)
    options = {"compression": "gzip", "compression_opts": 1, "shuffle": True} if compression else {}
    append_ms, flush_ms = [], []
    started = perf_counter()
    with h5py.File(path, "x") as f:
        f.attrs.update(is_synthetic=True, purpose="write_benchmark", assumed_block_hz=hz)
        datasets = []
        for key in range(blocks):
            g = f.create_group(f"blocks/block_{key+1:02d}")
            adc = g.create_dataset("raw_adc", shape=(0, rows, cols), maxshape=(None, rows, cols),
                                   chunks=(64, rows, cols), dtype="uint16", **options)
            time = g.create_dataset("host_time_ns", shape=(0,), maxshape=(None,),
                                    chunks=(64,), dtype="int64", **options)
            datasets.append((adc, time))
        for batch in range(batches):
            lo, hi = batch*batch_frames, (batch+1)*batch_frames
            before = perf_counter()
            for key, (adc, time) in enumerate(datasets):
                adc.resize(hi, axis=0)
                time.resize(hi, axis=0)
                adc[lo:hi] = source[key, lo:hi]
                time[lo:hi] = timestamps[lo:hi]
            if (batch+1) % 5 == 0:
                flush_started = perf_counter()
                f.flush()
                flush_ms.append((perf_counter()-flush_started)*1000)
            append_ms.append((perf_counter()-before)*1000)
    elapsed = perf_counter()-started
    with h5py.File(path, "r") as f:
        for key in range(blocks):
            g = f[f"blocks/block_{key+1:02d}"]
            np.testing.assert_array_equal(g["raw_adc"][:], source[key])
            np.testing.assert_array_equal(g["host_time_ns"][:], timestamps)
    return {
        "file": path.name, "compression": compression or "none", "block_count": blocks,
        "cells_per_block": rows*cols, "assumed_block_hz": hz, "virtual_duration_seconds": seconds,
        "block_frames_verified": frames*blocks, "all_values_verified": True,
        "raw_adc_bytes_per_virtual_second": blocks*rows*cols*2*hz,
        "adc_and_time_bytes_per_virtual_second": blocks*(rows*cols*2+8)*hz,
        "file_bytes": path.stat().st_size, "write_open_to_close_seconds": round(elapsed, 6),
        "batch_interval_budget_ms": 200, "batch_write_ms_median": round(float(np.median(append_ms)), 3),
        "batch_write_ms_p95": round(float(np.percentile(append_ms, 95)), 3),
        "batch_write_ms_max": round(max(append_ms), 3),
        "flush_ms_max": round(max(flush_ms), 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.output or ROOT / "artifacts" / ("h5-write-benchmark-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    directory.mkdir(parents=True, exist_ok=False)
    results = [run_case(directory, hz, compression) for hz in (20, 200) for compression in (None, "gzip")]
    report = {
        "is_synthetic": True, "h5py_version": h5py.__version__,
        "method": "14 blocks x 65 cells; raw uint16 ADC plus int64 times; batch every virtual 200 ms; HDF5 flush every virtual 1 s; 10 virtual seconds per case; no sleeping",
        "limitations": ["Short local I/O test, not long-term hardware validation.",
                        "OS disk caching is included; HDF5 flush is not an fsync/power-loss durability guarantee.",
                        "Excludes parsing, filtering, producer queue and concurrent UI rendering.",
                        "Input generation and readback verification excluded from write timing.",
                        "20/200 Hz per block are storage stress assumptions, not measured serial capacity."],
        "cases": results,
    }
    (directory / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"directory": str(directory), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
