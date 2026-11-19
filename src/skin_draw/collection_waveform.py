"""Lossy live previews and bounded, peak-preserving replay of CLOSED raw CSVs.

No Qt, no file writes, no filter/baseline changes. Each curve is the mean of
the valid raw ADC points of one block in the saved coordinate order. Live
previews are advisory; full review always re-reads the authoritative CSV.
"""

from collections import deque
import csv
import json
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class WaveformSnapshot:
    curves: dict
    latest_s: float
    status: str = ""


def saved_indices(plan):
    return {key: np.flatnonzero(block.transform(np.asarray(block.raw_valid_mask)).ravel())
            for key, block in plan.blocks.items()}


def envelope(points, bins=400, gap_s=.05):
    """Keep first/min/max/last per time bucket and mark receive gaps with NaNs."""
    if not points:
        return np.empty(0), np.empty(0)
    values = np.asarray(points, dtype=float)
    gaps = np.flatnonzero(np.diff(values[:, 0]) > gap_s) + 1
    gap_times = values[gaps, 0]
    if len(values) > bins * 4:
        width = max((values[-1, 0] - values[0, 0]) / bins, 1e-9)
        bucket = np.floor((values[:, 0] - values[0, 0]) / width).astype(np.int64)
        starts = np.unique(np.r_[0, np.flatnonzero(np.diff(bucket)) + 1, gaps])
        ends = np.r_[starts[1:], len(values)]
        sizes = ends - starts
        y, positions = values[:, 1], np.arange(len(values))
        lows = np.repeat(np.minimum.reduceat(y, starts), sizes)
        highs = np.repeat(np.maximum.reduceat(y, starts), sizes)
        low_ids = np.minimum.reduceat(np.where(y == lows, positions, len(values)), starts)
        high_ids = np.minimum.reduceat(np.where(y == highs, positions, len(values)), starts)
        values = values[np.unique(np.r_[starts, ends - 1, low_ids, high_ids])]
    if len(gaps):
        # Only original receive gaps create breaks. Aggregation must not invent
        # gaps merely because selected extrema are further apart in time.
        positions = np.searchsorted(values[:, 0], gap_times)
        values = np.insert(values, positions, np.c_[gap_times, np.full(len(gaps), np.nan)], axis=0)
    return values[:, 0], values[:, 1]


class LiveWaveform:
    """One bounded queue/worker per trial; receive never waits for this worker."""

    def __init__(self, plan, *, capacity=2048, seconds=10., max_points=4096):
        self.queue = Queue(maxsize=capacity)
        self.indices = saved_indices(plan)
        self.points = {key: deque(maxlen=max_points) for key in self.indices}
        self.seconds = seconds
        self.stop_event = Event()
        self.snapshot = WaveformSnapshot({}, 0.)
        self.thread = Thread(target=self._run, name="skin-waveform-preview", daemon=True)
        self.thread.start()

    def close(self):
        # No join on the UI/receive thread. It exits at most one short batch later.
        self.stop_event.set()

    def _run(self):
        last_publish = perf_counter()
        latest = 0.
        while not self.stop_event.is_set():
            try:
                row = self.queue.get(timeout=.02)
                key = int(row[2])
                indices = self.indices.get(key)
                if indices is not None and indices.size:
                    t = float(row[1])
                    value = float(np.mean([row[3 + int(i)] for i in indices]))
                    self.points[key].append((t, value))
                    latest = max(latest, t)
            except Empty:
                pass
            except (ValueError, IndexError, TypeError):
                continue  # Display failure cannot affect the original recording.
            if perf_counter() - last_publish >= .1:
                curves = {}
                for key, points in self.points.items():
                    while points and points[0][0] < latest - self.seconds:
                        points.popleft()
                    curves[key] = envelope(list(points), bins=160)
                # Immutable-by-convention object replacement; GUI takes no lock.
                self.snapshot = WaveformSnapshot(curves, latest)
                last_publish = perf_counter()


def load_waveform(plan, duration_s, cancel=None, bins=600):
    """Stream a completed file; memory O(blocks*bins), independent of row count.

    Four extrema per bucket retain narrow pulses. Buckets with an internal
    receive gap are deliberately shown as a gap rather than a false plateau.
    """
    # SampleResult's duration is rounded for its status label; annotation needs
    # the precise recorded boundary, especially for a very short tap.
    meta_path = plan.sample_dir / "meta.json"
    if meta_path.exists():
        duration_s = float(json.loads(meta_path.read_text(encoding="utf-8"))["duration_s"])
    indices = saved_indices(plan)
    width = max(float(duration_s) / bins, 1e-6)
    buckets = {key: {} for key in indices}
    last_time = {}
    gaps = {key: set() for key in indices}
    with (plan.sample_dir / "raw.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if cancel is not None and cancel.is_set():
                return None
            key, t = int(row["block_id"]), float(row["time_s"])
            selected = indices.get(key)
            if selected is None or not selected.size:
                continue
            if not np.isfinite(t) or t < 0 or t > duration_s + .001:
                raise ValueError("波形文件存在无效时间戳")
            bucket = min(bins - 1, int(t / width))
            if key in last_time and t < last_time[key]:
                raise ValueError("波形文件的逐块时间戳倒退")
            if key in last_time and t - last_time[key] > .05:
                gaps[key].add(bucket)
            last_time[key] = t
            raw = [float(row[f"adc_{int(i):03d}"]) for i in selected]
            y = float(np.mean(raw))
            if not np.isfinite(y):
                gaps[key].add(bucket)
                continue
            point = (t, y)
            values = buckets[key].get(bucket)
            if values is None:
                buckets[key][bucket] = [point, point, point, point]
            else:
                if y < values[1][1]:
                    values[1] = point
                if y > values[2][1]:
                    values[2] = point
                values[3] = point
    curves = {}
    for key, by_bucket in buckets.items():
        points = []
        previous = None
        for bucket, values in sorted(by_bucket.items()):
            if bucket in gaps[key]:
                points.extend([(values[0][0], np.nan), (values[-1][0], np.nan)])
            else:
                if previous is not None and values[0][0] - previous > .05:
                    points.append((values[0][0], np.nan))
                points.extend(sorted(set(values)))
            previous = values[-1][0]
        if points:
            a = np.asarray(points)
            curves[key] = (a[:, 0], a[:, 1])
        else:
            curves[key] = (np.empty(0), np.empty(0))
    return WaveformSnapshot(curves, float(duration_s), "完整波形 · 原始 ADC 有效点均值；缺口断线显示")
