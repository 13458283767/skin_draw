"""Bounded robust initialization and independent stable sampling of one row."""

from collections import deque
from copy import copy
import numpy as np


class CalibrationRow:
    # Calibration assumes no pressure. This band is not the live jump filter.
    tolerance = 100
    bootstrap_size = 7

    def __init__(self, columns, required_frames, minimum_seconds):
        self.required_frames = required_frames
        self.minimum_seconds = minimum_seconds
        self.history = deque(maxlen=min(self.bootstrap_size, required_frames))
        self.reference = None
        self.total = np.zeros(columns, dtype=float)
        self.count = 0
        self.seconds = 0.
        self.last_time = None
        self.rejections = 0
        self.restarts = 0
        self.rejected_streak = 0
        self.result = None
        if columns == 0:
            self.result = np.zeros(0)
            self.result.flags.writeable = False

    def snapshot(self):
        result = copy(self)
        result.total = self.total.copy()
        result.history = self.history.copy()  # samples and reference are read-only
        return result

    @property
    def progress(self):
        if self.result is not None:
            return 1.
        duration = min(1., self.seconds / self.minimum_seconds) if self.minimum_seconds else 1.
        return min(self.count / self.required_frames, duration, .99)

    def consume(self, values, now):
        if self.result is not None:
            return
        # Count observed valid sampling time, not a disconnect or time spent
        # displaying a held value. Slow streams still need at least 30 samples.
        delta = 0. if self.last_time is None else max(0., now - self.last_time)
        dt = min(delta, .1) if delta <= .25 else 0.
        self.last_time = now
        sample = np.array(values, dtype=float, copy=True)
        sample.flags.writeable = False
        self.history.append((sample, dt))
        if self.reference is not None:
            if np.all(np.abs(sample - self.reference) <= self.tolerance):
                self.total += sample
                self.count += 1
                self.seconds += dt
                self.rejected_streak = 0
            else:
                self.rejections += 1
                self.rejected_streak += 1
                if self.rejected_streak < self.history.maxlen:
                    return
                # Only calibration may replace a bad reference. Never relax
                # the runtime filter to accept a sustained corrupt ADC level.
                self.reference = None
                self.total.fill(0)
                self.count = 0
                self.seconds = 0.
                self.rejected_streak = 0
                self.restarts += 1
        if self.reference is None:
            if len(self.history) < self.history.maxlen:
                return
            samples = np.stack([value for value, _ in self.history])
            center = np.median(samples, axis=0)
            good = np.all(np.abs(samples - center) <= self.tolerance, axis=1)
            # Five consistent rows out of seven; a lone first-frame glitch
            # cannot become a permanent filtering reference.
            if int(good.sum()) < max(1, int(np.ceil(.7 * len(samples)))):
                return
            self.reference = center
            self.reference.flags.writeable = False
            self.total = samples[good].sum(axis=0)
            self.count = int(good.sum())
            accepted = np.flatnonzero(good)
            self.seconds = sum(self.history[int(i)][1] for i in accepted[1:])
        if self.count >= self.required_frames and self.seconds + 1e-9 >= self.minimum_seconds:
            self.result = self.total / self.count
            self.result.flags.writeable = False
            self.history.clear()
