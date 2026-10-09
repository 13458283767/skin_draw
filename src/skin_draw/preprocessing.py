"""Delivery-pack preprocessing pipeline: M3 filter -> slow baseline -> normalization.

Ported from the delivered online reference ``skin_preprocess.py``（数值语义逐帧
对齐）。Per point: raw sample -> PointFilter (M3 high-plateau sliding mean) ->
BaselineTracker (slow median EMA) -> normalization (none / A / B).

Empty points (mask False) and not-yet-ready points output NaN so the renderer
can paint them as missing. All output matrices are freshly allocated and
read-only.
"""

from collections import deque

import numpy as np

NORM_MODES = ("none", "A", "B")
DIRECTIONS = ("drop", "rise")
DEFAULT_FPS = 90.6  # delivery-pack measured full-board rate (poll mode)
DEFAULT_WINDOW = 30   # M3 sliding window length (frames)
DEFAULT_DELTA = 150   # M3 high-plateau band below the window maximum (LSB)
DEFAULT_MIN_KEEP = 3  # fewest kept samples before the plateau mean is trusted


def _median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else .5 * (ordered[middle - 1] + ordered[middle])


class PointFilter:
    """M3 high-plateau sliding-window mean for one point.

    Window = the latest ``w`` raw samples; the output is the mean of the samples
    within ``delta`` of the window maximum. Fewer than ``n_min`` kept samples
    keep the previous output (None before the first successful update).
    """

    def __init__(self, w=DEFAULT_WINDOW, delta=DEFAULT_DELTA, n_min=DEFAULT_MIN_KEEP):
        self.w = int(w)
        self.delta = float(delta)
        self.n_min = int(n_min)
        self.buf = deque(maxlen=self.w)
        self.last = None

    def update(self, value):
        value = float(value)
        self.buf.append(value)
        threshold = max(self.buf) - self.delta
        kept = [x for x in self.buf if x >= threshold]
        if len(kept) >= self.n_min:
            self.last = sum(kept) / len(kept)
        return self.last

    def reset(self):
        self.buf.clear()
        self.last = None


class BaselineTracker:
    """No-pressure baseline per point (delivery-pack semantics).

    mode="calib": median of the first ``calib_n`` valid filtered outputs, fixed.
    mode="slow" : median of the first ``calib_n`` samples, then every ``upd_n``
                  accumulated samples ``baseline += alpha * (median - baseline)``.
                  Self-heals the startup-convergence and slow-drift effects;
                  the block median rejects short presses (empty-load assumption).
    """

    def __init__(self, mode="slow", calib_n=270, upd_n=90, alpha=.2):
        if mode not in ("calib", "slow"):
            raise ValueError("基线模式必须为 calib 或 slow")
        self.mode = mode
        self.calib_n = max(1, int(calib_n))
        self.upd_n = max(1, int(upd_n))
        self.alpha = float(alpha)
        self.baseline = None
        self._calib = []
        self._recent = deque(maxlen=self.upd_n)

    def update(self, value):
        if value is None:
            return self.baseline
        value = float(value)
        if self.baseline is None:
            self._calib.append(value)
            if len(self._calib) >= self.calib_n:
                self.baseline = _median(self._calib)
            return self.baseline
        if self.mode == "slow":
            self._recent.append(value)
            if len(self._recent) >= self._recent.maxlen:
                self.baseline += self.alpha * (_median(self._recent) - self.baseline)
                self._recent.clear()
        return self.baseline

    @property
    def count(self):
        """Calibration samples collected so far (saturates at calib_n)."""
        return self.calib_n if self.baseline is not None else len(self._calib)

    @property
    def ready(self):
        return self.baseline is not None

    def reset(self):
        self.baseline = None
        self._calib = []
        self._recent.clear()


class BlockPreprocessor:
    """Whole-block per-point pipeline: M3 -> slow baseline -> normalization.

    - Empty points (``valid_mask`` False) are skipped and output NaN.
    - norm="none" outputs the M3-filtered raw value and never collects a
      baseline; switching to "A"/"B" starts collection from that moment.
    - norm="A" outputs the pressure deviation with negative values clamped to
      zero so it stays inside the renderer's 0-4095 contract.
    - norm="B" outputs ``clip(d / r_ref, 0, 1)``.
    - ``d = baseline - filtered`` for direction="drop" (pressed ADC decreases)
      and ``filtered - baseline`` for "rise".
    - ``fps`` converts the slow-tracking block length to time (``upd_sec``);
      pass the measured per-block rate.
    """

    def __init__(self, shape, valid_mask, *, norm="B", direction="drop",
                 fps=DEFAULT_FPS, upd_sec=1., calib_n=270, upd_n=90, alpha=.2,
                 r_ref=600., w=DEFAULT_WINDOW, delta=DEFAULT_DELTA, n_min=DEFAULT_MIN_KEEP):
        if norm not in NORM_MODES:
            raise ValueError("归一化模式必须为 none/A/B")
        if direction not in DIRECTIONS:
            raise ValueError("受压方向必须为 drop 或 rise")
        rows, cols = (int(n) for n in shape)
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != (rows, cols):
            raise ValueError("有效点位必须匹配数据尺寸")
        self.shape = (rows, cols)
        self.valid = np.frombuffer(valid.tobytes(), dtype=bool).reshape(valid.shape)
        self.norm = norm
        self.direction = direction
        self.r_ref = float(r_ref)
        self.w = int(w)
        if fps is not None and float(fps) > 0:
            upd_n = round(float(fps) * float(upd_sec))  # 块长按时间换算, 见交付包 2.4
        self.upd_n = max(1, int(upd_n))
        self.calib_n = max(1, int(calib_n))
        self.frames = 0
        self.filters = [[PointFilter(w, delta, n_min) for _ in range(cols)]
                        for _ in range(rows)]
        self.trackers = [[BaselineTracker("slow", calib_n, upd_n, alpha) for _ in range(cols)]
                        for _ in range(rows)]
        self.filtered = np.full(self.shape, np.nan, dtype=float)

    def update(self, values):
        """Advance one frame and return the read-only output matrix."""
        array = np.asarray(values)
        if array.shape != self.shape:
            raise ValueError("数据尺寸与预处理器不一致")
        self.frames += 1
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                if not self.valid[row, col]:
                    continue
                filtered = self.filters[row][col].update(array[row, col])
                if filtered is None:
                    continue
                self.filtered[row, col] = filtered
                if self.norm == "none" or self.frames < self.w:
                    # none never collects; during the warm-up frames the M3
                    # window is not full, so its output must not seed a baseline.
                    continue
                self.trackers[row][col].update(filtered)
        return self.recompose()

    def recompose(self):
        """Rebuild the output matrix from cached state (no new samples)."""
        output = np.full(self.shape, np.nan, dtype=float)
        if self.norm == "none":
            np.copyto(output, self.filtered)
        else:
            for row in range(self.shape[0]):
                for col in range(self.shape[1]):
                    value = self.filtered[row, col]
                    if not np.isfinite(value):
                        continue
                    baseline = self.trackers[row][col].baseline
                    if baseline is None:
                        continue
                    delta = (baseline - value) if self.direction == "drop" else (value - baseline)
                    if self.norm == "A":
                        output[row, col] = delta if delta > 0 else 0.
                    else:
                        scaled = delta / self.r_ref
                        output[row, col] = 0. if scaled < 0 else (1. if scaled > 1 else scaled)
        output.flags.writeable = False
        return output

    @property
    def progress(self):
        """0-100 calibration progress driven by the slowest valid point."""
        if self.norm == "none":
            return 100
        slowest = 1.
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                if self.valid[row, col]:
                    tracker = self.trackers[row][col]
                    slowest = min(slowest, tracker.count / self.calib_n)
        return int(100 * min(1., slowest))

    @property
    def ready(self):
        if self.norm == "none":
            return True
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                if self.valid[row, col] and not self.trackers[row][col].ready:
                    return False
        return True

    @property
    def samples(self):
        """Fewest calibration samples across valid points (progress detail)."""
        counts = [self.trackers[row][col].count
                  for row in range(self.shape[0]) for col in range(self.shape[1])
                  if self.valid[row, col]]
        return min(counts, default=self.calib_n)

    @property
    def baselines(self):
        """Current baseline matrix (NaN where missing), read-only."""
        result = np.full(self.shape, np.nan, dtype=float)
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                tracker = self.trackers[row][col]
                if self.valid[row, col] and tracker.baseline is not None:
                    result[row, col] = tracker.baseline
        result.flags.writeable = False
        return result

    def set_direction(self, direction):
        if direction not in DIRECTIONS:
            raise ValueError("受压方向必须为 drop 或 rise")
        self.direction = direction

    def set_norm(self, norm):
        if norm not in NORM_MODES:
            raise ValueError("归一化模式必须为 none/A/B")
        self.norm = norm

    def reset(self):
        self.frames = 0
        self.filtered.fill(np.nan)
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                self.filters[row][col].reset()
                self.trackers[row][col].reset()
