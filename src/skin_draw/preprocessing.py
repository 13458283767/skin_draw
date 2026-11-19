"""Preprocessing pipeline: M3 filter -> fixed baseline -> normalization.

Per point: raw sample -> PointFilter (M3 high-plateau sliding mean) ->
BaselineTracker (initial median, fixed until reset) -> normalization (none / A / B).

Empty points (mask False) and not-yet-ready points output NaN so the renderer
can paint them as missing. All output matrices are freshly allocated and
read-only.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

NORM_MODES = ("none", "A", "B")
DIRECTIONS = ("drop", "rise")
DEFAULT_FPS = 90.6  # delivery-pack measured full-board rate (poll mode)
DEFAULT_WINDOW = 30   # M3 sliding window length (frames)
DEFAULT_DELTA = 150   # M3 high-plateau band below the window maximum (LSB)
DEFAULT_MIN_KEEP = 3  # fewest kept samples before the plateau mean is trusted


@dataclass(frozen=True)
class BaselineGate:
    """连接时 3 秒标定的质量闸门。

    阈值与逐条基线（configs/collection_baseline.json）保持同一口径，只是这里判断
    的是“这一段无接触样本能不能固化成参考基线”。旧的实现直接取中位数，一旦操作员
    在标定窗口内碰到皮肤，这份错误的基线会被后续每条样本当作“未接触参考”，
    于是整场次的提前接触判定全部失真。
    """

    max_spread_counts: float = 30.    # 窗口 p95-p5，超过说明有接触或抖动
    max_drift_counts: float = 20.     # 首尾各 1/3 的中位数差，超过说明期间在滑动
    max_reference_delta: float = 100. # 与上一代基线的偏差，超过说明贴附状态变了
    min_samples: int = 30             # 样本太少时不做统计闸门，避免小窗口误判
    max_attempts: int = 3             # 连续被拒上限，达到后停止自动重试

    def __post_init__(self):
        for name in ("max_spread_counts", "max_drift_counts", "max_reference_delta"):
            value = getattr(self, name)
            if type(value) not in (int, float) or value <= 0:
                raise ValueError(f"标定闸门 {name} 必须为正数")
        for name in ("min_samples", "max_attempts"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"标定闸门 {name} 必须为正整数")


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
    """Median of the first ``calib_n`` no-contact samples, fixed until reset.

    固定前先过 ``BaselineGate``：窗口波动、漂移或偏离上一代基线时拒绝这一批样本
    并重新收集，避免把“手还在皮肤上”的读数固化成参考基线。被拒原因保留在
    ``last_issue`` 供界面直接显示；连续被拒达到上限后 ``blocked``，停止自动重试。
    """

    def __init__(self, calib_n=270, gate=None):
        self.calib_n = max(1, int(calib_n))
        self.gate = gate or BaselineGate()
        self.baseline = None
        self.previous = None      # 上一次成功固定的基线，用于发现重新贴附
        self.rejections = 0       # 连续被闸门拒绝的次数
        self.last_issue = ""      # 最近一次拒绝原因，空串表示没有待处理问题
        self.last_warning = ""    # 不阻断但需要提醒的情况（如重新贴附后基线跳变）
        self.blocked = False      # 连续被拒达到上限，需要人工确认后重置
        self._calib = []

    def update(self, value):
        if value is None:
            return self.baseline
        value = float(value)
        if self.baseline is not None:
            return self.baseline
        if self.blocked:
            return None           # 已停止重试，等用户确认后重置
        self._calib.append(value)
        if len(self._calib) < self.calib_n:
            return None
        baseline, issue = self._evaluate()
        self._calib.clear()
        if baseline is None:
            self.rejections += 1
            self.last_issue = issue
            self.blocked = self.rejections >= self.gate.max_attempts
            return None
        self.baseline = baseline
        self.rejections = 0
        self.last_issue = ""
        return self.baseline

    def _evaluate(self):
        """返回 (基线, 原因)；原因为空串表示通过闸门。"""
        gate = self.gate
        values = np.asarray(self._calib, dtype=float)
        self.last_warning = ""
        if values.size >= gate.min_samples:
            spread = float(np.percentile(values, 95) - np.percentile(values, 5))
            if spread > gate.max_spread_counts:
                return None, f"标定窗口波动 {spread:.0f} LSB，疑似接触或抖动"
            third = max(1, values.size // 3)
            drift = abs(float(np.median(values[-third:]) - np.median(values[:third])))
            if drift > gate.max_drift_counts:
                return None, f"标定窗口漂移 {drift:.0f} LSB，疑似缓慢接触或滑动"
        median = _median(self._calib)
        # 偏离上一代基线只提示不阻断：用户主动重标定常伴随贴附变化，
        # 直接拒绝会逼他把同一个动作点两次；这里保留证据交给操作员判断。
        if self.previous is not None:
            delta = abs(median - self.previous)
            if delta > gate.max_reference_delta:
                self.last_warning = f"新基线偏离上一代 {delta:.0f} LSB，请确认贴附未变"
        return median, ""

    @property
    def count(self):
        """Calibration samples collected so far (saturates at calib_n)."""
        return self.calib_n if self.baseline is not None else len(self._calib)

    @property
    def ready(self):
        return self.baseline is not None

    def reset(self, keep_reference=False):
        """
        ``keep_reference`` 只在用户主动重新校准时为真：把当前基线记为上一代，
        以便下次标定发现“贴附状态变了”。切换滤波、改点位等路径必须清空参考，
        否则处理方式变化本身就会被误判成贴附变化。已被阻断时再次重置视为
        强制重来，同样清空参考，避免同一个偏差反复拦截。
        """
        self.previous = (self.baseline or self.previous) if (keep_reference and not self.blocked) else None
        self.baseline = None
        self._calib = []
        self.rejections = 0
        self.last_issue = ""
        self.last_warning = ""
        self.blocked = False


class BlockPreprocessor:
    """Whole-block per-point pipeline: M3 -> fixed baseline -> normalization.

    - Empty points (``valid_mask`` False) are skipped and output NaN.
    - ``filter_enabled=False`` bypasses M3 and its warm-up entirely.
    - norm="none" outputs the optionally filtered raw value and never collects a
      baseline; switching to "A"/"B" starts collection from that moment.
    - norm="A" outputs the pressure deviation with negative values clamped to
      zero so it stays inside the renderer's 0-4095 contract.
    - norm="B" outputs ``clip(d / r_ref, 0, 1)``.
    - ``d = baseline - filtered`` for direction="drop" (pressed ADC decreases)
      and ``filtered - baseline`` for "rise".
    - The initial median baseline stays fixed until preprocessing is reset.
    """

    def __init__(self, shape, valid_mask, *, norm="B", direction="drop",
                 calib_n=270,
                 r_ref=600., w=DEFAULT_WINDOW, delta=DEFAULT_DELTA, n_min=DEFAULT_MIN_KEEP,
                 filter_enabled=True, gate=None):
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
        self.filter_enabled = bool(filter_enabled)
        self.direction = direction
        self.r_ref = float(r_ref)
        self.w = int(w)
        self.calib_n = max(1, int(calib_n))
        self.gate = gate or BaselineGate()
        self.frames = 0
        self.filters = [[PointFilter(w, delta, n_min) for _ in range(cols)]
                        for _ in range(rows)]
        self.trackers = [[BaselineTracker(calib_n, self.gate) for _ in range(cols)]
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
                # 关闭滤波时直接使用本帧读数，不更新滑窗，也不等待窗口暖机。
                filtered = (self.filters[row][col].update(array[row, col])
                            if self.filter_enabled else float(array[row, col]))
                if filtered is None:
                    continue
                self.filtered[row, col] = filtered
                if self.norm == "none" or (self.filter_enabled and self.frames < self.w):
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
    def baseline_issue(self):
        """首个未通过标定闸门的有效点原因；空串表示没有待处理问题。"""
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                if not self.valid[row, col]:
                    continue
                issue = self.trackers[row][col].last_issue
                if issue:
                    return f"行{row + 1}列{col + 1} {issue}"
        return ""

    @property
    def baseline_blocked(self):
        """连续被拒达到上限：需要人工确认无接触后点击重置，不再自动重试。"""
        return any(self.valid[row, col] and self.trackers[row][col].blocked
                   for row in range(self.shape[0]) for col in range(self.shape[1]))

    @property
    def baseline_warning(self):
        """不阻断但需要告知操作员的提示（如重新贴附后基线跳变）。"""
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                if self.valid[row, col]:
                    warning = self.trackers[row][col].last_warning
                    if warning:
                        return warning
        return ""

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

    def set_filter_enabled(self, enabled):
        """切换滤波后重新标定，避免新读数沿用另一种处理方式的基线。"""
        enabled = bool(enabled)
        if self.filter_enabled != enabled:
            self.filter_enabled = enabled
            self.reset()

    def set_norm(self, norm):
        if norm not in NORM_MODES:
            raise ValueError("归一化模式必须为 none/A/B")
        self.norm = norm

    def reset(self, keep_reference=False):
        self.frames = 0
        self.filtered.fill(np.nan)
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                self.filters[row][col].reset()
                self.trackers[row][col].reset(keep_reference)
