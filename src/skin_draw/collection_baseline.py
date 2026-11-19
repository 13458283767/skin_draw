"""Bounded first-second raw baseline; statistics run outside reception and Qt.

The recorder serializes observe/advance. Only immutable Block arrays are kept.
The worker owns the lists after advance; it publishes a completed result once.
No file I/O or changes to the display baseline occur here.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np


@dataclass(frozen=True)
class TrialBaselinePolicy:
    duration_s: float = 1.0
    min_samples: int = 20
    max_edge_gap_s: float = .15
    max_interval_s: float = .15
    max_spread_counts: float = 30.
    max_drift_counts: float = 20.
    max_reference_delta_counts: float = 100.
    contact_min_frames: int = 3
    max_samples_per_block: int = 2048

    def __post_init__(self):
        for name in ("duration_s", "max_edge_gap_s", "max_interval_s",
                     "max_spread_counts", "max_drift_counts", "max_reference_delta_counts"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"逐条基线参数 {name} 必须为有限正数")
        for name in ("min_samples", "contact_min_frames", "max_samples_per_block"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"逐条基线参数 {name} 必须为正整数")
        if not .5 <= self.duration_s <= 5 or self.max_edge_gap_s >= self.duration_s / 2:
            raise ValueError("逐条基线时长须为 0.5～5 秒，边缘空缺须小于半个窗口")
        if not 3 <= self.min_samples <= self.max_samples_per_block <= 8192:
            raise ValueError("逐条基线样本数须满足 3 ≤ 最少样本 ≤ 容量 ≤ 8192")


def load_trial_baseline_policy(path):
    path = Path(path)
    if not path.exists():
        return TrialBaselinePolicy()
    try:
        return TrialBaselinePolicy(**json.loads(path.read_text(encoding="utf-8")))
    except (TypeError, ValueError) as error:
        raise ValueError(f"逐条基线配置无效：{error}") from error


def validate_reference(blocks, calibration):
    """A complete startup reference is necessary to detect early contact."""
    if calibration is None:
        raise ValueError("请先保持无接触并完成连接标定，再开始逐条采集；归一化须开启")
    for key, block in blocks.items():
        mask = np.asarray(block.raw_valid_mask, dtype=bool)
        ref = np.asarray(calibration.baselines.get(key), dtype=float)
        if ref.shape != mask.shape or not mask.any() or not np.isfinite(ref[mask]).all():
            raise ValueError(f"皮肤块 {key} 的无接触标定未完成或没有有效点，请检查后重试")


class TrialBaseline:
    def __init__(self, blocks, reference, policy):
        validate_reference(blocks, reference)
        self.blocks, self.reference, self.policy = blocks, reference, policy
        self.rows = {key: [] for key in blocks}
        self.collecting = True
        self.error = ""
        self.result = None
        self.calibration = None
        self.event = Event()
        self.thread = Thread(target=self._run, name="skin-trial-baseline", daemon=True)
        self.thread.start()

    def observe(self, key, t, raw):
        if not self.collecting or not 0 <= t < self.policy.duration_s:
            return
        rows = self.rows[key]
        if len(rows) >= self.policy.max_samples_per_block:
            self.error = "基线缓存达到容量上限，请检查接收速率或调整配置"
            return
        rows.append((t, raw))

    def advance(self, duration):
        if self.collecting and duration >= self.policy.duration_s:
            self.collecting = False
            self.event.set()

    def finish(self, duration):
        """Called only by the background finalizer, after raw writer completion."""
        if self.collecting:
            if duration < self.policy.duration_s:
                self.error = "在逐条无接触基线采集完成前结束"
            self.collecting = False
            self.event.set()
        self.thread.join()
        return self.result

    def _run(self):
        self.event.wait()
        try:
            result, calibration = self._evaluate()
        except Exception as error:
            result = self._base_report()
            result.update(status="failed", reasons=[f"基线计算失败：{error}"])
            calibration = None
        self.calibration = calibration
        self.rows.clear()
        self.result = result  # Publication last; readers never see a partial report.

    def _base_report(self):
        return {"schema_version": "1.0", "method": "initial_raw_median_v1",
                "status": "collecting", "window_start_s": 0.,
                "window_end_s": self.policy.duration_s, "window_end_exclusive": True,
                "coordinates": "raw_row_col", "frozen_within_trial": True,
                "reference_calibration_id": self.reference.calibration_id,
                "policy": asdict(self.policy), "blocks": {}, "baselines": {}, "reasons": [],
                "contact_check": "stability_and_startup_reference_not_proof_of_no_contact"}

    def _evaluate(self):
        policy = self.policy
        report = self._base_report()
        failures = [self.error] if self.error else []
        baselines = {}
        for key, block in self.blocks.items():
            rows = self.rows[key]
            detail = {"samples": len(rows), "reasons": []}
            report["blocks"][str(key)] = detail
            reasons = detail["reasons"]
            if len(rows) < policy.min_samples:
                reasons.append(f"基线样本不足 {len(rows)}/{policy.min_samples}")
            if rows:
                times = np.array([row[0] for row in rows])
                gaps = np.diff(times)
                detail.update(first_time_s=float(times[0]), last_time_s=float(times[-1]),
                              max_gap_s=float(gaps.max()) if len(gaps) else None)
                if (times[0] > policy.max_edge_gap_s or
                        policy.duration_s - times[-1] > policy.max_edge_gap_s):
                    reasons.append("基线窗口首尾覆盖不足")
                if len(gaps) and (np.any(gaps <= 0) or gaps.max() > policy.max_interval_s):
                    reasons.append("基线时间重复、倒退或接收间断")
                mask = np.asarray(block.raw_valid_mask, dtype=bool)
                stack = np.stack([row[1] for row in rows]).astype(float)
                values = stack[:, mask]
                median = np.median(stack, axis=0)
                spread = float(np.max(np.percentile(values, 95, axis=0) - np.percentile(values, 5, axis=0)))
                n = max(1, len(rows) // 3)
                drift = float(np.max(np.abs(np.median(values[-n:], axis=0) - np.median(values[:n], axis=0))))
                reference = np.asarray(self.reference.baselines[key], dtype=float)[mask]
                ref_delta = np.abs(values - reference)
                contact_frames = int(np.count_nonzero(np.any(ref_delta >= policy.max_reference_delta_counts, axis=1)))
                median_delta = float(np.max(np.abs(median[mask] - reference)))
                detail.update(max_spread_counts=spread, max_drift_counts=drift,
                              max_median_reference_delta_counts=median_delta,
                              suspected_contact_frames=contact_frames)
                if spread > policy.max_spread_counts or drift > policy.max_drift_counts:
                    reasons.append("基线波动或漂移过大，请保持无接触")
                if contact_frames >= policy.contact_min_frames or median_delta >= policy.max_reference_delta_counts:
                    reasons.append("偏离连接基线，疑似提前接触；松开后仍异常请重新校准")
                median[~mask] = np.nan
                median.flags.writeable = False
                baselines[key] = median
                report["baselines"][str(key)] = [
                    [float(v) if np.isfinite(v) else None for v in row] for row in median]
            failures.extend(f"皮肤块 {key}：{reason}" for reason in reasons)
        report.update(status="ready" if not failures else "failed", reasons=failures)
        calibration = SimpleNamespace(baselines=baselines) if not failures else None
        return report, calibration
