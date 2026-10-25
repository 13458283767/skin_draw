"""接收完整性检查：只使用主机接收时间，不解释 ADC 大小或推算硬件丢帧数。"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


QUALITY_VERSION = "receive_integrity_v1"
LEVELS = {"pass": 0, "warn": 1, "fail": 2}
QUALITY_NAMES = {"pass": "通过完整性检查", "warn": "待复核", "fail": "不合格", "pending": "检查中"}


@dataclass(frozen=True)
class QualityPolicy:
    """试运行规则；每段开始时冻结并随元数据保存，允许通过配置调整。"""

    warning_floor_s: float = .2
    failure_floor_s: float = 1.
    warning_multiplier: float = 5.
    failure_multiplier: float = 20.
    min_reference_intervals: int = 10
    reference_window_s: float = 5.
    max_events: int = 100

    def __post_init__(self):
        for name in ("warning_floor_s", "failure_floor_s", "warning_multiplier", "failure_multiplier", "reference_window_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"完整性规则 {name} 必须为有限正数")
        if self.failure_floor_s < self.warning_floor_s or self.failure_multiplier < self.warning_multiplier:
            raise ValueError("严重异常阈值不能小于提醒阈值")
        for name in ("min_reference_intervals", "max_events"):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= 10000:
                raise ValueError(f"完整性规则 {name} 必须为 1～10000 的整数")
        if self.min_reference_intervals > 128:
            raise ValueError("接收参考样本数不能超过 128")


def load_quality_policy(path):
    """每次开始前读取一次；缺少配置用默认值，坏配置不能悄悄回退。"""
    path = Path(path)
    if not path.exists():
        return QualityPolicy()
    try:
        return QualityPolicy(**json.loads(path.read_text(encoding="utf-8")))
    except (TypeError, ValueError) as error:
        raise ValueError(f"采集完整性配置无效：{error}") from error


class IntegrityMonitor:
    """由录制器在同一把锁内调用；实时检查不修改计数，避免刷新重复累计异常。"""

    def __init__(self, ids, policy, references=()):
        self.policy = policy
        refs = {key: (interval, count) for key, interval, count in references}
        self.blocks = {}
        self.events = []
        self.event_count = 0
        for key in ids:
            interval, count = refs.get(key, (None, 0))
            if interval is not None and (not math.isfinite(interval) or interval <= 0):
                raise ValueError("接收间隔参考必须为有限正数")
            reliable = interval is not None and count >= policy.min_reference_intervals
            # 参考不足时仍可保留原始录制，采用固定下限并在结束时要求复核。
            used = interval if reliable else 0.
            self.blocks[key] = {
                "reference_interval_s": interval, "reference_count": count,
                "reference_source": "pre_recording_median" if reliable else "fallback_insufficient_or_unstable",
                "warning_threshold_s": max(policy.warning_floor_s, policy.warning_multiplier * used),
                "failure_threshold_s": max(policy.failure_floor_s, policy.failure_multiplier * used),
                "frames": 0, "first_time_s": None, "last_time_s": None,
                "max_interval_s": 0., "warning_count": 0, "failure_count": 0,
            }

    @staticmethod
    def _gap(key, block, kind, start, end):
        duration = max(0., end - start)
        # 边界相等仍算正常；小容差避免浮点累计将恰好等于阈值的间隔误判。
        if duration > block["failure_threshold_s"] + 1e-9:
            level = "fail"
        elif duration > block["warning_threshold_s"] + 1e-9:
            level = "warn"
        else:
            return None
        return {"block_id": key, "kind": kind, "level": level,
                "start_s": round(start, 6), "end_s": round(end, 6), "duration_s": round(duration, 6)}

    def receive(self, key, time_s):
        block = self.blocks[key]
        previous = block["last_time_s"]
        kind = "head_gap" if previous is None else "interval_gap"
        event = self._gap(key, block, kind, 0. if previous is None else previous, time_s)
        if event:
            block["failure_count" if event["level"] == "fail" else "warning_count"] += 1
            self.event_count += 1
            if len(self.events) < self.policy.max_events:
                self.events.append(event)
        if previous is None:
            block["first_time_s"] = time_s
        else:
            block["max_interval_s"] = max(block["max_interval_s"], time_s - previous)
        block["last_time_s"] = time_s
        block["frames"] += 1

    def evaluate(self, duration, *, final=False, failures=()):
        events = [dict(event) for event in self.events]
        total = self.event_count
        reports = {}
        reasons = list(failures)
        overall = "fail" if reasons else "pass"
        for key, original in self.blocks.items():
            block = dict(original)
            first, last = block["first_time_s"], block["last_time_s"]
            block["head_gap_s"] = max(0., duration if first is None else first)
            block["tail_gap_s"] = None if last is None else max(0., duration - last)
            current = self._gap(key, block, "no_data" if last is None else ("tail_gap" if final else "ongoing_gap"),
                                0. if last is None else last, duration)
            if current:
                total += 1
                block["failure_count" if current["level"] == "fail" else "warning_count"] += 1
                if len(events) < self.policy.max_events:
                    events.append(current)
            level = "fail" if block["failure_count"] else ("warn" if block["warning_count"] else "pass")
            notes = []
            if block["failure_count"]:
                notes.append(f"{block['failure_count']} 次严重接收空缺")
            elif block["warning_count"]:
                notes.append(f"{block['warning_count']} 次接收间隔异常")
            if current:
                notes.append(f"{'等待首帧' if last is None else '未更新'} {current['duration_s']:.2f} 秒")
            if final and block["frames"] < 2:
                level = "fail"
                notes.append("整段未收到数据" if not block["frames"] else "整段仅收到一帧")
            if final and block["reference_source"] != "pre_recording_median":
                if level == "pass":
                    level = "warn"
                notes.append("接收速度参考不足或不稳定，采用固定阈值")
            block["status"] = level
            block["message"] = "；".join(notes) or ("正常" if block["frames"] else "等待首帧")
            if level != "pass":
                reasons.append(f"BLOCK {key:02d}：{block['message']}")
            if LEVELS[level] > LEVELS[overall]:
                overall = level
            reports[str(key)] = block
        return {"status": overall, "final": final, "rule_version": QUALITY_VERSION,
                "policy": asdict(self.policy), "blocks": reports, "reasons": reasons,
                "events": events, "event_count": total,
                "events_truncated": total > len(events)}
