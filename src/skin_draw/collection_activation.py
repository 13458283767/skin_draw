"""接触激活摘要：用标定基线量出每块的最大响应，给出主接触块与区域集合。

只做可复算的数值摘要（不判断动作是否规范）；是否合规由 ``semantic_status``
按动作类型给出，两者都随 meta.json 一起保存，便于事后复核与训练筛选。

背景：这批数据要论证"动作识别与具体接触哪几块无关"，就必须从第一条样本起
记录 ``primary_block`` / ``region_set``，事后无法补。
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np


ACTIVATION_VERSION = "contact_activation_v1"


@dataclass(frozen=True)
class ActivationPolicy:
    """每段录制开始时冻结，并随 meta.json 保存；允许通过配置调整。"""

    # 单点相对标定基线的下降量超过该值，才算该块"被接触"。
    threshold_counts: float = 100.
    # 多区域动作（拥抱）至少要同时接触的区域数。
    hug_min_regions: int = 3
    # 单区域动作允许的主接触块切换次数。
    max_block_switches: int = 0
    multi_region_actions: tuple[str, ...] = ("hug",)

    def __post_init__(self):
        if (type(self.threshold_counts) not in (int, float)
                or not math.isfinite(self.threshold_counts) or self.threshold_counts <= 0):
            raise ValueError("接触判定阈值必须为有限正数")
        for name in ("hug_min_regions", "max_block_switches"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 32:
                raise ValueError(f"接触规则 {name} 必须为 0～32 的整数")
        actions = tuple(self.multi_region_actions)
        if any(not isinstance(name, str) or not name for name in actions):
            raise ValueError("多区域动作名称必须是非空字符串")
        object.__setattr__(self, "multi_region_actions", actions)

    def multi_region(self, action_name: str) -> bool:
        return action_name in self.multi_region_actions


def load_activation_policy(path) -> ActivationPolicy:
    """每次开始前读取一次；缺少配置用默认值，坏配置不能悄悄回退。"""
    path = Path(path)
    if not path.exists():
        return ActivationPolicy()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "multi_region_actions" in data:
            data["multi_region_actions"] = tuple(data["multi_region_actions"])
        return ActivationPolicy(**data)
    except (TypeError, ValueError) as error:
        raise ValueError(f"接触判定配置无效：{error}") from error


class ActivationTracker:
    """由录制器在同一把锁内逐帧调用；只累加每块最大响应，不做显示滤波。"""

    def __init__(self, blocks, calibration, policy: ActivationPolicy):
        self.policy = policy
        self._baseline: dict[int, np.ndarray] = {}
        self._mask: dict[int, np.ndarray] = {}
        self._max_delta: dict[int, float] = {}
        self._frames: dict[int, int] = {}
        self._switches = 0
        self._frames_evaluated = 0
        self._previous_block: int | None = None
        self._current: dict[int, float] = {}
        self.ready = bool(blocks)
        for key, block in blocks.items():
            mask = np.asarray(block.transform(np.asarray(block.raw_valid_mask)), dtype=bool).ravel()
            baseline = None
            # 内存中的快照用整数编号，JSON 里是字符串；两处都要兼容。
            raw = None
            if calibration is not None:
                raw = calibration.baselines.get(key)
                if raw is None:
                    raw = calibration.baselines.get(str(key))
            if raw is not None:
                array = np.asarray(raw, dtype=float)
                if array.shape == (block.spec.rows, block.spec.cols):
                    baseline = np.asarray(block.transform(array), dtype=float).ravel()
            if baseline is None or baseline.size != mask.size or not mask.any():
                self.ready = False
            self._baseline[key] = baseline
            self._mask[key] = mask
            self._max_delta[key] = 0.
            self._frames[key] = 0
            self._current[key] = 0.

    def observe(self, key, values):
        """记录一帧。``values`` 必须已是保存方向的行优先展开值。"""
        if not self.ready:
            return
        baseline, mask = self._baseline.get(key), self._mask.get(key)
        if baseline is None or mask is None:
            return
        frame = np.asarray(values, dtype=float).ravel()
        if frame.size != baseline.size:
            return
        # 无效点位不参与判定；用 -inf 屏蔽，避免把 0/1 的悬空读数当成响应。
        delta = np.where(mask, baseline - frame, -np.inf)
        peak = float(np.max(delta))
        if not math.isfinite(peak):
            return
        self._frames[key] += 1
        self._frames_evaluated += 1
        if peak > self._max_delta[key]:
            self._max_delta[key] = peak
        self._current[key] = peak
        strongest = max(self._current, key=lambda item: self._current[item])
        if self._current[strongest] < self.policy.threshold_counts:
            self._previous_block = None
            return
        if self._previous_block is not None and strongest != self._previous_block:
            self._switches += 1
        self._previous_block = strongest

    def summary(self) -> dict:
        regions = [key for key in sorted(self._max_delta)
                   if self._max_delta[key] >= self.policy.threshold_counts]
        primary = max(self._max_delta, key=lambda item: self._max_delta[item]) if regions else None
        return {
            "method": ACTIVATION_VERSION,
            "available": self.ready,
            "threshold_counts": self.policy.threshold_counts,
            "primary_block": primary,
            "region_ids": regions,
            "region_set": "".join(str(key) for key in regions) if self.ready else None,
            "region_count": len(regions),
            "max_delta_counts": {str(key): round(value, 1)
                                 for key, value in sorted(self._max_delta.items())},
            "block_switches": self._switches,
            "frames_evaluated": self._frames_evaluated,
        }


def semantic_status(activation: dict, action_name: str, policy: ActivationPolicy) -> dict:
    """按动作类型检查接触是否合乎采集规程；不知道就返回 unknown，绝不当成合格。"""
    if not activation.get("available"):
        return {"status": "unknown", "rule_version": ACTIVATION_VERSION,
                "reasons": [], "policy": asdict(policy)}
    reasons = []
    count = int(activation.get("region_count", 0))
    if count == 0:
        reasons.append("未检测到接触：所有块都低于判定阈值")
    elif policy.multi_region(action_name):
        if count < policy.hug_min_regions:
            reasons.append(f"多区域动作只检测到 {count} 个区域，少于 {policy.hug_min_regions}")
    else:
        switches = int(activation.get("block_switches", 0))
        if switches > policy.max_block_switches:
            reasons.append(f"单区域动作中途换块 {switches} 次")
    return {"status": "warn" if reasons else "pass", "rule_version": ACTIVATION_VERSION,
            "reasons": reasons, "policy": asdict(policy)}
