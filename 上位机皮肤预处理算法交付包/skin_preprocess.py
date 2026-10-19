#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上位机皮肤数据预处理模块（热力图实时显示用，纯 Python 无三方依赖）

流水线:  空点位掩膜 -> 单点位滤波(M3 高位滑窗均值) -> 归一化(none/A/B)

三个可复用类:
  PointFilter     单点位在线滤波（M3：窗口内取 >=max-150 的样本求均值，不足 3 个时保持）
  BaselineTracker 基线估计（calib=启动标定 / slow=标定+慢跟踪）
  SkinPreprocessor 整板预处理器（组织以上两者 + 归一化 A/B）

典型用法（上位机每收到一帧调用一次）::

    pre = SkinPreprocessor(rows=5, cols=3, empty={2, 3},
                           norm="B", baseline_mode="slow")   # 推荐配置
    for raw in frame_stream:                 # raw: 长度 rows*cols 的原始值列表
        out = pre.update_frame(raw)          # out: 同长度列表, float 或 None
        heatmap.render(out)                  # None=空点位/未就绪, 渲染为空白

启动时间线 (整板 90.6fps 时, calib_n=270 / upd_n=90):
  帧 0 ~ w-1      滤波暖机 (滑窗未满, 不参与基线收集)
  帧 w-1 起       基线标定收集 (calib_n 个样本, ≈3s), 期间归一化输出 None
  标定完成后      归一化输出上线; slow 模式之后每 upd_n 个样本用中位数
                  做 EMA 慢跟踪 (alpha=0.2)

【帧率换算】upd_n 表达的是慢跟踪块时长而非固定帧数: 90 帧在 90.6fps 下
≈1s, 高帧率下会过短 (单皮肤 376.7fps 时仅 0.24s, 持续按压会被过快跟踪
进基线)。构造时传入实测帧率 fps, 内部按 upd_n=round(fps*upd_sec) 自动
换算 (见 SkinPreprocessor 的 fps/upd_sec 参数)。calib_n 为样本数语义
(270 个样本取中位数), 与帧率无关, 无需换算。

【重要】部分皮肤块存在"上电悬空平台"现象: 上电后前 ~6.6s 输出悬空电平
(~4070), 之后渐降至正常平台 (~2500), 收敛历时 0.7s~30s 不等。固定标定
(calib) 会永久错位; slow 慢跟踪可自动校正 (实测见 _preprocess_eval.txt)。

归一化语义 (压力正向化; 物理前提: 按压使读数下降, 即磁铁靠近霍尔 -> 读数走低,
由"拆磁铁实验: 平台 2481 -> 2731 (+250)"推断; 若实测硬件方向相反, 改 direction="rise"):
  norm="none": 输出滤波原值 (LSB)
  norm="A"   : 输出 d = baseline_i - v_filt (压力偏差, LSB; 空载≈0, 按压>0)
  norm="B"   : 输出 clip(d / r_ref, 0, 1) (空载≈0 形成稳定底色, 按压量程按 r_ref 归一)

基线未就绪期间 (前 calib_n 个有效样本) 归一化输出 None。
"""

from collections import deque


def _median(x):
    s = sorted(x)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


class PointFilter:
    """单点位在线滤波（M3 高位滑窗均值）。

    窗口 = 最近 w 个原始样本; 取窗口内 >= (max-delta) 的样本求均值;
    有效样本 < n_min 时输出上次结果（无历史则为 None）。
    """

    def __init__(self, w=30, delta=150, n_min=3):
        self.w = int(w)
        self.delta = float(delta)
        self.n_min = int(n_min)
        self.buf = deque(maxlen=self.w)
        self.last = None

    def update(self, v):
        """输入 1 个原始样本, 返回滤波输出(float) 或 None(未就绪)。"""
        v = float(v)
        self.buf.append(v)
        th = max(self.buf) - self.delta
        kept = [x for x in self.buf if x >= th]
        if len(kept) >= self.n_min:
            self.last = sum(kept) / len(kept)
        return self.last

    def reset(self):
        self.buf.clear()
        self.last = None


class BaselineTracker:
    """无压力基线估计。

    mode='calib': 前 calib_n 个有效滤波输出取中位数作为基线, 之后固定。
    mode='slow' : calib 就绪后, 每积累 upd_n 个新样本用其中位数做 EMA 慢跟踪
                  (alpha=0.2 时时间常数约 5 次更新); 适用于装配变化/温漂场景,
                  要求数据大部分时间处于空载（中位数抗短时按压）。
                  【推荐】热力图实时显示用 slow: 可自愈启动收敛/基线漂移。
    """

    def __init__(self, mode="calib", calib_n=270, upd_n=90, alpha=0.2):
        self.mode = mode
        self.calib_n = int(calib_n)
        self.upd_n = int(upd_n)
        self.alpha = float(alpha)
        self.baseline = None
        self._calib = []
        self._recent = deque(maxlen=int(upd_n))

    def update(self, v):
        """输入 1 个有效滤波输出, 返回当前基线(float) 或 None(未就绪)。"""
        if v is None:
            return self.baseline
        v = float(v)
        if self.baseline is None:
            self._calib.append(v)
            if len(self._calib) >= self.calib_n:
                self.baseline = _median(self._calib)
            return self.baseline
        if self.mode == "slow":
            self._recent.append(v)
            if len(self._recent) >= self._recent.maxlen:
                med = _median(self._recent)
                self.baseline += self.alpha * (med - self.baseline)
                self._recent.clear()
        return self.baseline


class SkinPreprocessor:
    """整板/整块皮肤预处理器。

    rows, cols : 皮肤规格
    empty      : 空点位集合（1-based 点位号, 行优先）, 输出恒为 None
    norm       : "none" / "A" / "B"
    direction  : "drop"(默认, 按压使读数下降) / "rise"(按压使读数上升)
    baseline_mode : "slow"(默认, 推荐) / "calib"
    r_ref      : 方案 B 映射量程（LSB, 对应输出 1.0; 默认 600, 按压标定数据实测:
                 逐个按到底峰值 P90≈573 LSB, 向上圆整取 600; 整板多块皮肤建议复核）
    fps        : 实测帧率 (Hz); 传入时按 upd_n=round(fps*upd_sec) 换算慢跟踪块长
    upd_sec    : 慢跟踪块时长 (s), 默认 1.0 (仅当 fps 传入时生效)
    """

    def __init__(self, rows, cols, empty=(),
                 w=30, delta=150, n_min=3,
                 norm="A", baseline_mode="slow", direction="drop",
                 calib_n=270, upd_n=90, alpha=0.2, r_ref=600.0,
                 fps=None, upd_sec=1.0):
        self.rows = int(rows)
        self.cols = int(cols)
        self.npts = self.rows * self.cols
        self.empty = set(empty)
        self.norm = norm
        self.direction = direction
        self.r_ref = float(r_ref)
        self.w = int(w)
        if fps is not None and float(fps) > 0:
            upd_n = max(1, int(round(float(fps) * float(upd_sec))))
        self._nf = 0
        self.filt = [PointFilter(w, delta, n_min) for _ in range(self.npts)]
        self.base = [BaselineTracker(baseline_mode, calib_n, upd_n, alpha)
                     for _ in range(self.npts)]

    def update_frame(self, raw):
        """输入一帧原始值（list/tuple, 长度 rows*cols, 行优先）,
        返回同长度列表: float(归一化/滤波输出) 或 None(空点位或未就绪)。"""
        if len(raw) != self.npts:
            raise ValueError("frame length %d != %d" % (len(raw), self.npts))
        self._nf += 1
        out = [None] * self.npts
        for k in range(self.npts):
            if (k + 1) in self.empty:
                continue
            vf = self.filt[k].update(raw[k])
            if vf is None:
                continue
            if self.norm == "none":
                out[k] = vf
                continue
            if self._nf < self.w:
                continue                       # 滤波窗口未满: 不参与基线收集
            b = self.base[k].update(vf)
            if b is None:
                continue
            # 压力正向化: drop=读数下降为压力
            d = (b - vf) if self.direction == "drop" else (vf - b)
            if self.norm == "A":
                out[k] = d
            else:  # "B"
                z = d / self.r_ref
                out[k] = 0.0 if z < 0.0 else (1.0 if z > 1.0 else z)
        return out

    def reset(self):
        self._nf = 0
        for f in self.filt:
            f.reset()
        for b in self.base:
            b.baseline = None
            b._calib = []
            b._recent.clear()


if __name__ == "__main__":
    # 场景1: 空载 2500 -> 按压段 2250 -> 松开; 验证方向、量程与滑窗滞后
    import random
    random.seed(0)
    pre = SkinPreprocessor(1, 1, norm="B", calib_n=60, r_ref=500.0)
    for i in range(480):
        v = (2250.0 if 300 <= i < 360 else 2500.0) + random.uniform(-3, 3)
        b = pre.update_frame([v])[0]
        if i == 299:
            assert abs(b) < 0.02, b            # 空载: B ~ 0 (稳定底色)
        if i == 350:
            assert 0.44 < b < 0.56, b          # 按压尾部: d~250, r_ref=500 -> B~0.5
        if i == 430:
            assert abs(b) < 0.02, b            # 滑窗排空后: 回到 0
    # 场景2: 启动收敛 (前 600 帧悬空 4070, 之后落到 2500) -> slow 自愈
    pre2 = SkinPreprocessor(1, 1, norm="B", baseline_mode="slow", r_ref=500.0)
    tail = []
    for i in range(2400):
        v = (4070.0 if i < 600 else 2500.0) + random.uniform(-3, 3)
        b = pre2.update_frame([v])[0]
        if b is not None and i >= 2200:
            tail.append(abs(b))
    avg = sum(tail) / len(tail)
    assert avg < 0.08, avg                      # 慢跟踪把悬空基线拉回, 伪热消除
    # 场景3: A 模式 + 空点位接口
    pre3 = SkinPreprocessor(1, 2, empty={2}, norm="A", calib_n=60)
    for i in range(200):
        o = pre3.update_frame([2500.0, 999.0])
        assert o[1] is None
    # 场景4: fps 时间换算 (376.7fps -> upd_n = round(376.7*1.0) = 377)
    pre4 = SkinPreprocessor(1, 1, norm="A", baseline_mode="slow", fps=376.7)
    assert pre4.base[0].upd_n == 377, pre4.base[0].upd_n
    print("smoke test passed: direction/scale ok, slow self-heal ok, empty-point ok, "
          "fps-scale ok")
