"""单测：交付包预处理链路（M3 滤波 -> slow 基线 -> 归一化 A/B/none）。"""

import random

import numpy as np
import pytest

from skin_draw.preprocessing import BaselineTracker, BlockPreprocessor, PointFilter


def feed(pre, values, frames=1):
    """喂入同一帧值 ``frames`` 次，返回最后一次的输出矩阵。"""
    output = None
    for _ in range(frames):
        output = pre.update(values)
    return output


# ---------------------------------------------------------------- 交付包场景移植

def test_delivery_scenario_direction_and_scale():
    """场景1：空载 2500 -> 按压段 2250 -> 松开；B 模式方向、量程与滑窗滞后。"""
    rng = random.Random(0)
    pre = BlockPreprocessor((1, 1), [[True]], norm="B", calib_n=60, r_ref=500.)
    results = []
    for i in range(480):
        value = (2250. if 300 <= i < 360 else 2500.) + rng.uniform(-3, 3)
        results.append(pre.update([[value]])[0, 0])
    assert abs(results[299]) < 0.02          # 空载: B ~ 0 稳定底色
    assert 0.44 < results[350] < 0.56        # 按压尾部: d~250, r_ref=500 -> B~0.5
    assert abs(results[430]) < 0.02          # 滑窗排空后回到 0


def test_delivery_scenario_slow_self_heals_startup_convergence():
    """场景2：前 600 帧悬空 4070 后落到 2500，slow 慢跟踪自动校正。"""
    rng = random.Random(0)
    pre = BlockPreprocessor((1, 1), [[True]], norm="B", calib_n=270, r_ref=500.)
    tail = []
    for i in range(2400):
        value = (4070. if i < 600 else 2500.) + rng.uniform(-3, 3)
        out = pre.update([[value]])[0, 0]
        if np.isfinite(out) and i >= 2200:
            tail.append(abs(out))
    assert tail and sum(tail) / len(tail) < 0.08


def test_delivery_scenario_empty_points_and_norm_a():
    """场景3：A 模式 + 空点位恒为 NaN，按压为正偏差。"""
    pre = BlockPreprocessor((1, 2), [[True, False]], norm="A", calib_n=60)
    filled = feed(pre, [[2500., 999.]], frames=90)[0, 0]
    assert filled == pytest.approx(0., abs=0.5)      # 空载 A ~ 0
    out = feed(pre, [[2250., 999.]], frames=40)
    assert np.isnan(out[0, 1])                       # 空点位始终 NaN
    assert out[0, 0] > 200.                          # 按压偏差 ~250 LSB


def test_delivery_scenario_fps_time_scaling():
    """场景4：fps 与 calib_n 换算（376.7 fps -> upd_n = 377）。"""
    pre = BlockPreprocessor((1, 1), [[True]], norm="A", fps=376.7)
    assert pre.upd_n == 377
    fixed = BlockPreprocessor((1, 1), [[True]], fps=90.6)
    assert fixed.upd_n == 91


# ---------------------------------------------------------------------- M3 行为

def test_m3_rejects_low_rail_samples():
    pre = PointFilter()
    for _ in range(10):
        pre.update(2500)
    assert pre.update(1003) == pytest.approx(2500)   # 低电平被剔除，取高位均值


def test_m3_ignores_upward_spike():
    pre = PointFilter()
    for _ in range(5):
        pre.update(2500)
    assert pre.update(2900) == pytest.approx(2500)   # 尖峰抬高峰值，其余不足 n_min -> 保持


def test_m3_holds_without_min_keep():
    pre = PointFilter(n_min=3)
    assert pre.update(2000) is None
    assert pre.update(2000) is None
    assert pre.update(2000) == pytest.approx(2000)   # 第 3 帧暖机完成
    assert pre.update(1000) == pytest.approx(2000)   # 低电平剔除后 kept=3, 仍取高位


def test_m3_mean_skips_outliers_within_band():
    pre = PointFilter(w=10, delta=50, n_min=3)
    values = [2000, 2000, 2000, 2020, 1990, 2010, 2015, 2005, 1995, 2000]
    out = None
    for value in values:
        out = pre.update(value)
    assert out == pytest.approx(2003.5)              # 全部 >= 2020-50, 取全体均值


def test_point_filter_reset_clears_window():
    pre = PointFilter()
    for _ in range(5):
        pre.update(2500)
    pre.reset()
    assert pre.last is None and not pre.buf
    assert pre.update(1) is None


# -------------------------------------------------------------------- slow 基线

def test_slow_needs_full_calibration_before_baseline():
    tracker = BaselineTracker("slow", calib_n=270, upd_n=90, alpha=.2)
    for _ in range(269):
        assert tracker.update(2400) is None
    assert tracker.update(2400) == 2400
    assert tracker.count == 270 and tracker.ready


def test_slow_ema_steps_only_after_full_block():
    tracker = BaselineTracker("slow", calib_n=2, upd_n=4, alpha=.5)
    tracker.update(1000)
    tracker.update(1000)
    assert tracker.baseline == 1000
    for _ in range(3):
        tracker.update(800)
    assert tracker.baseline == 1000                  # 残量不足一个块，不更新
    tracker.update(800)                              # 第 4 个到齐: median=800
    assert tracker.baseline == pytest.approx(900.)   # 1000 + .5*(800-1000)


def test_slow_median_resists_short_press():
    tracker = BaselineTracker("slow", calib_n=1, upd_n=4, alpha=1.)
    tracker.update(1000)                             # 标定就绪
    tracker.update(500)                              # 短按一下
    for _ in range(3):
        tracker.update(1000)
    assert tracker.baseline == pytest.approx(1000.)  # 中位数仍是空载值


def test_calib_mode_freezes_after_initial_median():
    tracker = BaselineTracker("calib", calib_n=3, upd_n=2, alpha=.5)
    tracker.update(1000)
    tracker.update(2000)
    assert tracker.update(1500) == pytest.approx(1500.)
    assert tracker.update(900) == pytest.approx(1500.)  # calib 模式不再更新


# ----------------------------------------------------------------------- 归一化

def test_norm_a_clamps_negative_deviation():
    pre = BlockPreprocessor((1, 1), [[True]], norm="A", calib_n=1)
    assert feed(pre, [[2000.]], frames=35)[0, 0] == pytest.approx(0., abs=0.5)
    assert feed(pre, [[2100.]], frames=35)[0, 0] == 0.  # d 为负 -> 截断到 0


def test_norm_b_clips_to_unit_range():
    pre = BlockPreprocessor((1, 1), [[True]], norm="B", calib_n=1, r_ref=100.)
    assert feed(pre, [[2000.]], frames=35)[0, 0] == pytest.approx(0., abs=0.02)
    assert feed(pre, [[1950.]], frames=40)[0, 0] == pytest.approx(0.5)
    assert feed(pre, [[1900.]], frames=40)[0, 0] == pytest.approx(1.0)
    assert feed(pre, [[1850.]], frames=40)[0, 0] == pytest.approx(1.0)   # z>1 截断
    assert feed(pre, [[2050.]], frames=40)[0, 0] == pytest.approx(0.0)   # z<0 截断


def test_norm_none_passes_filtered_and_never_collects():
    pre = BlockPreprocessor((1, 1), [[True]], norm="none", calib_n=5)
    assert feed(pre, [[2400.]], frames=40)[0, 0] == pytest.approx(2400.)
    assert pre.ready and pre.progress == 100
    assert pre.trackers[0][0].baseline is None       # none 不收集基线
    assert pre.samples == 0


def test_not_ready_points_output_nan():
    pre = BlockPreprocessor((1, 1), [[True]], norm="B", calib_n=270)
    out = pre.update([[2000.]])
    assert np.isnan(out[0, 0])                       # M3 暖机期间
    assert np.isnan(feed(pre, [[2000.]], frames=29)[0, 0])   # 暖机结束, 标定未完成
    assert np.isnan(feed(pre, [[2000.]], frames=100)[0, 0])


def test_mask_empty_points_output_nan_every_mode():
    for norm in ("none", "A", "B"):
        pre = BlockPreprocessor((1, 3), [[True, False, True]], norm=norm, calib_n=1)
        out = feed(pre, [[2400., 999., 2600.]], frames=35)
        assert np.isnan(out[0, 1]), norm             # 空点位不处理
        assert out[0, 0] == pytest.approx(2400.) if norm == "none" else np.isfinite(out[0, 0])


def test_output_matrix_is_read_only():
    pre = BlockPreprocessor((1, 1), [[True]], norm="none")
    out = pre.update([[2400.]])
    assert not out.flags.writeable
    with pytest.raises(ValueError):
        out[0, 0] = 1.


# ------------------------------------------------------------------ 进度与重算

def test_progress_counts_calibration_samples():
    pre = BlockPreprocessor((1, 2), [[True, True]], norm="B", calib_n=4)
    feed(pre, [[2000., 2000.]], frames=29)           # 暖机 29 帧
    assert pre.progress == 0 and pre.samples == 0
    feed(pre, [[2000., 2000.]], frames=1)            # 第 30 帧起进入标定
    assert pre.progress == 25 and pre.samples == 1
    feed(pre, [[2000., 2000.]], frames=1)
    assert pre.progress == 50 and not pre.ready
    feed(pre, [[2000., 2000.]], frames=2)
    assert pre.progress == 100 and pre.ready


def test_set_norm_recomputes_from_cached_state():
    pre = BlockPreprocessor((1, 1), [[True]], norm="A", calib_n=1, r_ref=100.)
    feed(pre, [[2000.]], frames=35)
    assert feed(pre, [[1900.]], frames=40)[0, 0] == pytest.approx(100.)   # A: d=100 LSB
    pre.set_norm("B")
    assert pre.recompose()[0, 0] == pytest.approx(1.0)   # 缓存 vf/baseline 立即重算
    pre.set_norm("none")
    assert pre.recompose()[0, 0] == pytest.approx(1900.)


def test_set_direction_recomputes_immediately():
    pre = BlockPreprocessor((1, 1), [[True]], norm="A", calib_n=1)
    feed(pre, [[2000.]], frames=35)
    assert feed(pre, [[1900.]], frames=40)[0, 0] == pytest.approx(100.)
    pre.set_direction("rise")
    assert pre.recompose()[0, 0] == 0.               # rise 下 d 为负 -> 截断


def test_reset_clears_filters_and_baselines():
    pre = BlockPreprocessor((1, 1), [[True]], norm="B", calib_n=1)
    feed(pre, [[2000.]], frames=35)
    assert pre.ready
    pre.reset()
    assert not pre.ready and pre.progress == 0 and pre.samples == 0
    assert np.isnan(pre.update([[2000.]])[0, 0])     # 需要重新暖机与标定
    assert np.isnan(pre.baselines[0, 0])


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        BlockPreprocessor((1, 1), [[True]], norm="C")
    with pytest.raises(ValueError):
        BlockPreprocessor((1, 1), [[True]], direction="up")
    with pytest.raises(ValueError):
        BlockPreprocessor((1, 2), [[True]])
    pre = BlockPreprocessor((1, 1), [[True]])
    with pytest.raises(ValueError):
        pre.update([[1., 2.]])
    with pytest.raises(ValueError):
        pre.set_norm("D")
    with pytest.raises(ValueError):
        pre.set_direction("down")
