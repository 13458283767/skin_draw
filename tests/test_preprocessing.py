"""单测：M3 滤波 -> 固定基线 -> 归一化 A/B/none。"""

import random

import numpy as np
import pytest

from skin_draw.preprocessing import BaselineGate, BaselineTracker, BlockPreprocessor, PointFilter


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


@pytest.mark.parametrize("filter_enabled", [False, True])
@pytest.mark.parametrize("norm,expected", [("A", 300.), ("B", .5)])
@pytest.mark.parametrize("direction,pressed", [("drop", 2200.), ("rise", 2800.)])
def test_sustained_pressure_keeps_baseline_and_response(filter_enabled, norm, expected, direction, pressed):
    pre = BlockPreprocessor((1, 1), [[True]], norm=norm, direction=direction,
                            filter_enabled=filter_enabled)
    feed(pre, [[2500.]], frames=300)
    # Beyond the old slow tracker's settling time; every settled frame must hold.
    feed(pre, [[pressed]], frames=30)
    for _ in range(2700):
        assert pre.update([[pressed]])[0, 0] == pytest.approx(expected)
        assert pre.baselines[0, 0] == 2500.
    assert feed(pre, [[2500.]], frames=35)[0, 0] == pytest.approx(0.)


def test_delivery_scenario_empty_points_and_norm_a():
    """场景3：A 模式 + 空点位恒为 NaN，按压为正偏差。"""
    pre = BlockPreprocessor((1, 2), [[True, False]], norm="A", calib_n=60)
    filled = feed(pre, [[2500., 999.]], frames=90)[0, 0]
    assert filled == pytest.approx(0., abs=0.5)      # 空载 A ~ 0
    out = feed(pre, [[2250., 999.]], frames=40)
    assert np.isnan(out[0, 1])                       # 空点位始终 NaN
    assert out[0, 0] > 200.                          # 按压偏差 ~250 LSB


def test_new_zero_requires_explicit_reset():
    pre = BlockPreprocessor((1, 1), [[True]], norm="A", filter_enabled=False, calib_n=3)
    feed(pre, [[2500.]], frames=3)
    assert feed(pre, [[2200.]], frames=1000)[0, 0] == 300.
    pre.reset()
    assert not pre.ready
    assert np.isnan(feed(pre, [[2200.]], frames=2)[0, 0])
    assert pre.update([[2200.]])[0, 0] == 0.
    assert pre.baselines[0, 0] == 2200.


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


# -------------------------------------------------------------------- 固定基线

def test_needs_full_calibration_before_baseline():
    tracker = BaselineTracker(calib_n=270)
    for _ in range(269):
        assert tracker.update(2400) is None
    assert tracker.update(2400) == 2400
    assert tracker.count == 270 and tracker.ready


def test_fixed_baseline_survives_long_press_and_release():
    tracker = BaselineTracker(calib_n=2)
    tracker.update(1000)
    tracker.update(1000)
    assert tracker.baseline == 1000
    for value in [800] * 3000 + [1000] * 3000:
        assert tracker.update(value) == 1000


def test_fixed_baseline_resists_short_press():
    tracker = BaselineTracker(calib_n=1)
    tracker.update(1000)                             # 标定就绪
    tracker.update(500)                              # 短按一下
    for _ in range(3):
        tracker.update(1000)
    assert tracker.baseline == pytest.approx(1000.)  # 中位数仍是空载值


def test_calib_mode_freezes_after_initial_median():
    tracker = BaselineTracker(calib_n=3)
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


def feed_tracker(tracker, values):
    """依次喂入一窗标定样本，返回最后一次 update 的结果。"""
    baseline = None
    for value in values:
        baseline = tracker.update(value)
    return baseline


def test_gate_accepts_stable_window():
    tracker = BaselineTracker(calib_n=40)
    assert feed_tracker(tracker, [2400 + (index % 5) - 2 for index in range(40)]) == pytest.approx(2400)
    assert tracker.ready and not tracker.last_issue


def test_gate_rejects_window_with_contact():
    # 一半样本是按压态：波动远超阈值，不能固化成参考基线
    tracker = BaselineTracker(calib_n=40)
    assert feed_tracker(tracker, [2400] * 20 + [2000] * 20) is None
    assert not tracker.ready and tracker.rejections == 1
    assert "波动" in tracker.last_issue


def test_gate_rejects_slow_drift():
    # 波动只有 22 LSB（低于 30），但首尾中位数相差 22 LSB：说明在缓慢滑动
    tracker = BaselineTracker(calib_n=40)
    values = [2400] * 14 + [2410] * 13 + [2422] * 13
    assert feed_tracker(tracker, values) is None
    assert not tracker.ready and "漂移" in tracker.last_issue


def test_gate_warns_when_baseline_jumps_after_recalibration():
    tracker = BaselineTracker(calib_n=40)
    assert feed_tracker(tracker, [2400] * 40) == pytest.approx(2400)
    tracker.reset(keep_reference=True)          # 用户主动重标定：保留上一代基线
    assert tracker.previous == pytest.approx(2400)
    # 贴附变化会导致基线大幅跳变：只提醒不阻断，否则每次重新贴附都要点两次
    assert feed_tracker(tracker, [2200] * 40) == pytest.approx(2200)
    assert "偏离上一代" in tracker.last_warning
    assert not tracker.last_issue


def test_gate_blocks_after_repeated_rejections():
    tracker = BaselineTracker(calib_n=40, gate=BaselineGate(max_attempts=2))
    for _ in range(2):
        feed_tracker(tracker, [2400] * 20 + [2000] * 20)
    assert tracker.blocked and not tracker.ready
    # 阻断后不再自动重试：即使之后数据正常也不再收集
    assert feed_tracker(tracker, [2400] * 40) is None
    assert tracker.count == 0


def test_blocked_tracker_accepts_after_manual_reset():
    tracker = BaselineTracker(calib_n=40, gate=BaselineGate(max_attempts=2))
    for _ in range(2):
        feed_tracker(tracker, [2400] * 20 + [2000] * 20)      # 窗口内像在接触 -> 连续被拒
    assert tracker.blocked
    tracker.reset(keep_reference=True)                        # 阻断后重置视为强制重来
    assert tracker.previous is None and not tracker.blocked
    assert feed_tracker(tracker, [2200] * 40) == pytest.approx(2200)


def test_block_preprocessor_reports_gate_issue():
    # 关掉 M3 可以跳过暖机，让标定窗口直接吃到原始阶跃
    pre = BlockPreprocessor((1, 1), [[True]], norm="A", calib_n=40, filter_enabled=False)
    for _ in range(20):
        pre.update([[2400.]])
    for _ in range(20):
        pre.update([[2000.]])
    assert not pre.ready
    assert "行1列1" in pre.baseline_issue
    assert not pre.baseline_blocked                # 只拒一次，还在自动重试
    assert np.isnan(pre.recompose()[0, 0])         # 基线未固定，输出仍是缺失
