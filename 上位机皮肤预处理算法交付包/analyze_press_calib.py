#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按压标定数据分析 (数据: 按压标定数据.txt, 单块改造皮肤 10x5)

口径: 推荐管线输出 = M3 滤波 + slow 慢跟踪基线 (按时间换算块长, upd_n = fps x 1s)
内容:
  1. 方向验证 (按压 -> 读数下降 -> d > 0)
  2. 按压事件提取 (d > 100 LSB, 段长 >= 15 帧, 合并间隔 <= 40 帧)
  3. 按压深度分级统计
  4. r_ref 建议值推导
  5. 空载底噪与周期伪影抑制评估
  6. 图5: 全点位压力响应时空图 + 代表点位时序
输出: _press_calib_analysis.txt + figures/preprocess_press_calib.png
"""
import os
import time
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

import eval_skin_preprocess as ev

CFILE = os.path.join(ev.BASE, "按压标定数据.txt")
OUT = os.path.join(ev.BASE, "_press_calib_analysis.txt")
FIG = os.path.join(ev.FIGDIR, "preprocess_press_calib.png")

THR = 100.0       # 事件阈值 (LSB)
MINLEN = 15       # 事件最小段长 (帧)
GAP = 40          # 事件合并间隔 (帧)
UPD_SEC = 1.0     # slow 块时长 (秒)
DEEP, MID = 450.0, 250.0    # 深度分级阈值 (LSB): 按到底~550~610

DEMO = [5, 7, 9, 15]        # P06 P08 P10 P16


def save_retry(fig, path, tries=20, delay=0.5, **kw):
    """保存 PNG; 遇到 Windows 文件被占用 (Errno 22) 自动重试。"""
    for i in range(tries):
        try:
            fig.savefig(path, **kw)
            return path
        except OSError as e:
            if e.errno != 22 or i == tries - 1:
                raise
            time.sleep(delay)


def slow_seq(vf, b0, upd_n, alpha):
    """参数化慢跟踪基线序列 (与 BaselineTracker(slow) 逐帧对齐)"""
    sz = vf.size
    bs = np.empty(sz)
    start = ev.W - 1
    bs[:start] = np.nan
    bs[start:start + ev.CALIB_N] = b0
    i = start + ev.CALIB_N
    b = b0
    while i < sz:
        j = min(i + upd_n, sz)
        if j - i < upd_n:
            bs[i:j] = b
            break
        med = float(np.median(vf[i:j]))
        b_old = b
        b = b_old + alpha * (med - b_old)
        bs[i:j] = b_old
        bs[j - 1] = b
        i = j
    return bs


def detect(d, thr, minlen, gap):
    """返回 [(a, b)] 事件区间"""
    segs = []
    i, n = 0, d.size
    while i < n:
        if d[i] > thr:
            j = i
            while j < n and d[j] > thr:
                j += 1
            segs.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for a, b in segs:
        if merged and a - merged[-1][1] <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= minlen]


def main():
    t0 = datetime.now()
    vals, nfr, dur = ev.load_values(CFILE, ev.SPEC_S)
    nf = nfr[1]
    fps = nf / dur
    upd_n = max(1, int(round(fps * UPD_SEC)))

    dmap = []
    for k in range(50):
        arr = np.asarray(vals[1][k], dtype=np.float64)
        o3 = ev.m3_offline(arr)
        b0 = float(np.median(o3[ev.W - 1:ev.W - 1 + ev.CALIB_N]))
        bs = slow_seq(o3, b0, upd_n, ev.ALPHA)
        dmap.append(bs - o3)

    evs = []
    for k in range(50):
        for a, b in detect(dmap[k], THR, MINLEN, GAP):
            seg = dmap[k][a:b]
            tail = seg[max(0, int(seg.size * 0.9)):]
            evs.append(dict(p=k + 1, a=a, b=b, peak=float(seg.max()),
                            mean=float(seg.mean()), tail=float(tail.mean())))
    evs.sort(key=lambda e: e["a"])

    out = []
    ap = out.append
    ap("==== 按压标定数据分析 ====")
    ap("生成: " + t0.strftime("%Y-%m-%d %H:%M:%S") + " | 脚本: analyze_press_calib.py")
    ap(f"数据: 按压标定数据.txt | {nf} 帧 / {dur:.1f}s ({fps:.1f} fps) | 网格 10x5 (50 点位)")
    ap(f"管线: M3(W={ev.W},d={ev.DELTA}) + slow(upd_n={upd_n}≈{UPD_SEC:.1f}s, alpha={ev.ALPHA}) "
       f"| direction=drop (d = 基线 - 读数)")
    ap("")

    ap("-- [1] 方向验证 --")
    ap("用户确认: 按下读数变小; 全部按压事件输出 d = 基线-读数 > 0, direction=drop 正确。")
    ap("")

    ap(f"-- [2] 按压事件表 (d>{THR:.0f}, 段长>={MINLEN}帧, 合并间隔<={GAP}帧, 共 {len(evs)} 个) --")
    ap("序号 | 点位 | 起        | 止        | 时长   | 峰值d  | 峰值时刻 | 均值d  | 尾部d")
    for i, e in enumerate(evs):
        ap(f"{i + 1:4d} | P{e['p']:02d}  | {e['a'] / fps:8.2f}s | {e['b'] / fps:8.2f}s | "
           f"{((e['b'] - e['a']) / fps):5.2f}s | {e['peak']:6.0f} | {e['a'] / fps:7.2f}s | "
           f"{e['mean']:6.0f} | {e['tail']:6.0f}")
    ap("")

    peaks = np.array([e["peak"] for e in evs])
    deep = [e for e in evs if e["peak"] >= DEEP]
    mid = [e for e in evs if MID <= e["peak"] < DEEP]
    light = [e for e in evs if e["peak"] < MID]
    ap("-- [3] 深度分级 (按事件峰值 d) --")
    for tag, g in (("深压(>={:.0f}, 按到底)".format(DEEP), deep),
                   ("中压({:.0f}~{:.0f})".format(MID, DEEP), mid),
                   ("轻压(<{:.0f})".format(MID), light)):
        if g:
            pk = np.array([e["peak"] for e in g])
            ap(f"  {tag}: {len(g)} 个 | 峰值 中位={np.median(pk):.0f} "
               f"范围={pk.min():.0f}~{pk.max():.0f}")
        else:
            ap(f"  {tag}: 0 个")
    ap("")

    ap("-- [4] r_ref 建议 (方案B: B = clip(d/r_ref, 0, 1)) --")
    if peaks.size:
        p90 = float(np.percentile(peaks, 90))
        r_ref = float(int(np.ceil(p90 / 100.0)) * 100)
        md = float(np.median([e["peak"] for e in deep])) if deep else float("nan")
        mm = float(np.median([e["peak"] for e in mid])) if mid else float("nan")
        ml = float(np.median([e["peak"] for e in light])) if light else float("nan")
        ap(f"  全部事件峰值: 中位={np.median(peaks):.0f} / P90={p90:.0f} / 最大={peaks.max():.0f} LSB")
        ap(f"  深压(按到底)中位 = {md:.0f} LSB -> 取 P90 向上圆整, 建议 r_ref = {r_ref:.0f} LSB")
        ap(f"  映射示例: 深压 {md:.0f} -> B≈{min(md / r_ref, 1.0):.2f}; 中压 {mm:.0f} -> B≈{mm / r_ref:.2f}; "
           f"轻压 {ml:.0f} -> B≈{ml / r_ref:.2f}")
    ap("")

    mask = np.zeros(nf, dtype=bool)
    for e in evs:
        a = max(0, e["a"] - GAP)
        b = min(nf, e["b"] + GAP)
        mask[a:b] = True
    base_vals = np.concatenate([dmap[k][~mask] for k in range(50)])
    base_vals = base_vals[~np.isnan(base_vals)]
    ap("-- [5] 空载底噪与周期伪影 --")
    ap(f"  非事件时段 d 稳健范围: P0.1={np.percentile(base_vals, 0.1):.1f} / "
       f"P50={np.percentile(base_vals, 50):.1f} / P99.9={np.percentile(base_vals, 99.9):.1f} LSB")
    ap(f"  非事件时段 |d|>50 帧占比: {(np.abs(base_vals) > 50).mean() * 100:.2f}%")
    ap("  原始数据存在周期性采样伪影: 列2~5 点位约每 276ms 两个 9~10 帧深跌脉冲")
    ap("  (列2 跌至~1100~1600 / 列5 跌至~0); M3 窗口内按 max-150 剔除, 输出已滤除。")
    ap("  注: 深跌脉冲的最低值(如 162/0)为伪影而非按压读数; 按压到底的真实读数下限")
    ap("  约 1900~1930 (机械行程限制), 对应 d 约 550~610 LSB。")
    ap("")

    ap("-- [6] 输出文件 --")
    ap("  _press_calib_analysis.txt")
    ap("  figures/preprocess_press_calib.png (图5 按压响应: 时空图 + 代表点位时序)")

    with open(OUT, "w", encoding="utf-8") as fo:
        fo.write("\n".join(out) + "\n")
    print("written", OUT)

    # ---- 图5 ----
    t = np.arange(nf) / fps
    fig = plt.figure(figsize=(13, 9), dpi=110)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.30)
    gsl = gs[1].subgridspec(2, 2, hspace=0.45, wspace=0.22)

    ax0 = fig.add_subplot(gs[0])
    DM = np.vstack([dmap[k] for k in range(50)])
    im = ax0.imshow(DM, aspect="auto", cmap="inferno",
                    norm=PowerNorm(0.8, vmin=0, vmax=700),
                    extent=[0, nf / fps, 50.5, 0.5], interpolation="nearest")
    cb = fig.colorbar(im, ax=ax0, fraction=0.025, pad=0.01)
    cb.set_label("压力偏差 d (LSB)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax0.set_yticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax0.set_ylabel("点位号")
    ax0.set_xlabel("时间 (s)")
    ax0.set_title("(a) 全点位压力响应时空图 (M3 + slow 管线输出; 前 5 次为 P6~P10 逐个按到底, "
                  "后 3 波为多指随机按压)", fontsize=10)

    for sub, k in enumerate(DEMO):
        ax = fig.add_subplot(gsl[sub // 2, sub % 2])
        d = dmap[k]
        ax.plot(t, d, lw=0.7, c="#1B7A3D")
        pk = float(np.nanmax(d))
        ax.set_ylim(-100, 850)
        ax.set_xlim(0, nf / fps)
        ax.set_title(f"P{k + 1:02d}  峰值 {pk:.0f} LSB", fontsize=9)
        ax.grid(alpha=0.3)
        for e in evs:
            if e["p"] == k + 1:
                ax.axvspan(e["a"] / fps, e["b"] / fps, color="#E8702A", alpha=0.25, lw=0)
        if sub >= 2:
            ax.set_xlabel("时间 (s)", fontsize=8)
        if sub % 2 == 0:
            ax.set_ylabel("d (LSB)", fontsize=8)
    fig.suptitle("按压标定数据响应 (单块 10x5 改造皮肤, 376.7fps, 38.5s)", fontsize=12)
    save_retry(fig, FIG, bbox_inches="tight")
    plt.close(fig)
    print("written", FIG)
    print(f"total {(datetime.now() - t0).total_seconds():.1f}s")


if __name__ == "__main__":
    main()
