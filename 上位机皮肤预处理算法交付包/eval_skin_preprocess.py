#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上位机滤波 + 归一化算法离线评测脚本

被评测对象: skin_preprocess.py (PointFilter 高位滑窗均值 / BaselineTracker 基线 / SkinPreprocessor)

数据集:
  7块皮肤测试_03.txt          主验证 (7 块, 166 有效点位, 21551 帧/块)
  7块皮肤测试_04.txt          复核 (方案A ROW_SETTLE=2 后采集)
  单皮肤拆除单行电容实验.txt   边界用例 (ID1 10x5, 25026 帧, 长时坏行)

评测内容:
  [1] 在线实现与既有离线口径一致性 (analyze_7skin_test03_plan.py 的 M3)
  [2] M1(区间滤波+保持) vs M3(高位滑窗均值) 全点位指标: sigma / |bias| / lowfrac
  [3] 归一化方案 A(基线偏差) / B(偏差映射 0~1) 量化评测
  [4] 图表: 时序对比 / sigma 分布 / 基线统一效果 / 热力图快照四联

口径 (与既有脚本对齐):
  truth = arr[arr>=P95].mean(); sigma 从第 W 帧起 ddof=1;
  lowfrac = P(out < truth-200); W=30, DELTA=150, LO/HI=1600/2700

输出: _preprocess_eval.txt + figures/preprocess_*.png
"""
import os
import re
import time
from array import array
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skin_preprocess import PointFilter, BaselineTracker, SkinPreprocessor

BASE = os.path.dirname(os.path.abspath(__file__))
F03 = os.path.join(BASE, "7块皮肤测试_03.txt")
F04 = os.path.join(BASE, "7块皮肤测试_04.txt")
FS = os.path.join(BASE, "单皮肤拆除单行电容实验.txt")
OUT = os.path.join(BASE, "_preprocess_eval.txt")
FIGDIR = os.path.join(BASE, "figures")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

TS_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d{3}) R\(HEX\)$")

SPEC = {1: (5, 3), 2: (5, 3), 3: (7, 3), 4: (7, 3), 5: (7, 4), 6: (7, 4), 7: (10, 5)}
SPEC_S = {1: (10, 5)}
EMPTY = {5: {2, 3, 4, 26, 27, 28}, 6: {2, 3, 4, 26, 27, 28}}

W, DELTA, NMIN = 30, 150, 3
LO, HI = 1600, 2700
ROW_TH = 2200
CALIB_N, UPD_N, ALPHA = 270, 90, 0.2
R_REF = 600.0
RREFS = (300, 450, 600, 900, 1200)
STAT_FROM = 3600

DEMO = {(1, 1), (7, 1), (7, 21), (5, 19), (5, 1), (3, 3)}
DEMO_FIG = [(1, 1), (7, 21), (5, 19), (3, 3)]

OFFS = {1: (0, 0), 2: (0, 4), 3: (0, 8), 4: (0, 12), 5: (8, 0), 6: (8, 5), 7: (8, 10)}
CANVAS = (18, 15)


# ======================== 日志解析 (与既有脚本一致) ========================

def parse_log(path):
    segs = []
    with open(path, "rb") as f:
        text = f.read().decode("utf-8", errors="replace")
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        m = TS_RE.match(s)
        if m:
            yy, mo, dd, hh, mi, ss, ms = (int(x) for x in m.groups())
            segs.append([datetime(2000 + yy, mo, dd, hh, mi, ss, ms * 1000), bytearray()])
            continue
        toks = s.split()
        ok = len(toks) > 0
        buf = bytearray()
        for t in toks:
            if len(t) != 2:
                ok = False
                break
            try:
                buf.append(int(t, 16))
            except ValueError:
                ok = False
                break
        if not ok:
            continue
        if not segs:
            segs.append([None, bytearray()])
        segs[-1][1].extend(buf)
    return segs


def find_first_frame(stream, spec, flen):
    i = 0
    n = len(stream)
    while True:
        i = stream.find(b"\xAA", i)
        if i < 0 or i + 5 > n:
            return -1, None
        fid = stream[i + 1]
        if fid in spec:
            m, nn = spec[fid]
            f = flen[fid]
            if (i + f <= n and stream[i + 2] == m and stream[i + 3] == nn
                    and stream[i + f - 1] == 0x0A):
                nxt = i + f
                if nxt + 1 >= n or (stream[nxt] == 0xAA and stream[nxt + 1] in spec):
                    return i, fid
        i += 1


def decode_frames(stream, first_off, first_id, spec, flen, ids):
    frames = []
    i = first_off
    expected = first_id
    pos_of = {fid: j for j, fid in enumerate(ids)}
    n = len(stream)
    while i < n:
        f = flen[expected]
        m, nn = spec[expected]
        if (i + f <= n and stream[i] == 0xAA and stream[i + 1] == expected
                and stream[i + 2] == m and stream[i + 3] == nn
                and stream[i + f - 1] == 0x0A):
            frames.append((i, expected))
            i += f
            expected = ids[(pos_of[expected] + 1) % len(ids)]
        else:
            j = stream.find(b"\xAA" + bytes([expected]), i + 1)
            if j < 0:
                break
            i = j
    return frames


def load_values(path, spec):
    """解析日志 -> values[fid][k] = array('H'); 返回 (values, nfr, duration_s)"""
    segs = parse_log(path)
    stream = bytearray()
    for _, data in segs:
        stream.extend(data)
    ids = sorted(spec)
    flen = {k: 5 + 2 * spec[k][0] * spec[k][1] for k in spec}
    off, fid0 = find_first_frame(stream, spec, flen)
    if off < 0:
        raise RuntimeError("no valid frame found in " + path)
    frames = decode_frames(stream, off, fid0, spec, flen, ids)
    values = {fid: [array("H") for _ in range(spec[fid][0] * spec[fid][1])] for fid in ids}
    for o, fid in frames:
        m, n = spec[fid]
        base = o + 4
        ss = values[fid]
        for k in range(m * n):
            ss[k].append((stream[base] << 8) | stream[base + 1])
            base += 2
    nfr = {fid: len(values[fid][0]) for fid in ids}
    ts = [t for t, _ in segs if t is not None]
    dur = (ts[-1] - ts[0]).total_seconds() if len(ts) >= 2 else float("nan")
    return values, nfr, dur


# ======================== 离线算法 (与 analyze_7skin_test03_plan.py 一致) ========================

def smooth_hold(values, valid):
    """带保持的前向填充: values 仅在 valid 处可信, 其余沿用最近有效值。"""
    pos = np.where(valid, np.arange(values.size), 0)
    np.maximum.accumulate(pos, out=pos)
    return values[pos]


def m1_offline(arr):
    """M1: 合理区间 [LO,HI] 过滤 + 保持"""
    return smooth_hold(arr, (arr >= LO) & (arr <= HI))


def m3_offline(arr):
    """M3: 高位滑窗均值 (W, DELTA, NMIN) + 保持; 暖机前 W-1 帧用均值填充"""
    sw = np.lib.stride_tricks.sliding_window_view(arr, W)
    wmax = sw.max(axis=1)
    th = wmax - DELTA
    mask = sw >= th[:, None]
    kept_cnt = mask.sum(axis=1)
    kept_sum = np.where(mask, sw, 0.0).sum(axis=1)
    body = smooth_hold(kept_sum / np.maximum(kept_cnt, 1), kept_cnt >= NMIN)
    return np.concatenate([np.full(W - 1, arr[:W - 1].mean()), body])


def slow_baseline_seq(vf, b0):
    """慢跟踪基线序列, 与 BaselineTracker(slow) 逐帧对齐:
    标定从滤波窗口满后开始 (帧 W-1 起 calib_n 个样本), 之后每 UPD_N 帧
    用块中位数做 EMA 更新 (新值挂在块尾帧上); 末尾不足 UPD_N 的残块
    在线不会更新 (len(_recent)<maxlen), 此处保持旧基线。"""
    sz = vf.size
    bs = np.empty(sz)
    start = W - 1
    bs[:start] = np.nan
    bs[start:start + CALIB_N] = b0
    i = start + CALIB_N
    b = b0
    while i < sz:
        j = min(i + UPD_N, sz)
        if j - i < UPD_N:              # 残块: 样本数不足, 在线不更新
            bs[i:j] = b
            break
        med = float(np.median(vf[i:j]))
        b_old = b
        b = b_old + ALPHA * (med - b_old)
        bs[i:j] = b_old
        bs[j - 1] = b
        i = j
    return bs


def eval_metrics(values, spec, empty):
    """全点位 M1/M3 指标表: (fid, pk, tag, sigma, bias, lowfrac, truth)"""
    rows = []
    for fid in sorted(spec):
        m, n = spec[fid]
        for k in range(m * n):
            pk = k + 1
            if pk in empty.get(fid, ()):
                continue
            arr = np.asarray(values[fid][k], dtype=np.float64)
            truth = float(arr[arr >= np.percentile(arr, 95)].mean())
            o1 = m1_offline(arr)
            o3 = m3_offline(arr)
            for tag, o in (("M1", o1), ("M3", o3)):
                oo = o[W:]
                rows.append((fid, pk, tag, float(oo.std(ddof=1)),
                             float(oo.mean() - truth), float((oo < truth - 200).mean()), truth))
    return rows


def table_lines(rows, ntot):
    lines = ["tag  | sigma均值 | sigma中位 | |bias|均值 | |bias|中位 | lowfrac均值 | lowfrac中位 | sigma<10 占比"]
    for tag in ("M1", "M3"):
        sigs = np.array([r[3] for r in rows if r[2] == tag])
        bias = np.array([abs(r[4]) for r in rows if r[2] == tag])
        lof = np.array([r[5] for r in rows if r[2] == tag])
        c10 = int((sigs < 10).sum())
        lines.append(f"{tag:4s} | {sigs.mean():9.1f} | {np.median(sigs):9.1f} | {bias.mean():10.1f} | "
                     f"{np.median(bias):11.1f} | {lof.mean():11.4f} | {np.median(lof):11.4f} | "
                     f"{100.0 * c10 / ntot:5.1f}%")
    return lines


def win_counts(rows):
    d = {}
    for r in rows:
        d.setdefault((r[0], r[1]), {})[r[2]] = r
    ws = sum(1 for v in d.values() if v["M3"][3] < v["M1"][3])
    wl = sum(1 for v in d.values() if v["M3"][5] <= v["M1"][5] + 1e-12)
    return ws, wl, len(d)


# ======================== 图表 ========================

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

def board_canvas(dd, spec, empty):
    c = np.full(CANVAS, np.nan)
    for fid in sorted(spec):
        m, n = spec[fid]
        r0, c0 = OFFS[fid]
        for k in range(m * n):
            if (k + 1) in empty.get(fid, ()):
                continue
            rr, cc = divmod(k, n)
            c[r0 + rr, c0 + cc] = dd[(fid, k + 1)]
    return c


def nan_cmap(name):
    cm = plt.get_cmap(name).copy()
    cm.set_bad("#d9d9d9")
    return cm


def fig1_pipeline(demo_data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=110)
    for ax, key in zip(axes.ravel(), DEMO_FIG):
        d = demo_data[key]
        arr, o1, o3, truth = d["arr"], d["m1"], d["m3"], d["truth"]
        a = 400
        b = min(1400, arr.size)
        x = np.arange(a, b)
        ax.scatter(x, arr[a:b], s=3, c="#bbbbbb", alpha=0.55, label="原始读数", zorder=1)
        ax.plot(x, o1[a:b], c="#E8702A", lw=1.0, label="M1 区间滤波+保持", zorder=2)
        ax.plot(x, o3[a:b], c="#1B7A3D", lw=1.2, label="M3 高位滑窗均值", zorder=3)
        ax.axhline(truth, color="k", ls="--", lw=0.8, label="平台参考 truth")
        ax.set_title(f"ID{key[0]}-P{key[1]}   sigma: M1={d['sig1']:.1f} -> M3={d['sig3']:.1f} LSB",
                     fontsize=10)
        ax.set_ylim(1400, 3100)
        ax.set_xlabel("数据帧序号")
        ax.set_ylabel("读数 (LSB)")
        ax.legend(fontsize=7, loc="lower right", ncol=2)
    fig.suptitle("滤波流水线时序对比 (7块皮肤测试_03, 帧 400~1400)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(FIGDIR, "preprocess_pipeline_compare.png")
    save_retry(fig, p)
    plt.close(fig)
    return p


def fig2_sigma(rows03, rows04):
    s1 = np.array([r[3] for r in rows03 if r[2] == "M1"])
    s3 = np.array([r[3] for r in rows03 if r[2] == "M3"])
    s4 = np.array([r[3] for r in rows04 if r[2] == "M3"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=110)
    bins = np.linspace(-0.15, 2.65, 57)
    ax1.hist(np.log10(s1 + 1), bins=bins, color="#E8702A", alpha=0.55,
             label=f"M1 (中位 {np.median(s1):.1f} LSB)")
    ax1.hist(np.log10(s3 + 1), bins=bins, color="#1B7A3D", alpha=0.55,
             label=f"M3 (中位 {np.median(s3):.1f} LSB)")
    ax1.set_xlabel("log10(sigma+1)   sigma 单位: LSB")
    ax1.set_ylabel("点位数")
    ax1.set_title("sigma 分布对比 (7块皮肤测试_03, 166 点位)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    for s, tag, c, ls in ((s1, "M1 (_03)", "#E8702A", "-"),
                          (s3, "M3 (_03)", "#1B7A3D", "-"),
                          (s4, "M3 (_04 复核)", "#1B7A3D", "--")):
        xs = np.sort(s)
        ys = np.arange(1, xs.size + 1) / xs.size
        ax2.plot(np.log10(xs + 1), ys, c=c, ls=ls, lw=1.5, label=tag)
    ax2.axvline(np.log10(11), color="gray", ls=":", lw=0.9)
    ax2.text(np.log10(11) + 0.03, 0.06, "sigma=10 LSB", fontsize=8, color="gray")
    ax2.set_xlabel("log10(sigma+1)")
    ax2.set_ylabel("累计占比 (sigma <= x)")
    ax2.set_title("sigma 累计分布")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "preprocess_sigma_hist.png")
    save_retry(fig, p)
    plt.close(fig)
    return p


def fig3_baseline(rec):
    keys = sorted(rec)
    pm = np.array([rec[k]["pm"] for k in keys])
    dm = np.array([rec[k]["dm"] for k in keys])
    cpm = board_canvas({k: rec[k]["pm"] for k in keys}, SPEC, EMPTY)
    cdm = board_canvas({k: rec[k]["dm"] for k in keys}, SPEC, EMPTY)
    v0, v1 = np.nanpercentile(cpm, [5, 95])
    dv = max(5.0, np.nanpercentile(np.abs(cdm), 95))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=110)
    im0 = axes[0].imshow(cpm, cmap=nan_cmap("inferno"), vmin=v0, vmax=v1, interpolation="nearest")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    axes[0].set_title(f"(a) M3 平台均值空间分布 (LSB)\n跨点位 std = {pm.std():.1f}", fontsize=10)
    im1 = axes[1].imshow(cdm, cmap=nan_cmap("RdBu_r"), vmin=-dv, vmax=dv, interpolation="nearest")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    axes[1].set_title(f"(b) 方案A 偏差均值 d (slow 基线, LSB)\nstd = {dm.std():.2f}", fontsize=10)
    order = np.argsort(pm)
    axc = axes[2]
    xs = np.arange(1, pm.size + 1)
    axc.plot(xs, pm[order], color="#888888", marker=".", ms=3, lw=0.8, label="M3 平台均值")
    axc.set_ylabel("M3 平台均值 (LSB)", color="#666666")
    axc.set_xlabel("点位序号 (按平台值排序)")
    axc2 = axc.twinx()
    axc2.plot(xs, dm[order], color="#1B7A3D", marker=".", ms=3, lw=0.8, label="方案A 偏差 d")
    axc2.set_ylabel("方案A 偏差 d (slow 基线, LSB)", color="#1B7A3D")
    axc.set_title(f"(c) 跨点位离散: std {pm.std():.0f} -> {dm.std():.1f} LSB", fontsize=10)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([]) if ax in axes[:2] else None
    axes[2].set_xticks(np.linspace(0, pm.size, 6).astype(int))
    axes[2].set_yticks(axes[2].get_yticks())
    fig.tight_layout()
    p = os.path.join(FIGDIR, "preprocess_baseline_calib.png")
    save_retry(fig, p)
    plt.close(fig)
    return p


def fig4_snapshot(snap_raw, snap_m3, snap_a, snap_b, idx):
    panels = [
        (f"(a) 原始单帧 (帧 {idx})", snap_raw, "inferno", 2100, 3000),
        ("(b) M3 滤波输出 (同帧)", snap_m3, "inferno", 2100, 3000),
        ("(c) 方案A 偏差 d (slow 基线, LSB)", snap_a, "RdBu_r", -300, 300),
        ("(d) 方案B 映射 0~1 (slow 基线)", snap_b, "inferno", 0.0, 1.0),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 5.6), dpi=110)
    for ax, (ttl, dd, cmname, v0, v1) in zip(axes, panels):
        c = board_canvas(dd, SPEC, EMPTY)
        im = ax.imshow(c, cmap=nan_cmap(cmname), vmin=v0, vmax=v1, interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7)
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("热力图快照对比 (7 块拼接布局, 灰=空点位; 原始/滤波/方案A/方案B)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(FIGDIR, "preprocess_heatmap_snapshot.png")
    save_retry(fig, p)
    plt.close(fig)
    return p


# ======================== 主流程 ========================

def main():
    t0 = time.time()
    print("[1/5] parsing ...")
    v03, nf03, dur03 = load_values(F03, SPEC)
    v04, nf04, dur04 = load_values(F04, SPEC)
    vs, nfs, durs = load_values(FS, SPEC_S)
    nf_min = min(nf03.values())
    npt03 = sum(SPEC[f][0] * SPEC[f][1] - len(EMPTY.get(f, ())) for f in SPEC)
    nptS = SPEC_S[1][0] * SPEC_S[1][1]

    out = []
    ap = out.append
    ap("==== 上位机滤波与归一化算法离线评测 ====")
    ap("生成: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " | 脚本: eval_skin_preprocess.py")
    ap(f"数据: _03: {nf03[1]} 帧/块 x 7 块 ({dur03:.1f}s) | _04: {nf04[1]} 帧/块 ({dur04:.1f}s) | "
       f"单皮肤: {nfs[1]} 帧 ({durs:.1f}s)")
    ap(f"参数: W={W} DELTA={DELTA} NMIN={NMIN} LO/HI={LO}/{HI} | 归一化: CALIB_N={CALIB_N} "
       f"UPD_N={UPD_N} ALPHA={ALPHA} R_REF={R_REF}")
    ap("")

    # ---- 快照帧选择 (7 块合计坏行最多) ----
    print("[2/5] pick snapshot frame ...")
    low_total = np.zeros(nf_min)
    for fid in sorted(SPEC):
        m, n = SPEC[fid]
        V = np.empty((nf_min, m * n))
        for k in range(m * n):
            V[:, k] = np.asarray(v03[fid][k][:nf_min], dtype=np.float64)
        lc = (V.reshape(nf_min, m, n).mean(axis=2) < ROW_TH).sum(axis=1)
        low_total += lc
    SNAP_IDX = int(np.argmax(low_total[STAT_FROM:])) + STAT_FROM
    nrow_all = sum(SPEC[f][0] for f in SPEC)
    ap(f"-- [0] 快照帧: idx={SNAP_IDX} (7 块合计坏行 {int(low_total[SNAP_IDX])}/{nrow_all}, "
       f"行均值<{ROW_TH})")
    ap("")

    # ---- _03 全点位指标 + 归一化记录 ----
    print("[3/5] metrics _03 + normalization ...")
    demo_data = {}
    rec = {}
    rows03 = []
    snap_raw, snap_m3, snap_a, snap_b = {}, {}, {}, {}
    for fid in sorted(SPEC):
        m, n = SPEC[fid]
        for k in range(m * n):
            pk = k + 1
            if pk in EMPTY.get(fid, ()):
                continue
            arr = np.asarray(v03[fid][k], dtype=np.float64)
            truth = float(arr[arr >= np.percentile(arr, 95)].mean())
            o1 = m1_offline(arr)
            o3 = m3_offline(arr)
            sig1 = float(o1[W:].std(ddof=1))
            sig3 = float(o3[W:].std(ddof=1))
            rows03.append((fid, pk, "M1", sig1, float(o1[W:].mean() - truth),
                           float((o1[W:] < truth - 200).mean()), truth))
            rows03.append((fid, pk, "M3", sig3, float(o3[W:].mean() - truth),
                           float((o3[W:] < truth - 200).mean()), truth))
            b0 = float(np.median(o3[W - 1:W - 1 + CALIB_N]))
            bs = slow_baseline_seq(o3, b0)
            d = bs[STAT_FROM:] - o3[STAT_FROM:]          # slow 模式偏差 (推荐)
            dc = b0 - o3[STAT_FROM:]                     # calib 模式偏差 (对照)
            edges = np.linspace(STAT_FROM, arr.size, 11).astype(int)
            cseg = [float(dc[a - STAT_FROM:c - STAT_FROM].mean())
                    for a, c in zip(edges[:-1], edges[1:])]
            sseg = [float(d[a - STAT_FROM:c - STAT_FROM].mean())
                    for a, c in zip(edges[:-1], edges[1:])]
            r05 = {rr: float((np.clip(d / rr, 0.0, 1.0) > 0.05).mean()) for rr in RREFS}
            bz = np.clip(d / R_REF, 0.0, 1.0)
            rec[(fid, pk)] = dict(
                pm=float(o3[STAT_FROM:].mean()), dm=float(d.mean()),
                siga=float(d.std(ddof=1)), dmc=float(dc.mean()),
                sigc=float(dc.std(ddof=1)), sig1=sig1, sig3=sig3,
                cseg=cseg, sseg=sseg,
                b05=float((bz > 0.05).mean()), b10=float((bz > 0.10).mean()), r05=r05)
            if (fid, pk) in DEMO:
                demo_data[(fid, pk)] = dict(arr=arr, m1=o1, m3=o3, truth=truth,
                                            sig1=sig1, sig3=sig3)
            v_raw = float(arr[SNAP_IDX])
            v_m3 = float(o3[SNAP_IDX])
            b_snap = float(bs[SNAP_IDX])
            snap_raw[(fid, pk)] = v_raw
            snap_m3[(fid, pk)] = v_m3
            snap_a[(fid, pk)] = b_snap - v_m3
            snap_b[(fid, pk)] = float(np.clip((b_snap - v_m3) / R_REF, 0.0, 1.0))

    # ---- 在线一致性 ----
    print("[4/5] online vs offline ...")
    maxdiff1 = 0.0
    maxdiff2 = 0.0
    for key in sorted(DEMO):
        arr = demo_data[key]["arr"]
        o3 = demo_data[key]["m3"]
        f = PointFilter(W, DELTA, NMIN)
        on_f = []
        for vv in arr:
            vf = f.update(vv)
            on_f.append(np.nan if vf is None else vf)
        on_f = np.asarray(on_f)
        maxdiff1 = max(maxdiff1, float(np.nanmax(np.abs(on_f[W:] - o3[W:]))))
        # 端到端: SkinPreprocessor(slow) 整链 vs 向量化模拟
        pre = SkinPreprocessor(1, 1, norm="A", baseline_mode="slow",
                               calib_n=CALIB_N, upd_n=UPD_N, alpha=ALPHA)
        on_a = []
        for vv in arr:
            r = pre.update_frame([vv])[0]
            on_a.append(np.nan if r is None else r)
        on_a = np.asarray(on_a)
        b0 = float(np.median(o3[W - 1:W - 1 + CALIB_N]))
        ref = slow_baseline_seq(o3, b0) - o3
        maxdiff2 = max(maxdiff2,
                       float(np.nanmax(np.abs(on_a[STAT_FROM:] - ref[STAT_FROM:]))))

    # ---- _04 与单皮肤 ----
    print("[5/5] metrics _04 / single + figures ...")
    rows04 = eval_metrics(v04, SPEC, EMPTY)
    rowsS = eval_metrics(vs, SPEC_S, {})

    # ---- 写入统计 ----
    ws3, wl3, tot3 = win_counts(rows03)
    ws4, wl4, tot4 = win_counts(rows04)
    ap("-- [1] 在线实现 vs 离线口径 一致性 (6 代表点位, 全时长) --")
    ap(f"  PointFilter vs 离线 M3               : 最大绝对差 = {maxdiff1:.3e}")
    ap(f"  SkinPreprocessor(slow) vs 向量化模拟 : 最大绝对差 = {maxdiff2:.3e} (帧>={STAT_FROM})")
    ap("")
    ap(f"-- [2] 全点位指标 (_03, {npt03} 有效点位) --")
    out.extend(table_lines(rows03, npt03))
    ap(f"M3 sigma 低于 M1: {ws3}/{tot3} | M3 lowfrac 不高于 M1: {wl3}/{tot3}")
    ap("(对照 _plan03.txt: M1 sigma中位=3.4, M3 sigma中位=0.9, M3 sigma<10=69.9%, "
       "sigma胜出129/166, lowfrac胜出166/166)")
    ap("")
    ap(f"-- [3] 复核 (_04 方案A后采集, {tot4} 有效点位) --")
    out.extend(table_lines(rows04, tot4))
    ap(f"M3 sigma 低于 M1: {ws4}/{tot4} | M3 lowfrac 不高于 M1: {wl4}/{tot4}")
    ap("")
    ap(f"-- [4] 单皮肤边界 (拆除单行电容+磁铁, {nptS} 点位, 长时坏行) --")
    out.extend(table_lines(rowsS, nptS))
    ap("")
    ap("-- [5] 归一化方案 A/B 评测 (_03) --")
    pm = np.array([v["pm"] for v in rec.values()])
    dm = np.array([v["dm"] for v in rec.values()])
    siga = np.array([v["siga"] for v in rec.values()])
    ap("[5a] 跨点位平台差异 (基线统一效果, 帧>=%d)" % STAT_FROM)
    dmc = np.array([v["dmc"] for v in rec.values()])
    sigc = np.array([v["sigc"] for v in rec.values()])
    ap(f"  原始 M3 输出平台均值(即热力图原始色阶): mean={pm.mean():.1f} std={pm.std():.1f} "
       f"min={pm.min():.0f} max={pm.max():.0f} 极差={pm.max() - pm.min():.0f} LSB")
    ap(f"  slow 偏差 d (推荐): mean={dm.mean():.2f} std={dm.std():.2f} min={dm.min():.1f} "
       f"max={dm.max():.1f} 极差={dm.max() - dm.min():.1f} LSB")
    ap(f"  calib 偏差 d (对照): mean={dmc.mean():.2f} std={dmc.std():.2f} min={dmc.min():.1f} "
       f"max={dmc.max():.1f} (启动收敛点位基线错位于悬空平台 ~4070)")
    ap(f"  -> 跨点位离散度 std: 原始 {pm.std():.1f} -> slow {dm.std():.2f} LSB "
       f"(缩小 {pm.std() / max(dm.std(), 1e-9):.0f} 倍)")
    ap(f"  偏差 sigma 中位: slow={np.median(siga):.2f} / calib={np.median(sigc):.2f} LSB")
    ap("")
    b05 = np.array([v["b05"] for v in rec.values()])
    b10 = np.array([v["b10"] for v in rec.values()])
    ap(f"[5b] 空载 B 输出 (slow 基线, r_ref={R_REF:.0f}; 空载应≈0 形成稳定底色)")
    ap(f"  sigma_B: 中位={np.median(siga) / R_REF:.4f} 均值={siga.mean() / R_REF:.4f}")
    ap(f"  B>0.05 占比: 中位={np.median(b05):.5f} 均值={b05.mean():.5f} 点位最高={b05.max():.4f}")
    ap(f"  B>0.10 占比: 中位={np.median(b10):.5f} 均值={b10.mean():.5f}")
    ap(f"  B>0.05 占比超过 1% 的点位数: {int((b05 > 0.01).sum())}/{npt03} "
       f"(此类点位空载即可能显示伪热斑)")
    ap("")
    ap("[5c] r_ref 敏感性 (方案B; r_ref=满量程读数差, d=r_ref 时输出 1.0)")
    ap("r_ref | sigma_B中位 | B>0.05占比中位 | B>0.05占比均值")
    for rr in RREFS:
        rv = np.array([v["r05"][rr] for v in rec.values()])
        ap(f"{rr:5d} | {np.median(siga) / rr:10.4f} | {np.median(rv):13.5f} | {rv.mean():12.5f}")
    ap("")
    cseg = np.array([v["cseg"] for v in rec.values()])
    sseg = np.array([v["sseg"] for v in rec.values()])
    rng_c = np.ptp(cseg, axis=1)
    rng_s = np.ptp(sseg, axis=1)
    seg_span = (nf_min - STAT_FROM) / 10.0 / 90.6
    ap(f"[5d] 时间稳定性 (帧>={STAT_FROM} 分 10 段, 每段≈{seg_span:.0f}s; 检验慢漂移)")
    ap("段号    : " + " ".join(f"{i + 1:6d}" for i in range(10)))
    ap("calib段均: " + " ".join(f"{v:6.1f}" for v in cseg.mean(axis=0)) + "  (166 点位 d 均值)")
    ap("slow 段均: " + " ".join(f"{v:6.1f}" for v in sseg.mean(axis=0)))
    ap(f"  单点位段均值极差: calib 中位={np.median(rng_c):.1f} / slow 中位={np.median(rng_s):.1f} LSB")
    ap("  (若 calib 段间存在明显趋势/极差, 说明平台缓漂, 建议启用 slow 模式)")
    ap("")
    ap("-- [6] 输出文件 --")
    ap("  _preprocess_eval.txt")
    ap("  figures/preprocess_pipeline_compare.png  (图1 时序对比)")
    ap("  figures/preprocess_sigma_hist.png        (图2 sigma 分布)")
    ap("  figures/preprocess_baseline_calib.png    (图3 基线统一)")
    ap("  figures/preprocess_heatmap_snapshot.png  (图4 热力图快照四联)")

    with open(OUT, "w", encoding="utf-8") as fo:
        fo.write("\n".join(out) + "\n")
    print("written", OUT)

    p1 = fig1_pipeline(demo_data)
    p2 = fig2_sigma(rows03, rows04)
    p3 = fig3_baseline(rec)
    p4 = fig4_snapshot(snap_raw, snap_m3, snap_a, snap_b, SNAP_IDX)
    for p in (p1, p2, p3, p4):
        print("written", p)
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
