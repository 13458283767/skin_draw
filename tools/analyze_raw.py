"""Audit captured bytes independently, summarize row faults."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    folder = args.folder.resolve()
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    raw = (folder / "rx.bin").read_bytes()
    raw.decode("ascii")  # fail rather than silently replace corrupted bytes
    (folder / "rx.txt").write_bytes(raw)
    lines = raw.splitlines()
    recorded = [json.loads(line) for line in (folder / "blocks.jsonl").read_text().splitlines()]
    audited = []
    # Deliberately independent of BlockParser: full header + declared CSV rows.
    for index, line in enumerate(lines):
        match = re.fullmatch(rb"BLOCK_(\d+),(\d+),(\d+):", line)
        if not match:
            continue
        key, rows, cols = map(int, match.groups())
        values = []
        for data in lines[index + 1:index + 1 + rows]:
            fields = data.split(b",")
            if len(fields) != cols or not all(re.fullmatch(rb"\d+", value) for value in fields):
                break
            values.append(list(map(int, fields)))
        if len(values) == rows:
            assert all(0 <= value <= 4095 for row in values for value in row)
            audited.append({"block_id": key, "values": values, "header_line": index + 1})
    assert len(audited) == len(recorded), "raw audit and streaming parser disagree on block count"
    for block, record in zip(audited, recorded):
        assert block["block_id"] == record["block_id"] and block["values"] == record["values"]
        record["header_line"] = block["header_line"]

    groups = defaultdict(list)
    for record in recorded:
        groups[record["block_id"]].append(record)
    stats, events = [], []
    for key, records in sorted(groups.items()):
        values = np.array([r["values"] for r in records], dtype=int)
        times = np.array([r["seconds"] for r in records])
        changed = np.any(np.diff(values, axis=0) != 0, axis=(1, 2))
        starts = np.r_[0, np.flatnonzero(changed) + 1]
        lengths = np.diff(np.r_[starts, len(values)])
        low = np.all(values <= 1300, axis=2)
        item = {"block_id": key, "frames": len(values), "shape": list(values.shape[1:]),
                "received_hz": (len(values) - 1) / (times[-1] - times[0]),
                "changed_matrices": int(changed.sum()),
                "content_change_hz": int(changed.sum()) / (times[-1] - times[0]),
                "duplicate_run_median": float(np.median(lengths)), "duplicate_run_max": int(lengths.max()),
                "bad_frames": int(low.any(axis=1).sum()), "bad_rows": low.sum(axis=0).tolist(),
                "min_adc": int(values.min()), "max_adc": int(values.max())}
        stats.append(item)
        for row in range(values.shape[1]):
            indices = np.flatnonzero(low[:, row])
            if not len(indices):
                continue
            runs = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
            for run in runs:
                first, last = int(run[0]), int(run[-1])
                events.append({"block_id": key, "row": row + 1, "first_block_frame": first + 1,
                               "start_seconds": float(times[first]), "consecutive_frames": len(run),
                               "bad_values": values[first, row].tolist(),
                               "previous_values": values[first - 1, row].tolist() if first else None,
                               "next_values": values[last + 1, row].tolist() if last + 1 < len(values) else None,
                               "raw_line": records[first]["header_line"] + row + 1})
    events.sort(key=lambda event: event["start_seconds"])

    summary = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
               "independent_audit_blocks": len(audited), "stats": stats, "events": events}
    (folder / "analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# 原始串口异常行检查", "", "来源：COM6，115200/8N1，连续被动接收约 60 秒，未发送指令，未经过校准或显示过滤。",
              "采集时已提示保持静置，但是否实际完全无按压未自动验证。", "",
              f"保存 {len(raw):,} 字节、{len(recorded):,} 个完整 BLOCK。独立按头部/行数重读原始字节，与在线解析结果逐项一致；在线解析错误为 {metadata['parse_errors']}。末尾未完成行 {metadata['partial_line_bytes_at_end']} 字节（采集截止边界）。",
              "", "| 块 | 报文数 | 收包频率 | 矩阵内容变化频率 | 连续相同矩阵中位数 | 含整行异常报文数 |",
              "|---|---:|---:|---:|---:|---:|"]
    for stat in stats:
        report.append(f"| {stat['block_id']} | {stat['frames']} | {stat['received_hz']:.2f} Hz | {stat['content_change_hz']:.2f} 次/秒 | {stat['duplicate_run_median']:.0f} | {stat['bad_frames']} |")
    report += ["", "异常行统计判据：该行所有原始 ADC ≤ 1300，用于定位原始数据中的整行故障。",
               "", "## 可复核的异常片段", "", "| 时间/秒 | 块/行 | 连续报文数 | 前一有效行 | 异常行 | 恢复行 | 原始文本行号 |",
               "|---:|---|---:|---|---|---|---:|"]
    for event in events:
        report.append(f"| {event['start_seconds']:.3f} | {event['block_id']} / {event['row']} | {event['consecutive_frames']} | {event['previous_values']} | {event['bad_values']} | {event['next_values']} | {event['raw_line']} |")
    report += ["", "## 判断", "",
               "1. 整行异常数字就在保存的串口字节中，独立解析也能重现，因此不是热力图着色或矩阵索引制造的。",
               "2. BLOCK 2 的五行均出现过整行同时降低；当前片段的异常集中在该块，不是单个固定传感器。",
               "3. 收包约 27 Hz，内容变化约 3.9 次/秒，矩阵通常重复 7 个报文。说明收包帧率不能直接当作新采样更新率；现有协议没有采样序号/时间戳，不能仅凭相同值证明 ADC 内部采样频率。",
               "4. 异常行也连续重复。若把第二份相近报文当作有效的新水平，异常就会再次进入显示。",
               "5. 协议没有 CRC。合法数字且按整行变化更值得检查下位机的行扫描与发送数据更新路径，但仅凭串口不能区分电路、扫描时序、接触问题或固件缓存问题，不能据此锁定具体元件。",
               "", "文件：rx.bin / rx.txt 为相同原始字节；chunks.jsonl 记录每次接收时间和字节偏移；blocks.jsonl 为未过滤矩阵；analysis.json 保存统计与事件。",
               "", "SHA-256: " + summary["sha256"]]
    (folder / "analysis.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"stats": stats, "episodes": len(events),
                      "independent_audit_blocks": len(audited)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
