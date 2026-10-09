"""Create and verify an explicitly synthetic 3 s / 14 block recording."""

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
try:
    import h5py
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT / "artifacts" / "_h5_demo_deps"))
    import h5py


# Diagram numbers are DEMO block IDs, NOT verified hardware bindings.
# Occupancy is described top-to-bottom in display coordinates, then transposed
# into this demo's raw matrix orientation (long dimension first).
PROFILES = [
    ("右肩", 5, 3, [3]*5), ("左肩", 5, 3, [3]*5),
    ("右上臂", 7, 3, [7]*3), ("左上臂", 7, 3, [7]*3),
    ("右下臂", 7, 4, [7, 5, 5, 5]), ("左下臂", 7, 4, [7, 5, 5, 5]),
    ("胸部", 10, 5, [4, 6, 10, 10, 10]), ("腹部", 10, 5, [10]*5),
    ("右大腿", 13, 5, [13, 11, 11, 9, 9]),
    ("左大腿", 13, 5, [13, 11, 11, 9, 9]),
    ("右手", 6, 3, [6]*3), ("左手", 6, 3, [6]*3),
    ("背部", 10, 5, [4, 6, 10, 10, 10]), ("腰部", 10, 5, [10]*5),
]


def sensor_mask(key, rows, cols, widths):
    if key <= 2:
        return np.ones((rows, cols), dtype=bool)
    display = np.zeros((cols, rows), dtype=bool)
    for row, width in enumerate(widths):
        start = (rows-width)//2
        display[row, start:start+width] = True
    return display.T.copy()


def add(group, name, values, unit, axes, compressed=True):
    options = dict(compression="gzip", compression_opts=1, shuffle=True,
                   fletcher32=True, chunks=values.shape) if compressed else {}
    ds = group.create_dataset(name, data=values, **options)
    ds.attrs["unit"] = unit
    ds.attrs["axes"] = axes
    return ds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "h5_demo_14blocks_3s")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("raw.h5", "meta.json", "README.md", "preview.html", "validation.json")):
        raise SystemExit("Output already exists; use another --output directory.")
    rng = np.random.default_rng(20260924)
    meta = {
        "schema_version": "skindraw.demo.1", "sample_id": "synthetic_14blocks_press_3s_001",
        "is_synthetic": True, "data_source": "simulation", "subject_id": None,
        "session_id": "synthetic_demo_001", "label_id": -1,
        "label_name": "synthetic_all_blocks_press", "label_cn": "模拟：全部皮肤按压后释放",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_ns": 3_000_000_000, "nominal_block_hz": 10,
        "frames_per_block": 30, "total_block_frames": 420,
        "calibration_frames_per_block": 30, "calibration_in_recording": False,
        "pressure_direction": "decrease", "data_kind": "raw_adc_uint16",
        "filter_threshold_adc": 800, "filter_applied_to_raw": False,
        "timestamps": {"unit": "ns", "origin": "recording_start", "interval": "[0, 3000000000)",
                       "kind": "simulated_host_receive_completion", "hardware_sample_time_available": False,
                       "calibration": "negative times before recording; all timestamps are simulated"},
        "action_intervals": [
            {"phase": "rest", "start_ns": 0, "end_ns": 500_000_000},
            {"phase": "press_ramp", "start_ns": 500_000_000, "end_ns": 1_000_000_000},
            {"phase": "full_press", "start_ns": 1_000_000_000, "end_ns": 1_800_000_000},
            {"phase": "release", "start_ns": 1_800_000_000, "end_ns": 2_400_000_000},
            {"phase": "rest", "start_ns": 2_400_000_000, "end_ns": 3_000_000_000}],
        "mapping_verified": False,
        "assumptions": ["BLOCK 1-14 temporarily follow diagram numbers, not actual hardware IDs.",
                        "Raw orientation and centered masks are illustrative; validate wiring before real use.",
                        "Every active sensor is pressed for a format demo, not a real physical trial.",
                        "10 Hz per block is a simulation setting, not measured hardware performance.",
                        "No corrupt frames or missing blocks injected; rejected_rows are all false.",
                        "Press intensity is dimensionless, not force in N or pressure in kPa."],
        "files": {"data": "raw.h5", "labels": "meta.json", "preview": "preview.html"},
        "blocks": [],
    }
    preview_sections, structure = [], []
    record_bytes = calibration_bytes = 0
    wire_offset = 0.
    with h5py.File(output / "raw.h5", "x") as f:
        f.attrs.update(schema_version=meta["schema_version"], sample_id=meta["sample_id"],
                       is_synthetic=True, completion_status="complete", duration_ns=meta["duration_ns"])
        f.create_group("blocks")
        f.create_group("calibration")
        for key, (body, rows, cols, widths) in enumerate(PROFILES, 1):
            mask = sensor_mask(key, rows, cols, widths)
            block_name = f"block_{key:02d}"
            # One round every 100 ms; use packet lengths only to illustrate
            # sequential receive completion, not actual ADC sampling times.
            wire_offset += (5+2*rows*cols)/11520
            time_ns = np.rint((np.arange(30)/10 + wire_offset)*1e9).astype(np.int64)
            cal_time = time_ns - 3_000_000_000
            nominal = rng.integers(2480, 2521, size=(rows, cols))
            cal_raw = (nominal[None] + rng.integers(-3, 4, size=(30, rows, cols)))
            cal_raw = np.where(mask, cal_raw, 0).astype(np.uint16)
            baseline = cal_raw.mean(axis=0)
            full_press = rng.integers(1920, 1981, size=(rows, cols))
            t = time_ns / 1e9
            intensity = np.minimum(np.clip((t-.5)/.5, 0, 1), np.clip((2.4-t)/.6, 0, 1))
            adc = baseline[None] - intensity[:, None, None]*(baseline-full_press)[None]
            adc += rng.integers(-3, 4, size=adc.shape)
            raw = np.where(mask, np.rint(adc), 0).astype(np.uint16)
            group = f["blocks"].create_group(block_name)
            group.attrs.update(block_id=key, body_part_demo=body, mapping_verified=False,
                               rotation_deg=0, flip_x=False, flip_y=False)
            add(group, "raw_adc", raw, "ADC count", "time,raw_row,raw_col")
            add(group, "host_time_ns", time_ns, "ns relative to recording start", "time")
            add(group, "event_index", (np.arange(30)*14+key-1).astype(np.uint64), "software delivery index; not hardware sequence", "time")
            add(group, "sensor_mask", mask, "true=physical sensor in this demo", "raw_row,raw_col", False)
            add(group, "rejected_rows", np.zeros((30, rows), dtype=bool), "true=row rejected by display filter", "time,raw_row")
            cal = f["calibration"].create_group(block_name)
            add(cal, "raw_adc", cal_raw, "ADC count", "time,raw_row,raw_col")
            add(cal, "host_time_ns", cal_time, "ns relative to recording start", "time")
            add(cal, "baseline_adc", baseline, "ADC count; mean of 30 preceding samples", "raw_row,raw_col", False)
            meta["blocks"].append({"id": key, "body_part_demo": body, "raw_shape": [rows, cols],
                                   "frame_count": 30, "sensor_count": int(mask.sum()),
                                   "slot_count": rows*cols, "rotation_deg": 0, "flip_x": False, "flip_y": False})
            record_bytes += raw.nbytes
            calibration_bytes += cal_raw.nbytes
            examples = []
            for index, title in ((0, "静置"), (12, "完全按压"), (28, "释放后")):
                grid = []
                for row in range(rows):
                    cells = []
                    for col in range(cols):
                        value = int(raw[index, row, col])
                        alpha = float(np.clip((baseline[row, col]-value)/650, 0, 1))
                        bg = f"rgba(210,35,40,{alpha:.3f})" if mask[row, col] else "#e2e8f0"
                        text = str(value) if mask[row, col] else "—"
                        cells.append(f'<td style="background:{bg}">{text}</td>')
                    grid.append("<tr>"+"".join(cells)+"</tr>")
                examples.append(f'<div><h3>{title} · {t[index]:.3f}s</h3><table>{"".join(grid)}</table></div>')
            preview_sections.append(f'<section><h2>BLOCK {key:02d} · {html.escape(body)}（假设） · {rows}×{cols}</h2><div class="frames">{"".join(examples)}</div></section>')
        f.create_dataset("metadata_json", data=json.dumps(meta, ensure_ascii=False), dtype=h5py.string_dtype("utf-8"))
        f.flush()

    # Reopen from disk and validate actual persisted data and ordering.
    events = []
    with h5py.File(output / "raw.h5", "r") as f:
        assert bool(f.attrs["is_synthetic"])
        assert len(f["blocks"]) == 14
        assert json.loads(f["metadata_json"].asstr()[()]) == meta
        for name, group in f["blocks"].items():
            raw, times, mask = group["raw_adc"][:], group["host_time_ns"][:], group["sensor_mask"][:]
            baseline = f[f"calibration/{name}/baseline_adc"][:]
            assert raw.dtype == np.dtype("uint16") and raw.shape[0] == 30
            assert np.all(np.diff(times)>0) and 0 <= times.min() < times.max() < 3_000_000_000
            assert np.all(raw[:, ~mask] == 0)
            assert not group["rejected_rows"][:].any()
            rest = raw[(times < 500_000_000) | (times >= 2_400_000_000)][:, mask]
            pressed = raw[(times >= 1_000_000_000) & (times < 1_800_000_000)][:, mask]
            assert rest.min() >= 2470 and rest.max() <= 2530
            assert pressed.min() >= 1900 and pressed.max() <= 2000
            np.testing.assert_array_equal(baseline, f[f"calibration/{name}/raw_adc"][:].mean(axis=0))
            np.testing.assert_array_equal(np.abs(np.diff(raw.astype(np.int32), axis=0)) > 800, False)
            events.extend(zip(group["event_index"][:].tolist(), times.tolist()))
        events.sort()
        assert [e[0] for e in events] == list(range(420))
        assert all(a[1] < b[1] for a, b in zip(events, events[1:]))
        def describe(name, obj):
            if isinstance(obj, h5py.Dataset):
                structure.append(f"/{name}  shape={obj.shape}  dtype={obj.dtype}")
        f.visititems(describe)
    size = (output / "raw.h5").stat().st_size
    digest = hashlib.sha256((output / "raw.h5").read_bytes()).hexdigest()
    validation = {"passed": True, "is_synthetic": True, "recording_block_frames": 420,
                  "calibration_block_frames": 420, "slots": sum(b["slot_count"] for b in meta["blocks"]),
                  "active_sensors": sum(b["sensor_count"] for b in meta["blocks"]),
                  "recording_adc_bytes": record_bytes, "calibration_adc_bytes": calibration_bytes,
                  "h5_file_bytes": size, "sha256": digest,
                  "checks": ["14 blocks", "30 recording frames per block", "timestamps and event order",
                             "zero padding mask", "rest and pressed ADC ranges", "baseline exact mean",
                             "no jumps above 800", "embedded metadata matches sidecar"]}
    (output / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (output / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (output / "preview.html").write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>模拟电子皮肤 H5 样例</title>
<style>body{font:15px "Microsoft YaHei",sans-serif;background:#f1f5f9;color:#183049;margin:32px}section{background:white;padding:20px;margin:20px 0;border-radius:10px}.frames{display:flex;gap:32px;flex-wrap:wrap}table{border-collapse:collapse}td{border:2px solid white;min-width:43px;height:30px;text-align:center;font-size:13px}h3{font-size:15px}p{max-width:1000px;line-height:1.8}</style>
<h1>14块皮肤 · 3秒 · 模拟数据</h1><p>不是实物测量，不能计入论文实验。每块10帧/秒、30帧，先前30帧校准单独保存。下方矩阵直接读取生成时的数据，不做旋转；灰色“—”代表掩码空位，文件原值为0。部位与BLOCK绑定仅为假设。颜色仅用于预览，H5保存的是ADC数值。</p>'''+"".join(preview_sections)+"</html>", encoding="utf-8")
    table = "\n".join(f'| {b["id"]:02d} | {b["body_part_demo"]}（假设） | [30, {b["raw_shape"][0]}, {b["raw_shape"][1]}] | {b["sensor_count"]} |' for b in meta["blocks"])
    readme = f'''# 14块皮肤、3秒 HDF5 模拟样例

**纯模拟数据，不能作为真实受试者数据或论文实验结果。** 固定随机种子20260924。

每块10帧/秒、3秒各30帧，总计420个块帧。ADC约2500静置，逐点下降至1900–2000完全按压，再恢复。全部有效传感器一起按压只是便于观察格式。校准在录制之前，另存每块30帧，时间戳为负值；不占这3秒。

模型时间：0–0.5s静置，0.5–1.0s逐渐按压，1.0–1.8s完全按压，1.8–2.4s释放，2.4–3.0s静置。各块按ID轮询，时间错开；并非同步全身帧。时间轴为模拟接收完成时刻，纳秒只是单位，不能推断真实采样精度。

## 文件

- raw.h5：原始uint16矩阵、接收时间、0起始全局事件编号、空位掩码、异常行标记，以及前置校准矩阵和float64基线。
- meta.json：本样例的标签及配置；H5内metadata_json是生成时相同内容的只读快照。
- preview.html：用浏览器打开，看所有块静置、按压、释放后的原始矩阵。
- validation.json：重新打开H5后的检查结果、大小和SHA256。

## 尺寸

| BLOCK | 示例部位 | raw_adc维度 | 有效点数 |
|---|---|---|---:|
{table}

合计494槽位、438个模拟有效传感器。56个空位始终存0，sensor_mask=False。真实ADC为0与空位须用掩码区分。图纸编号与硬件ID、原始行列方向均未核实，不可把本例映射直接用于硬件。为展示现有长边在前的矩阵，本例将图纸网格转置；不是正式的人体方向配置。行过滤仍按原始矩阵行。

## 容量

3秒原始ADC：{record_bytes:,}字节；前置校准ADC：{calibration_bytes:,}字节。完整H5为 **{size:,}字节（{size/1024:.1f} KiB）**，含时间、基线、gzip无损压缩和文件结构开销。小样例结构开销明显，不能按文件大小线性推算大型数据集，也不保证比CSV更小。

本例14块一轮二进制v0.5负载及帧结构共1058字节，10轮/秒约10580字节/秒；这只是模拟配置，真实采样和总线开销可能使14块达不到10Hz。不假定可沿用少数块时的18–19Hz。

## 读取

安装numpy、h5py后：

```python
import h5py
with h5py.File("raw.h5", "r") as f:
    adc = f["blocks/block_07/raw_adc"][:10]  # 仅胸部前10帧，形状[10,10,5]
    time_ns = f["blocks/block_07/host_time_ns"][:10]
    mask = f["blocks/block_07/sensor_mask"][:]
    baseline = f["calibration/block_07/baseline_adc"][:].astype("float32")
    delta = baseline[None] - adc.astype("float32")
    delta[:, ~mask] = 0  # 保留mask给网络；相减前转浮点，避免uint16下溢
```

meta中的label_id=-1意为模拟/未纳入正式类别，不应直接送入分类损失。3秒数据是一条模拟试验，不能把420个块帧算成420次独立动作。未注入异常，rejected_rows全False；raw_adc不经过显示过滤。生成器：tools/create_h5_demo.py；本次h5py安装在artifacts/_h5_demo_deps，不修改应用运行依赖。

## 文件实际结构

```text
{chr(10).join(structure)}
```
'''
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), **validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
