"""Read-only CSV/metadata audit. Writes results only to --output (never dataset)."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def audit(root):
    root = Path(root).resolve()
    counts = {name: Counter() for name in ("schema", "action", "subject", "session_key",
                                         "accepted", "recording_status", "quality_status", "labels_id")}
    errors, warnings, files, hashes = [], [], [], defaultdict(list)
    total_rows = 0
    annotation_count = calibration_count = baseline_count = auto_annotation_count = 0
    labels = read(root / "labels.json")
    label_map = {item["id"]: item["name"] for item in labels["classes"]}
    for path in sorted(root.glob("*/*/*/*/meta.json")):
        if path.relative_to(root).parts[0].startswith("_"):
            continue
        rel = path.parent.relative_to(root).as_posix()
        problems, notes = set(), set()
        try:
            meta = read(path)
            counts["schema"][meta.get("schema_version")] += 1
            counts["action"][meta["action"]["name"]] += 1
            counts["subject"][meta["subject_id"]] += 1
            counts["session_key"][meta["subject_id"] + "__" + meta["session_id"]] += 1
            for key in ("accepted", "recording_status", "quality_status", "labels_id"):
                counts[key][str(meta.get(key))] += 1
            if meta["action"]["name"] != label_map.get(meta["action"]["id"]):
                problems.add("action_id_name_mismatch")
            if meta["labels_id"] != labels["labels_id"]:
                notes.add("historical_labels_version_differs")
            layout = read(root / (meta["layout_id"] + ".json"))
            expected = set(meta["expected_block_ids"])
            session = read(path.parents[2] / "session.json")
            calibration = session.get("calibrations", {}).get(meta.get("calibration_id"))
            calibration_count += calibration is not None
            if meta.get("calibration_id") and calibration is None:
                problems.add("missing_calibration_reference")
            if meta.get("baseline"):
                baseline_count += 1
                base = read(path.parent / meta["baseline"]["file"])
                if base.get("status") != "ready":
                    notes.add("trial_baseline_not_ready")
            annotation = meta.get("annotation", {})
            start, end = annotation.get("action_start_s"), annotation.get("action_end_s")
            if start is not None and end is not None:
                annotation_count += 1
                # 自动检测写入的区间未经人工确认，单独计数以便区分可信度。
                if annotation.get("status") == "auto":
                    auto_annotation_count += 1
                    if annotation.get("auto", {}).get("open_at_end"):
                        notes.add("auto_bounds_open_at_end")
                if not 0 <= start < end <= meta["duration_s"]:
                    problems.add("invalid_annotation_bounds")
            else:
                notes.add("no_contact_bounds")
            raw = path.parent / meta["data_file"]
            hashes[hashlib.sha256(raw.read_bytes()).hexdigest()].append(rel)
            frames = Counter()
            previous = {}
            gaps = defaultdict(float)
            row_count = 0
            with raw.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader)
                columns = meta["max_adc_columns"]
                if header != ["record_index", "time_s", "block_id"] + [f"adc_{i:03}" for i in range(columns)]:
                    problems.add("invalid_csv_header")
                for index, row in enumerate(reader):
                    row_count += 1
                    if len(row) != columns + 3:
                        problems.add("invalid_row_width")
                        continue
                    if int(row[0]) != index:
                        problems.add("non_contiguous_record_index")
                    t, key = float(row[1]), int(row[2])
                    if not math.isfinite(t) or not 0 <= t <= meta["duration_s"] + .001:
                        problems.add("invalid_time")
                    if key in previous:
                        if t < previous[key]:
                            problems.add("block_time_reversal")
                        gaps[key] = max(gaps[key], t - previous[key])
                    previous[key] = t
                    frames[key] += 1
                    if key not in expected or str(key) not in layout["blocks"]:
                        problems.add("unknown_block")
                        continue
                    spec = layout["blocks"][str(key)]
                    size = spec["rows"] * spec["cols"]
                    values = row[3:3 + size]
                    if any(not value or not 0 <= int(value) <= 4095 for value in values):
                        problems.add("invalid_adc")
                    if any(row[3 + size:]):
                        problems.add("nonempty_padding")
            total_rows += row_count
            if row_count != meta["summary"]["records"]:
                problems.add("row_count_mismatch")
            if dict(frames) != {int(k): v for k, v in meta["summary"]["frames_by_block"].items()}:
                problems.add("block_count_mismatch")
            if expected != set(frames):
                problems.add("missing_block")
            if meta.get("quality", {}).get("dropped_records", 0):
                problems.add("reported_dropped_rows")
            files.append({"sample": rel, "rows": row_count, "max_gap_s": max(gaps.values(), default=0.),
                          "subject_id": meta["subject_id"], "session_id": meta["session_id"],
                          "action": meta["action"]["name"], "errors": sorted(problems), "notes": sorted(notes)})
        except Exception as error:
            problems.add(f"read_or_parse_error: {error}")
        if problems:
            errors.append({"sample": rel, "errors": sorted(problems)})
        if notes:
            warnings.append({"sample": rel, "notes": sorted(notes)})
    return {"dataset": str(root), "counts": {key: dict(value) for key, value in counts.items()},
            "samples": sum(counts["schema"].values()), "raw_rows": total_rows,
            "annotated_samples": annotation_count, "auto_annotation_samples": auto_annotation_count,
            "connection_calibration_samples": calibration_count,
            "trial_baseline_samples": baseline_count, "errors": errors, "warnings": warnings,
            "duplicate_raw_groups": [paths for paths in hashes.values() if len(paths) > 1],
            "files": files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.dataset)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {key: value for key, value in result.items() if key not in ("files", "warnings")}
    print(json.dumps(compact, ensure_ascii=True, indent=2))
