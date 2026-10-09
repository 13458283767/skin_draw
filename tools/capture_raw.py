"""Bounded passive serial capture before calibration or display filtering."""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter

from skin_draw.protocol import BlockParser
from skin_draw.transport import open_port


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parents[1] / "artifacts" / ("raw-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    folder.mkdir(parents=True)
    meta = {"port": args.port, "baud": 115200, "format": "8N1", "mode": "passive",
            "started_local": datetime.now().isoformat(), "requested_seconds": args.seconds,
            "operator_instruction": "保持静置；是否实际无按压未自动验证", "filters": "none"}
    print(str(folder), flush=True)
    started = perf_counter()
    decoder = BlockParser()
    counts, low_rows = Counter(), Counter()
    total_bytes = sequence = 0
    last_status = 0.
    try:
        with open_port(args.port) as port, (folder / "rx.bin").open("wb") as binary, \
                (folder / "chunks.jsonl").open("w", encoding="utf-8") as chunks, \
                (folder / "blocks.jsonl").open("w", encoding="utf-8") as frames:
            # Keep even startup bytes. Never reset/drain input or send commands.
            while perf_counter() - started < args.seconds:
                data = port.read(max(1, min(port.in_waiting, 4096)))
                elapsed = perf_counter() - started
                if data:
                    binary.write(data)
                    chunks.write(json.dumps({"seconds": elapsed, "offset": total_bytes, "size": len(data)}) + "\n")
                    total_bytes += len(data)
                    for block in decoder.feed(data):
                        sequence += 1
                        counts[block.block_id] += 1
                        low_rows[block.block_id] += int((block.values.max(axis=1) <= 1300).sum())
                        frames.write(json.dumps({"sequence": sequence, "seconds": elapsed,
                                                 "block_id": block.block_id, "values": block.values.tolist()}) + "\n")
                if elapsed - last_status >= 10:
                    last_status = elapsed
                    binary.flush()
                    frames.flush()
                    chunks.flush()
                    print(json.dumps({"seconds": round(elapsed, 1), "blocks": counts,
                                      "rows_all_le_1300": low_rows, "parse_errors": decoder.errors}), flush=True)
    except Exception as error:
        meta["error"] = str(error)
        raise
    finally:
        meta.update({"elapsed_seconds": perf_counter() - started, "bytes": total_bytes,
                     "block_counts": counts, "parse_errors": decoder.errors,
                     "incomplete_block_at_end": decoder.current is not None,
                     "partial_line_bytes_at_end": len(decoder.buffer)})
        (folder / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
