"""Read a bounded number of real scans; never calibrates or changes firmware."""

import argparse
import json
from threading import Event, Timer

from skin_draw.acquisition import SerialAcquisition

parser = argparse.ArgumentParser()
parser.add_argument("--port", required=True)
parser.add_argument("--scans", type=int, default=3)
parser.add_argument("--timeout", type=float, default=45)
parser.add_argument("--mode", choices=("auto", "poll", "stream"), default="auto")
args = parser.parse_args()
stop = Event()
deadline = Timer(args.timeout, stop.set)
deadline.daemon = True
records = []


def receive(scan):
    record = {"sequence": scan.frame.sequence, "expected_ids": scan.expected_ids,
              "blocks": [{"id": b.block_id, "shape": b.values.shape} for b in scan.frame.blocks],
              "complete": scan.complete, "seconds": round(scan.elapsed, 3),
              "protocol_errors": scan.protocol_errors}
    records.append(record)
    print(json.dumps(record), flush=True)
    if len(records) >= args.scans:
        stop.set()


deadline.start()
try:
    SerialAcquisition(args.port, stop, receive, lambda message: None, mode=args.mode).run()
finally:
    stop.set()
    deadline.cancel()
if len(records) != args.scans:
    raise SystemExit("Did not receive the requested number of scans before timeout")
