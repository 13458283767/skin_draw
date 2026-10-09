"""Bounded serial timing probe: no calibration or firmware writes."""

import argparse
import json
from time import monotonic, sleep

from skin_draw.protocol import BlockParser
from skin_draw.transport import open_port


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", type=int, default=1)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    with open_port(args.port) as port:
        port.timeout = .005
        port.reset_input_buffer()
        for sequence in range(args.count):
            decoder = BlockParser()
            started = monotonic()
            port.write(f"hello num {args.id}\r\n".encode("ascii"))
            received = bytearray()
            blocks = []
            first_byte = None
            while monotonic() - started < .8:
                data = port.read(max(1, min(port.in_waiting, 4096)))
                if data and first_byte is None:
                    first_byte = monotonic() - started
                received.extend(data)
                blocks.extend(decoder.feed(data))
                if blocks and (received.endswith(b"END\r\n") or not data):
                    break
            print(json.dumps({"sample": sequence + 1,
                              "first_byte_ms": round(first_byte * 1000, 2) if first_byte is not None else None,
                              "elapsed_ms": round((monotonic() - started) * 1000, 2),
                              "bytes": len(received),
                              "blocks": [{"id": b.block_id, "shape": b.values.shape} for b in blocks],
                              "errors": decoder.errors,
                              "prefix": bytes(received[:100]).decode("ascii", errors="replace")}), flush=True)
            sleep(.003)


if __name__ == "__main__":
    main()
