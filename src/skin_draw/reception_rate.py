"""Exact counts of valid block matrices received in the last second.

Owned by LiveBuffer and accessed under its lock. Independent of rendering,
baseline collection and outlier filtering; no Qt or extra worker thread.
"""

from collections import deque


class ReceptionRate:
    def __init__(self):
        self.arrivals: dict[int, deque[float]] = {}

    @staticmethod
    def _expire(times: deque[float], now: float):
        while times and times[0] <= now - 1.0:
            times.popleft()

    def record(self, block_id: int, now: float):
        times = self.arrivals.setdefault(block_id, deque())
        self._expire(times, now)
        times.append(now)

    def counts(self, now: float) -> dict[int, int]:
        result = {}
        for block_id, times in sorted(self.arrivals.items()):
            self._expire(times, now)
            result[block_id] = len(times)
        return result
