"""Match robot events to camera frames by timestamp, like matching spans in a trace.

Each camera keeps a short ring buffer of (stamp_ns, frame). When a robot event
arrives, we take the frame closest in time, and refuse it if the gap is larger
than the tolerance (a stale frame would make the check lie).
"""
from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Match(Generic[T]):
    item: T
    stamp_ns: int
    gap_ms: float


class StampBuffer(Generic[T]):
    def __init__(self, maxlen: int = 120):
        self._buf: deque[tuple[int, T]] = deque(maxlen=maxlen)

    def push(self, stamp_ns: int, item: T) -> None:
        if self._buf and stamp_ns < self._buf[-1][0]:
            # out-of-order frame: insert in place to keep the buffer sorted
            items = list(self._buf)
            idx = bisect.bisect([s for s, _ in items], stamp_ns)
            items.insert(idx, (stamp_ns, item))
            self._buf = deque(items[-self._buf.maxlen:], maxlen=self._buf.maxlen)
        else:
            self._buf.append((stamp_ns, item))

    def nearest(self, stamp_ns: int, tolerance_ms: float) -> Match[T] | None:
        if not self._buf:
            return None
        best = min(self._buf, key=lambda si: abs(si[0] - stamp_ns))
        gap_ms = abs(best[0] - stamp_ns) / 1e6
        if gap_ms > tolerance_ms:
            return None
        return Match(best[1], best[0], gap_ms)

    def latest_after(self, stamp_ns: int) -> Match[T] | None:
        """First frame at or after an event: use for post-place, when the scene settles after the event."""
        for s, item in self._buf:
            if s >= stamp_ns:
                return Match(item, s, (s - stamp_ns) / 1e6)
        return None

    def __len__(self) -> int:
        return len(self._buf)
