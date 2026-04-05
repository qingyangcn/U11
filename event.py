from __future__ import annotations

import heapq
from typing import List, Tuple

from core.entities import Event
from core.enums import EVENT_PRIORITIES


def _event_tie_key(event: Event) -> tuple[str, str, str]:
    payload = event.payload or {}
    return (
        str(payload.get("drone_id", "")),
        str(payload.get("order_id", "")),
        str(payload.get("station_id", "")),
    )


class EventQueue:
    def __init__(self) -> None:
        self._heap: List[Tuple[float, int, tuple[str, str, str], int, Event]] = []
        self._counter = 0

    def push(self, event: Event) -> None:
        priority = event.priority
        if priority == 100:
            priority = EVENT_PRIORITIES.get(event.event_type, 100)
        tie_key = _event_tie_key(event)
        heapq.heappush(self._heap, (event.time, priority, tie_key, self._counter, event))
        self._counter += 1

    def pop(self) -> Event:
        return heapq.heappop(self._heap)[-1]

    def peek(self) -> Event | None:
        if not self._heap:
            return None
        return self._heap[0][-1]

    def empty(self) -> bool:
        return not self._heap

    def clear(self) -> None:
        self._heap.clear()
