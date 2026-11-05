"""Tree-area KV slots. Padding uses a slot too, and rollback frees all of them."""

from __future__ import annotations


class TreeKV:
    """Track one request's draft slots above its prefix.

    ``alloc(width)`` must return ``width`` slot ids or fewer when the budget
    is exhausted. A short allocation is not kept.
    """

    def __init__(self, alloc, free) -> None:
        self._alloc = alloc
        self._free = free
        self.layers: list[list[int]] = []

    def alloc_layer(self, width: int) -> list[int] | None:
        slots = list(self._alloc(int(width)) or [])
        if len(slots) < int(width):
            if slots:
                self._free(slots)
            return None
        self.layers.append(slots)
        return slots

    def rollback(self) -> int:
        freed = 0
        while self.layers:
            slots = self.layers.pop()
            self._free(slots)
            freed += len(slots)
        return freed
