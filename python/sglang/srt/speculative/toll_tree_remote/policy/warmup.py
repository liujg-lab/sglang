"""Rotate draft width and verify width until the cost model can be identified."""

from __future__ import annotations


class WarmupSchedule:
    """Pairs ``(W, K)`` from the configured buckets.

    A single batch size still sees more than one layer width and more than
    one verify width, which is what ``ready`` asks for.
    """

    def __init__(self, cfg) -> None:
        widths = [int(value) for value in cfg.width_buckets]
        verifies = [int(value) for value in cfg.verify_buckets]
        pairs = [(widths[i % len(widths)], verifies[i % len(verifies)]) for i in range(max(len(widths), len(verifies)))]
        extra = [(widths[(i + 1) % len(widths)], verifies[i % len(verifies)]) for i in range(len(verifies))]
        for pair in extra:
            if pair not in pairs:
                pairs.append(pair)
        self.pairs = tuple(pairs)
        self._cursor = 0

    def next(self) -> tuple[int, int]:
        pair = self.pairs[self._cursor % len(self.pairs)]
        self._cursor += 1
        return pair
