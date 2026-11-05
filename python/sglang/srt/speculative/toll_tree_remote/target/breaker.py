"""Target RPC breaker. A timeout opens it; cooldown allows one probe."""

from __future__ import annotations


class TollRpcBreaker:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 3, cooldown_steps: int = 32) -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_steps = max(1, int(cooldown_steps))
        self.state = self.CLOSED
        self.consecutive_failures = 0
        self.steps_in_open = 0

    def should_send(self) -> bool:
        return self.state != self.OPEN

    def note_skipped_step(self) -> None:
        if self.state != self.OPEN:
            return
        self.steps_in_open += 1
        if self.steps_in_open >= self.cooldown_steps:
            self.state = self.HALF_OPEN

    def record_success(self) -> None:
        self.state = self.CLOSED
        self.consecutive_failures = 0
        self.steps_in_open = 0

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.state == self.HALF_OPEN or self.consecutive_failures >= self.failure_threshold:
            self.state = self.OPEN
            self.steps_in_open = 0
