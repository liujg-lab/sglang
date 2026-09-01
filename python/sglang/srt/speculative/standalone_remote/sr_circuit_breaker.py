"""Sync-RPC circuit breaker for STANDALONE_REMOTE Target.

Timeouts skip the next drafts instead of blocking rpc_timeout_ms every step.
REJECT / stale replies are not failures: Draft is still alive.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class SRRpcBreaker:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_steps: int = 32,
    ) -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_steps = max(1, int(cooldown_steps))
        self.state = self.CLOSED
        self.consecutive_failures = 0
        self.steps_in_open = 0

    def reset(self) -> None:
        self.state = self.CLOSED
        self.consecutive_failures = 0
        self.steps_in_open = 0

    def should_send(self) -> bool:
        if self.state == self.CLOSED:
            return True
        if self.state == self.HALF_OPEN:
            return True
        return False

    def note_skipped_step(self) -> None:
        if self.state != self.OPEN:
            return
        self.steps_in_open += 1
        if self.steps_in_open >= self.cooldown_steps:
            self.state = self.HALF_OPEN
            logger.info(
                "[SR] breaker OPEN -> HALF_OPEN after %s skipped steps",
                self.steps_in_open,
            )

    def record_success(self) -> None:
        if self.state != self.CLOSED:
            logger.info("[SR] breaker %s -> CLOSED", self.state)
        self.state = self.CLOSED
        self.consecutive_failures = 0
        self.steps_in_open = 0

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.state == self.OPEN:
            # PREFILL still probes while OPEN; do not reset decode cooldown.
            return
        if self.state == self.HALF_OPEN:
            self.state = self.OPEN
            self.steps_in_open = 0
            logger.info("[SR] breaker HALF_OPEN -> OPEN (probe timeout)")
            return
        if self.consecutive_failures >= self.failure_threshold:
            logger.info(
                "[SR] breaker CLOSED -> OPEN after %s timeouts, cooldown %s steps",
                self.consecutive_failures,
                self.cooldown_steps,
            )
            self.state = self.OPEN
            self.steps_in_open = 0
