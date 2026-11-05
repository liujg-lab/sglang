"""One-line [TOLL round] summary. No device counters."""

from __future__ import annotations


class TollMetrics:
    """Keep the latest round line for tests and logs."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def record(
        self,
        *,
        lam: float,
        elapsed_ms: float,
        expected_alen: float,
        actual_gain: float,
        layers,
        k: int,
        stop_reasons: dict[str, int],
        rtt_ms: float,
        target_ms: float,
        ready: bool,
        cost,
    ) -> str:
        widths = ",".join(str(int(layer.width)) for layer in layers) or "-"
        active = ",".join(str(int(layer.active)) for layer in layers) or "-"
        reasons = ",".join(f"{name}:{count}" for name, count in sorted(stop_reasons.items()))
        line = (
            f"[TOLL round] lam={float(lam):.4f} T={float(elapsed_ms):.3f} "
            f"alen={float(expected_alen):.4f} W={widths} active={active} "
            f"K={int(k)} stop={reasons or '-'} ghat={float(expected_alen):.4f} "
            f"gain={float(actual_gain):.4f} R={float(rtt_ms):.3f} "
            f"target={float(target_ms):.3f} w={float(cost.w):.4f} "
            f"beta={float(cost.beta):.4f} ready={bool(ready)}"
        )
        self.lines.append(line)
        return line
