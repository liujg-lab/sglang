"""Scripted draft. It does not load a model.

Four checks: a correct greedy chain is accepted, every wrong token accepts
nothing, a real token parked in a padding slot is not accepted, and a silent
draft opens the breaker and then lets it recover.
"""

from __future__ import annotations

from sglang.srt.speculative.toll_tree_remote.target.breaker import TollRpcBreaker
from sglang.srt.speculative.toll_tree_remote.target.worker import pack_draft_tokens
from sglang.srt.speculative.toll_tree_remote.tree.layout import build_verify_layout


def accepted_draft_tokens(gold, tokens, parents, num_verify: int, packed=None) -> int:
    """Walk the linked verify slots. Padding is not part of that walk."""
    layout = build_verify_layout([list(parents)], [max(len(list(gold)), 1)], int(num_verify))
    row = packed if packed is not None else pack_draft_tokens([0], [list(tokens)], int(num_verify))
    nxt = layout.retrive_next_token[0]
    node = 0
    accepted = 0
    for expected in list(gold):
        jump = int(nxt[node])
        if jump <= 0 or jump > len(list(tokens)):
            break
        if int(row[jump]) != int(expected):
            break
        accepted += 1
        node = jump
    return accepted


def padding_holds(tokens, planted, num_verify: int) -> list[int]:
    """Put ``planted`` in the first padding slot of the packed verify row."""
    packed = pack_draft_tokens([0], [list(tokens)], int(num_verify))
    slot = len(list(tokens)) + 1
    if slot < int(num_verify):
        packed[slot] = int(planted)
    return packed


class RoundStats:
    def __init__(self) -> None:
        self.verify_rounds = 0
        self.accepted_draft_tokens = 0
        self.emitted = 0
        self.opened = False
        self.recovered = False


def run_scripted(mode: str, rounds: int = 4, threshold: int = 3, cooldown: int = 2) -> RoundStats:
    """One scripted target loop. ``mode`` is gold, wrong, pad, or silent."""
    gold = [10, 11, 12]
    chain = ([10, 11, 12], [-1, 0, 1])
    wrong = ([99, 98, 97], [-1, 0, 1])
    pad = ([10], [-1])
    num_verify = 4
    breaker = TollRpcBreaker(failure_threshold=threshold, cooldown_steps=cooldown)
    stats = RoundStats()
    for _ in range(int(rounds)):
        if not breaker.should_send():
            breaker.note_skipped_step()
            stats.emitted += 1
            if breaker.state == TollRpcBreaker.OPEN:
                stats.opened = True
            continue
        if mode == "silent":
            breaker.record_failure()
            stats.emitted += 1
            if breaker.state == TollRpcBreaker.OPEN:
                stats.opened = True
            continue
        tokens, parents = {"gold": chain, "wrong": wrong, "pad": pad}[mode]
        packed = padding_holds(tokens, gold[1], num_verify) if mode == "pad" else None
        if packed is not None and gold[1] not in packed:
            raise RuntimeError("padding slot did not receive the planted token")
        accepted = accepted_draft_tokens(gold, tokens, parents, num_verify, packed)
        breaker.record_success()
        if breaker.state == TollRpcBreaker.CLOSED and stats.opened:
            stats.recovered = True
        stats.verify_rounds += 1
        stats.accepted_draft_tokens += accepted
        stats.emitted += accepted + 1
    if mode == "silent":
        # Cooldown already advanced on the skipped sends. One success probe closes it.
        if breaker.state == TollRpcBreaker.HALF_OPEN:
            breaker.record_success()
            stats.recovered = breaker.state == TollRpcBreaker.CLOSED
        elif breaker.state == TollRpcBreaker.OPEN:
            while breaker.state == TollRpcBreaker.OPEN:
                breaker.note_skipped_step()
            breaker.record_success()
            stats.recovered = breaker.state == TollRpcBreaker.CLOSED
            stats.opened = True
    return stats


def check() -> None:
    gold = run_scripted("gold")
    if gold.verify_rounds <= 0 or gold.accepted_draft_tokens <= 0:
        raise SystemExit("gold chain did not verify any draft token")
    if gold.accepted_draft_tokens != 3 * 4:
        raise SystemExit(f"gold accepted {gold.accepted_draft_tokens}, want 12")
    wrong = run_scripted("wrong")
    if wrong.accepted_draft_tokens != 0 or wrong.emitted != 4:
        raise SystemExit(f"wrong tokens accepted {wrong.accepted_draft_tokens} emitted {wrong.emitted}")
    pad = run_scripted("pad", rounds=1)
    if pad.accepted_draft_tokens != 1:
        raise SystemExit(f"padding token was accepted, count {pad.accepted_draft_tokens}")
    silent = run_scripted("silent", rounds=3, threshold=3, cooldown=2)
    if not silent.opened or not silent.recovered:
        raise SystemExit(f"breaker opened={silent.opened} recovered={silent.recovered}")
    if silent.verify_rounds != 0:
        raise SystemExit("a silent draft still counted a verify round")


if __name__ == "__main__":
    check()
    print("toll fake draft: 4 cases ok")
