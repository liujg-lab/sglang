"""Draft peer for a live Toll target. It does not load a model.

Run the target alone first and save its greedy token ids:

    python3 test/manual/toll_fake_draft.py --save-baseline /tmp/toll-gold.json --tokens 1,2,3,10,11

Then start this process before the target connects, and point the target at
``--addr`` / ``--port``. Read ``[TOLL target]`` in the target log.

* gold: the chain is the baseline after ``commit_len``. Output matches the
  target-only run, ``verify_rounds > 0``, and the accept count equals the depth.
* wrong: every draft token is off by one. Output still matches, accept count is 0,
  and each round emits one token.
* pad: only the first future token is linked. Output matches, and the unlinked
  padding slot is not accepted.
* silent: no reply. Output matches ordinary decode, the breaker opens, and it
  closes again after this process is restarted in another mode.
"""

from __future__ import annotations

import argparse
import json

from sglang.srt.speculative.toll_tree_remote.protocol import (
    PROTO_VERSION,
    TollAction,
    TollReply,
    TollReplyItem,
    TollStatus,
)


def save_baseline(path: str, tokens: list[int]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([int(token) for token in tokens], handle)


def load_baseline(path: str) -> list[int]:
    with open(path, encoding="utf-8") as handle:
        return [int(token) for token in json.load(handle)]


def draft_window(mode: str, baseline: list[int], commit_len: int, depth: int):
    """Return ``(tokens, parents)``. Silent mode returns ``None``."""
    if mode == "silent":
        return None
    depth = max(1, int(depth))
    start = max(0, int(commit_len))
    future = list(baseline)[start : start + depth]
    if mode == "gold":
        tokens = future
    elif mode == "wrong":
        tokens = [int(token) + 1 for token in future] or [1] * depth
    elif mode == "pad":
        tokens = future[:1]
    else:
        raise ValueError(f"unknown mode {mode}")
    if not tokens:
        return [], []
    parents = [-1] + list(range(len(tokens) - 1))
    return tokens, parents


def serve(url: str, mode: str, baseline: list[int], depth: int) -> None:
    from sglang.srt.speculative.toll_tree_remote.transport import TollDraftServer

    server = TollDraftServer(url)
    print(f"toll fake draft listening on {url} mode={mode}", flush=True)
    try:
        while True:
            request = server.recv_request(timeout_ms=1000)
            if request is None or not request.reqs:
                continue
            item = request.reqs[0]
            if request.action in (TollAction.PREFILL, TollAction.FINISH, TollAction.ABORT):
                server.send_reply(_empty(request, TollStatus.OK))
                continue
            window = draft_window(mode, baseline, item.commit_len, depth)
            if window is None:
                continue
            tokens, parents = window
            status = TollStatus.OK if tokens else TollStatus.EMPTY
            server.send_reply(
                TollReply(
                    proto=PROTO_VERSION,
                    session=request.session,
                    seq=int(request.seq),
                    verify_bucket=max(len(tokens), 1),
                    residence_ns=server.residence_ns(),
                    reqs=[
                        TollReplyItem(
                            rid=item.rid,
                            base_len=int(item.commit_len),
                            tokens=[int(token) for token in tokens],
                            parents=[int(parent) for parent in parents],
                            status=status,
                        )
                    ],
                )
            )
    finally:
        server.close()


def _empty(request, status: str) -> TollReply:
    item = request.reqs[0]
    return TollReply(
        proto=PROTO_VERSION,
        session=request.session,
        seq=int(request.seq),
        verify_bucket=1,
        residence_ns=0,
        reqs=[
            TollReplyItem(
                rid=item.rid,
                base_len=int(item.commit_len),
                tokens=[],
                parents=[],
                status=status,
            )
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30039)
    parser.add_argument("--mode", choices=("gold", "wrong", "pad", "silent"), default="gold")
    parser.add_argument("--baseline", default="")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--save-baseline", default="")
    parser.add_argument("--tokens", default="")
    args = parser.parse_args()
    if args.save_baseline:
        tokens = [int(part) for part in args.tokens.split(",") if part]
        save_baseline(args.save_baseline, tokens)
        print(f"wrote {len(tokens)} tokens to {args.save_baseline}")
        return
    if not args.baseline:
        raise SystemExit("--baseline is required unless --save-baseline is set")
    serve(f"tcp://{args.addr}:{args.port}", args.mode, load_baseline(args.baseline), args.depth)


if __name__ == "__main__":
    main()
