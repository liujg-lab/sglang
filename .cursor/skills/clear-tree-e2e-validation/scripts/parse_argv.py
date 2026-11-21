#!/usr/bin/env python3
"""Parse a ClearTree server argv and print the resolved settings.

Does not start a server. Requires ``PYTHONPATH=python``. Draft uses
``on_npu=False``; target uses ``on_npu=True``.
"""

from __future__ import annotations

import argparse
import sys


def cli_role(argv: list[str]) -> str | None:
    for index, item in enumerate(argv):
        if item == "--clear-tree-role" and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith("--clear-tree-role="):
            return item.split("=", 1)[1]
    return None


def comma(values) -> str:
    if not values:
        return ""
    return ",".join(str(item) for item in values)


def main() -> None:
    if "--" not in sys.argv:
        raise SystemExit(
            "usage: parse_argv.py --role {draft|target} -- <server argv>"
        )
    cut = sys.argv.index("--")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("draft", "target"))
    args = parser.parse_args(sys.argv[1:cut])
    server_argv = sys.argv[cut + 1 :]
    given = cli_role(server_argv)
    if given is not None and given != args.role:
        raise SystemExit(f"role flag is {args.role}, argv role is {given}")
    from sglang.srt.server_args import ServerArgs, prepare_server_args
    from sglang.srt.speculative.clear_tree_remote.clear_tree_args import (
        apply_clear_tree_settings,
    )

    on_npu = args.role == "target"

    def apply_for_role(server_args) -> None:
        apply_clear_tree_settings(server_args, on_npu=on_npu)

    # prepare_server_args applies ClearTree rules from the local device.
    # This check follows --role instead, so a target argv is page 128 even
    # when the parser runs on the draft machine.
    ServerArgs._handle_clear_tree_remote = apply_for_role
    try:
        server_args = prepare_server_args(server_argv)
        if server_args.clear_tree_role != args.role:
            raise SystemExit(
                f"role flag is {args.role}, argv role is {server_args.clear_tree_role}"
            )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    rows = (
        ("role", server_args.clear_tree_role),
        ("capacity", server_args.speculative_num_draft_tokens),
        ("widths", comma(server_args.clear_tree_widths)),
        ("ks", comma(server_args.clear_tree_ks)),
        ("depth", server_args.clear_tree_max_depth),
        ("steps", server_args.speculative_num_steps),
        ("cuda_graph_bs", comma(server_args.cuda_graph_bs)),
        ("page", server_args.page_size),
        ("attention_backend", server_args.attention_backend),
        ("mm_attention_backend", server_args.mm_attention_backend),
    )
    sys.stdout.write("".join(f"{key} {value}\n" for key, value in rows))


if __name__ == "__main__":
    main()
