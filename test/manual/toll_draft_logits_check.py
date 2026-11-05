"""Compare one draft layer with ordinary decode. No target process is involved.

W=2. Row 0 is the real token and must match the decode logits of that same
token. Row 1 is padding and is ignored. A mismatch here is a KV, mask, or
padding bug, not a weak draft model.

Save the two tensors from a live draft run, then:

    python3 test/manual/toll_draft_logits_check.py \
        --layer /tmp/toll-layer.pt --decode /tmp/toll-decode.pt
"""

from __future__ import annotations

import argparse

import torch


def row_matches_decode(layer_logits, decode_logits, atol: float = 1e-4, rtol: float = 1e-4) -> bool:
    """True when the first layer row matches the ordinary-decode distribution."""
    layer = torch.as_tensor(layer_logits).float()
    decode = torch.as_tensor(decode_logits).float().reshape(-1, layer.shape[-1])
    if layer.ndim != 2 or layer.shape[0] < 1:
        raise ValueError(f"layer logits must be [width, vocab], got {tuple(layer.shape)}")
    return bool(torch.allclose(layer[0], decode[-1], atol=float(atol), rtol=float(rtol)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", required=True)
    parser.add_argument("--decode", required=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    layer = torch.load(args.layer, map_location="cpu")
    decode = torch.load(args.decode, map_location="cpu")
    if not row_matches_decode(layer, decode, atol=args.atol, rtol=args.rtol):
        raise SystemExit("draft layer row 0 does not match ordinary decode")
    print("toll draft logits: row 0 matches, padding row ignored")


if __name__ == "__main__":
    main()
