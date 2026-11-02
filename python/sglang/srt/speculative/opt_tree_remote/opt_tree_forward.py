"""One OPT-Tree forward per layer.

The committed tail is scored once. Each non-final ``update`` then runs N
tokens at one shared position under ``ancestor_mask``. Speculative KV is
appended past the committed prefix and dropped by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch

from sglang.srt.speculative.opt_tree_remote.opt_tree_config import OptTreeConfig
from sglang.srt.speculative.opt_tree_remote.opt_tree_grow import grow_opt_tree


def layer_positions(prefix_len: int, depth: int, width: int) -> torch.Tensor:
    """Absolute token index shared by one OPT layer.

    ``depth`` is 1-based. The first frontier sits at ``prefix_len``, which is
    the next index after the committed tail.
    """
    if int(depth) < 1 or int(width) < 1:
        raise ValueError(f"invalid OPT layer depth={depth} width={width}")
    return torch.full(
        (int(width),),
        int(prefix_len) + int(depth) - 1,
        dtype=torch.long,
    )


def layer_attention_mask(prefix_len: int, ancestor: torch.Tensor) -> torch.Tensor:
    """Flattened TARGET_VERIFY mask: committed prefix, then the ancestor mask.

    ``ancestor`` is ``[N, depth * N]`` with 1 for a visible tree column.
    Every query sees the whole committed prefix. Tree columns stay
    ancestor-only (``leak=False``).
    """
    if ancestor.dim() != 2:
        raise ValueError(
            f"ancestor mask must be [N, cols], got {tuple(ancestor.shape)}"
        )
    n = int(ancestor.shape[0])
    prefix = torch.ones(
        (n, int(prefix_len)),
        dtype=torch.bool,
        device=ancestor.device,
    )
    tree = ancestor.to(dtype=torch.bool)
    return torch.cat([prefix, tree], dim=1).reshape(-1)


def materialize_missing_prefix(scheduler, reqs) -> None:
    """Build draft KV before ingest.

    ``_sr_kv_len`` treats a request with no pool slot as already cached, so
    OPT checks ``req_pool_idx`` itself and does not change that helper.
    """
    missing = [req for req in reqs if req.req_pool_idx is None]
    if not missing:
        return
    scheduler._sr_materialize_prefix_batch(missing)
    for req in missing:
        if req.req_pool_idx is None:
            raise RuntimeError(f"OPT draft {req.rid} prefix KV was not materialized")


def produce_opt_windows(scheduler, action, pairs) -> list:
    """PREFILL only materializes KV. STEP materializes, ingests, then grows."""
    from sglang.srt.speculative.standalone_remote.sr_protocol import SRAction

    if not pairs:
        return []
    reqs = [req for _, req in pairs]
    if action == SRAction.PREFILL:
        materialize_missing_prefix(scheduler, reqs)
        return [([], None, None)] * len(pairs)
    if action != SRAction.STEP:
        return [([], None, None)] * len(pairs)
    materialize_missing_prefix(scheduler, reqs)
    scheduler._sr_ingest_committed_batch(reqs)
    scheduler._sr_replay_grammars(reqs)
    return [scheduler._opt_expand_one(req) for req in reqs]


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    """Temperature-1 full-vocabulary probabilities. No top-N renormalization."""
    return torch.softmax(logits.detach().float(), dim=-1)


def grow_with_forwards(
    config: OptTreeConfig,
    tail_logits: torch.Tensor,
    forward_layer: Callable,
) -> Tuple[List[int], List[int], List[int]]:
    """Seed from the committed tail, then one ``forward_layer`` per kept layer.

    ``forward_layer(step)`` receives the non-final step and returns logits for
    its N tokens. The stopping ``update`` is top-N only and is not forwarded.
    """

    def next_probs(step, state):
        if step is None or state is None:
            return softmax_probs(tail_logits)
        return softmax_probs(forward_layer(step))

    return grow_opt_tree(config, next_probs)


@dataclass
class OptLayerSpec:
    """Fields TARGET_VERIFY attention already reads. Not an EAGLE verify input."""

    custom_mask: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    num_tokens_per_req: int
    capture_hidden_mode: object = None

    def is_draft_input(self) -> bool:
        return False

    def get_spec_adjusted_global_num_tokens(self, batch):
        return batch.global_num_tokens, batch.global_num_tokens_for_logprob


class OptLayerForward:
    """Tail logits plus one N-token extend per OPT layer. Batch size is 1."""

    def __init__(self, scheduler, req, prefix_len: int):
        self.scheduler = scheduler
        self.req = req
        self.prefix_len = int(prefix_len)
        self.cached_len = int(prefix_len)

    def logits_at_tail(self) -> torch.Tensor:
        """Recompute the committed tail in its existing KV slot."""
        if self.prefix_len < 1:
            raise RuntimeError(f"OPT draft {self.req.rid} has an empty prefix")
        if self.req.req_pool_idx is None:
            raise RuntimeError(f"OPT draft {self.req.rid} has no KV pool slot")
        pool = self.scheduler.req_to_token_pool.req_to_token
        slot = pool[self.req.req_pool_idx, self.prefix_len - 1]
        if int(slot.item()) < 0:
            raise RuntimeError(f"OPT draft {self.req.rid} tail KV slot is empty")
        batch = self.scheduler._sr_make_decode_batch([self.req])
        batch.input_ids = batch.output_ids
        batch.out_cache_loc = slot.reshape(1)
        batch.output_ids = None
        _keep_raw_logits(batch)
        return _row_logits(self._run(batch), 1, tail=True)

    def forward_layer(self, step) -> torch.Tensor:
        """Forward the N tokens ``update`` just kept. KV stays uncommitted."""
        tokens = step.tokens
        if not torch.is_tensor(tokens):
            tokens = torch.tensor(list(tokens), dtype=torch.long)
        n = int(tokens.shape[0])
        width = int(self.scheduler.server_args.speculative_num_draft_tokens)
        if n != width:
            raise RuntimeError(
                f"OPT layer has {n} tokens but attention was built for {width}"
            )
        depth = int(step.positions.reshape(-1)[0].item())
        ancestor = step.ancestor_mask
        if int(ancestor.shape[0]) != n or int(ancestor.shape[1]) != depth * n:
            raise RuntimeError(
                f"OPT ancestor mask {tuple(ancestor.shape)} != [{n}, {depth * n}]"
            )
        if self.cached_len + n != self.prefix_len + depth * n:
            raise RuntimeError(
                f"OPT cache length {self.cached_len} is not layer {depth} "
                f"after prefix {self.prefix_len}"
            )
        device = self.scheduler.req_to_token_pool.device
        positions = layer_positions(self.prefix_len, depth, n).to(device)
        mask = layer_attention_mask(self.prefix_len, ancestor).to(device)
        out_cache_loc = self._alloc_layer_slots(n)
        self.req.kv_allocated_len = self.cached_len + n
        batch = self._verify_batch(tokens.to(device), positions, mask, out_cache_loc)
        logits = _row_logits(self._run(batch), n, tail=False)
        self.cached_len += n
        return logits

    def _alloc_layer_slots(self, n: int) -> torch.Tensor:
        from sglang.srt.mem_cache.common import (
            alloc_paged_token_slots_extend,
            alloc_token_slots,
            get_last_loc,
        )
        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        scheduler = self.scheduler
        tree_cache = scheduler.tree_cache
        page = int(getattr(tree_cache, "page_size", 1) or 1)
        device = scheduler.req_to_token_pool.device
        seq = torch.tensor([self.cached_len], dtype=torch.int64, device=device)
        pool_index = torch.tensor(
            [int(self.req.req_pool_idx)], dtype=torch.int64, device=device
        )
        if page == 1:
            out_cache_loc = alloc_token_slots(tree_cache, n)
        else:
            seq_cpu = torch.tensor([self.cached_len], dtype=torch.int64)
            end = seq + n
            end_cpu = seq_cpu + n
            last = get_last_loc(
                scheduler.req_to_token_pool.req_to_token,
                pool_index,
                seq,
            )
            out_cache_loc = alloc_paged_token_slots_extend(
                tree_cache,
                seq,
                seq_cpu,
                end,
                end_cpu,
                last,
                n,
            )
        if int(out_cache_loc.numel()) != n:
            raise RuntimeError(
                f"OPT layer allocated {int(out_cache_loc.numel())} slots for {n} tokens"
            )
        assign_req_to_token_pool_func(
            pool_index,
            scheduler.req_to_token_pool.req_to_token,
            seq,
            seq + n,
            out_cache_loc,
            1,
        )
        return out_cache_loc

    def _verify_batch(self, tokens, positions, mask, out_cache_loc):
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardMode,
        )

        n = int(tokens.shape[0])
        batch = self.scheduler._sr_make_decode_batch([self.req])
        device = batch.seq_lens.device
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.global_forward_mode = ForwardMode.TARGET_VERIFY
        batch.input_ids = tokens.to(dtype=torch.int64, device=device).reshape(-1)
        batch.output_ids = None
        batch.seq_lens = torch.tensor([self.cached_len], dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor([self.cached_len], dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(
            [self.cached_len], dtype=torch.int32, device=device
        )
        batch.seq_lens_sum = int(self.cached_len)
        batch.out_cache_loc = out_cache_loc
        batch.extend_lens = [n]
        batch.prefix_lens = [self.cached_len]
        batch.extend_num_tokens = n
        batch.extend_logprob_start_lens = [self.cached_len]
        batch.spec_info = OptLayerSpec(
            custom_mask=mask.reshape(-1),
            positions=positions.to(dtype=torch.int64, device=device).reshape(-1),
            draft_token_num=n,
            num_tokens_per_req=n,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        _keep_raw_logits(batch)
        return batch

    def _run(self, batch) -> torch.Tensor:
        self.scheduler.cur_batch = batch
        result = self.scheduler.run_batch(batch)
        logits = getattr(getattr(result, "logits_output", None), "next_token_logits", None)
        if logits is None:
            raise RuntimeError(f"OPT draft {self.req.rid} forward returned no logits")
        return logits


def _keep_raw_logits(batch) -> None:
    """Leave next-token logits unscaled so OPT can softmax at temperature 1.

    The sampler turns non-greedy logits into probabilities in place. OPT's
    path score is the full-vocabulary temperature-1 softmax, so this batch
    only argmaxes and discards that token.
    """
    info = batch.sampling_info
    info.is_all_greedy = True
    info.need_top_p_sampling = False
    info.need_top_k_sampling = False
    info.need_min_p_sampling = False
    info.temperatures = torch.ones_like(info.temperatures)
    batch.has_grammar = False
    batch.return_logprob = False
    batch.return_hidden_states = False


def _row_logits(logits: torch.Tensor, n: int, *, tail: bool) -> torch.Tensor:
    t = logits.detach()
    if t.dim() == 3 and int(t.shape[0]) == 1:
        t = t[0]
    if tail:
        if t.dim() == 2:
            return t[-1]
        if t.dim() != 1:
            raise RuntimeError(f"OPT tail logits rank {t.dim()} is not 1 or 2")
        return t
    if t.dim() != 2 or int(t.shape[0]) != n:
        raise RuntimeError(f"OPT layer logits shape {tuple(t.shape)} != [{n}, vocab]")
    return t
