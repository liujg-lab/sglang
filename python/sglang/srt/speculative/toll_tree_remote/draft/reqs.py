"""Draft request and the prefix / layer forwards. No other algorithm is imported."""

from __future__ import annotations


def layer_positions(prefix_len: int, depth: int, width: int) -> list[int]:
    """One position per physical row. Depth 1 sits on the committed tail."""
    position = int(prefix_len) + int(depth) - 1
    return [position] * int(width)


def make_draft_req(rid: str, input_ids: list[int]):
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    origin = [int(token) for token in input_ids]
    req = Req(rid, "", origin, SamplingParams(temperature=0.0, top_k=1))
    req.origin_input_ids = origin
    req.output_ids = []
    req.fill_ids = list(origin)
    return req


def keep_raw_logits(batch) -> None:
    """Leave next-token logits as a temperature-1 distribution.

    The sampler rewrites non-greedy logits in place. Penalty, logit bias,
    grammar, and logprob are cleared on this batch only.
    """
    info = batch.sampling_info
    info.is_all_greedy = True
    info.need_top_p_sampling = False
    info.need_top_k_sampling = False
    info.need_min_p_sampling = False
    info.temperatures = info.temperatures.new_ones(info.temperatures.shape)
    info.acc_additive_penalties = None
    info.acc_scaling_penalties = None
    info.penalizer_orchestrator = None
    info.logit_bias = None
    info.vocab_mask = None
    batch.has_grammar = False
    batch.return_logprob = False
    batch.return_hidden_states = False


def _device(scheduler):
    device = getattr(scheduler, "device", None)
    if device is None:
        device = scheduler.tp_worker.model_runner.device
    return device


def _batch(scheduler, req, seq_len: int):
    import torch

    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

    device = _device(scheduler)
    length = int(seq_len)
    batch = ScheduleBatch(
        reqs=[req],
        req_to_token_pool=scheduler.req_to_token_pool,
        token_to_kv_pool_allocator=scheduler.token_to_kv_pool_allocator,
        tree_cache=scheduler.tree_cache,
        model_config=scheduler.model_config,
        enable_overlap=False,
        spec_algorithm=scheduler.spec_algorithm,
        forward_mode=ForwardMode.DECODE,
        device=device,
    )
    batch.req_pool_indices = torch.tensor(
        [int(req.req_pool_idx)], dtype=torch.int64, device=device
    )
    batch.seq_lens = torch.tensor([length], dtype=torch.int64, device=device)
    batch.seq_lens_cpu = torch.tensor([length], dtype=torch.int64)
    batch.orig_seq_lens = torch.tensor([length], dtype=torch.int32, device=device)
    batch.seq_lens_sum = length
    tail = req.fill_ids[-1] if req.fill_ids else 0
    batch.output_ids = torch.tensor([int(tail)], dtype=torch.int64, device=device)
    batch.return_logprob = False
    batch.multimodal_inputs = [getattr(req, "multimodal_inputs", None)]
    batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
        batch, scheduler.model_config.vocab_size
    )
    keep_raw_logits(batch)
    return batch


def _assign(scheduler, req, start: int, slots) -> None:
    import torch

    from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

    device = scheduler.req_to_token_pool.device
    n = len(slots)
    assign_req_to_token_pool_func(
        torch.tensor([int(req.req_pool_idx)], dtype=torch.int64, device=device),
        scheduler.req_to_token_pool.req_to_token,
        torch.tensor([int(start)], dtype=torch.int64, device=device),
        torch.tensor([int(start) + n], dtype=torch.int64, device=device),
        torch.tensor(list(slots), dtype=torch.int64, device=device),
        1,
    )


def _logits_of(result, rid: str):
    logits = getattr(getattr(result, "logits_output", None), "next_token_logits", None)
    if logits is None:
        raise RuntimeError(f"draft {rid} forward returned no logits")
    return logits


def ingest_prefix(scheduler, req, slots):
    """Write the committed prefix into KV, then one extend over those tokens."""
    if scheduler is None or not hasattr(scheduler, "req_to_token_pool"):
        return None
    import torch

    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    if req.req_pool_idx is None:
        allocated = scheduler.req_to_token_pool.alloc([req])
        if not allocated:
            raise RuntimeError(f"draft {req.rid} has no request pool slot")
    tokens = [int(token) for token in req.fill_ids]
    if tokens and slots:
        _assign(scheduler, req, 0, slots)
    req.kv_committed_len = len(tokens)
    if not tokens:
        return None
    return extend_committed(scheduler, req, 0, tokens, slots)


def extend_committed(scheduler, req, start: int, tokens, slots):
    """EXTEND ``[start, start + len(tokens))``. The last row is the next distribution."""
    import torch

    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    tokens = [int(token) for token in tokens]
    if not tokens:
        return None
    if slots:
        _assign(scheduler, req, int(start), slots)
    end = int(start) + len(tokens)
    batch = _batch(scheduler, req, end)
    device = batch.seq_lens.device
    batch.forward_mode = ForwardMode.EXTEND
    batch.global_forward_mode = ForwardMode.EXTEND
    batch.input_ids = torch.tensor(tokens, dtype=torch.int64, device=device)
    batch.output_ids = None
    batch.seq_lens = torch.tensor([end], dtype=torch.int64, device=device)
    batch.seq_lens_cpu = torch.tensor([end], dtype=torch.int64)
    batch.orig_seq_lens = torch.tensor([end], dtype=torch.int32, device=device)
    batch.seq_lens_sum = end
    batch.out_cache_loc = torch.tensor(list(slots), dtype=torch.int64, device=device)
    batch.extend_lens = [len(tokens)]
    batch.prefix_lens = [int(start)]
    batch.extend_num_tokens = len(tokens)
    batch.extend_logprob_start_lens = [int(start)]
    result = scheduler.run_batch(batch)
    req.kv_committed_len = end
    req.kv_allocated_len = end
    return _logits_of(result, req.rid)


def run_layer_logits(scheduler, req, tokens, positions, mask, slots, cached_len: int):
    """One layer extend. ``cached_len`` already includes earlier physical slots."""
    import torch

    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
    from sglang.srt.speculative.toll_tree_remote.draft.spec_input import TollLayerSpec

    width = len(tokens)
    cached = int(cached_len)
    if slots:
        _assign(scheduler, req, cached, slots)
    batch = _batch(scheduler, req, cached)
    device = batch.seq_lens.device
    batch.forward_mode = ForwardMode.TARGET_VERIFY
    batch.global_forward_mode = ForwardMode.TARGET_VERIFY
    batch.input_ids = torch.tensor(list(tokens), dtype=torch.int64, device=device)
    batch.output_ids = None
    batch.seq_lens = torch.tensor([cached], dtype=torch.int64, device=device)
    batch.seq_lens_cpu = torch.tensor([cached], dtype=torch.int64)
    batch.orig_seq_lens = torch.tensor([cached], dtype=torch.int32, device=device)
    batch.seq_lens_sum = cached
    batch.out_cache_loc = torch.tensor(list(slots), dtype=torch.int64, device=device)
    batch.extend_lens = [width]
    batch.prefix_lens = [cached]
    batch.extend_num_tokens = width
    batch.extend_logprob_start_lens = [cached]
    batch.spec_info = TollLayerSpec(
        custom_mask=torch.as_tensor(mask, dtype=torch.bool, device=device).reshape(-1),
        positions=torch.tensor(list(positions), dtype=torch.int64, device=device),
        draft_token_num=width,
        num_tokens_per_req=width,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )
    result = scheduler.run_batch(batch)
    return _logits_of(result, req.rid)
