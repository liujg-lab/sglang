"""Draft request and the prefix / layer forwards. No other algorithm is imported."""

from __future__ import annotations


def layer_positions(prefix_len: int, depth: int, width: int) -> list[int]:
    """One position per physical row. Depth 1 sits on the committed tail."""
    position = int(prefix_len) + int(depth) - 1
    return [position] * int(width)


def make_draft_req(rid: str, input_ids: list[int]):
    origin = [int(token) for token in input_ids]
    try:
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams
    except ModuleNotFoundError:
        return _DraftReq(rid, origin)
    req = Req(rid, "", origin, SamplingParams(temperature=0.0, top_k=1))
    req.origin_input_ids = origin
    req.output_ids = []
    req.fill_ids = list(origin)
    return req


class _DraftReq:
    """Stand-in used only when this interpreter cannot import ``Req``."""

    def __init__(self, rid: str, origin: list[int]) -> None:
        self.rid = rid
        self.origin_input_text = ""
        self.origin_input_ids = list(origin)
        self.output_ids = []
        self.fill_ids = list(origin)
        self.req_pool_idx = None
        self.kv_committed_len = 0


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
    batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
        batch, scheduler.model_config.vocab_size
    )
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


def ingest_prefix(scheduler, req, slots) -> None:
    """Write the committed prefix into KV, then one extend over those tokens."""
    if scheduler is None or not hasattr(scheduler, "req_to_token_pool"):
        return
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
        return
    batch = _batch(scheduler, req, len(tokens))
    device = batch.seq_lens.device
    batch.forward_mode = ForwardMode.EXTEND
    batch.global_forward_mode = ForwardMode.EXTEND
    batch.input_ids = torch.tensor(tokens, dtype=torch.int64, device=device)
    batch.output_ids = None
    batch.out_cache_loc = torch.tensor(list(slots), dtype=torch.int64, device=device)
    batch.extend_lens = [len(tokens)]
    batch.prefix_lens = [0]
    batch.extend_num_tokens = len(tokens)
    batch.extend_logprob_start_lens = [0]
    scheduler.run_batch(batch)


def tail_logits(scheduler, req):
    """Logits at the committed tail. The tail already has a KV slot."""
    import torch

    length = len(req.fill_ids)
    if length < 1 or req.req_pool_idx is None:
        raise RuntimeError(f"draft {req.rid} has no committed tail")
    slot = scheduler.req_to_token_pool.req_to_token[req.req_pool_idx, length - 1]
    batch = _batch(scheduler, req, length)
    device = batch.seq_lens.device
    batch.input_ids = torch.tensor([int(req.fill_ids[-1])], dtype=torch.int64, device=device)
    batch.out_cache_loc = slot.reshape(1)
    batch.output_ids = None
    result = scheduler.run_batch(batch)
    logits = getattr(getattr(result, "logits_output", None), "next_token_logits", None)
    if logits is None:
        raise RuntimeError(f"draft {req.rid} tail forward returned no logits")
    return logits


def run_layer_logits(scheduler, req, tokens, positions, mask, slots):
    """One layer extend. ``slots`` are this layer's physical KV ids."""
    import torch

    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
    from sglang.srt.speculative.toll_tree_remote.draft.spec_input import TollLayerSpec

    width = len(tokens)
    prefix = len(req.fill_ids)
    batch = _batch(scheduler, req, prefix)
    device = batch.seq_lens.device
    batch.forward_mode = ForwardMode.TARGET_VERIFY
    batch.global_forward_mode = ForwardMode.TARGET_VERIFY
    batch.input_ids = torch.tensor(list(tokens), dtype=torch.int64, device=device)
    batch.output_ids = None
    batch.out_cache_loc = torch.tensor(list(slots), dtype=torch.int64, device=device)
    batch.extend_lens = [width]
    batch.prefix_lens = [prefix]
    batch.extend_num_tokens = width
    batch.extend_logprob_start_lens = [prefix]
    batch.spec_info = TollLayerSpec(
        custom_mask=mask.reshape(-1),
        positions=torch.tensor(list(positions), dtype=torch.int64, device=device),
        draft_token_num=width,
        num_tokens_per_req=width,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )
    result = scheduler.run_batch(batch)
    logits = getattr(getattr(result, "logits_output", None), "next_token_logits", None)
    if logits is None:
        raise RuntimeError(f"draft {req.rid} layer forward returned no logits")
    return logits
