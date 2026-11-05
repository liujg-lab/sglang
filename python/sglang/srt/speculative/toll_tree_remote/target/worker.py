"""Target verify worker. Padding tokens are 0 and stay inside the verify width."""

from __future__ import annotations

import logging

import torch

from sglang.srt.speculative.toll_tree_remote.tree.layout import build_verify_layout

logger = logging.getLogger(__name__)


def tokens_in_vocab(tokens, vocab_size: int) -> bool:
    limit = int(vocab_size)
    return all(0 <= int(token) < limit for token in tokens)


def empty_draft(parent_rows):
    return [[] for _ in parent_rows], [[] for _ in parent_rows]


def filter_vocab(token_rows, parent_rows, vocab_size: int):
    """Drop a row whose draft token is outside the vocabulary."""
    kept_tokens = []
    kept_parents = []
    for tokens, parents in zip(token_rows, parent_rows):
        if tokens_in_vocab(tokens, vocab_size):
            kept_tokens.append(list(tokens))
            kept_parents.append(list(parents))
        else:
            kept_tokens.append([])
            kept_parents.append([])
    return kept_tokens, kept_parents


def pack_draft_tokens(tails, token_rows, num_verify: int) -> list[int]:
    """One flat vector, length ``batch * (K + 1)``. Padding is 0."""
    packed = []
    width = int(num_verify)
    for tail, tokens in zip(tails, token_rows):
        row = [int(tail)] + [int(token) for token in tokens]
        row = row[:width]
        row.extend([0] * (width - len(row)))
        packed.extend(row)
    return packed


def walk_accept(layout, spec_steps: int) -> None:
    """Raise if a walk of ``spec_steps`` leaves the verify slots."""
    nxt = layout.retrive_next_token
    limit = int(layout.num_verify_tokens)
    for batch in range(int(nxt.shape[0])):
        node = 0
        for _ in range(int(spec_steps)):
            if node < 0 or node >= limit:
                raise RuntimeError(f"accept index {node} is outside {limit}")
            jump = int(nxt[batch, node])
            if jump < 0:
                break
            if jump >= limit:
                raise RuntimeError(f"accept index {jump} is outside {limit}")
            node = jump


def committed_prefix(req) -> int:
    """KV length already stored. Fall back to the tokens kept on the request."""
    committed = int(getattr(req, "kv_committed_len", 0) or 0)
    if committed > 0:
        return committed
    origin = list(getattr(req, "origin_input_ids", []) or [])
    output = list(getattr(req, "output_ids", []) or [])
    return max(0, len(origin) + len(output))


def is_health_check(req) -> bool:
    return str(getattr(req, "rid", "")).startswith("HEALTH_CHECK")


def _host_len_cache(batch, name: str, n: int, dtype: torch.dtype) -> torch.Tensor:
    cache = getattr(batch, name, None)
    if (
        cache is None
        or cache.numel() < n
        or cache.dtype != dtype
        or cache.device.type != "cpu"
    ):
        cache = torch.empty(max(n, 1), dtype=dtype, device="cpu")
        setattr(batch, name, cache)
    return cache


def _copy_committed_lens(dest: torch.Tensor, host: torch.Tensor, n: int) -> torch.Tensor:
    view = host[:n]
    if dest.shape == view.shape and dest.dtype == view.dtype:
        dest.copy_(view)
        return dest
    moved = view.to(device=dest.device, dtype=dest.dtype)
    if moved.data_ptr() == view.data_ptr():
        return moved.clone()
    return moved


def align_seq_lens_to_committed(batch) -> None:
    """Point ``seq_lens`` at the committed prefix before a decode or verify."""
    committed = [committed_prefix(req) for req in batch.reqs]
    n = len(committed)
    source = torch.tensor(committed, dtype=torch.int64)
    host = _host_len_cache(batch, "_toll_committed_len_host", n, batch.seq_lens.dtype)
    cpu_host = _host_len_cache(batch, "_toll_committed_len_cpu", n, torch.int64)
    if n:
        host[:n].copy_(source)
        if cpu_host.data_ptr() != host.data_ptr():
            cpu_host[:n].copy_(source)
    batch.seq_lens = _copy_committed_lens(batch.seq_lens, host, n)
    if batch.seq_lens_cpu is not None:
        batch.seq_lens_cpu = _copy_committed_lens(batch.seq_lens_cpu, cpu_host, n)
    else:
        batch.seq_lens_cpu = cpu_host[:n].clone()
    if batch.orig_seq_lens is not None:
        orig_host = _host_len_cache(
            batch, "_toll_committed_len_orig", n, batch.orig_seq_lens.dtype
        )
        if n and orig_host.data_ptr() != host.data_ptr():
            orig_host[:n].copy_(source)
        batch.orig_seq_lens = _copy_committed_lens(batch.orig_seq_lens, orig_host, n)
    batch.seq_lens_sum = int(source.sum())


def sync_kv_from_cpu_lengths(batch, seq_lens_cpu_pre, accept_length_per_req_cpu) -> None:
    """Set KV bounds from the pre-verify length plus accepted draft tokens and the bonus."""
    if seq_lens_cpu_pre is None:
        raise RuntimeError("seq_lens_cpu snapshot is missing")
    pre_list = (
        seq_lens_cpu_pre.tolist()
        if isinstance(seq_lens_cpu_pre, torch.Tensor)
        else list(seq_lens_cpu_pre)
    )
    accepted = list(accept_length_per_req_cpu or [])
    for index, req in enumerate(batch.reqs):
        if index >= len(pre_list):
            raise RuntimeError("seq_lens_cpu snapshot shorter than the verify batch")
        if is_health_check(req):
            continue
        extra = int(accepted[index]) if index < len(accepted) else 0
        committed = int(pre_list[index]) + extra + 1
        req.kv_committed_len = committed
        req.kv_allocated_len = committed


def note_decode_token(req, token: int) -> None:
    """Append one ordinary-decode token and apply the request's own stop rule."""
    req.output_ids.append(int(token))
    req.spec_cnt = int(getattr(req, "spec_cnt", 0) or 0) + 1
    req.len_output_ids = len(req.output_ids)
    checker = getattr(req, "check_finished", None)
    if callable(checker):
        checker()


def account_verify_step(batch) -> None:
    for req in getattr(batch, "reqs", None) or ():
        if is_health_check(req):
            continue
        req.spec_cnt = int(getattr(req, "spec_cnt", 0) or 0) + 1
        req.len_output_ids = len(getattr(req, "output_ids", []) or [])


class TollTreeRemoteWorker:
    """Target verifier. ``verify`` uses Eagle's path and does not import SR."""

    def __init__(self, server_args=None, target_worker=None, **kwargs) -> None:
        self.server_args = server_args
        self.target_worker = target_worker
        page = 1 if server_args is None else getattr(server_args, "page_size", 1)
        self.page_size = int(page or 1)
        self._req_to_token_pool = None
        self._kv_allocator = None
        getter = getattr(target_worker, "get_memory_pool", None)
        if callable(getter):
            self._req_to_token_pool, self._kv_allocator = getter()
        self.verify_rounds = 0
        self.accepted_draft_tokens = 0
        self.ar_fallback_rounds = 0

    @property
    def req_to_token_pool(self):
        return self._req_to_token_pool

    @property
    def draft_model_runner(self):
        return None

    @property
    def model_runner(self):
        return self.target_worker.model_runner

    @property
    def model_config(self):
        return self.target_worker.model_config

    @property
    def token_to_kv_pool_allocator(self):
        if self._kv_allocator is not None:
            return self._kv_allocator
        return self.model_runner.token_to_kv_pool_allocator

    def clear_cache_pool(self) -> None:
        return None

    def forward_batch_generation(self, batch):
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        if batch.forward_mode.is_extend() or getattr(batch, "is_extend_in_batch", False):
            return self._forward_extend(batch)
        reply = getattr(batch.reqs[0], "toll_reply", None) if batch.reqs else None
        if reply is None or not reply.reqs or not reply.reqs[0].tokens:
            return self._forward_normal_decode(batch)
        return self._forward_verify(batch, reply, CaptureHiddenMode.NULL)

    def _forward_extend(self, batch):
        from sglang.srt.managers.utils import GenerationBatchResult
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.NULL
        result = self.target_worker.forward_batch_generation(model_worker_batch)
        return GenerationBatchResult(
            logits_output=result.logits_output,
            next_token_ids=result.next_token_ids,
            num_accepted_tokens=0,
            can_run_cuda_graph=False,
        )

    def _forward_normal_decode(self, batch):
        from sglang.srt.managers.utils import GenerationBatchResult
        from sglang.srt.mem_cache.common import alloc_for_decode
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode

        last_token_ids = [self._tail_token(req) for req in batch.reqs]
        device = batch.seq_lens.device
        info = getattr(batch, "sampling_info", None)
        orchestrator = getattr(info, "penalizer_orchestrator", None) if info else None
        if orchestrator is not None and getattr(orchestrator, "is_required", False):
            orchestrator.cumulate_output_tokens(
                torch.tensor(last_token_ids, dtype=torch.int64, device=device)
            )
        batch.input_ids = torch.tensor(last_token_ids, dtype=torch.int32, device=device)
        batch.output_ids = None
        batch.forward_mode = ForwardMode.DECODE
        batch.spec_info = None
        bs = len(batch.reqs)
        if batch.global_num_tokens is not None:
            batch.global_num_tokens = [bs]
        if batch.global_num_tokens_for_logprob is not None:
            batch.global_num_tokens_for_logprob = [bs]
        align_seq_lens_to_committed(batch)
        batch.out_cache_loc = alloc_for_decode(batch, token_per_req=1)
        for req in batch.reqs:
            req.kv_committed_len = int(getattr(req, "kv_committed_len", 0) or 0) + 1
            req.kv_allocated_len = int(getattr(req, "kv_allocated_len", 0) or 0) + 1
        batch.seq_lens.add_(1)
        batch.seq_lens_cpu.add_(1)
        if batch.orig_seq_lens is not None:
            batch.orig_seq_lens.add_(1)
        batch.seq_lens_sum = int(getattr(batch, "seq_lens_sum", 0) or 0) + bs
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.NULL
        result = self.target_worker.forward_batch_generation(model_worker_batch)
        next_ids = result.next_token_ids
        if hasattr(next_ids, "tolist"):
            next_ids = next_ids.tolist()
        counted = False
        for index, req in enumerate(batch.reqs):
            if is_health_check(req):
                continue
            if index < len(next_ids) and next_ids[index] is not None:
                note_decode_token(req, int(next_ids[index]))
            else:
                req.spec_cnt = int(getattr(req, "spec_cnt", 0) or 0) + 1
                req.len_output_ids = len(req.output_ids)
            counted = True
        if counted:
            self.ar_fallback_rounds += 1
            self._log_target()
        return GenerationBatchResult(
            logits_output=result.logits_output,
            next_token_ids=result.next_token_ids,
            num_accepted_tokens=0,
            accept_length_per_req_cpu=[1] * bs,
            can_run_cuda_graph=False,
        )

    def _forward_verify(self, batch, reply, hidden_mode):
        from sglang.srt.managers.utils import GenerationBatchResult
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        align_seq_lens_to_committed(batch)
        bs = len(batch.reqs)
        batch.seq_lens_sum = int(batch.seq_lens_cpu[:bs].sum())
        seq_lens_pre = batch.seq_lens_cpu.clone()
        spec = self.build_verify_input(
            batch,
            [reply.reqs[0].tokens],
            [reply.reqs[0].parents],
            [self._tail_token(batch.reqs[0])],
            int(self.model_runner.model_config.vocab_size),
            int(self.server_args.toll_tree_max_depth),
        )
        if spec is None:
            batch.draft_num_tokens = 1
            return self._forward_normal_decode(batch)
        spec.capture_hidden_mode = hidden_mode
        spec.prepare_for_verify(batch, self.page_size)
        spec.num_tokens_per_req = spec.draft_token_num
        batch.return_hidden_states = False
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = spec
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = hidden_mode
        result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        verified = spec.verify(
            batch,
            result.logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            None,
            prepare_local_draft_hidden=False,
            sr_accept_state=None,
        )
        lengths = list(verified.accept_length_per_req_cpu)
        sync_kv_from_cpu_lengths(batch, seq_lens_pre, lengths)
        account_verify_step(batch)
        logits = result.logits_output
        if logits is not None and getattr(logits, "next_token_logits", None) is not None:
            logits.next_token_logits = logits.next_token_logits[verified.accepted_indices]
        batch.forward_mode = ForwardMode.DECODE
        batch.spec_info = verified.draft_input
        self.verify_rounds += 1
        self.accepted_draft_tokens += sum(lengths)
        self._log_target()
        return GenerationBatchResult(
            logits_output=logits,
            next_token_ids=verified.verified_id,
            num_accepted_tokens=sum(lengths),
            accept_length_per_req_cpu=lengths,
            can_run_cuda_graph=result.can_run_cuda_graph,
        )

    def _log_target(self) -> None:
        logger.info(
            "[TOLL target] verify_rounds=%s accepted_draft_tokens=%s ar_fallback_rounds=%s",
            self.verify_rounds,
            self.accepted_draft_tokens,
            self.ar_fallback_rounds,
        )

    @staticmethod
    def _tail_token(req) -> int:
        output = list(getattr(req, "output_ids", []) or [])
        if output:
            return int(output[-1])
        origin = list(getattr(req, "origin_input_ids", []) or [])
        return int(origin[-1]) if origin else 0

    def build_verify_input(self, batch, token_rows, parent_rows, tails, vocab_size: int, max_depth: int):
        tokens, parents = filter_vocab(token_rows, parent_rows, vocab_size)
        if not any(tokens):
            return None
        num_verify = int(self.server_args.speculative_num_draft_tokens)
        seq_lens = [int(v) for v in batch.seq_lens_cpu.tolist()]
        layout = build_verify_layout(parents, seq_lens, num_verify)
        walk_accept(layout, int(max_depth))
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
        from sglang.srt.speculative.eagle_info import EagleVerifyInput

        device = batch.seq_lens.device
        draft = torch.tensor(
            pack_draft_tokens(tails, tokens, num_verify),
            dtype=torch.int64,
            device=device,
        )
        return EagleVerifyInput(
            draft_token=draft,
            custom_mask=torch.tensor(layout.tree_mask, dtype=torch.bool, device=device),
            positions=torch.tensor(layout.positions, dtype=torch.int64, device=device),
            retrive_index=torch.tensor(layout.retrive_index, dtype=torch.int64, device=device),
            retrive_next_token=torch.tensor(layout.retrive_next_token, dtype=torch.int64, device=device),
            retrive_next_sibling=torch.tensor(layout.retrive_next_sibling, dtype=torch.int64, device=device),
            retrive_cum_len=None,
            spec_steps=int(max_depth),
            topk=max(num_verify - 1, 1),
            draft_token_num=num_verify,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=int(batch.seq_lens_sum),
            seq_lens_cpu=batch.seq_lens_cpu,
        )
