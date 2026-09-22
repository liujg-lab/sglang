"""Fixed-capacity SR Target accept postprocess.

CPU-importable. NPU kernels load only after static admission decides to build
the workspace. ``A`` is the final output count including the bonus token.
``accept_length_per_req_cpu`` stays ``A - 1`` for the existing API. KV bounds
move by ``A``.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

SR_FIXED_ACCEPT_ENV = "SGLANG_NPU_SR_FIXED_ACCEPT"
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_KERNEL_MODULE = (
    "sglang.srt.speculative.standalone_remote.verifier.sr_fixed_accept_kernels"
)


def read_sr_fixed_accept_env(env=None) -> bool:
    """Read once at SR Target init. Default on. ``0/false/no/off`` keeps V1."""
    environ = os.environ if env is None else env
    raw = environ.get(SR_FIXED_ACCEPT_ENV, "1")
    return str(raw).strip().lower() in _TRUTHY


def accept_control_decision(admitted: bool, resolved_verify: str, simulate: bool) -> str:
    """Route after penalty and verify have already run once.

    ``fresh_v1``: workspace was not bound.
    ``finalizer``: greedy result, enter the fixed-capacity postprocess.
    ``v1_workspace``: workspace holds the verify result; run V1 postprocess only.
    """
    if not admitted:
        return "fresh_v1"
    if resolved_verify == "greedy" and not simulate:
        return "finalizer"
    return "v1_workspace"


def run_accept_stages(
    *,
    admitted: bool,
    simulate: bool,
    penalty: Callable[[], None],
    verify: Callable[[], str],
    v1: Callable[[], object],
    finalizer: Callable[[], object],
):
    """Penalty, verify, then exactly one postprocess. No second pass."""
    penalty()
    resolved = verify()
    decision = accept_control_decision(admitted, resolved, simulate)
    if decision == "finalizer":
        return finalizer()
    return v1()


def first_free_pos(committed_end: int, page_size: int) -> int:
    page = int(page_size)
    return ((int(committed_end) + page - 1) // page) * page


def num_free_pages(prefix_len: int, accepted: int, width: int, page_size: int) -> int:
    """Whole pages after the committed end, inside the verify window."""
    page = int(page_size)
    committed_end = int(prefix_len) + int(accepted)
    extended_end = int(prefix_len) + int(width)
    return max(
        0,
        (extended_end + page - 1) // page - (committed_end + page - 1) // page,
    )


def free_page_row_offsets(
    prefix_len: int, accepted: int, width: int, page_size: int
) -> List[int]:
    """Representative token offsets. Empty when nothing is freed.

    Does not touch ``out_cache_loc``. Offsets are in-range or this raises
    before a caller is allowed to read them.
    """
    count = num_free_pages(prefix_len, accepted, width, page_size)
    if count <= 0:
        return []
    page = int(page_size)
    committed_end = int(prefix_len) + int(accepted)
    base = first_free_pos(committed_end, page) - int(prefix_len)
    offsets = []
    for j in range(count):
        row_offset = base + j * page
        if row_offset < 0 or row_offset >= int(width):
            raise RuntimeError(
                f"free-page slot {row_offset} outside verify width {width}"
            )
        offsets.append(row_offset)
    return offsets


def length_update_mode(finished: Sequence[bool]) -> str:
    """``all`` updates every seq_lens row. ``none`` leaves both tensors alone.

    Unfinished filtering is a separate decision and does not select rows here.
    """
    if not finished or all(finished):
        return "none"
    return "all"


_QWEN3_VL_ARCHS = frozenset(
    {
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
    }
)
_INTEGER_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)


def _integer_delta_layout(delta) -> bool:
    """Metadata only: shape ``[1, 1]`` and an integer dtype. No value read."""
    if not torch.is_tensor(delta) or delta.dtype not in _INTEGER_DTYPES:
        return False
    shape = delta.shape
    return len(shape) == 2 and int(shape[0]) == 1 and int(shape[1]) == 1


def multimodal_accept_reject_reason(batch, *, bs: int) -> Optional[str]:
    """只读取元数据及已有 CPU 长度；不修改请求，不回读设备张量。"""
    reqs = getattr(batch, "reqs", None)
    if reqs is None:
        reqs = []
    if all(getattr(req, "multimodal_inputs", None) is None for req in reqs):
        return None

    hf_config = getattr(getattr(batch, "model_config", None), "hf_config", None)
    architectures = getattr(hf_config, "architectures", None)
    if (
        not architectures
        or architectures[0] not in _QWEN3_VL_ARCHS
    ):
        return "multimodal_model"

    forward_mode = getattr(batch, "forward_mode", None)
    is_target_verify = getattr(forward_mode, "is_target_verify", None)
    if not callable(is_target_verify) or not is_target_verify():
        return "multimodal_phase"

    rows = getattr(batch, "multimodal_inputs", None)
    if (
        rows is None
        or len(reqs) != int(bs)
        or len(rows) != int(bs)
    ):
        return "multimodal_rows"
    for req, row in zip(reqs, rows):
        if getattr(req, "multimodal_inputs", None) is not row:
            return "multimodal_rows"

    seq_lens_cpu = getattr(batch, "seq_lens_cpu", None)
    if (
        not torch.is_tensor(seq_lens_cpu)
        or seq_lens_cpu.device.type != "cpu"
        or seq_lens_cpu.dim() != 1
        or int(seq_lens_cpu.shape[0]) != int(bs)
        or seq_lens_cpu.dtype not in _INTEGER_DTYPES
    ):
        return "multimodal_prefix"

    for index, req in enumerate(reqs):
        mm_input = rows[index]
        if mm_input is None:
            continue
        origin = getattr(req, "origin_input_ids", None)
        if origin is None or int(seq_lens_cpu[index]) < len(origin):
            return "multimodal_prefill"
        if not _integer_delta_layout(getattr(mm_input, "mrope_position_delta", None)):
            return "multimodal_mrope"
    return None


def conservative_mode_reason(verify_mode: Optional[str], is_all_greedy: bool) -> Optional[str]:
    mode = verify_mode or "auto"
    if mode == "greedy":
        return None
    if mode == "auto" and is_all_greedy:
        return None
    return "mode"


def _clone_tensor(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    return value


def apply_free_unique_pages(allocator, page_ids: torch.Tensor) -> None:
    """Device-side sorted prepend used by ``NPUPagedTokenToKVPoolAllocator``.

    Empty input returns immediately. The caller guarantees uniqueness, so this
    does not dedup and does not copy page ids back to the CPU.
    """
    if page_ids.numel() == 0:
        return
    if not bool(getattr(allocator, "is_not_in_free_group", False)):
        raise RuntimeError("free_unique_pages is forbidden inside a free group")
    sorted_pages, _ = torch.sort(page_ids)
    if allocator.need_sort:
        sorted_pages = sorted_pages.to(dtype=allocator.release_pages.dtype)
        allocator.release_pages = torch.cat((sorted_pages, allocator.release_pages))
    else:
        sorted_pages = sorted_pages.to(dtype=allocator.free_pages.dtype)
        allocator.free_pages = torch.cat((sorted_pages, allocator.free_pages))
    if getattr(allocator, "debug_mode", False):
        assert len(torch.unique(allocator.free_pages)) == len(allocator.free_pages)


def detach_verify_output(output):
    """Copy results off storage the next round may reuse."""
    if output is None:
        return output
    output.verified_id = _clone_tensor(output.verified_id)
    output.accepted_indices = _clone_tensor(output.accepted_indices)
    draft = getattr(output, "draft_input", None)
    if draft is None:
        return output
    for name in (
        "verified_id",
        "accept_length",
        "seq_lens_for_draft_extend",
        "seq_lens_for_draft_extend_cpu",
        "req_pool_indices_for_draft_extend",
    ):
        if hasattr(draft, name):
            setattr(draft, name, _clone_tensor(getattr(draft, name)))
    return output


def validate_packed_rows(
    indices: Sequence[Sequence[int]],
    pre_lengths: Sequence[int],
    errors: Sequence[int],
) -> None:
    """Reject the whole batch before any request appends output."""
    if not (len(indices) == len(pre_lengths) == len(errors)):
        raise RuntimeError("packed accept batch rank mismatch")
    for row_i, row in enumerate(indices):
        if int(errors[row_i]) != 0:
            raise RuntimeError(f"packed accept error on request {row_i}")
        seen_end = False
        valid = 0
        for idx in row:
            idx = int(idx)
            if idx < -1:
                raise RuntimeError(f"packed accept index {idx} on request {row_i}")
            if seen_end:
                if idx != -1:
                    raise RuntimeError(
                        f"packed accept hole on request {row_i}"
                    )
                continue
            if idx == -1:
                seen_end = True
                continue
            valid += 1
        pre = int(pre_lengths[row_i])
        if valid == 0:
            if pre != 0:
                raise RuntimeError(
                    f"packed accept length {pre} disagrees with empty row {row_i}"
                )
        elif pre != valid - 1:
            raise RuntimeError(
                f"packed accept length {pre} disagrees with row {row_i}"
            )


def apply_cpu_acceptance(
    reqs,
    rows: Sequence[Sequence[int]],
    tokens: Sequence[Sequence[int]],
    think_end_id,
) -> Tuple[List[int], List[bool], List[List[int]]]:
    """Append like V1. Returns ``A`` (with bonus), finished flags, truncated rows.

    Histogram and ``spec_accepted_tokens`` use the original row, including
    tokens after an EOS cut. ``kv_committed_len`` increases by the appended
    count only. Callers still SET from the pre-verify snapshot.
    """
    if len(reqs) != len(rows) or len(reqs) != len(tokens):
        raise RuntimeError("accept row count does not match requests")
    accepted = []
    finished = []
    truncated = []
    for req, row, token_row in zip(reqs, rows, tokens):
        original = [int(idx) for idx in row]
        appended = 0
        cut = None
        for j, idx in enumerate(original):
            if idx == -1:
                break
            appended += 1
            token = int(token_row[j])
            req.output_ids.append(token)
            if getattr(req, "require_reasoning", False) and think_end_id is not None:
                req.update_reasoning_tokens(token, think_end_id)
            req.check_finished()
            if req.finished():
                cut = j + 1
                break
        req.kv_committed_len = int(getattr(req, "kv_committed_len", 0) or 0) + appended
        req.kv_allocated_len = req.kv_committed_len
        req.spec_verify_ct = int(getattr(req, "spec_verify_ct", 0) or 0) + 1
        raw_draft = sum(1 for idx in original if idx != -1) - 1
        req.spec_accepted_tokens = int(getattr(req, "spec_accepted_tokens", 0) or 0) + raw_draft
        req.update_spec_acceptance_histogram(raw_draft)
        copied = list(original)
        if cut is not None:
            for k in range(cut, len(copied)):
                copied[k] = -1
        accepted.append(appended)
        finished.append(bool(req.finished()))
        truncated.append(copied)
    return accepted, finished, truncated


def _timed(metrics, name, fn, device=False):
    """Account a stage without retrying ``fn`` or replacing its exception."""
    start = time.perf_counter()
    event_pair = None
    try:
        if (
            device
            and metrics is not None
            and getattr(metrics, "active", False)
            and getattr(metrics, "device_module", None) is not None
            and int(getattr(metrics, "rounds", 0)) % 32 == 0
            and len(getattr(metrics, "pending", ())) < 16
        ):
            factory = metrics.device_module.Event
            ev_start = factory(enable_timing=True)
            ev_end = factory(enable_timing=True)
            ev_start.record()
            event_pair = (ev_start, ev_end)
    except Exception:
        event_pair = None
    completed = False
    try:
        result = fn()
        completed = True
        return result
    finally:
        try:
            if metrics is not None and getattr(metrics, "active", False):
                metrics.add_host(name, time.perf_counter() - start)
            if event_pair is not None and completed:
                event_pair[1].record()
                metrics.pending.append((name, event_pair[0], event_pair[1]))
        except Exception:
            pass


def _allocator_supported(allocator) -> Optional[str]:
    if allocator is None or not callable(getattr(allocator, "free_unique_pages", None)):
        return "allocator missing free_unique_pages"
    names = [cls.__name__ for cls in type(allocator).mro()]
    if "NPUPagedTokenToKVPoolAllocator" not in names:
        return "allocator type"
    return None


def _kv_buffer(worker):
    allocator = getattr(worker, "token_to_kv_pool_allocator", None)
    if allocator is None or not hasattr(allocator, "get_kvcache"):
        return None
    try:
        cache = allocator.get_kvcache()
    except Exception:
        return None
    return getattr(cache, "kv_buffer", None)


def static_disable_reason(worker) -> Optional[str]:
    """Reason to skip workspace construction. Does not import kernels."""
    if bool(getattr(worker, "_hybrid_needs_hidden", False)):
        return "hybrid"
    try:
        topk = int(getattr(worker, "topk", 0) or 0)
        page_size = int(getattr(worker, "page_size", 0) or 0)
        width = int(getattr(worker, "speculative_num_draft_tokens", 0) or 0)
        steps = int(getattr(worker, "speculative_num_steps", 0) or 0)
        batch_cap = int(getattr(worker, "_verify_max_bs", 0) or 0)
    except (TypeError, ValueError):
        return "capacity"
    if topk <= 1:
        return "topk"
    if page_size <= 1:
        return "page_size"
    if width < 1 or steps < 0 or batch_cap < 1 or steps + 1 > width:
        return "capacity"
    allocator = getattr(worker, "token_to_kv_pool_allocator", None)
    alloc_reason = _allocator_supported(allocator)
    if alloc_reason is not None:
        return alloc_reason
    try:
        alloc_page = int(allocator.page_size)
    except (TypeError, ValueError, AttributeError):
        return "page size mismatch"
    buf = _kv_buffer(worker)
    if not torch.is_tensor(buf) or buf.dim() != 6 or int(buf.shape[0]) != 2:
        return "not mha 6d"
    try:
        pool_page = int(buf.shape[3])
    except (TypeError, ValueError):
        return "page size mismatch"
    if len({pool_page, alloc_page, page_size}) != 1:
        return "page size mismatch"
    if buf.device.type != "npu":
        return "not npu"
    return None


def _load_kernels():
    from sglang.srt.speculative.standalone_remote.verifier import (
        sr_fixed_accept_kernels,
    )

    return sr_fixed_accept_kernels


@dataclass
class FixedAcceptResult:
    """Stable accept result. Tensors do not alias the reusable workspace."""

    verified_id: torch.Tensor
    accepted_indices: torch.Tensor
    accept_length_per_req_cpu: List[int]
    draft_verified_id: torch.Tensor
    draft_accept_length: torch.Tensor
    draft_accept_length_cpu: List[int]
    seq_lens_for_draft: Optional[torch.Tensor]
    seq_lens_for_draft_cpu: Optional[torch.Tensor]
    req_pool_indices: Optional[torch.Tensor]
    tree_paths: List
    idle: bool
    logits_output: object


class SRFixedAcceptState:
    """Per-target workspace. Views passed to greedy stay contiguous."""

    def __init__(self, batch_cap: int, path_cap: int, width: int, page_size: int, device):
        self.B_cap = int(batch_cap)
        self.L = int(path_cap)
        self.W = int(width)
        self.page_size = int(page_size)
        self.device = torch.device(device)
        self.metrics = None
        self.inject_error = None
        self._tree_paths: List = []
        self.kernels = _load_kernels()
        self.predict = torch.empty(
            (self.B_cap * self.W + 1,), dtype=torch.int32, device=self.device
        )
        self.accept_index = torch.full(
            (self.B_cap, self.L), -1, dtype=torch.int32, device=self.device
        )
        self.accept_length = torch.empty(
            (self.B_cap,), dtype=torch.int32, device=self.device
        )
        self.pack_buf = torch.empty(
            (self.B_cap, self.L * 2 + 2), dtype=torch.int64, device=self.device
        )
        self.control_buf = torch.empty(
            (self.B_cap, 6), dtype=torch.int64, device=self.device
        )
        max_out = self.B_cap * self.L
        max_free = self.B_cap * (self.W + self.page_size - 1) // self.page_size + self.B_cap
        self.src_buf = torch.empty((max_out,), dtype=torch.int64, device=self.device)
        self.tgt_buf = torch.empty((max_out,), dtype=torch.int64, device=self.device)
        self.page_buf = torch.empty((max(max_free, 1),), dtype=torch.int64, device=self.device)
        self.index_buf = torch.empty((max_out,), dtype=torch.int64, device=self.device)

    def note_path(self, name: str) -> None:
        metrics = self.metrics
        if metrics is None:
            return
        try:
            metrics.paths[name] += 1
        except Exception:
            pass

    def reject_before_alloc(
        self,
        *,
        bs: int,
        verify_mode,
        is_all_greedy: bool,
        has_grammar: bool,
        vocab_mask,
        return_logprob: bool,
        prepare_hidden: bool,
        has_custom_logit_processor: bool,
        multimodal_reject_reason: Optional[str],
        simulate_acc_len: float,
        sampling_rows: int,
        seq_lens_cpu,
        allocator,
        draft_token_num: int,
        spec_steps: int,
        logits,
        out_cache_loc,
    ) -> Optional[str]:
        reason = conservative_mode_reason(verify_mode, is_all_greedy)
        if reason:
            return reason
        if has_grammar or vocab_mask is not None:
            return "grammar"
        if return_logprob:
            return "logprob"
        if prepare_hidden:
            return "hidden"
        if has_custom_logit_processor:
            return "logit_processor"
        if multimodal_reject_reason is not None:
            return multimodal_reject_reason
        if float(simulate_acc_len) > 0.0:
            return "simulate"
        if int(sampling_rows) != int(bs):
            return "sampling_rows"
        if (
            seq_lens_cpu is None
            or not torch.is_tensor(seq_lens_cpu)
            or seq_lens_cpu.dim() < 1
            or int(seq_lens_cpu.shape[0]) != int(bs)
        ):
            return "seq_lens_cpu"
        if allocator is None or not bool(getattr(allocator, "is_not_in_free_group", False)):
            return "free_group"
        if int(bs) > self.B_cap or int(bs) < 0:
            return "batch_cap"
        if int(draft_token_num) != self.W:
            return "width"
        if int(spec_steps) + 1 != self.L:
            return "path_cap"
        if (
            not torch.is_tensor(logits)
            or logits.dim() < 1
            or int(logits.shape[0]) != int(bs) * self.W
        ):
            return "logits_rows"
        if not torch.is_tensor(out_cache_loc) or int(out_cache_loc.numel()) != int(bs) * self.W:
            return "cache_loc"
        if out_cache_loc.dtype not in (torch.int32, torch.int64):
            return "dtype"
        if logits.device != self.device or out_cache_loc.device != self.device:
            return "device"
        if not logits.is_contiguous() or not out_cache_loc.is_contiguous():
            return "noncontiguous"
        return None

    def bind_verify_buffers(self, bs: int):
        predict = self.predict[: int(bs) * self.W + 1]
        accept_index = self.accept_index[: int(bs)]
        accept_length = self.accept_length[: int(bs)]
        if not (
            predict.is_contiguous()
            and accept_index.is_contiguous()
            and accept_length.is_contiguous()
        ):
            raise RuntimeError("fixed accept workspace view is not contiguous")
        accept_index.fill_(-1)
        accept_length.zero_()
        return predict, accept_index, accept_length

    def warmup_scratch(self) -> None:
        """Compile pack/commit against private tensors. Does not touch requests or live KV."""
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            copy_paged_kv_buffer_by_slot,
        )

        bs = 1
        predict, accept_index, accept_length = self.bind_verify_buffers(bs)
        accept_index.fill_(-1)
        if self.L > 0:
            accept_index[0, 0] = 0
        predict.zero_()
        if predict.numel() > 0:
            predict[0] = 1
        packed = self.pack_buf[:bs]
        self.kernels.pack_accept(accept_index, predict, accept_length, packed)
        packed.detach().to("cpu")
        width = self.W
        cache = torch.arange(width, device=self.device, dtype=torch.int64)
        pages = width // self.page_size + 2
        kv = torch.zeros(
            (2, 1, pages, self.page_size, 1, 1),
            dtype=torch.float32,
            device=self.device,
        )
        if width > 0:
            copy_paged_kv_buffer_by_slot(kv, cache[:1], cache[:1])
        self.accept_index.fill_(-1)
        self.accept_length.zero_()
        self.predict.zero_()

    def finalize(
        self,
        batch,
        logits_output,
        page_size: int,
        topk: int,
        allocator,
        accept_index: torch.Tensor,
        predict: torch.Tensor,
        accept_length: torch.Tensor,
        prepare_local_draft_hidden: bool = False,
    ):
        del prepare_local_draft_hidden  # fast path never gathers target hidden
        bs = int(accept_index.shape[0])
        try:
            if self.inject_error == "pack":
                raise RuntimeError("fixed accept pack failed")
            packed = _timed(
                self.metrics,
                "fixed_accept_pack",
                lambda: self._pack(bs, accept_index, predict, accept_length),
                device=True,
            )
            if self.inject_error == "readback":
                raise RuntimeError("fixed accept readback failed")
            cpu_pack = _timed(
                self.metrics,
                "fixed_accept_readback",
                lambda: self._readback(packed),
                device=False,
            )
            rows, token_rows, pre_lengths, errors = _parse_pack(cpu_pack, bs, self.L)
            validate_packed_rows(rows, pre_lengths, errors)
        except Exception:
            self.note_path("fixed_accept_error")
            raise

        try:
            think_end_id = getattr(getattr(batch, "model_config", None), "think_end_id", None)
            accepted, finished, truncated = _timed(
                self.metrics,
                "fixed_accept_cpu",
                lambda: apply_cpu_acceptance(
                    batch.reqs, rows, token_rows, think_end_id
                ),
                device=False,
            )
            self._export_paths(batch, truncated)
            output = _timed(
                self.metrics,
                "fixed_accept_kv_commit",
                lambda: self._commit(
                    batch,
                    logits_output,
                    allocator,
                    page_size,
                    topk,
                    accepted,
                    finished,
                    truncated,
                    token_rows,
                ),
                device=True,
            )
        except Exception:
            self.note_path("fixed_accept_error")
            raise
        self.note_path("fixed_accept_hit")
        if any(
            getattr(req, "multimodal_inputs", None) is not None
            for req in (getattr(batch, "reqs", None) or ())
        ):
            self.note_path("fixed_accept_multimodal_hit")
        return output

    def _pack(self, bs, accept_index, predict, accept_length):
        packed = self.pack_buf[:bs]
        self.kernels.pack_accept(accept_index, predict, accept_length, packed)
        return packed

    def _readback(self, packed: torch.Tensor) -> torch.Tensor:
        cpu = torch.empty(packed.shape, dtype=packed.dtype, device="cpu")
        cpu.copy_(packed)
        return cpu

    def _export_paths(self, batch, truncated) -> None:
        """Record this round's paths.

        A skipped export leaves request attributes unchanged and publishes
        nothing. A failed export clears both.
        """
        self._tree_paths = []
        spec_algo = getattr(batch, "spec_algorithm", None)
        export = bool(
            spec_algo is not None
            and callable(getattr(spec_algo, "is_standalone_remote", None))
            and spec_algo.is_standalone_remote()
        )
        if not export:
            return
        try:
            from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
                export_accepted_tree_candidate_indices,
            )

            paths = export_accepted_tree_candidate_indices(truncated)
            stored = [list(path) for path in paths]
            for req, path in zip(batch.reqs, stored):
                req.sr_accepted_tree_candidate_indices = path
            self._tree_paths = stored
        except Exception:
            self._tree_paths = []
            for req in batch.reqs:
                req.sr_accepted_tree_candidate_indices = None

    def _commit(
        self,
        batch,
        logits_output,
        allocator,
        page_size,
        topk,
        accepted,
        finished,
        truncated,
        token_rows,
    ):
        if int(page_size) != self.page_size or int(topk) <= 1:
            raise RuntimeError("fixed accept commit saw an unsupported layout")
        if not bool(getattr(allocator, "is_not_in_free_group", False)):
            raise RuntimeError("fixed accept allocator entered a free group")
        bs = len(accepted)
        if any(int(count) > self.W for count in accepted):
            raise RuntimeError("fixed accept output longer than the verify width")
        prefixes = _prefix_lengths(batch, bs)
        src_index: List[int] = []
        tgt_index: List[int] = []
        page_index: List[int] = []
        compact = 0
        for i, (count, row, prefix) in enumerate(zip(accepted, truncated, prefixes)):
            for idx in row:
                if int(idx) < 0:
                    break
                src_index.append(int(idx))
            for j in range(int(count)):
                tgt_index.append(i * self.W + j)
            for offset in free_page_row_offsets(prefix, count, self.W, self.page_size):
                page_index.append(i * self.W + offset)
            compact += int(count)
        if compact != len(src_index) or compact != len(tgt_index):
            raise RuntimeError("fixed accept compact length mismatch")
        n_free = len(page_index)
        cache = batch.out_cache_loc.reshape(-1)
        limit = int(cache.numel())
        if any(idx < 0 or idx >= limit for idx in src_index + tgt_index + page_index):
            raise RuntimeError("fixed accept slot index outside out_cache_loc")
        if self.inject_error == "move":
            raise RuntimeError("fixed accept kv move failed")
        src, tgt, pages = self.kernels.gather_commit_slots(
            cache,
            torch.tensor(src_index, dtype=torch.int64),
            torch.tensor(tgt_index, dtype=torch.int64),
            torch.tensor(page_index, dtype=torch.int64),
            self.page_size,
            self.src_buf,
            self.tgt_buf,
            self.page_buf,
        )
        if int(pages.numel()) != n_free:
            raise RuntimeError("fixed accept page count mismatch")
        # Assemble host-visible results while the stream is idle after readback.
        # The KV move is last so a later host sync does not wait for it.
        result = self._publish(
            batch,
            logits_output,
            accepted,
            finished,
            truncated,
            token_rows,
            tgt,
            int(topk),
        )
        kv = allocator.get_kvcache().kv_buffer
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            copy_paged_kv_buffer_by_slot,
        )

        if src.numel() > 0:
            copy_paged_kv_buffer_by_slot(kv, src, tgt)
        allocator.free_unique_pages(pages)
        return result

    def _copy_control(self, bs, accepted, finished, prefixes) -> None:
        """Device control rows. The hot path does not call this."""
        rows = []
        compact = 0
        for i, count in enumerate(accepted):
            rows.append(
                (
                    int(count),
                    compact,
                    0 if finished[i] else 1,
                    num_free_pages(prefixes[i], count, self.W, self.page_size),
                    first_free_pos(prefixes[i] + count, self.page_size),
                    int(count) - 1,
                )
            )
            compact += int(count)
        control = torch.tensor(rows, dtype=torch.int64, device="cpu")
        if control.numel() == 0:
            return
        self.control_buf[:bs].copy_(control.to(self.device))

    def _publish(
        self, batch, logits_output, accepted, finished, truncated, token_rows, tgt, topk
    ):
        del topk
        mode = length_update_mode(finished)
        accept_length_list = [int(count) - 1 for count in accepted]
        device = tgt.device
        length_tensor = torch.tensor(
            accept_length_list, dtype=torch.int32, device=device
        )
        if mode == "all":
            delta = torch.tensor(
                accepted, dtype=batch.seq_lens.dtype, device=batch.seq_lens.device
            )
            batch.seq_lens.add_(delta)
            cpu_delta = torch.tensor(accepted, dtype=batch.seq_lens_cpu.dtype)
            batch.seq_lens_cpu.add_(cpu_delta)
        verified = _compact_tokens(token_rows, truncated, device)
        flat_index = _compact_indices(truncated, device)
        unfinished = [i for i, flag in enumerate(finished) if not flag]
        idle = not unfinished
        if idle:
            draft_verified = verified[:0]
            draft_length = length_tensor[:0]
            draft_length_cpu: List[int] = []
            draft_seq = None
            draft_seq_cpu = None
            draft_req = None
            returned_cpu = list(accept_length_list)
        elif len(unfinished) == len(finished):
            batch.out_cache_loc = tgt
            draft_verified = verified
            draft_length = length_tensor
            draft_length_cpu = list(accept_length_list)
            draft_seq = batch.seq_lens.clone()
            draft_seq_cpu = batch.seq_lens_cpu.clone()
            draft_req = batch.req_pool_indices.clone()
            returned_cpu = draft_length_cpu
        else:
            pieces = []
            cursor = 0
            unfinished_set = set(unfinished)
            for i, count in enumerate(accepted):
                if i in unfinished_set:
                    pieces.append(tgt[cursor : cursor + int(count)])
                cursor += int(count)
            batch.out_cache_loc = torch.cat(pieces) if pieces else tgt[:0]
            index = torch.tensor(unfinished, dtype=torch.int64, device=device)
            draft_verified = _select_rows(verified, accepted, unfinished, device)
            draft_length = length_tensor.index_select(0, index)
            draft_length_cpu = [accept_length_list[i] for i in unfinished]
            draft_seq = batch.seq_lens.index_select(0, index)
            draft_seq_cpu = batch.seq_lens_cpu.index_select(
                0, torch.tensor(unfinished, dtype=torch.int64)
            )
            draft_req = batch.req_pool_indices.index_select(0, index)
            returned_cpu = list(accept_length_list)
        tree_paths = [list(path) for path in self._tree_paths]
        return FixedAcceptResult(
            verified_id=verified,
            accepted_indices=flat_index,
            accept_length_per_req_cpu=returned_cpu,
            draft_verified_id=draft_verified,
            draft_accept_length=draft_length,
            draft_accept_length_cpu=draft_length_cpu,
            seq_lens_for_draft=draft_seq,
            seq_lens_for_draft_cpu=draft_seq_cpu,
            req_pool_indices=draft_req,
            tree_paths=tree_paths,
            idle=idle,
            logits_output=logits_output,
        )


def _prefix_lengths(batch, bs: int) -> List[int]:
    cpu = batch.seq_lens_cpu
    values = cpu.detach().to("cpu").tolist()
    if len(values) < bs:
        raise RuntimeError("seq_lens_cpu shorter than the verify batch")
    return [int(v) for v in values[:bs]]


def _parse_pack(cpu_pack: torch.Tensor, bs: int, path_cap: int):
    rows = []
    tokens = []
    pre = []
    errors = []
    for i in range(bs):
        row = cpu_pack[i, :path_cap].tolist()
        token_row = cpu_pack[i, path_cap : path_cap * 2].tolist()
        rows.append([int(v) for v in row])
        tokens.append([int(v) for v in token_row])
        pre.append(int(cpu_pack[i, path_cap * 2]))
        errors.append(int(cpu_pack[i, path_cap * 2 + 1]))
    return rows, tokens, pre, errors


def _compact_tokens(token_rows, truncated, device) -> torch.Tensor:
    values = []
    for tokens, row in zip(token_rows, truncated):
        for j, idx in enumerate(row):
            if int(idx) < 0:
                break
            values.append(int(tokens[j]))
    if not values:
        return torch.empty((0,), dtype=torch.int32, device=device)
    return torch.tensor(values, dtype=torch.int32, device=device)


def _compact_indices(truncated, device) -> torch.Tensor:
    values = []
    for row in truncated:
        for idx in row:
            if int(idx) < 0:
                break
            values.append(int(idx))
    if not values:
        return torch.empty((0,), dtype=torch.int64, device=device)
    return torch.tensor(values, dtype=torch.int64, device=device)


def _select_rows(verified, accepted, unfinished, device) -> torch.Tensor:
    pieces = []
    cursor = 0
    unfinished_set = set(unfinished)
    for i, count in enumerate(accepted):
        if i in unfinished_set:
            pieces.append(verified[cursor : cursor + int(count)])
        cursor += int(count)
    if not pieces:
        return torch.empty((0,), dtype=verified.dtype, device=device)
    return torch.cat(pieces)


def build_fixed_accept_state(worker, env=None):
    """Construct the workspace once. ``None`` keeps every batch on V1.

    An explicit off switch does not import kernels. Unsupported static
    conditions are logged and also skip the import.
    """
    requested = read_sr_fixed_accept_env(env)
    device = getattr(worker, "device", None)
    buf = _kv_buffer(worker)
    if torch.is_tensor(buf):
        device = buf.device
    if not requested:
        logger.info(
            "[SR] fixed accept requested=False effective=False device=%s reason=env off",
            device,
        )
        return None
    reason = static_disable_reason(worker)
    if reason is not None:
        logger.info(
            "[SR] fixed accept requested=True effective=False device=%s reason=%s",
            device,
            reason,
        )
        return None
    try:
        _load_kernels()
    except Exception as exc:
        logger.info(
            "[SR] fixed accept requested=True effective=False device=%s reason=kernel import failed: %s",
            device,
            exc,
        )
        return None
    state = SRFixedAcceptState(
        batch_cap=int(worker._verify_max_bs),
        path_cap=int(worker.speculative_num_steps) + 1,
        width=int(worker.speculative_num_draft_tokens),
        page_size=int(worker.page_size),
        device=buf.device,
    )
    logger.info(
        "[SR] fixed accept requested=True effective=True device=%s reason=",
        state.device,
    )
    return state
