"""SR candidate storage. Existing softmax/topk arithmetic and ordering are kept."""

from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext

import torch

from sglang.srt.speculative.standalone_remote.sr_transfer_staging import (
    SRTransferUnresolved,
)

logger = logging.getLogger(__name__)


class CandidatePreflightDeclined(RuntimeError):
    """Completed private comparisons declined the workspace; safe to fall back."""


def _candidate_device(device):
    device = torch.device(device)
    if device.type != "cpu" and device.index is None:
        device = torch.device(
            device.type, torch.get_device_module(device.type).current_device()
        )
    return device


@contextmanager
def candidate_capture_scope(worker, workspace):
    """Bind an explicit capture decision, including a safe legacy decision."""
    previous = getattr(worker, "_capture_candidate_prepared", False)
    previous_workspace = getattr(worker, "_capture_candidate_workspace", None)
    scope = workspace.capture_scope() if workspace is not None else nullcontext()
    with scope:
        worker._capture_candidate_prepared = True
        worker._capture_candidate_workspace = workspace
        try:
            yield
        finally:
            worker._capture_candidate_workspace = previous_workspace
            worker._capture_candidate_prepared = previous


def draft_candidate_workspace(worker, batch, seed):
    if getattr(worker, "_capture_candidate_prepared", False):
        return worker._capture_candidate_workspace
    return candidate_workspace(worker, batch, seed)


def _topk_out(value, k, values, indices):
    if k == 1:
        torch.max(value, dim=-1, keepdim=True, out=(values, indices))
    else:
        torch.topk(value, k, dim=-1, out=(values, indices))


class SRTreeCandidateWorkspace:
    """One stream/domain owns step outputs until reply packing is queued.

    Graphs retain this object. Eager workspaces are keyed by shape and stream,
    so growing another round never replaces a captured address. Probabilities
    can be reused per step; selected scores, tokens and parents cannot.
    """

    def __init__(self, device, batch, topk, steps, width, vocab, dtype, *, graph=False):
        if min(batch, topk, steps, width, vocab) < 1:
            raise ValueError("invalid candidate workspace geometry")
        device = self.device = _candidate_device(device)
        self.batch, self.topk, self.steps = batch, topk, steps
        self.width, self.vocab, self.dtype = width, vocab, dtype
        self.graph, self.unresolved = graph, False
        self.holds = None
        self._inside_capture = False
        self.probability_out = True
        # CUDA already compiles candidate selection; preserve that fused path.
        # Fixed tables still replace cat and the final selection uses out=.
        self.selection_out = device.type != "cuda"
        self.validated = self.device.type == "cpu"
        self.stream = self._stream()
        total = topk + (steps - 1) * topk * topk
        if width - 1 > total:
            raise ValueError("draft window exceeds candidate count")
        self.score_table = torch.empty((batch, total), dtype=dtype, device=device)
        self.token_table = torch.empty((batch, total), dtype=torch.int64, device=device)
        self.parent_table = (
            torch.empty(
                (batch, topk + 1 + max(steps - 2, 0) * topk),
                dtype=torch.int64,
                device=device,
            )
            if steps > 1
            else torch.empty((batch, 0), dtype=torch.float32, device=device)
        )
        self.probs = torch.empty((batch * topk, vocab), dtype=dtype, device=device)
        self.top_values = torch.empty(
            (max(steps - 1, 1), batch * topk, topk), dtype=dtype, device=device
        )
        self.top_indices = torch.empty_like(self.top_values, dtype=torch.int64)
        self.scores = torch.empty((steps, batch, topk), dtype=dtype, device=device)
        self.select_indices = torch.empty(
            (steps, batch, topk), dtype=torch.int64, device=device
        )
        self.input_ids = torch.empty_like(self.select_indices)
        self.parent_rows = torch.empty_like(self.select_indices)
        self.parent_nodes = torch.empty_like(self.select_indices)
        self.base_rows = (
            torch.arange(batch, dtype=torch.int64, device=device).reshape(-1, 1) * topk
        )
        self.initial_parents = (
            torch.arange(-1, topk, dtype=torch.int64, device=device)
            .reshape(1, -1)
            .repeat(batch, 1)
        )
        self.final_values = torch.empty((batch, width - 1), dtype=dtype, device=device)
        self.final_indices_unsorted = torch.empty_like(
            self.final_values, dtype=torch.int64
        )
        self.final_indices = torch.empty_like(self.final_indices_unsorted)
        self.sort_order = torch.empty_like(self.final_indices)
        self.final_tokens = torch.empty_like(self.final_indices)
        self._storage = self._storage_key()

    def _stream(self):
        if self.device.type == "cpu":
            return None
        return torch.get_device_module(self.device.type).current_stream(self.device)

    def _storage_key(self):
        return tuple(
            (t.data_ptr(), tuple(t.shape), tuple(t.stride()))
            for t in self.__dict__.values()
            if isinstance(t, torch.Tensor)
        )

    def check(self, batch, *, check_stream=True, check_storage=True):
        if self.unresolved:
            raise SRTransferUnresolved("candidate workspace completion is unresolved")
        if not 0 < batch <= self.batch or (
            check_storage and self._storage_key() != self._storage
        ):
            raise RuntimeError("candidate workspace capacity/storage changed")
        if check_stream and not self._inside_capture and self._stream() != self.stream:
            raise RuntimeError("candidate workspace belongs to another stream")

    @contextmanager
    def capture_scope(self):
        if not self.graph or not self.validated:
            raise RuntimeError("candidate workspace must be prepared before capture")
        self.check(self.batch, check_stream=False)
        self._inside_capture = True
        try:
            yield self
        finally:
            self._inside_capture = False

    def _submitted(self, function, *inputs):
        if self.device.type != "cpu":
            self.holds = inputs
        try:
            return function()
        except BaseException as exc:
            if self.device.type != "cpu":
                self.unresolved = True
                if isinstance(exc, Exception) and not isinstance(
                    exc, SRTransferUnresolved
                ):
                    raise SRTransferUnresolved(
                        "candidate operation failed after submission"
                    ) from exc
            raise

    def probabilities(self, logits, step):
        if (
            logits.ndim != 2
            or logits.shape[1] != self.vocab
            or logits.dtype != self.dtype
            or logits.device != self.device
        ):
            raise RuntimeError("candidate logits layout/dtype mismatch")
        rows = logits.shape[0]
        if rows % self.topk or not 0 <= step < self.steps - 1:
            raise RuntimeError("candidate logits rows/step mismatch")
        self.check(rows // self.topk, check_storage=False)
        return self._submitted(lambda: self._probabilities(logits, step), logits)

    def _probabilities(self, logits, step):
        rows = logits.shape[0]
        values, indices = self.top_values[step, :rows], self.top_indices[step, :rows]
        if self.probability_out:
            torch.softmax(logits, dim=-1, out=self.probs[:rows])
            _topk_out(self.probs[:rows], self.topk, values, indices)
        else:
            # Selected data still lives in step-specific workspace storage.
            probs = torch.softmax(logits, dim=-1)
            result = (
                torch.max(probs, dim=-1, keepdim=True)
                if self.topk == 1
                else torch.topk(probs, self.topk, dim=-1)
            )
            values.copy_(result.values)
            indices.copy_(result.indices)
        return values, indices

    def select(self, step, p, indices, previous):
        k = self.topk
        batch = p.shape[0] if step == 0 else p.shape[0] // k
        self.check(batch, check_storage=False)
        expected = (batch, k) if step == 0 else (batch * k, k)
        if (
            not 0 <= step < self.steps
            or tuple(p.shape) != expected
            or tuple(indices.shape) != expected
            or p.dtype != self.dtype
            or indices.dtype != torch.int64
            or p.device != self.device
            or indices.device != self.device
            or (step > 0 and (previous is None or previous.shape != (batch, k)))
        ):
            raise RuntimeError("candidate step input mismatch")
        return self._submitted(
            lambda: self._select(step, p, indices, previous, batch),
            p,
            indices,
            previous,
        )

    def _select(self, step, p, indices, previous, batch):
        k = self.topk
        scores = self.scores[step, :batch]
        if step == 0:
            self.score_table[:batch, :k].copy_(p)
            self.token_table[:batch, :k].copy_(indices)
            scores.copy_(p)
            if self.steps > 1:
                self.parent_table[:batch, : k + 1].copy_(self.initial_parents[:batch])
            return indices.flatten(), scores, self.initial_parents[:batch], None
        offset = k + (step - 1) * k * k
        if not self.selection_out:
            from sglang.srt.speculative.spec_utils import select_top_k_tokens

            ids, _, selected_scores, info, selected_parents = select_top_k_tokens(
                step,
                p,
                indices,
                None,
                previous,
                k,
            )
            self.score_table[:batch, offset : offset + k * k].copy_(info[0].flatten(1))
            self.token_table[:batch, offset : offset + k * k].copy_(info[1])
            scores.copy_(selected_scores)
            self.input_ids[step, :batch].copy_(ids.reshape(batch, k))
            self.parent_rows[step, :batch].copy_(selected_parents.reshape(batch, k))
            self.parent_nodes[step, :batch].copy_(info[2])
            if step < self.steps - 1:
                start = k + 1 + (step - 1) * k
                self.parent_table[:batch, start : start + k].copy_(info[2])
            return (
                self.input_ids[step, :batch].flatten(),
                scores,
                self.parent_nodes[step, :batch],
                self.parent_rows[step, :batch].flatten(),
            )
        expanded = self.score_table[:batch, offset : offset + k * k].view(batch, k, k)
        torch.mul(previous.unsqueeze(2), p.reshape(batch, k, k), out=expanded)
        token_view = self.token_table[:batch, offset : offset + k * k]
        token_view.copy_(indices.reshape(batch, k * k))
        selected = self.select_indices[step, :batch]
        _topk_out(expanded.flatten(1), k, scores, selected)
        ids = self.input_ids[step, :batch]
        torch.gather(indices.reshape(batch, k * k), 1, selected, out=ids)
        parents = self.parent_rows[step, :batch]
        torch.div(selected, k, rounding_mode="floor", out=parents)
        parents.add_(self.base_rows[:batch])
        nodes = self.parent_nodes[step, :batch]
        torch.add(selected, k * k * (step - 1) + k, out=nodes)
        if step < self.steps - 1:
            start = k + 1 + (step - 1) * k
            self.parent_table[:batch, start : start + k].copy_(nodes)
        return ids.flatten(), scores, nodes, parents.flatten()

    def finish(self, batch):
        self.check(batch, check_storage=False)
        return self._submitted(lambda: self._finish(batch))

    def _finish(self, batch):
        n = self.width - 1
        if not self.selection_out and self.device.type != "cuda":
            indices = torch.sort(
                torch.topk(self.score_table[:batch], n, dim=-1).indices
            ).values
            self.final_indices[:batch].copy_(indices)
            self.final_tokens[:batch].copy_(
                torch.gather(self.token_table[:batch], 1, indices)
            )
            return (
                self.parent_table[:batch],
                self.final_indices[:batch],
                self.final_tokens[:batch],
            )
        values, unsorted = (
            self.final_values[:batch],
            self.final_indices_unsorted[:batch],
        )
        torch.topk(self.score_table[:batch], n, dim=-1, out=(values, unsorted))
        selected = self.final_indices[:batch]
        torch.sort(unsorted, dim=-1, out=(selected, self.sort_order[:batch]))
        torch.gather(
            self.token_table[:batch], 1, selected, out=self.final_tokens[:batch]
        )
        return self.parent_table[:batch], selected, self.final_tokens[:batch]

    def _sync(self):
        if self.device.type == "cpu":
            return
        torch.get_device_module(self.device.type).current_stream(
            self.device
        ).synchronize()

    def warm(self):
        """Only completed numerical comparisons may choose original operators.

        Operator exceptions do not establish that no device work was submitted.
        Keep private inputs and the owner's workspace on any uncertain failure.
        """
        if self.unresolved:
            raise SRTransferUnresolved("candidate preflight completion is unresolved")
        if self.validated:
            return
        # Imports are preflight metadata work, before private device submissions.
        from sglang.srt.speculative.eagle_utils import organize_draft_results
        from sglang.srt.speculative.spec_utils import select_top_k_tokens

        self.holds = []
        try:
            self._warm_compare(organize_draft_results, select_top_k_tokens)
        except CandidatePreflightDeclined:
            self.holds = None
            raise
        except BaseException as exc:
            if self.device.type != "cpu" or isinstance(exc, SRTransferUnresolved):
                self.unresolved = True
                if isinstance(exc, Exception) and not isinstance(
                    exc, SRTransferUnresolved
                ):
                    raise SRTransferUnresolved(
                        "candidate preflight failed after device submission"
                    ) from exc
            raise

    def _warm_compare(self, organize_draft_results, select_top_k_tokens):
        # Deterministic values with ties and near ties, without RNG side effects.
        row = torch.arange(self.vocab, dtype=torch.float32, device=self.device)
        self.holds.append(row)
        logits = (
            ((row % 97) * 0.03125)
            .to(self.dtype)
            .reshape(1, -1)
            .repeat(self.batch * self.topk, 1)
        )
        self.holds.append(logits)
        p = torch.softmax(logits, dim=-1)
        self.holds.append(p)
        expected = (
            torch.max(p, dim=-1, keepdim=True)
            if self.topk == 1
            else torch.topk(p, self.topk, dim=-1)
        )
        self.holds.append(expected)
        self.probability_out = True
        values, indices = self._probabilities(logits, 0)
        self._sync()
        probability_match = (
            torch.equal(self.probs, p)
            and torch.equal(values, expected.values)
            and torch.equal(indices, expected.indices)
        )
        self.probability_out = bool(probability_match)
        # Validate strided score regions and all selection/order out variants.
        # A mismatch here cannot silently change the captured tree.
        seed_p, seed_i = expected.values[: self.batch], expected.indices[: self.batch]

        def validate_selection():
            rp, ri = seed_p, seed_i
            wp, wi = seed_p, seed_i
            reference_scores = workspace_scores = None
            score_list, token_list, parent_list = [], [], []
            compared = []
            self.holds.extend((score_list, token_list, parent_list, compared))
            for step in range(self.steps):
                ids, _, reference_scores, info, parents = select_top_k_tokens(
                    step,
                    rp,
                    ri,
                    None,
                    reference_scores,
                    self.topk,
                )
                self.holds.append((ids, reference_scores, info, parents))
                wids, workspace_scores, nodes, wparents = self._select(
                    step, wp, wi, workspace_scores, self.batch
                )
                next_probs = None
                if step < self.steps - 1:
                    rp, ri = expected.values, expected.indices
                    wp, wi = self._probabilities(logits, step)
                    next_probs = (wp, wi, rp, ri)
                compared.append(
                    (
                        ids,
                        wids,
                        reference_scores,
                        workspace_scores,
                        info[2],
                        nodes,
                        parents,
                        wparents,
                        next_probs,
                    )
                )
                score_list.append(info[0])
                token_list.append(info[1])
                parent_list.append(info[2])
            ref = organize_draft_results(
                score_list, token_list, parent_list, self.width
            )
            self.holds.append(ref)
            got = self._finish(self.batch)
            self._sync()
            for (
                ids,
                wids,
                reference_scores,
                workspace_scores,
                nodes,
                wnodes,
                parents,
                wparents,
                next_probs,
            ) in compared:
                if not (
                    torch.equal(ids, wids)
                    and torch.equal(reference_scores, workspace_scores)
                    and torch.equal(nodes, wnodes)
                    and (parents is None or torch.equal(parents, wparents))
                ):
                    return False
                if next_probs is not None and not (
                    torch.equal(next_probs[0], next_probs[2])
                    and torch.equal(next_probs[1], next_probs[3])
                ):
                    return False
            return all(torch.equal(a, b) for a, b in zip(ref, got))

        self.selection_out = self.device.type != "cuda"
        if not validate_selection():
            self.selection_out = False
            if not validate_selection():
                self.holds = None
                raise CandidatePreflightDeclined(
                    "candidate workspace changed tree semantics"
                )
        self.holds = None
        self.validated = True
        logger.info(
            "[SR] candidate workspace probability_out=%s selection_out=%s graph=%s batch=%s",
            self.probability_out,
            self.selection_out,
            self.graph,
            self.batch,
        )


def candidate_workspace(worker, batch, seed, *, graph=False):
    device = seed.device
    if device.type not in ("npu", "cuda"):
        return None
    backend = torch.npu if device.type == "npu" else torch.cuda
    stream = backend.current_stream(device)
    from sglang.srt.speculative.standalone_remote.sr_kv_copy import (
        _capturing,
        _stream_key,
    )

    # Exact row shapes also keep sort input/output storage descriptors aligned.
    # Previously seen eager shapes retain their own buffers until shutdown.
    capacity = int(batch)
    key = (str(device), _stream_key(stream), bool(graph), capacity, seed.dtype)
    workspaces = getattr(worker, "_candidate_workspaces", None)
    if workspaces is None:
        workspaces = worker._candidate_workspaces = {}
    if key not in workspaces:
        if _capturing(device):
            raise RuntimeError(
                "candidate workspace cannot allocate during graph capture"
            )
        workspace = SRTreeCandidateWorkspace(
            device,
            capacity,
            worker.topk,
            worker.speculative_num_steps,
            worker.speculative_num_draft_tokens,
            worker.model_config.vocab_size,
            seed.dtype,
            graph=graph,
        )
        # The owner must retain the object even if warmup fails after submission.
        workspaces[key] = workspace
        try:
            workspace.warm()
        except CandidatePreflightDeclined:
            logger.warning(
                "[SR] candidate workspace preflight declined; using original draft assembly",
                exc_info=True,
            )
            # Remember the decline so a failed preflight is not launched again.
            workspaces[key] = None
        except BaseException:
            workspace.unresolved = True
            raise
        else:
            workspaces[key] = workspace
            metrics = getattr(
                getattr(worker, "scheduler", None), "_sr_round_metrics", None
            )
            if metrics is not None:
                metrics.counts["tree_candidate_workspace_grow"] = (
                    metrics.counts.get("tree_candidate_workspace_grow", 0) + 1
                )
                metrics.counts["tree_candidate_bytes"] = sum(
                    sum(
                        t.numel() * t.element_size()
                        for t in w.__dict__.values()
                        if isinstance(t, torch.Tensor)
                    )
                    for w in workspaces.values()
                    if w is not None
                )
    installed = workspaces[key]
    if installed is None:
        return None
    installed.check(batch)
    return installed
