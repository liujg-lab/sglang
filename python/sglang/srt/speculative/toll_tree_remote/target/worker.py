"""Target verify worker. Padding tokens are 0 and stay inside the verify width."""

from __future__ import annotations

import torch

from sglang.srt.speculative.toll_tree_remote.tree.layout import build_verify_layout


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


class TollTreeRemoteWorker:
    """Target verifier. ``verify`` uses Eagle's path and does not import SR."""

    def __init__(self, server_args=None, target_worker=None, **kwargs) -> None:
        self.server_args = server_args
        self.target_worker = target_worker
        self.page_size = int(getattr(target_worker, "page_size", 1) or 1)
        self.verify_rounds = 0
        self.accepted_draft_tokens = 0

    @property
    def model_runner(self):
        return self.target_worker.model_runner

    @property
    def token_to_kv_pool_allocator(self):
        return self.model_runner.token_to_kv_pool_allocator

    def forward_batch_generation(self, batch):
        from sglang.srt.managers.utils import GenerationBatchResult
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode

        if batch.forward_mode.is_extend() or getattr(batch, "is_extend_in_batch", False):
            return self._forward_plain(batch, CaptureHiddenMode.NULL)
        reply = getattr(batch.reqs[0], "toll_reply", None) if batch.reqs else None
        if reply is None or not reply.reqs or not reply.reqs[0].tokens:
            return self._forward_plain(batch, CaptureHiddenMode.NULL)
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
            return self._forward_plain(batch, CaptureHiddenMode.NULL)
        spec.capture_hidden_mode = CaptureHiddenMode.NULL
        spec.prepare_for_verify(batch, self.page_size)
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = spec
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.NULL
        result = self.target_worker.forward_batch_generation(model_worker_batch, is_verify=True)
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
        self.verify_rounds += 1
        self.accepted_draft_tokens += sum(lengths)
        return GenerationBatchResult(
            logits_output=result.logits_output,
            next_token_ids=verified.verified_id,
            num_accepted_tokens=sum(lengths),
            accept_length_per_req_cpu=lengths,
            can_run_cuda_graph=result.can_run_cuda_graph,
        )

    def _forward_plain(self, batch, hidden_mode):
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = hidden_mode
        return self.target_worker.forward_batch_generation(model_worker_batch)

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
