"""Target verifier for CLEAR_TREE_REMOTE.

The draft list is this round's K, one of the configured verify counts. The verify forward
also contains the committed tail at slot 0, so ``draft_token_num`` is K + 1.
``draft_token`` is 1-D with that many entries per request. A parent of -2 is
an unlinked pad. The mask comes from ``build_tree_from_parents``.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.clear_tree_remote.clear_tree_layout import (
    align_draft_row,
    build_tree_from_parents,
    clear_opt_no_draft_links,
    pack_verify_draft_token,
)
from sglang.srt.speculative.standalone_remote.verifier.sr_worker import (
    StandaloneRemoteWorker,
)


class ClearTreeRemoteWorker(StandaloneRemoteWorker):
    def __init__(
        self,
        server_args,
        gpu_id: int,
        tp_rank: int,
        dp_rank,
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker,
    ):
        self.fixed_accept_skip = "clear_tree"
        super().__init__(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

    def construct_draft_input(
        self,
        batch: ScheduleBatch,
        draft_num_tokens: Optional[int] = None,
        spec_steps: Optional[int] = None,
    ) -> EagleVerifyInput:
        from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
            VERIFY_KS,
        )
        from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
            TARGET_CAPACITY,
        )

        max_depth = int(self.server_args.clear_tree_max_depth)
        capacity = int(self.server_args.speculative_num_draft_tokens)
        if capacity != TARGET_CAPACITY:
            raise RuntimeError(
                f"CLEAR_TREE_REMOTE verify capacity {capacity} != {TARGET_CAPACITY}"
            )
        tree_topk = max(int(self.server_args.speculative_eagle_topk or 2), 2)
        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                tree_topk,
                max_depth,
                VERIFY_KS[0] + 1,
                device=batch.device,
            )

        bs = batch.batch_size()
        device = batch.device
        cpu_lens = batch.seq_lens_cpu
        if (
            cpu_lens is None
            or not torch.is_tensor(cpu_lens)
            or cpu_lens.device.type != "cpu"
        ):
            batch.seq_lens_cpu = batch.seq_lens.detach().to("cpu")
            cpu_lens = batch.seq_lens_cpu
        batch.seq_lens_sum = int(cpu_lens[:bs].sum())

        roots: List[int] = []
        raw_tokens: List[List[int]] = []
        raw_parents: List[List[int]] = []
        for req in list(batch.reqs)[:bs]:
            if len(req.output_ids) > 0:
                roots.append(int(req.output_ids[-1]))
            else:
                roots.append(int(req.origin_input_ids[-1]))
            dtl = req.draft_tokens_and_logits or {}
            tokens = [int(x) for x in (dtl.get("draft_tokens") or [])]
            parents = dtl.get("parent_list")
            if parents is None:
                parents = [-1] * len(tokens)
            raw_tokens.append(tokens)
            raw_parents.append([int(p) for p in parents])
        present = [len(row) for row in raw_tokens if len(row) in VERIFY_KS]
        verify_k = max(present) if present else VERIFY_KS[0]
        if verify_k + 1 > capacity:
            raise RuntimeError(
                f"CLEAR_TREE_REMOTE verify length {verify_k + 1} exceeds "
                f"capacity {capacity}"
            )
        token_rows: List[List[int]] = []
        parent_rows: List[List[int]] = []
        no_draft: List[bool] = []
        for tokens, parents in zip(raw_tokens, raw_parents):
            tokens, parents, missing = align_draft_row(tokens, parents, verify_k)
            no_draft.append(missing)
            token_rows.append(tokens)
            parent_rows.append(parents)
        num_verify = verify_k + 1

        parent_index = torch.tensor(parent_rows, dtype=torch.long)
        layout = build_tree_from_parents(parent_index, cpu_lens[:bs].long())
        clear_opt_no_draft_links(
            layout.retrive_next_token, layout.retrive_next_sibling, no_draft
        )
        draft_token = pack_verify_draft_token(roots, token_rows).to(device=device)
        if batch.sampling_info.penalizer_orchestrator.is_required:
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                draft_token.view(bs, num_verify)[:, 0]
            )

        tree_mask = layout.tree_mask.to(device=device)
        positions = layout.positions.to(device=device)
        mask_buf, pos_buf, wrote_mask, wrote_pos = self._verify_mask_position_bufs(
            int(tree_mask.numel()), int(positions.numel()), device
        )
        if wrote_mask:
            mask_buf.copy_(tree_mask.reshape(-1))
            tree_mask = mask_buf
        if wrote_pos:
            pos_buf.copy_(positions.reshape(-1))
            positions = pos_buf

        spec_info = EagleVerifyInput(
            draft_token=draft_token,
            custom_mask=tree_mask,
            positions=positions,
            retrive_index=layout.retrive_index.to(device=device),
            retrive_next_token=layout.retrive_next_token.to(device=device),
            retrive_next_sibling=layout.retrive_next_sibling.to(device=device),
            retrive_cum_len=None,
            spec_steps=max_depth,
            topk=tree_topk,
            draft_token_num=num_verify,
            capture_hidden_mode=self._target_capture_hidden_mode(batch),
            seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
        )
        self._opt_rpd_owner = self._attach_rpd_input(
            spec_info, batch, roots, token_rows, parent_rows, no_draft, device
        )
        if not wrote_mask or not wrote_pos:
            self._maybe_update_graph_verify_buffers(spec_info)
        return spec_info

    def _attach_rpd_input(
        self, spec_info, batch, roots, token_rows, parent_rows, no_draft, device
    ):
        from sglang.srt.speculative.clear_tree_remote.clear_tree_rpd import (
            attach_opt_rpd_input,
        )

        return attach_opt_rpd_input(
            getattr(self, "_opt_rpd_owner", None),
            self.server_args,
            spec_info,
            batch,
            roots,
            token_rows,
            parent_rows,
            no_draft,
            device,
        )

    def verify(self, batch, spec_info):
        """Parent verify, then a local cost line. The reply bytes are unchanged."""
        from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
            ledger_enabled,
        )

        enabled = ledger_enabled()
        metrics = getattr(batch, "sr_round_metrics", None)
        frozen = self._freeze_verify_shape(batch, spec_info)
        seq_lens = list(frozen["seq_lens"])
        if enabled and metrics is not None:
            from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
                install_clear_device_emit,
            )

            install_clear_device_emit(metrics)
            metrics._clear_ledger_device = True
        ok = False
        try:
            result = super().verify(batch, spec_info)
            ok = True
        finally:
            if enabled and metrics is not None:
                metrics._clear_ledger_device = False
                if not ok:
                    metrics._clear_device_open.clear()
        if ok:
            self._store_verify_report(batch, spec_info, result, metrics, frozen)
        if not enabled:
            return result
        try:
            from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
                consume_verify_device,
                next_verify_id,
                read_verify_host_us,
                runtime_kv_buckets,
                target_verify_record,
            )
            from sglang.srt.speculative.tree_attn_fallback import (
                select_tree_kv_bucket,
            )

            k = int(frozen["k"])
            ran_graph = bool(result[3]) if len(result) > 3 else False
            buckets = runtime_kv_buckets()
            max_ell = max(seq_lens) if seq_lens else 0
            bucket = select_tree_kv_bucket(max_ell + k + 1, buckets)
            size_fn = getattr(batch, "batch_size", None)
            size = size_fn() if callable(size_fn) else len(seq_lens or [])
            host_us = None if metrics is None else read_verify_host_us(metrics)
            verify_id = None if metrics is None else next_verify_id(metrics)
            inflight = getattr(batch, "sr_step_inflight", None)
            rpc_seq = getattr(inflight, "rpc_seq", None) if inflight is not None else None
            device_us = None
            if metrics is not None:
                device_us = consume_verify_device(
                    metrics,
                    {
                        "verify_id": verify_id,
                        "rpc_seq": None if rpc_seq is None else int(rpc_seq),
                        "sum_ell": sum(seq_lens or []),
                        "kv_bucket": None if bucket is None else int(bucket),
                        "b": int(size),
                        "k": int(k),
                        "ran_graph": bool(ran_graph),
                    },
                )
            if metrics is not None and device_us is not None:
                from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
                    push_verify_sample,
                )

                push_verify_sample(
                    metrics, int(size), int(k), bucket, int(device_us), ran_graph
                )
            target_verify_record(
                rank=int(getattr(self, "tp_rank", 0) or 0),
                batch_size=int(size),
                k=k,
                seq_lens=seq_lens or [],
                ran_graph=ran_graph,
                kv_bucket=bucket,
                verify_host_us=host_us,
                device_us=device_us,
                verify_id=verify_id,
                rpc_seq=None if rpc_seq is None else int(rpc_seq),
            )
        except Exception:
            logger.warning("[CLEAR ledger] target verify log failed", exc_info=True)
        return result

    def _freeze_verify_shape(self, batch, spec_info) -> dict:
        """Snapshot the reply being verified, before accept sends the next STEP.

        Sequence lengths and K are the pre-verify values. The rpc is the
        reply key still held by the Target client.
        """
        try:
            seq_lens = _seq_lens_before_verify(batch)
        except Exception:
            seq_lens = []
        k = max(int(getattr(spec_info, "draft_token_num", 1) or 1) - 1, 0)
        size_fn = getattr(batch, "batch_size", None)
        try:
            size = int(size_fn()) if callable(size_fn) else len(seq_lens)
        except Exception:
            size = len(seq_lens)
        session = None
        rpc = None
        scheduler = getattr(self, "_clear_target_scheduler", None)
        if scheduler is not None:
            live = getattr(scheduler, "sr_session_id", None)
            session = None if live in (None, "") else str(live)
            client = getattr(scheduler, "sr_client", None)
            key = None if client is None else getattr(client, "_last_reply_key", None)
            if (
                session is not None
                and isinstance(key, tuple)
                and len(key) == 2
                and key[0] == session
                and type(key[1]) is int
            ):
                rpc = int(key[1])
        return {
            "session": session,
            "rpc_seq": rpc,
            "b": int(size) if int(size) > 0 else len(seq_lens),
            "k": int(k),
            "seq_lens": [int(item) for item in seq_lens],
        }

    def _store_verify_report(self, batch, spec_info, result, metrics, frozen) -> None:
        """Remember the frozen shape so a later STEP can name this verify.

        ``fallback`` is only a verify that did not use draft tokens. Eager
        verify and an over-bucket prefix stay distinct: neither is a graph
        sample. The bucket uses the length frozen before accept.
        """
        del spec_info
        if metrics is None or not isinstance(frozen, dict):
            return
        mode = getattr(batch, "forward_mode", None)
        if mode is not None and bool(getattr(mode, "is_idle", lambda: False)()):
            return
        try:
            from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
                complete_verify_report,
            )

            ran_graph = (
                bool(result[3])
                if isinstance(result, (tuple, list)) and len(result) > 3
                else False
            )
            metrics._clear_verify_report = complete_verify_report(frozen, ran_graph)
        except Exception:
            logger.warning("[CLEAR] verify report failed", exc_info=True)


def _seq_lens_before_verify(batch) -> List[int]:
    cpu = getattr(batch, "seq_lens_cpu", None)
    if cpu is not None:
        try:
            return [int(length) for length in list(cpu)[: batch.batch_size()]]
        except Exception:
            pass
    lens = getattr(batch, "seq_lens", None)
    if lens is None:
        return []
    if hasattr(lens, "detach"):
        values = lens.detach().cpu().tolist()
    else:
        values = list(lens)
    return [int(length) for length in values]
