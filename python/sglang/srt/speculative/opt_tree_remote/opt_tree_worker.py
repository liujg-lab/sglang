"""Target verifier for OPT_TREE_REMOTE.

Draft tokens stay a width-N list. The verify forward also contains the
committed tail at slot 0, so ``draft_token_num`` is N + 1. ``draft_token``
is 1-D with that many entries per request: ``prepare_for_verify`` allocates
``len(input_ids)`` KV slots. Parents use -1 for that root. The mask comes
from ``build_tree_from_parents``.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.opt_tree_remote.opt_tree_layout import (
    build_tree_from_parents,
    pack_verify_draft_token,
)
from sglang.srt.speculative.standalone_remote.verifier.sr_worker import (
    StandaloneRemoteWorker,
)


class OptTreeRemoteWorker(StandaloneRemoteWorker):
    def construct_draft_input(
        self,
        batch: ScheduleBatch,
        draft_num_tokens: Optional[int] = None,
        spec_steps: Optional[int] = None,
    ) -> EagleVerifyInput:
        width = int(self.server_args.opt_tree_width)
        max_depth = int(self.server_args.opt_tree_max_depth)
        num_verify = width + 1
        draft_tokens = int(self.server_args.speculative_num_draft_tokens)
        if num_verify != draft_tokens:
            raise RuntimeError(
                f"OPT verify length {num_verify} != speculative_num_draft_tokens "
                f"{draft_tokens}"
            )
        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                max(width, 1),
                max_depth,
                num_verify,
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
        token_rows: List[List[int]] = []
        parent_rows: List[List[int]] = []
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
            parents = [int(p) for p in parents]
            if len(tokens) != width or len(parents) != width:
                tokens = (tokens + [0] * width)[:width]
                parents = (parents + [-1] * width)[:width]
            token_rows.append(tokens)
            parent_rows.append(parents)

        parent_index = torch.tensor(parent_rows, dtype=torch.long)
        layout = build_tree_from_parents(parent_index, cpu_lens[:bs].long())
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
            topk=max(width, 1),
            draft_token_num=num_verify,
            capture_hidden_mode=self._target_capture_hidden_mode(batch),
            seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
        )
        if not wrote_mask or not wrote_pos:
            self._maybe_update_graph_verify_buffers(spec_info)
        return spec_info
