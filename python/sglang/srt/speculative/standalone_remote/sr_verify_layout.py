"""CPU-safe verify-layout helpers for STANDALONE_REMOTE trees.

Kept free of sglang.srt.utils so unit tests can import it without torchvision.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.speculative.standalone_remote.sr_transfer_staging import (
    SRTransferUnresolved,
    alloc_host,
    submit_copy,
    wait_event,
)


def advance_tree_draft_positions(
    positions: Optional[torch.Tensor],
    mrope_positions: Optional[torch.Tensor] = None,
) -> None:
    """Advance RoPE by one tree-decode step in-place.

    EAGLE ``draft_forward`` only does ``positions.add_(1)``. Qwen2/3-VL ignores
    that 1D tensor and reads ``mrope_positions`` (shape ``(3, n)``). After the
    vision prefix, all three M-RoPE axes increment together during text decode,
    so both tensors must move by +1 on every spec step. Leaving M-RoPE frozen
    makes layer-0 drafts match and deeper nodes garbage (accept len stuck at 2).
    """
    if positions is not None:
        positions.add_(1)
    if mrope_positions is not None:
        mrope_positions.add_(1)


def advance_tree_draft_positions_for_step(
    step_i: int,
    positions: Optional[torch.Tensor],
    mrope_positions: Optional[torch.Tensor] = None,
) -> None:
    """Advance RoPE only after the first tree forward.

    STANDALONE initializes ``positions`` / M-RoPE at ``seq_lens``, which is
    already the correct id for the first draft tokens. EAGLE CUDA still
    ``add_(1)`` on every step including ``i==0``; SR full models must not.
    """
    if int(step_i) <= 0:
        return
    advance_tree_draft_positions(positions, mrope_positions)


def export_accepted_tree_candidate_indices(
    accept_index,
    retrive_index=None,
) -> List[List[int]]:
    """Map 2D ``accept_index`` to reply candidate indices (exclude root).

    ``accept_index[b, j]`` stores ``retrive_index`` values. Column 0 is the
    tree root. Later columns that are not ``-1`` are accepted draft nodes.
    Bonus lives in ``predict[last_accepted]``, not an extra accept slot.

    When ``retrive_index`` is omitted, each CPU row uses ``row[0]`` as root.
    Empty rows or invalid roots yield an empty path. Never default root to 0.
    """
    if accept_index is None:
        return []
    if hasattr(accept_index, "detach"):
        rows = accept_index.detach().to("cpu").tolist()
    else:
        rows = list(accept_index)
    if not rows:
        return []
    roots = None
    if retrive_index is not None:
        if hasattr(retrive_index, "detach"):
            roots = retrive_index[:, 0].detach().to("cpu").reshape(-1).tolist()
        else:
            roots = [int(row[0]) for row in retrive_index]
    out: List[List[int]] = []
    for b, row in enumerate(rows):
        seq = row if isinstance(row, (list, tuple)) else [row]
        if roots is not None:
            root = int(roots[b]) if b < len(roots) else None
        elif not seq:
            root = None
        else:
            try:
                root = int(seq[0])
            except (TypeError, ValueError):
                root = None
        if root is None or root < 0:
            out.append([])
            continue
        path: List[int] = []
        for j, idx in enumerate(seq):
            if j == 0:
                continue
            if idx is None or int(idx) < 0:
                break
            local = int(idx) - root
            if local >= 1:
                path.append(local - 1)
        out.append(path)
    return out


def chain_tree_structure(
    num_draft_tokens: int, spec_steps: int, device=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    dev = device if device is not None else torch.device("cpu")
    parent_list = torch.arange(-1, spec_steps - 1, dtype=torch.int64, device=dev)
    top_scores_index = torch.arange(
        max(num_draft_tokens - 1, 0), dtype=torch.int64, device=dev
    )
    return parent_list, top_scores_index


_CHAIN_TEMPLATES: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def cached_chain_template(
    num_draft_tokens: int, spec_steps: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read-only CPU chain topology for one ``(num_draft_tokens, spec_steps)``.

    Callers copy out of the tensors. Filling a packet must not write them.
    """
    key = (int(num_draft_tokens), int(spec_steps))
    cached = _CHAIN_TEMPLATES.get(key)
    if cached is None:
        cached = chain_tree_structure(key[0], key[1], device="cpu")
        _CHAIN_TEMPLATES[key] = cached
    return cached


class DraftPacketPlan:
    """This round's widths. Capacity elsewhere may be larger."""

    def __init__(
        self,
        bs: int,
        token_width: int,
        parent_w: int,
        index_w: int,
        chain_only: bool,
        chain_parents: torch.Tensor,
        chain_indices: torch.Tensor,
        token_rows: Sequence,
        parsed_parents: Sequence[Optional[torch.Tensor]],
        parsed_indices: Sequence[Optional[torch.Tensor]],
    ) -> None:
        self.bs = int(bs)
        self.token_width = int(token_width)
        self.parent_w = int(parent_w)
        self.index_w = int(index_w)
        self.chain_only = bool(chain_only)
        self.chain_parents = chain_parents
        self.chain_indices = chain_indices
        self.token_rows = token_rows
        self.parsed_parents = parsed_parents
        self.parsed_indices = parsed_indices

    @property
    def used(self) -> int:
        return self.bs * (1 + self.token_width + self.parent_w + self.index_w)


def plan_draft_rows(
    token_rows: Sequence,
    parent_rows: Sequence[Optional[Sequence[int]]],
    index_rows: Sequence[Optional[Sequence[int]]],
    topk: int,
    spec_steps: int,
    num_draft_tokens: int,
) -> DraftPacketPlan:
    """Choose this round's row widths without writing a destination buffer.

    ``topk <= 1`` ignores supplied topology. A row uses chain when parent or
    index is ``None``. An empty list is present topology, not ``None``.
    """
    bs = len(token_rows)
    token_width = max(int(num_draft_tokens) - 1, 1)
    chain_parents, chain_indices = cached_chain_template(num_draft_tokens, spec_steps)
    parsed_parents: List[Optional[torch.Tensor]] = [None] * bs
    parsed_indices: List[Optional[torch.Tensor]] = [None] * bs
    n_tree = 0
    if int(topk) > 1:
        for i in range(bs):
            parent = parent_rows[i] if i < len(parent_rows) else None
            index = index_rows[i] if i < len(index_rows) else None
            if parent is None or index is None:
                continue
            n_tree += 1
            parsed_parents[i] = torch.as_tensor(parent, dtype=torch.int64)
            parsed_indices[i] = torch.as_tensor(index, dtype=torch.int64)
    chain_only = int(topk) <= 1 or n_tree == 0
    parent_w = int(chain_parents.numel())
    index_w = int(chain_indices.numel())
    if not chain_only:
        for parent, index in zip(parsed_parents, parsed_indices):
            if parent is not None:
                parent_w = max(parent_w, int(parent.numel()))
            if index is not None:
                index_w = max(index_w, int(index.numel()))
    return DraftPacketPlan(
        bs,
        token_width,
        parent_w,
        index_w,
        chain_only,
        chain_parents,
        chain_indices,
        token_rows,
        parsed_parents,
        parsed_indices,
    )


def packet_segment_views(
    storage: torch.Tensor, plan: DraftPacketPlan
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """View ``verified, tokens, parents, indices`` inside ``storage[:used]``.

    Offsets follow this round's widths, not the buffer's historical capacity.
    """
    bs = plan.bs
    used = plan.used
    flat = storage[:used]
    offset = 0
    verified = flat[offset : offset + bs]
    offset += bs
    tokens = flat[offset : offset + bs * plan.token_width].view(bs, plan.token_width)
    offset += bs * plan.token_width
    parents = flat[offset : offset + bs * plan.parent_w].view(bs, plan.parent_w)
    offset += bs * plan.parent_w
    indices = flat[offset : offset + bs * plan.index_w].view(bs, plan.index_w)
    return verified, tokens, parents, indices


def slice_decode_batch_row(
    tensor: Optional[torch.Tensor],
    row: int,
    n_rows: int,
    token_lens: Optional[Sequence[int]] = None,
) -> Optional[torch.Tensor]:
    """Pick one request's last-token row from decode or packed-extend tensors.

    DECODE logits/hidden are ``[n_rows, ...]``. A packed EXTEND hidden is
    ``[sum(token_lens), ...]`` (FULL capture); take the last token of ``row``
    using ``token_lens`` (same as logits_processor ``cumsum(extend_seq_lens)-1``).
    A single-request EXTEND without lens takes the last row.
    """
    if tensor is None or n_rows <= 0 or row < 0 or row >= n_rows:
        return None
    if tensor.dim() == 0:
        return tensor
    leading = int(tensor.shape[0])
    if leading == n_rows:
        return tensor[row : row + 1]
    if token_lens is not None and len(token_lens) == n_rows:
        lens = [int(x) for x in token_lens]
        if lens[row] > 0 and sum(lens) == leading:
            end = sum(lens[: row + 1])
            return tensor[end - 1 : end]
    if n_rows == 1 and leading >= 1:
        return tensor[-1:]
    return None


def copy_tokens_into_row(
    dest: torch.Tensor, src, token_width: int
) -> None:
    if src is None or token_width <= 0:
        return
    if isinstance(src, torch.Tensor):
        flat = src.detach().reshape(-1)
        if flat.device != dest.device:
            flat = flat.to(dest.device)
        n = min(int(flat.numel()), token_width)
        if n:
            dest[:n] = flat[:n]
        return
    n = min(len(src), token_width)
    if n:
        dest[:n] = torch.as_tensor(src[:n], dtype=torch.int64)


def _export_assembled_rows(
    cpu_tensor: torch.Tensor,
    out: Optional[torch.Tensor],
    device,
) -> torch.Tensor:
    bs, width = int(cpu_tensor.shape[0]), int(cpu_tensor.shape[1])
    if out is not None and out.shape[0] >= bs and out.shape[1] >= width:
        view = out[:bs, :width]
        if width:
            view.copy_(cpu_tensor, non_blocking=view.device.type != "cpu")
        return view
    if device is None or torch.device(device).type == "cpu":
        return cpu_tensor
    return cpu_tensor.to(device=device, non_blocking=True)


def _copy_vector_into_rows(dest: torch.Tensor, vector: torch.Tensor) -> None:
    """Copy a read-only 1D template into every row, or into a prefix."""
    if dest.numel() == 0 or vector.numel() == 0 or dest.dim() != 2:
        return
    width = int(dest.shape[1])
    count = min(width, int(vector.numel()))
    if count == 0:
        return
    source = vector[:count].unsqueeze(0).expand(int(dest.shape[0]), count)
    if count == width:
        dest.copy_(source)
    else:
        dest[:, :count].copy_(source)


def _copy_vector_into_row(dest_row: torch.Tensor, vector: torch.Tensor) -> None:
    count = min(int(dest_row.numel()), int(vector.numel()))
    if count:
        dest_row[:count].copy_(vector[:count])


def write_draft_regions(
    tokens: torch.Tensor,
    parents: torch.Tensor,
    indices: torch.Tensor,
    plan: DraftPacketPlan,
) -> None:
    """Initialize this round's active region, then copy each row.

    Tokens, parents, and indices start as ``0 / -1 / 0`` when the chain
    template does not already cover the whole row. Chain templates are only
    read. ``torch.as_tensor`` for present rows happens in ``plan_draft_rows``.
    """
    if plan.bs == 0:
        return
    tokens.zero_()
    for i in range(plan.bs):
        src = plan.token_rows[i] if i < len(plan.token_rows) else None
        copy_tokens_into_row(tokens[i], src, plan.token_width)
    if plan.chain_only:
        _copy_vector_into_rows(parents, plan.chain_parents)
        _copy_vector_into_rows(indices, plan.chain_indices)
        return
    if parents.numel():
        parents.fill_(-1)
    if indices.numel():
        indices.zero_()
    for i, (parent, index) in enumerate(zip(plan.parsed_parents, plan.parsed_indices)):
        if parent is None or index is None:
            _copy_vector_into_row(parents[i], plan.chain_parents)
            _copy_vector_into_row(indices[i], plan.chain_indices)
            continue
        _copy_vector_into_row(parents[i], parent.reshape(-1))
        _copy_vector_into_row(indices[i], index.reshape(-1))


def devices_compatible(stored: torch.device, requested: torch.device) -> bool:
    """Whether an existing buffer can serve this round's device.

    A request without an index (``"npu"`` / ``"cuda"``) means the current
    device of that type, so a buffer already on ``npu:0`` stays reusable.
    ``npu:0`` is not reusable for an explicit ``npu:1``.
    """
    if stored.type != requested.type:
        return False
    if requested.index is None:
        return True
    if stored.index is None:
        return False
    return stored.index == requested.index


class VerifyInputPacket:
    """Reusable host packet for one SR verify input upload.

    The worker keeps this object. ``event is None`` means the previous
    ``submit_copy()`` returned ``None`` after a finished copy. A copy that was
    submitted but whose completion cannot be confirmed sets ``unresolved`` and
    must not be treated as finished just because ``event`` is missing.

    Synchronous SR orders one stream as this round's H2D, then every consumer
    of the returned views, then the next round's H2D. Consumers include
    penalty accumulation on ``verified_id`` and tree build. The event only
    guards rewriting this host packet. Another stream or overlap needs its
    own consume-complete dependency. Returned views are borrowed for this
    round; the next upload may overwrite them. Graph mask and position
    buffers are not stored here.
    """

    def __init__(self) -> None:
        self.host_packet: Optional[torch.Tensor] = None
        self.device_packet: Optional[torch.Tensor] = None
        self.capacity = 0
        self.event = None
        self.unresolved = False
        self.generation = 0
        self.rpd_input = None

    def _raise_if_unresolved(self) -> None:
        if self.unresolved:
            raise RuntimeError(
                "SR verify packet is unresolved; refusing to fill, grow, or replace it"
            )

    def on_accelerator(self, device) -> bool:
        dev = device if isinstance(device, torch.device) else torch.device(device)
        return dev.type != "cpu"

    def new_device_packet(self, capacity: int, device) -> torch.Tensor:
        return torch.empty((capacity,), dtype=torch.int64, device=device)

    def _note(self, metrics, kind: str, name: str, value) -> None:
        if metrics is None:
            return
        if kind == "host":
            metrics.add_host(name, value)
            return
        bucket = getattr(metrics, kind, None)
        if bucket is None:
            return
        bucket[name] += value

    def wait_for_host(self, metrics) -> None:
        event = self.event
        if event is None:
            return
        start = time.perf_counter()
        try:
            wait_event(event)
        except SRTransferUnresolved:
            self.unresolved = True
            self._note(metrics, "host", "verify_packet_wait", time.perf_counter() - start)
            raise
        self.event = None
        self._note(metrics, "host", "verify_packet_wait", time.perf_counter() - start)

    def ensure(self, used: int, device, metrics) -> None:
        """Grow host and, on an accelerator, device storage. Never shrinks."""
        self._raise_if_unresolved()
        dev = device if isinstance(device, torch.device) else torch.device(device)
        on_accel = self.on_accelerator(dev)
        host = self.host_packet
        device_buf = self.device_packet
        host_ok = (
            host is not None and host.dtype == torch.int64 and int(host.numel()) >= used
        )
        device_ok = not on_accel or (
            device_buf is not None
            and device_buf.dtype == torch.int64
            and devices_compatible(device_buf.device, dev)
            and int(device_buf.numel()) >= used
        )
        if host_ok and device_ok:
            return
        new_cap = max(int(self.capacity), 1)
        while new_cap < max(int(used), 1):
            new_cap *= 2
        new_host = host
        if not host_ok:
            new_host, _pinned = alloc_host((new_cap,), torch.int64, dev)
        new_device = device_buf
        if on_accel and not device_ok:
            new_device = self.new_device_packet(new_cap, dev)
        self.host_packet = new_host
        self.device_packet = new_device
        self.capacity = int(new_host.numel())
        self._note(metrics, "counts", "verify_packet_grow", 1)

    def _empty_views(self, plan: DraftPacketPlan, device):
        dev = device if isinstance(device, torch.device) else torch.device(device)
        verified = torch.empty((0,), dtype=torch.int64, device=dev)
        parents = torch.empty((0, plan.parent_w), dtype=torch.int64, device=dev)
        indices = torch.empty((0, plan.index_w), dtype=torch.int64, device=dev)
        tokens = torch.empty((0, plan.token_width), dtype=torch.int64, device=dev)
        return verified, parents, indices, tokens

    def load(
        self,
        verified_ids: Sequence[int],
        token_rows: Sequence,
        parent_rows: Sequence,
        index_rows: Sequence,
        topk: int,
        spec_steps: int,
        num_draft_tokens: int,
        device,
        metrics=None,
        *,
        rpd_vocab=None,
        rpd_batch_key=(),
    ):
        """Fill this round and upload ``[:used]`` once on an accelerator.

        CPU callers get host views and do not call ``submit_copy()``. An empty
        batch does not upload. ``SRTransferUnresolved`` from submit or wait
        leaves the existing buffers in place and blocks later fills or growth.
        """
        self._raise_if_unresolved()
        self.generation += 1
        self.rpd_input = None
        plan = plan_draft_rows(
            token_rows,
            parent_rows,
            index_rows,
            topk,
            spec_steps,
            num_draft_tokens,
        )
        if plan.bs == 0:
            return self._empty_views(plan, device)
        self.wait_for_host(metrics)
        self.ensure(plan.used, device, metrics)
        fill_start = time.perf_counter()
        verified, tokens, parents, indices = packet_segment_views(self.host_packet, plan)
        for i, value in enumerate(verified_ids):
            verified[i] = int(value)
        write_draft_regions(tokens, parents, indices, plan)
        used = plan.used
        if rpd_vocab is not None:
            from sglang.srt.speculative.standalone_remote.sr_rpd import (
                build_sr_rpd_input,
            )

            context = build_sr_rpd_input(
                verified,
                tokens,
                parents,
                indices,
                topk=topk,
                steps=spec_steps,
                width=num_draft_tokens,
                vocab=rpd_vocab,
                owner=self,
                generation=self.generation,
                batch_key=rpd_batch_key,
            )
            if context is not None:
                used += int(context.edge_index_cpu.numel())
                # Growing storage replaces it. Preserve only this round's
                # initialized input region, then append the edge indices.
                if used > self.capacity:
                    saved = self.host_packet[: plan.used].clone()
                    self.ensure(used, device, metrics)
                    self.host_packet[: plan.used].copy_(saved)
                    verified, tokens, parents, indices = packet_segment_views(
                        self.host_packet, plan
                    )
                self.host_packet[plan.used : used].copy_(
                    context.edge_index_cpu.reshape(-1)
                )
                self.rpd_input = context
                self._note(
                    metrics,
                    "counts",
                    "rpd_input_edge_bytes",
                    context.edge_index_cpu.numel() * 8,
                )
        self._note(metrics, "host", "verify_packet_fill", time.perf_counter() - fill_start)
        if not self.on_accelerator(device):
            if self.rpd_input is not None:
                self.rpd_input.edge_index = self.host_packet[plan.used : used].view(
                    2, -1
                )
            return verified, parents, indices, tokens
        host = self.host_packet
        device_buf = self.device_packet
        try:
            event = submit_copy(device_buf[:used], host[:used])
        except SRTransferUnresolved:
            self.unresolved = True
            raise
        self.event = event
        self._note(metrics, "counts", "verify_packet_upload", 1)
        if self.rpd_input is not None:
            self.rpd_input.edge_index = device_buf[plan.used : used].view(2, -1)
        verified, tokens, parents, indices = packet_segment_views(device_buf, plan)
        return verified, parents, indices, tokens


def assemble_draft_rows(
    token_rows: Sequence,
    parent_rows: Sequence[Optional[Sequence[int]]],
    index_rows: Sequence[Optional[Sequence[int]]],
    topk: int,
    spec_steps: int,
    num_draft_tokens: int,
    device=None,
    out_tokens: Optional[torch.Tensor] = None,
    out_parents: Optional[torch.Tensor] = None,
    out_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (parent_list, top_scores_index, draft_tokens). Never fakes a bush."""
    dev = device if device is not None else torch.device("cpu")
    plan = plan_draft_rows(
        token_rows,
        parent_rows,
        index_rows,
        topk,
        spec_steps,
        num_draft_tokens,
    )
    draft_cpu = torch.empty((plan.bs, plan.token_width), dtype=torch.int64)
    parents_cpu = torch.empty((plan.bs, plan.parent_w), dtype=torch.int64)
    indices_cpu = torch.empty((plan.bs, plan.index_w), dtype=torch.int64)
    write_draft_regions(draft_cpu, parents_cpu, indices_cpu, plan)
    return (
        _export_assembled_rows(parents_cpu, out_parents, dev),
        _export_assembled_rows(indices_cpu, out_indices, dev),
        _export_assembled_rows(draft_cpu, out_tokens, dev),
    )
