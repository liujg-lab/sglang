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


def rows_with_draft_tree_version(reqs) -> List[int]:
    """Rows whose Draft can reuse a cached tree KV lease.

    ``sr_draft_tree_version is None`` means the Draft cannot reuse the tree,
    as with CUDA page size 1 or a request that fell back. Accepted candidate
    indices for those rows are unused.
    """
    return [
        i
        for i, req in enumerate(reqs)
        if getattr(req, "sr_draft_tree_version", None) is not None
    ]


def publish_sr_tree_paths(batch, rows) -> List[List[int]]:
    """Write accepted candidate paths for rows the Draft can reuse.

    Callers skip non-standalone-remote batches. A row whose
    ``sr_draft_tree_version`` is missing is cleared and omitted. Returns an
    empty list when nothing is reusable or export fails. On failure every
    request path is cleared and ``batch.sr_round_metrics`` records the error.
    """
    reqs = batch.reqs
    try:
        reusable = rows_with_draft_tree_version(reqs)
        if not reusable:
            for req in reqs:
                req.sr_accepted_tree_candidate_indices = None
            return []
        paths = export_accepted_tree_candidate_indices([rows[i] for i in reusable])
        stored: List[List[int]] = [[] for _ in reqs]
        reusable_set = set(reusable)
        for i, path in zip(reusable, paths):
            stored[i] = list(path)
            reqs[i].sr_accepted_tree_candidate_indices = stored[i]
        for i, req in enumerate(reqs):
            if i not in reusable_set:
                req.sr_accepted_tree_candidate_indices = None
        return stored
    except Exception as exc:
        for req in reqs:
            req.sr_accepted_tree_candidate_indices = None
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            note_sr_tree_path_export_error,
        )

        note_sr_tree_path_export_error(getattr(batch, "sr_round_metrics", None), exc)
        return []


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


def kernel_parent_width(topk: int, steps: int) -> int:
    """Column stride of ``build_tree_efficient``: ``topk * (depth - 1) + 1``."""
    if int(steps) <= 1:
        return 1
    return int(topk) * (int(steps) - 1) + 1


def organized_parent_len(topk: int, steps: int) -> int:
    """Parent-row length produced by ``organize_draft_results``.

    ``steps == 1`` drops the only parent tensor and yields an empty row. Longer
    trees keep every parent tensor except the last, which is ``topk * (steps - 1) + 1``.
    """
    if int(steps) <= 1:
        return 0
    return int(topk) * (int(steps) - 1) + 1


def draft_index_limit(topk: int, steps: int) -> int:
    """Candidate-pool size indexed by ``top_scores_index``."""
    topk = int(topk)
    steps = max(int(steps), 1)
    return topk + (steps - 1) * topk * topk


def _as_int_list(values) -> Optional[List[int]]:
    if values is None:
        return None
    if torch.is_tensor(values):
        flat = values.detach().reshape(-1)
        try:
            return [int(x) for x in flat.tolist()]
        except (TypeError, ValueError):
            return None
    try:
        return [int(x) for x in values]
    except (TypeError, ValueError):
        return None


def tree_row_valid(tokens, parents, indices, topk: int, steps: int, num_draft_tokens: int) -> bool:
    """Whether one supplied tree row matches the verify kernel's fixed stride.

    ``tokens``, ``parents``, and ``indices`` are already int sequences. An empty
    parent list is present topology and is checked here. A selected child whose
    parent candidate was not selected is a normal EAGLE row: the kernel skips
    that token, so the row stays valid even when that parent id is larger than
    the child. A parent token of ``-1`` at ``parent_tb > 0`` does not stop the
    mask thread and is invalid.

    A selected parent must have a candidate id strictly less than the child.
    Draft trees number candidates by step, so every real edge decreases the id
    and cannot form a cycle.
    """
    topk = int(topk)
    steps = int(steps)
    width = int(num_draft_tokens)
    if topk < 1 or steps < 1 or width < 1:
        return False
    if len(tokens) != width - 1 or len(indices) != width - 1:
        return False
    if len(parents) != organized_parent_len(topk, steps):
        return False
    pos_of: Dict[int, int] = {}
    for node_i, node in enumerate(indices):
        if node not in pos_of:
            pos_of[node] = node_i
    limit = draft_index_limit(topk, steps)
    for idx in indices:
        if idx < 0 or idx >= limit:
            return False
        parent_tb = idx // topk
        if parent_tb <= 0:
            continue
        if parent_tb >= len(parents):
            return False
        parent_token = parents[parent_tb]
        if parent_token == -1:
            return False
        if parent_token in pos_of and parent_token >= idx:
            return False
    return True


def repair_ungrounded_parents(
    parents, indices, topk: int
) -> Tuple[List[int], List[int], List[int]]:
    """Point missing parents at a grounded node so the mask thread can stop.

    A node is grounded when ``parent_tb == 0``, or its parent token is a
    grounded selected id. ``parents`` and ``indices`` are already int sequences.
    Selected parents have a smaller candidate id than their children, so one
    ascending scan marks the same grounded set as a fixed-point loop. Other
    nodes stay in the row, but their parent column is rewritten to the first
    grounded root-child id. The mask thread finds that id and stops on the next
    ``parent_tb == 0``. When nothing is grounded, every index becomes 0 and the
    mask thread does not read ``parent_list``.

    Returns ``(parents, indices, ungrounded draft slots)``. A rewritten parent
    row is a new list. The caller's sequences are not modified. The ungrounded
    slots must not be accepted.
    """
    parent_row = parents
    index_row = indices
    topk = int(topk)
    grounded = [idx // topk <= 0 for idx in index_row]
    pos_of: Dict[int, int] = {}
    for slot, node in enumerate(index_row):
        if node not in pos_of:
            pos_of[node] = slot
    for slot in sorted(range(len(index_row)), key=index_row.__getitem__):
        if grounded[slot]:
            continue
        idx = index_row[slot]
        parent_tb = idx // topk
        if parent_tb <= 0 or parent_tb >= len(parent_row):
            continue
        found = pos_of.get(parent_row[parent_tb])
        if found is not None and grounded[found]:
            grounded[slot] = True
    ungrounded = [slot for slot, flag in enumerate(grounded) if not flag]
    if not ungrounded:
        return parent_row, index_row, []
    if not any(grounded):
        return parent_row, [0] * len(index_row), ungrounded
    anchor = next(
        index_row[slot]
        for slot, flag in enumerate(grounded)
        if flag and index_row[slot] // topk <= 0
    )
    grounded_ids = {index_row[slot] for slot, flag in enumerate(grounded) if flag}
    rewritten = list(parent_row)
    for slot in ungrounded:
        parent_tb = index_row[slot] // topk
        if parent_tb <= 0 or parent_tb >= len(rewritten):
            continue
        if rewritten[parent_tb] not in grounded_ids:
            rewritten[parent_tb] = anchor
    return rewritten, index_row, ungrounded


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
        parsed_parents: Sequence,
        parsed_indices: Sequence,
        invalid_tree_rows: int = 0,
        no_draft: Optional[Sequence[bool]] = None,
        disconnected: Optional[Sequence[Sequence[int]]] = None,
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
        self.invalid_tree_rows = int(invalid_tree_rows)
        if no_draft is None:
            self.no_draft = [False] * self.bs
        else:
            self.no_draft = [bool(flag) for flag in no_draft]
        if disconnected is None:
            self.disconnected = [[] for _ in range(self.bs)]
        else:
            self.disconnected = [list(slots) for slots in disconnected]

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

    Widths follow the verify kernel, not the longest supplied row:
    ``token_width == index_w == max(W - 1, 0)`` and
    ``parent_w == topk * (steps - 1) + 1`` (1 when ``steps == 1``).

    ``topk <= 1`` ignores supplied topology and uses the chain template.
    ``topk > 1`` keeps a valid tree. A child whose parent was not selected stays
    in the row; ``repair_ungrounded_parents`` rewrites that parent column so the
    mask thread can stop, and records the draft slot in ``disconnected``.
    A missing topology or a row that fails ``tree_row_valid`` is ``no_draft``:
    parents stay ``-1`` and indices stay ``0``, with no chain template. Only a
    failed row clears tokens and increments ``invalid_tree_rows``.
    """
    bs = len(token_rows)
    width = int(num_draft_tokens)
    topk_i = int(topk)
    steps_i = int(spec_steps)
    token_width = max(width - 1, 0)
    index_w = token_width
    parent_w = kernel_parent_width(topk_i, steps_i)
    chain_parents, chain_indices = cached_chain_template(num_draft_tokens, spec_steps)
    parsed_parents: List[Optional[Sequence[int]]] = [None] * bs
    parsed_indices: List[Optional[Sequence[int]]] = [None] * bs
    token_rows_out: List = list(token_rows)
    no_draft = [False] * bs
    disconnected: List[List[int]] = [[] for _ in range(bs)]
    cleared = (0,) * token_width
    invalid = 0
    if topk_i > 1:
        for i in range(bs):
            parent = parent_rows[i] if i < len(parent_rows) else None
            index = index_rows[i] if i < len(index_rows) else None
            if parent is None or index is None:
                no_draft[i] = True
                continue
            tokens = token_rows[i] if i < len(token_rows) else None
            token_row = _as_int_list(tokens)
            parent_row = _as_int_list(parent)
            index_row = _as_int_list(index)
            if (
                token_row is None
                or parent_row is None
                or index_row is None
                or not tree_row_valid(
                    token_row, parent_row, index_row, topk_i, steps_i, width
                )
            ):
                invalid += 1
                token_rows_out[i] = cleared
                no_draft[i] = True
                continue
            repaired_parents, repaired_indices, slots = repair_ungrounded_parents(
                parent_row, index_row, topk_i
            )
            parsed_parents[i] = repaired_parents
            parsed_indices[i] = repaired_indices
            disconnected[i] = slots
    return DraftPacketPlan(
        bs,
        token_width,
        parent_w,
        index_w,
        topk_i <= 1,
        chain_parents,
        chain_indices,
        token_rows_out,
        parsed_parents,
        parsed_indices,
        invalid,
        no_draft,
        disconnected,
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


def _assign_ints(dest: torch.Tensor, values) -> None:
    """Write a host integer sequence into ``dest`` without a fresh tensor."""
    if torch.is_tensor(values):
        flat = values.detach().reshape(-1)
        n = min(int(flat.numel()), int(dest.numel()))
        if n:
            dest[:n].copy_(flat[:n].to(dtype=torch.int64, device=dest.device))
        return
    n = min(len(values), int(dest.numel()))
    if n <= 0:
        return
    chunk = dest[:n]
    seq = values[:n]
    try:
        chunk.numpy()[:] = seq
    except Exception:
        chunk.copy_(torch.as_tensor(seq, dtype=torch.int64))


def _numpy_prefix(row, values) -> None:
    if values is None:
        return
    if torch.is_tensor(values):
        flat = values.detach().reshape(-1).cpu().numpy()
        n = min(int(flat.shape[0]), int(row.shape[0]))
        if n:
            row[:n] = flat[:n]
        return
    n = min(len(values), int(row.shape[0]))
    if n:
        row[:n] = values[:n]


def _broadcast_cpu_prefix(dest, vector: torch.Tensor) -> None:
    """Write a read-only 1D CPU template into every row prefix."""
    if dest.size == 0 or vector.numel() == 0:
        return
    flat = vector.detach().reshape(-1).numpy()
    n = min(int(flat.shape[0]), int(dest.shape[1]))
    if n:
        dest[:, :n] = flat[:n]


def write_draft_regions(
    tokens: torch.Tensor,
    parents: torch.Tensor,
    indices: torch.Tensor,
    plan: DraftPacketPlan,
) -> None:
    """Initialize this round's active region, then copy each row.

    The three regions are contiguous CPU tensors. Writes go through their
    ``numpy`` views, so a reused host packet is cleared before prefixes are
    filled. Tokens, parents, and indices start as ``0 / -1 / 0``. Chain
    templates are only read and broadcast into every row. Present rows stay
    host sequences; ``plan_draft_rows`` does not build a tensor per row.
    ``no_draft`` rows keep parent ``-1`` and index ``0``.
    """
    if plan.bs == 0:
        return
    token_view = tokens.numpy()
    parent_view = parents.numpy()
    index_view = indices.numpy()
    if token_view.size:
        token_view.fill(0)
        for i in range(plan.bs):
            src = plan.token_rows[i] if i < len(plan.token_rows) else None
            _numpy_prefix(token_view[i], src)
    if plan.chain_only:
        if parent_view.size:
            parent_view.fill(-1)
            _broadcast_cpu_prefix(parent_view, plan.chain_parents)
        if index_view.size:
            index_view.fill(0)
            _broadcast_cpu_prefix(index_view, plan.chain_indices)
        return
    if parent_view.size:
        parent_view.fill(-1)
    if index_view.size:
        index_view.fill(0)
    for i, (parent, index) in enumerate(zip(plan.parsed_parents, plan.parsed_indices)):
        if parent is None or index is None:
            continue
        _numpy_prefix(parent_view[i], parent)
        _numpy_prefix(index_view[i], index)


def _skip_removed_link(node: int, removed: set, old_sib: Sequence[int]) -> int:
    """Walk the pre-rewrite sibling chain past nodes in ``removed``."""
    seen = set()
    while node in removed:
        if node < 0 or node in seen or node >= len(old_sib):
            return -1
        seen.add(node)
        node = int(old_sib[node])
    return node


def _index_copy_rows(
    dest: torch.Tensor, rows_t: torch.Tensor, values: Sequence[Sequence[int]]
) -> None:
    updated = torch.tensor(
        [list(row) for row in values], dtype=dest.dtype, device=dest.device
    )
    dest.index_copy_(0, rows_t, updated)


def unlink_disconnected_retrieve_nodes(
    retrive_next_token: Optional[torch.Tensor],
    retrive_next_sibling: Optional[torch.Tensor],
    disconnected: Optional[Sequence[Sequence[int]]],
) -> None:
    """Drop ungrounded draft nodes from the retrieve child and sibling lists.

    ``disconnected[row]`` holds draft slots. Tree position ``slot + 1`` is the
    node the verify kernel linked after the parent column was rewritten. Splicing
    it out leaves grounded nodes reachable and the ungrounded node unreachable.

    Affected rows are gathered with one ``index_select`` and one host copy,
    rewritten together, then written back with one ``index_copy_``.
    """
    if not disconnected or retrive_next_token is None or retrive_next_token.numel() == 0:
        return
    limit = int(retrive_next_token.shape[0])
    affected = [
        row
        for row, slots in enumerate(disconnected)
        if slots and row < limit
    ]
    if not affected:
        return

    rows_t = torch.tensor(affected, dtype=torch.long, device=retrive_next_token.device)
    next_host = retrive_next_token.index_select(0, rows_t).cpu().tolist()

    sib_limit = 0
    if retrive_next_sibling is not None and retrive_next_sibling.numel() != 0:
        sib_limit = int(retrive_next_sibling.shape[0])
    with_sib = [row for row in affected if row < sib_limit]
    sib_by_row: Dict[int, List[int]] = {}
    sib_rows_t = None
    if with_sib:
        if (
            len(with_sib) == len(affected)
            and retrive_next_sibling.device == retrive_next_token.device
        ):
            sib_rows_t = rows_t
        else:
            sib_rows_t = torch.tensor(
                with_sib, dtype=torch.long, device=retrive_next_sibling.device
            )
        for row, old in zip(
            with_sib, retrive_next_sibling.index_select(0, sib_rows_t).cpu().tolist()
        ):
            sib_by_row[row] = old

    new_next: List[List[int]] = []
    new_sib: List[List[int]] = []
    for local_i, row in enumerate(affected):
        removed = {int(slot) + 1 for slot in disconnected[row]}
        old_next = [int(node) for node in next_host[local_i]]
        if row in sib_by_row:
            old_sib = [int(node) for node in sib_by_row[row]]
        else:
            old_sib = [-1] * len(old_next)
        new_next.append(
            [_skip_removed_link(node, removed, old_sib) for node in old_next]
        )
        if row in sib_by_row:
            new_sib.append(
                [_skip_removed_link(node, removed, old_sib) for node in old_sib]
            )

    _index_copy_rows(retrive_next_token, rows_t, new_next)
    if with_sib:
        _index_copy_rows(retrive_next_sibling, sib_rows_t, new_sib)


def clear_no_draft_retrieve_links(
    retrive_next_token: Optional[torch.Tensor],
    retrive_next_sibling: Optional[torch.Tensor],
    no_draft: Optional[Sequence[bool]],
) -> None:
    """Drop draft children on rows that must not accept speculative tokens.

    The tree kernel still links index 0 to the root. Clearing both retrieve
    rows leaves only the root bonus for those requests.
    """
    if not no_draft:
        return
    rows = [i for i, flag in enumerate(no_draft) if flag]
    if not rows:
        return
    cached_key = None
    cached_rows_t = None
    for tensor in (retrive_next_token, retrive_next_sibling):
        if tensor is None or tensor.numel() == 0:
            continue
        limit = int(tensor.shape[0])
        kept = [row for row in rows if row < limit]
        if not kept:
            continue
        key = (str(tensor.device), tuple(kept))
        if cached_rows_t is None or key != cached_key:
            cached_rows_t = torch.tensor(kept, dtype=torch.long, device=tensor.device)
            cached_key = key
        tensor.index_fill_(0, cached_rows_t, -1)


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
        self.no_draft_rows: Tuple[bool, ...] = ()
        self.disconnected_slots: Tuple[Tuple[int, ...], ...] = ()

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
        self.no_draft_rows = tuple(plan.no_draft)
        self.disconnected_slots = tuple(tuple(slots) for slots in plan.disconnected)
        if plan.invalid_tree_rows:
            self._note(metrics, "counts", "sr_invalid_tree_rows", plan.invalid_tree_rows)
        if plan.bs == 0:
            return self._empty_views(plan, device)
        self.wait_for_host(metrics)
        self.ensure(plan.used, device, metrics)
        fill_start = time.perf_counter()
        verified, tokens, parents, indices = packet_segment_views(self.host_packet, plan)
        _assign_ints(verified, verified_ids)
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
                no_draft=plan.no_draft,
                disconnected=plan.disconnected,
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
