"""Worker-owned references to KV whose consumers or release are uncertain."""

from dataclasses import dataclass, field

from sglang.srt.speculative.standalone_remote.sr_transfer_staging import (
    SRTransferUnresolved,
)


@dataclass(eq=False)
class IsolatedKV:
    owner: object
    units: list
    event: object = None
    states: list = field(default_factory=list)
    # Set by the single completion query. Reclaim must not query again.
    completion_done: object = None
    # First allocator.free failure. A later poll must not retry or free siblings.
    release_error: object = None
    alloc_generation: object = None
    row_identity: object = None
    # Sealed pages are reclaimed from the saved ids. The live pool row may be gone.
    sealed: bool = False
    req: object = None
    # A deferred FINISH whose session, generation, or pool row no longer matches.
    stale: bool = False

    def __post_init__(self):
        self.states = ["unsubmitted"] * len(self.units)


class KVQuarantine:
    """No event means unknown completion, never permission to reclaim.

    Records survive session rotation and worker close. Only confirmed release
    or an explicitly confirmed full allocator reset can drop these references.
    Process teardown finally releases records from a poisoned device context.
    """

    def __init__(self):
        self.records = []
        self.generation = 0

    def retain(self, owner, units=(), event=None):
        for record in self.records:
            if record.owner is owner:
                return record
        record = IsolatedKV(owner, list(units), event)
        self.records.append(record)
        return record

    def reclaim(self, allocator):
        reclaimed = 0
        for record in list(self.records):
            if not record_may_release(record, allocator):
                continue
            # A tree abandon records one query result. A second query must
            # not decide release on its own.
            checked = getattr(record, "completion_done", None)
            if checked is None:
                if record.event is None:
                    continue
                try:
                    done = record.event.query()
                except BaseException as exc:
                    record.event = None
                    raise SRTransferUnresolved(
                        "quarantined KV completion unknown"
                    ) from exc
                if not done:
                    continue
            elif not checked:
                continue
            # Tree pages stay until the request map has been reconciled.
            # That pass restores only slots still holding the abandoned ids.
            owner = record.owner
            if (
                isinstance(owner, dict)
                and "mapping_snaps" in owner
                and not owner.get("mapping_restored")
            ):
                continue
            # A consumer event never authorizes releasing a request-pool row.
            if not record.units:
                continue
            for i, unit in enumerate(record.units):
                if record.states[i] != "unsubmitted":
                    continue
                record.states[i] = "unknown"  # Mark BEFORE allocator submission.
                try:
                    allocator.free(unit)
                except BaseException as exc:
                    if record.release_error is None:
                        record.release_error = exc
                    raise SRTransferUnresolved("quarantined KV release failed") from exc
                record.states[i] = "done"
                record.units[i] = None
            if all(state == "done" for state in record.states):
                self.records.remove(record)
                reclaimed += 1
        return reclaimed

    def after_allocator_reset(self, *, completion_confirmed):
        if not completion_confirmed:
            raise SRTransferUnresolved("cannot discard unresolved KV at reset")
        # Do not free old slots: the allocator now owns every page again.
        self.records.clear()
        self.generation += 1


def record_may_release(record, allocator) -> bool:
    """False when a free already failed, or the page ids name a new lifetime.

    ``unknown`` is sticky for the whole record: later polls neither retry that
    unit nor submit the record's other frees. A generation or request-row
    mismatch means those ids may already belong to a later allocation.
    """
    if getattr(record, "stale", False):
        return False
    if getattr(record, "release_error", None) is not None:
        return False
    if any(state == "unknown" for state in record.states):
        return False
    owner = record.owner
    generation = getattr(record, "alloc_generation", None)
    if generation is None and isinstance(owner, dict):
        generation = owner.get("alloc_generation")
    current = getattr(allocator, "generation", None) if allocator is not None else None
    if generation is not None and current is not None and int(current) != int(generation):
        return False
    if getattr(record, "sealed", False):
        return True
    saved = getattr(record, "row_identity", None)
    if saved is None and isinstance(owner, dict):
        saved = owner.get("row_identity")
    if saved is not None and isinstance(owner, dict):
        reqs = owner.get("reqs") or []
        if len(saved) != len(reqs):
            return False
        for item, req in zip(saved, reqs):
            saved_id, saved_idx = item
            if id(req) != saved_id or getattr(req, "req_pool_idx", None) != saved_idx:
                return False
    return True


def release_unknown(record) -> bool:
    return getattr(record, "release_error", None) is not None or any(
        state == "unknown" for state in getattr(record, "states", ())
    )


def handoff_request_pages(bag, req, owned_pages, page_size) -> None:
    """Drop pages this request's FINISH release will free.

    The allocator frees a whole physical page from any of its slots. A page
    named by the released map therefore leaves the quarantine record without
    another ``free``. Pages the map does not name stay quarantined. A record
    whose release already failed is left unchanged.
    """
    if bag is None:
        return
    page_size = max(1, int(page_size or 1))
    owned = {int(page) for page in owned_pages}

    for record in list(bag.records):
        if release_unknown(record):
            continue
        owner = record.owner
        reqs = owner.get("reqs") if isinstance(owner, dict) else ()
        if req not in tuple(reqs or ()) and getattr(record, "req", None) is not req:
            continue
        units = []
        states = []
        for unit, state in zip(record.units, record.states):
            if unit is None or state == "done":
                continue
            ids = _flat_ids(unit)
            pages = {}
            for slot in ids:
                pages.setdefault(_page_of(slot, page_size), []).append(slot)
            keep = []
            for page, slots in pages.items():
                if page in owned:
                    continue
                keep.extend(slots)
            if not keep:
                continue
            if len(keep) == len(ids):
                units.append(unit)
                states.append(state)
            else:
                units.append(_take_ids(unit, keep))
                states.append(state)
        record.units = units
        record.states = states
        if not record.units:
            bag.records.remove(record)


def _flat_ids(unit):
    if unit is None:
        return []
    detach = getattr(unit, "detach", None)
    if callable(detach):
        return [int(x) for x in unit.detach().view(-1).tolist()]
    return [int(x) for x in unit]


def _take_ids(unit, ids):
    new_tensor = getattr(unit, "new_tensor", None)
    if callable(new_tensor):
        return unit.new_tensor(ids)
    return list(ids)


def _page_of(slot, page_size):
    slot = int(slot)
    return slot if page_size == 1 else slot // page_size


def record_belongs_to(record, req) -> bool:
    if getattr(record, "req", None) is req:
        return True
    owner = record.owner
    if owner is req:
        return True
    if isinstance(owner, dict) and req in (owner.get("reqs") or ()):
        return True
    if isinstance(owner, tuple) and owner and owner[0] is req:
        return True
    return False


def iter_request_records(owner, req):
    drafter = getattr(owner, "sr_tree_drafter", None)
    seen = set()
    for holder in (owner, drafter):
        if holder is None or id(holder) in seen:
            continue
        seen.add(id(holder))
        bag = getattr(holder, "_sr_kv_quarantine", None)
        if bag is None:
            continue
        for record in list(bag.records):
            if record_belongs_to(record, req):
                yield holder, bag, record


def consumer_unconfirmed(record) -> bool:
    """True when this record's consumer is not confirmed finished.

    Queries only the record's existing event. A missing event or a failed
    query stays unconfirmed and does not synchronize the device.
    """
    if release_unknown(record):
        return True
    event = record.event
    if event is None:
        return True
    try:
        done = bool(event.query())
    except Exception as exc:
        record.event = None
        record.completion_done = None
        if record.release_error is None:
            record.release_error = exc
        return True
    record.completion_done = done
    return not done


def request_consumer_unconfirmed(owner, req) -> bool:
    return any(
        consumer_unconfirmed(record)
        for _holder, _bag, record in iter_request_records(owner, req)
    )


def defer_finish(owner, req) -> None:
    found = False
    for _holder, _bag, record in iter_request_records(owner, req):
        record.req = req
        found = True
    if found:
        return
    bag = quarantine(owner)
    hold = bag.retain(("sr-finish-deferred", req), (), None)
    hold.req = req


def note_deferred_finish(owner, rid, req, state, release_mm, generation) -> None:
    """Remember one FINISH until its consumer event can be queried again.

    The entry lives on the worker, not in ``sr_state``. A repeated FINISH or
    ABORT in the same session only updates ``release_mm``. It does not replace
    the request, generation, row, or quarantine records captured at defer time.
    """
    queue = getattr(owner, "_sr_deferred_finish", None)
    if not isinstance(queue, dict):
        queue = {}
        owner._sr_deferred_finish = queue
    current = queue.get(rid)
    if current is not None:
        current["release_mm"] = bool(current.get("release_mm")) or bool(release_mm)
        return
    session_id = getattr(state, "session_id", None)
    if session_id is None:
        session_id = getattr(getattr(owner, "sr_state", None), "session_id", None)
    pages, record_ids = _deferred_page_snapshot(owner, req)
    gen = int(generation or 0)
    queue[rid] = {
        "session_id": session_id,
        "rid": rid,
        "req": req,
        "state": state,
        "release_mm": bool(release_mm),
        "generation": gen,
        "allocator_generation": gen,
        "row_identity": (
            (id(req), getattr(req, "req_pool_idx", None)) if req is not None else None
        ),
        "record_ids": record_ids,
        "pages": pages,
        "stale": False,
    }


def _deferred_page_snapshot(owner, req):
    """Copy page ids already stored on this request's quarantine records."""
    if req is None:
        return [], []
    alloc = _finish_allocator(owner)
    page_size = max(1, int(getattr(alloc, "page_size", 1) or 1))
    pages = []
    record_ids = []
    seen = set()
    for _holder, _bag, record in iter_request_records(owner, req):
        record_ids.append(id(record))
        for unit in record.units:
            for slot in _flat_ids(unit):
                page = _page_of(slot, page_size)
                if page not in seen and page != 0:
                    seen.add(page)
                    pages.append(page)
    return pages, record_ids


def _finish_allocator(owner):
    alloc = getattr(owner, "token_to_kv_pool_allocator", None)
    if alloc is None:
        drafter = getattr(owner, "sr_tree_drafter", None)
        alloc = getattr(drafter, "token_to_kv_pool_allocator", None)
    return alloc


def _live_record_ids(owner):
    found = set()
    drafter = getattr(owner, "sr_tree_drafter", None)
    for holder in (owner, drafter):
        bag = getattr(holder, "_sr_kv_quarantine", None)
        if bag is None:
            continue
        for record in bag.records:
            found.add(id(record))
    return found


def _row_taken_by_other(owner, req, saved_idx) -> bool:
    if saved_idx is None:
        return False
    manager = getattr(owner, "sr_state", None)
    active = getattr(manager, "active", None)
    candidates = []
    if isinstance(active, dict):
        candidates.extend(active.values())
    for name in ("draft_paused_reqs", "waiting_queue"):
        items = getattr(owner, name, None)
        if isinstance(items, list):
            candidates.extend(items)
    running = getattr(owner, "running_batch", None)
    reqs = getattr(running, "reqs", None)
    if isinstance(reqs, list):
        candidates.extend(reqs)
    for item in candidates:
        other = getattr(item, "req_object", item)
        if other is req or other is None:
            continue
        if getattr(other, "req_pool_idx", None) == saved_idx:
            return True
    return False


def deferred_release_allowed(owner, entry) -> bool:
    """False when a deferred FINISH would name a newer pool lifetime."""
    if not isinstance(entry, dict) or entry.get("stale"):
        return False
    req = entry.get("req")
    if req is None:
        return False
    alloc = _finish_allocator(owner)
    current = getattr(alloc, "generation", None) if alloc is not None else None
    saved_gen = entry.get("allocator_generation", entry.get("generation"))
    if saved_gen is not None and current is not None and int(current) != int(saved_gen):
        return False
    session = entry.get("session_id")
    current_session = getattr(getattr(owner, "sr_state", None), "session_id", None)
    if session is not None and current_session is not None and session != current_session:
        return False
    saved = entry.get("row_identity")
    if saved is not None:
        saved_id, saved_idx = saved
        if id(req) != saved_id or getattr(req, "req_pool_idx", None) != saved_idx:
            return False
        if _row_taken_by_other(owner, req, saved_idx):
            return False
    record_ids = entry.get("record_ids") or []
    if record_ids:
        live = _live_record_ids(owner)
        if any(record_id not in live for record_id in record_ids):
            return False
    return True


def quarantine_stale_deferred(owner, entry) -> None:
    """Keep the old FINISH without reading or freeing the live pool row."""
    if not isinstance(entry, dict):
        return
    entry["stale"] = True
    req = entry.get("req")
    key = entry.get("stale_owner")
    if key is None:
        key = ("sr-deferred-stale", req if req is not None else entry.get("rid"))
        entry["stale_owner"] = key
    pages = [int(page) for page in (entry.get("pages") or [])]
    record = quarantine(owner).retain(key, [pages] if pages else [], None)
    record.stale = True
    record.req = req
    record.alloc_generation = entry.get("allocator_generation", entry.get("generation"))
    record.row_identity = entry.get("row_identity")


def freeze_deferred_finishes(owner) -> None:
    """Convert queued FINISH entries before an allocator reset touches pools."""
    queue = getattr(owner, "_sr_deferred_finish", None)
    if not isinstance(queue, dict):
        return
    for entry in list(queue.values()):
        quarantine_stale_deferred(owner, entry)


def drop_confirmed_deferred_finishes(owner) -> None:
    """Drop deferred entries only after a confirmed allocator reset."""
    queue = getattr(owner, "_sr_deferred_finish", None)
    if isinstance(queue, dict):
        queue.clear()


def pop_deferred_finish(owner, rid):
    queue = getattr(owner, "_sr_deferred_finish", None)
    if not isinstance(queue, dict):
        return None
    return queue.pop(rid, None)


def finish_has_device_hold(owner, req) -> bool:
    """True when FINISH still has a pool row, lease, page, or failed release."""
    if getattr(req, "req_pool_idx", None) is not None:
        return True
    rid = getattr(req, "rid", None)
    store = getattr(owner, "sr_tree_leases", None)
    leases = getattr(store, "_leases", None)
    if isinstance(leases, dict) and rid in leases:
        return True
    for _holder, _bag, record in iter_request_records(owner, req):
        if release_unknown(record) or record.event is None:
            return True
        for unit, state in zip(record.units, record.states):
            if unit is not None and state != "done":
                return True
    return False


def _saved_ids_for_req(owner, req):
    if not isinstance(owner, dict):
        return None
    reqs = owner.get("reqs") or []
    slots = owner.get("page_slots") or []
    if req not in reqs:
        return None
    if len(slots) == len(reqs):
        return set(_flat_ids(slots[reqs.index(req)]))
    if len(reqs) == 1:
        ids = []
        for slot in slots:
            ids.extend(_flat_ids(slot))
        return set(ids)
    return None


def _pop_index(owner, key, index):
    values = owner.get(key)
    if isinstance(values, list) and len(values) > index:
        values.pop(index)


def _isolate_request_record(bag, record, req):
    """Give this request its own record when a batch lease holds several."""
    owner = record.owner
    if not isinstance(owner, dict):
        record.req = req
        return record
    reqs = list(owner.get("reqs") or [])
    if len(reqs) <= 1:
        record.req = req
        return record
    mine = _saved_ids_for_req(owner, req)
    if not mine:
        record.req = req
        return record
    index = reqs.index(req)
    my_units, my_states = [], []
    other_units, other_states = [], []
    for unit, state in zip(record.units, record.states):
        ids = _flat_ids(unit)
        if ids and set(ids) <= mine:
            my_units.append(unit)
            my_states.append(state)
        elif ids and (set(ids) & mine):
            keep = [slot for slot in ids if slot in mine]
            rest = [slot for slot in ids if slot not in mine]
            if keep:
                my_units.append(_take_ids(unit, keep))
                my_states.append(state)
            if rest:
                other_units.append(_take_ids(unit, rest))
                other_states.append(state)
        else:
            other_units.append(unit)
            other_states.append(state)
    child_owner = {"reqs": [req]}
    for key in ("starts", "mapping_snaps", "spec_snaps", "page_slots", "row_identity"):
        values = owner.get(key)
        if isinstance(values, list) and len(values) > index:
            child_owner[key] = [values[index]]
    if "spec_snaps" not in owner:
        child_owner.pop("spec_snaps", None)
    child_owner["mapping_restored"] = owner.get("mapping_restored")
    child_owner["alloc_generation"] = owner.get("alloc_generation")
    child = IsolatedKV(child_owner, my_units, record.event)
    child.states = list(my_states) if my_states else child.states
    child.completion_done = record.completion_done
    child.alloc_generation = record.alloc_generation
    child.row_identity = child_owner.get("row_identity")
    child.req = req
    bag.records.append(child)
    for key in ("starts", "mapping_snaps", "spec_snaps", "page_slots", "row_identity"):
        _pop_index(owner, key, index)
    owner["reqs"] = [item for item in reqs if item is not req]
    record.units = other_units
    record.states = other_states
    if not record.units and not owner.get("reqs"):
        bag.records.remove(record)
    return child


def _seal_speculative(bag, record, req, owned_pages, page_size, generation):
    owned = {int(page) for page in owned_pages}
    spec_units = []
    owned_units = []
    owned_states = []
    for unit, state in zip(list(record.units), list(record.states)):
        if unit is None or state == "done":
            continue
        ids = _flat_ids(unit)
        pages = {}
        for slot in ids:
            pages.setdefault(_page_of(slot, page_size), []).append(slot)
        spec, keep = [], []
        for page, slots in pages.items():
            if page in owned or page == 0:
                keep.extend(slots)
            else:
                spec.extend(slots)
        if spec:
            spec_units.append(unit if len(spec) == len(ids) else _take_ids(unit, spec))
        if keep:
            owned_units.append(unit if len(keep) == len(ids) else _take_ids(unit, keep))
            owned_states.append(state)
    record.units = owned_units
    record.states = owned_states
    record.req = req
    if not spec_units:
        return None
    sealed = IsolatedKV({"sealed": True}, spec_units, record.event)
    sealed.completion_done = True
    sealed.sealed = True
    sealed.req = req
    sealed.alloc_generation = (
        generation if generation is not None else record.alloc_generation
    )
    bag.records.append(sealed)
    return sealed


def prepare_finish_pages(owner, req, owned_pages, page_size, generation):
    """Split this request's pages without dropping the ones FINISH will free."""
    key = getattr(req, "_sr_finish_owner", None)
    if key is None:
        key = ("sr-finish", req)
        req._sr_finish_owner = key
    hold = quarantine(owner).retain(key, (), None)
    if release_unknown(hold):
        raise SRTransferUnresolved("finish release already failed") from getattr(
            hold, "release_error", None
        )
    hold.req = req
    hold.alloc_generation = generation
    hold.owned_pages = {int(page) for page in owned_pages}
    for _holder, bag, record in list(iter_request_records(owner, req)):
        if record is hold or getattr(record, "sealed", False):
            continue
        if release_unknown(record):
            raise SRTransferUnresolved("finish release already failed") from getattr(
                record, "release_error", None
            )
        isolated = _isolate_request_record(bag, record, req)
        _seal_speculative(bag, isolated, req, owned_pages, page_size, generation)
    return hold


def commit_finish_pages(owner, req, owned_pages, page_size) -> None:
    seen = set()
    for _holder, bag, record in list(iter_request_records(owner, req)):
        if id(record) in seen or getattr(record, "sealed", False):
            continue
        seen.add(id(record))
        handoff_request_pages(bag, req, owned_pages, page_size)
    key = getattr(req, "_sr_finish_owner", None)
    bag = getattr(owner, "_sr_kv_quarantine", None)
    if bag is None or key is None:
        return
    for record in list(bag.records):
        if record.owner is key and not record.units:
            bag.records.remove(record)


def fail_finish_release(owner, req, exc: BaseException) -> None:
    from sglang.srt.speculative.standalone_remote.sr_align import (
        is_device_context_error,
    )

    for _holder, _bag, record in list(iter_request_records(owner, req)):
        if getattr(record, "sealed", False):
            continue
        if record.release_error is None:
            record.release_error = exc
        if record.states:
            record.states = [
                "unknown" if state in ("unsubmitted", "held") else state
                for state in record.states
            ]
        else:
            record.states = ["unknown"]
    if isinstance(exc, SRTransferUnresolved) or is_device_context_error(exc):
        raise exc
    wrapped = SRTransferUnresolved("finish KV release failed")
    wrapped.req = req
    raise wrapped from exc


def reconcile_request_mapping(owner, req) -> None:
    """Restore an abandoned window while this request still owns its row."""
    drafter = getattr(owner, "sr_tree_drafter", None)
    restore = getattr(drafter, "_restore_abandoned_mapping", None)
    if not callable(restore):
        return
    alloc = getattr(owner, "token_to_kv_pool_allocator", None)
    if alloc is None and drafter is not None:
        alloc = getattr(drafter, "token_to_kv_pool_allocator", None)
    for _holder, _bag, record in list(iter_request_records(owner, req)):
        lease = record.owner
        if not isinstance(lease, dict) or "mapping_snaps" not in lease:
            continue
        if lease.get("mapping_restored") or "spec_snaps" not in lease:
            continue
        if not record_may_release(record, alloc):
            continue
        restore(lease)
        lease["mapping_restored"] = True


def quarantine(owner):
    value = getattr(owner, "_sr_kv_quarantine", None)
    if value is None:
        value = owner._sr_kv_quarantine = KVQuarantine()
    return value


class HostCompletion:
    """CPU work is finished before abandon returns. ``query`` is always done."""

    def query(self):
        return True


def _staging_unresolved(owner) -> bool:
    return any(
        slot.get("unresolved", False)
        for slot in getattr(owner, "_sr_int_staging", {}).values()
    )


def has_unresolved_resources(owner, *, include_quarantine=True):
    """Host-only check used before FINISH/session cleanup touches a device.

    Queryable consumer events are reclaimed first. A record that is only
    waiting on a live event does not block other requests. ``event is None``
    still forbids returning those pages. A poisoned device stays unresolved
    and this check does not rewrite the request map or free pages; a dead
    context must not recycle those slots.

    The draft worker's quarantine holds abandoned tree pages whose request
    map must be restored before those pages return to the allocator. That bag
    is reclaimed through the drafter, not by freeing the slots directly.

    ``include_quarantine=False`` is the per-request finish check. Records
    already retained on this worker belong to other requests; they must not
    stop the request being finished from releasing its own KV. A release
    that already failed stays unresolved for every cleanup entry, including
    that per-request check. Session reset keeps the default and still refuses
    an allocator reset while any record is unresolved.
    """
    if getattr(owner, "_sr_device_poisoned", False):
        return True
    if _staging_unresolved(owner):
        return True
    store = getattr(owner, "sr_tree_leases", None)
    leases = getattr(store, "_leases", {})
    if isinstance(leases, dict) and any(
        lease.copy_unresolved for lease in leases.values()
    ):
        return True
    allocator = getattr(owner, "token_to_kv_pool_allocator", None)
    if any(
        lease.copy_unresolved
        for lease in getattr(allocator, "_sr_tree_lease_pending_frees", ())
    ):
        return True
    if not include_quarantine:
        return _release_unknown_on(owner)
    drafter = getattr(owner, "sr_tree_drafter", None)
    if allocator is None and drafter is not None:
        allocator = getattr(drafter, "token_to_kv_pool_allocator", None)
    seen = set()
    for holder in (owner, drafter):
        if holder is None or id(holder) in seen:
            continue
        seen.add(id(holder))
        bag = getattr(holder, "_sr_kv_quarantine", None)
        if bag is None or not bag.records:
            continue
        if allocator is not None and not getattr(owner, "_sr_device_poisoned", False):
            try:
                restore = getattr(holder, "_reclaim_abandoned_leases", None)
                if holder is drafter and callable(restore):
                    restore()
                else:
                    bag.reclaim(allocator)
            except SRTransferUnresolved:
                return True
        if any(
            record.event is None or release_unknown(record) for record in bag.records
        ):
            return True
    return False


def _release_unknown_on(owner) -> bool:
    drafter = getattr(owner, "sr_tree_drafter", None)
    seen = set()
    for holder in (owner, drafter):
        if holder is None or id(holder) in seen:
            continue
        seen.add(id(holder))
        bag = getattr(holder, "_sr_kv_quarantine", None)
        if bag is None:
            continue
        if any(release_unknown(record) for record in bag.records):
            return True
    return False
