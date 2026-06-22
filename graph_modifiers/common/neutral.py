"""Neutral (backend-agnostic) schedule format.

Schedulers emit ``NeutralSchedule`` — a pure list of prefetch / evict /
cold-start decisions indexed by ``(graph_id, launch_id)``. Nothing in
this format is pytorch-specific; the same JSON can be:

  * fed into cg-sim's simulator / validator (which understands per-graph
    launch ids natively),
  * converted to the PyTorch ``jit_sim_prune_schedule.json`` schema for
    runtime consumption via :func:`neutral_to_pytorch`.

JSON layout (format_version = "1.0"):

    {
      "format_version": "1.0",
      "graph_order": [0, 1, 2, 3],
      "compilation_hashes": {"0": "abc...", "1": "def...", ...},
      "tensors": [
        {"uid": 0, "graph_id": 2, "compiled_tensor_id": 5,
         "graph_input_name": "primals_1", "size_bytes": 4096,
         "dtype": "torch.float16",
         "used_by_launch_ids": [10, 20, 500]},
        ...
      ],
      "prefetches": [
        {"tensor_uid": 0, "issue_launch_id": 0, "wait_launch_id": 10,
         "transfer_start_ns": 0, "transfer_end_ns": 40000,
         "reason": "belady_h2d"},
        ...
      ],
      "evicts": [
        {"tensor_uid": 0, "issue_launch_id": 500,
         "transfer_start_ns": 502000, "transfer_end_ns": 540000,
         "reason": "belady_tail_evict"},
        ...
      ],
      "cold_starts": [
        {"tensor_uid": 42, "anchor_launch_id": 0, "reason": "locked"},
        ...
      ],
      "meta": {"io_model": "ct_milp_zerostall", "summary": {...}}
    }

The prefetch ``issue_launch_id`` semantics:

  * ``issue_launch_id == wait_launch_id`` -> synchronous H2D (load
    immediately before the consumer kernel).
  * ``issue_launch_id <  wait_launch_id`` -> async start at the earlier
    launch, consumer waits at the later launch.
  * ``issue_launch_id == -1``              -> sync at ``wait_launch_id``
    (same as equal, but with an explicit sentinel).

Evict ``issue_launch_id`` is always "after this launch completes".
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FORMAT_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class NeutralTensor:
    uid: int
    graph_id: int
    compiled_tensor_id: int
    graph_input_name: str
    size_bytes: int
    dtype: str
    used_by_launch_ids: list[int] = field(default_factory=list)
    # Carried for the injector's per-node tid resolver. Empty list/None
    # for legacy schedules; modern schedulers populate them so the
    # position-based shape-twin tiebreaker (``_resolve_tid_for_node``)
    # works for synthetic entries that share a shape with the real
    # compile-side tensor.
    shape: list[Any] = field(default_factory=list)
    graph_input_idx: int | None = None
    storage_group_id: Any = None
    # Trace-runtime tids that this tensor backs (one or more cgsim_tids
    # that share the same storage_group_id at runtime). Populated by
    # ``build_neutral_schedule_from_timeline`` from
    # ``GlobalTensor.trace_tids`` so ``resolve_neutral_cgsim_tids`` can
    # pre-populate cgsim_tids without invoking the per-node
    # shape-disambiguation resolver. Empty list means the tensor's
    # trace tid was not known at timeline-build time and the resolver
    # has to fall back to per-node heuristics (the legacy path).
    trace_tids: list[int] = field(default_factory=list)


@dataclass
class NeutralPrefetch:
    tensor_uid: int
    issue_launch_id: int
    wait_launch_id: int
    transfer_start_ns: int = 0
    transfer_end_ns: int = 0
    reason: str = ""
    # Exact cg-sim node ids for repeated-launch timelines. PyTorch-facing
    # consumers can ignore these; cg-sim injectors use them to avoid mapping
    # every repeated ``(graph_id, launch_id)`` to the first occurrence.
    issue_node_id: int = -1
    wait_node_id: int = -1
    # Resolved cg-sim tensor id (the integer key into ``trace.tensor_map``)
    # for the *specific* runtime tensor instance this prefetch targets.
    # Populated at emit time by ``resolve_neutral_cgsim_tids`` so the
    # injector can skip its (gid, ctid) → cgsim_tid disambiguation
    # entirely. ``-1`` means "not resolved at emit time"; the injector
    # falls back to per-node resolution for legacy schedules.
    cgsim_tid: int = -1
    # Scheduler attests: the FIFO ordering of this async reload vs prior
    # evicts is proven safe (no race). The PyTorch wrapper's safety-net
    # will treat this as providing residency at ``wait_launch_id`` instead
    # of re-injecting a sync H2D. Only schedulers that verify ordering
    # (e.g., ``ct_milp_oracle`` via prefix-sum H2D deadlines) should set
    # this to True.
    trusted_async: bool = False
    # Cross-graph async: the graph in whose wrapper the async start is
    # issued. -1 = same-graph (issue in the tensor's own graph). Set to a
    # different graph_id to emit the async start from an EARLIER graph's
    # wrapper and the wait in the consumer graph's wrapper. Oracle uses
    # this when backward-ALAP required_start falls before the tensor's
    # own graph begins.
    issue_graph_id: int = -1
    # Cross-iter same-graph wrap: fire at end of iter N (issue_launch_id),
    # wait at start of iter N+1 (wait_launch_id). Used when the consumer
    # graph runs more than once per pipeline call (e.g., UNet in SDXL
    # diffusion); a single cross-graph pre-fire would only cover iter 1.
    # The wrapper handles this correctly because each iter's fire
    # populates h2d_events for the next iter's wait.
    cross_iter: bool = False
    # Per-iter mask for partial schedules. Empty list (default) = fire on
    # every iter (legacy semantic). Non-empty = fire only on iter indices
    # in this list (0-indexed within the consumer graph's pipeline-local
    # iter counter). Allows multi-iter tensors to be reloaded only in a
    # subset of iters.
    iter_mask: list[int] = field(default_factory=list)
    # Structural swap-in gate (SwapAdvisor control edge W_swapout→W_swapin):
    # cg-sim node ids of the eviction trigger-nodes whose freed bytes this
    # swap-in reuses (a fixed-pool producer/consumer pairing — see the
    # planner). DAV defers firing this prefetch's H2D (and the VRAM claim)
    # until every gate node has retired (NodeStatus.DONE). Gating on the
    # evict NODE — not on the victim being "absent" — is what makes the
    # bound real: the trigger node consumes the victim as an input, so it
    # can only retire after the victim was resident and then freed, which
    # genuinely accounts the reused byte and ties the load to tl-progression
    # (it cannot run ahead of the evictions it depends on). Empty = no gate.
    gate_evict_node_ids: list[int] = field(default_factory=list)


@dataclass
class NeutralEvict:
    tensor_uid: int
    issue_launch_id: int
    transfer_start_ns: int = 0
    transfer_end_ns: int = 0
    reason: str = ""
    issue_node_id: int = -1
    # Per-iter mask for partial schedules. Same semantics as
    # NeutralPrefetch.iter_mask: empty = every iter; non-empty = subset.
    iter_mask: list[int] = field(default_factory=list)
    # Resolved cg-sim tensor id. See ``NeutralPrefetch.cgsim_tid``.
    cgsim_tid: int = -1


@dataclass
class NeutralColdStart:
    tensor_uid: int
    anchor_launch_id: int
    reason: str = ""
    # Resolved cg-sim tensor ids — coldstart applies to ALL runtime
    # instances of the (gid, ctid), so this is a list rather than a
    # single id. Empty list ⇒ not resolved at emit time.
    cgsim_tids: list[int] = field(default_factory=list)


@dataclass
class NeutralSchedule:
    graph_order: list[int] = field(default_factory=list)
    compilation_hashes: dict[int, str] = field(default_factory=dict)
    tensors: list[NeutralTensor] = field(default_factory=list)
    prefetches: list[NeutralPrefetch] = field(default_factory=list)
    evicts: list[NeutralEvict] = field(default_factory=list)
    cold_starts: list[NeutralColdStart] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    format_version: str = FORMAT_VERSION

    def tensor_by_uid(self) -> dict[int, NeutralTensor]:
        return {t.uid: t for t in self.tensors}


# ---------------------------------------------------------------------------
# Builder helper (from a UnifiedTimeline)
# ---------------------------------------------------------------------------


def build_neutral_schedule_from_timeline(
    tl,
    *,
    prefetches: list[NeutralPrefetch],
    evicts: list[NeutralEvict],
    cold_starts: list[NeutralColdStart],
    meta: dict[str, Any],
) -> "NeutralSchedule":
    """Package schedule decisions + tensor table extracted from a UnifiedTimeline.

    Callers build ``prefetches`` / ``evicts`` / ``cold_starts`` against
    ``tl.tensors[uid]`` (using the unified-timeline UID as ``tensor_uid``)
    and supply scheduler-specific metadata. This helper copies the tensor
    table into the neutral format with per-graph launch_ids restored.
    """
    tensors: list[NeutralTensor] = []
    for t in tl.tensors:
        sgid = getattr(t, "storage_group_id", None)
        tt_tids = list(getattr(t, "trace_tids", []) or [])
        entry = t.entry if isinstance(t.entry, dict) else {}
        tensors.append(NeutralTensor(
            uid=t.uid,
            graph_id=t.graph_id,
            compiled_tensor_id=t.compiled_tensor_id,
            graph_input_name=t.graph_input_name,
            size_bytes=t.size_bytes,
            dtype=t.dtype,
            used_by_launch_ids=[int(tl.tasks[pos].launch_id) for pos in t.uses],
            shape=list(entry.get("shape", []) or []),
            graph_input_idx=entry.get("graph_input_idx"),
            storage_group_id=int(sgid) if sgid is not None else None,
            trace_tids=tt_tids,
        ))
    return NeutralSchedule(
        graph_order=list(tl.graph_order),
        compilation_hashes=dict(tl.per_graph_hash),
        tensors=tensors,
        prefetches=list(prefetches),
        evicts=list(evicts),
        cold_starts=list(cold_starts),
        meta=dict(meta),
    )


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------


def resolve_neutral_cgsim_tids(schedule: NeutralSchedule, trace) -> None:
    """Mutate ``schedule`` in place: populate ``cgsim_tid`` on every
    prefetch / evict / cold_start by resolving its (gid, ctid) against
    the runtime tensors actually read by the referenced trace nodes.

    Once a schedule has resolved tids, the injector can skip its own
    (gid, ctid) → cgsim_tid disambiguation entirely — the schedule
    speaks runtime-space identifiers directly. Eliminates the
    ``ambiguous_skipped`` failure mode for models with repeated
    same-shape blocks.

    Per-entry tids are derived from the entry's anchor node:
      - Prefetch:   ``wait_node_id`` (the consumer that reads this tid)
      - Evict:      ``issue_node_id`` (the node after which the tid frees)
      - ColdStart:  every GPU node in any of the tensor's
        ``used_by_launch_ids`` (cold-start applies to every runtime
        instance the schedule isn't covering with explicit prefetches)

    Entries we cannot resolve (no anchor node, no matching candidate)
    are left with ``cgsim_tid = -1`` / empty list. The injector treats
    those as legacy and falls back to its in-process resolver.
    """
    from .tid_resolve import (
        build_lid_ctid_order,
        build_lid_to_gpu_node_ids,
        build_lid_to_node_id,
        resolve_tid_for_node,
    )

    tensors_by_uid = schedule.tensor_by_uid()
    tensor_metas: dict[tuple[int, int], dict[Any, Any]] = {}
    for t in tensors_by_uid.values():
        gid = int(t.graph_id)
        ctid = int(t.compiled_tensor_id)
        if gid < 0 or ctid < 0:
            continue
        tensor_metas[(gid, ctid)] = {
            "used_by_launch_ids": list(t.used_by_launch_ids),
            "shape": list(t.shape) if t.shape else None,
            "size_bytes": int(t.size_bytes) if t.size_bytes else None,
            "dtype": t.dtype or None,
            "graph_input_idx": t.graph_input_idx,
        }

    lid_ctid_order = build_lid_ctid_order(tensor_metas)
    lid_to_node = build_lid_to_node_id(trace)
    lid_to_gpu_nodes = build_lid_to_gpu_node_ids(trace)

    # --- prefetches ---
    for pf in schedule.prefetches:
        if pf.cgsim_tid >= 0:
            continue  # already resolved
        meta = tensors_by_uid.get(pf.tensor_uid)
        if meta is None:
            continue
        # Fast path (RC1/RC4): if the timeline recorded the trace tid
        # for this storage and the consumer's node reads exactly one
        # such tid (the typical case for non-aliased storages), pick
        # it directly without shape disambiguation.
        if meta.trace_tids:
            tt_set = set(int(t) for t in meta.trace_tids)
            if pf.wait_node_id >= 0:
                node = trace.node_map.get(int(pf.wait_node_id))
                if node is not None:
                    cand = [
                        int(t) for t in (node.input_tensors or [])
                        if int(t) in tt_set
                    ]
                    if len(cand) == 1:
                        pf.cgsim_tid = cand[0]
                        continue
                    if len(cand) >= 1:
                        # Multiple matches (storage aliasing) — pick the
                        # first as a deterministic tie-break.
                        pf.cgsim_tid = cand[0]
                        continue
            # No wait_node_id or no match in its inputs: if only one
            # trace tid is associated with this storage, use it.
            if len(tt_set) == 1:
                pf.cgsim_tid = next(iter(tt_set))
                continue
        gid = int(meta.graph_id)
        ctid = int(meta.compiled_tensor_id)
        # Prefer the schedule's exact wait_node_id; fall back to first
        # GPU node of (gid, wait_launch_id).
        node_id = pf.wait_node_id if pf.wait_node_id >= 0 else lid_to_node.get(
            (gid, pf.wait_launch_id), -1,
        )
        if node_id < 0:
            continue
        cg = resolve_tid_for_node(
            trace, node_id, gid, pf.wait_launch_id, ctid,
            tensor_metas, lid_ctid_order,
        )
        if cg is not None:
            pf.cgsim_tid = int(cg)

    # --- evicts ---
    for ev in schedule.evicts:
        if ev.cgsim_tid >= 0:
            continue
        meta = tensors_by_uid.get(ev.tensor_uid)
        if meta is None:
            continue
        if meta.trace_tids:
            tt_set = set(int(t) for t in meta.trace_tids)
            if ev.issue_node_id >= 0:
                node = trace.node_map.get(int(ev.issue_node_id))
                if node is not None:
                    cand = [
                        int(t) for t in (node.input_tensors or [])
                        if int(t) in tt_set
                    ]
                    if cand:
                        ev.cgsim_tid = cand[0]
                        continue
            if len(tt_set) == 1:
                ev.cgsim_tid = next(iter(tt_set))
                continue
        gid = int(meta.graph_id)
        ctid = int(meta.compiled_tensor_id)
        node_id = ev.issue_node_id if ev.issue_node_id >= 0 else lid_to_node.get(
            (gid, ev.issue_launch_id), -1,
        )
        if node_id < 0:
            continue
        cg = resolve_tid_for_node(
            trace, node_id, gid, ev.issue_launch_id, ctid,
            tensor_metas, lid_ctid_order,
        )
        if cg is not None:
            ev.cgsim_tid = int(cg)

    # --- cold_starts: applies to EVERY runtime instance of (gid, ctid) ---
    for cs in schedule.cold_starts:
        if cs.cgsim_tids:
            continue
        meta = tensors_by_uid.get(cs.tensor_uid)
        if meta is None:
            continue
        # Fast path (RC1): timeline already knows the trace tids that
        # back this NeutralTensor — use them directly. Cold-start
        # applies to every runtime instance of the storage, so the
        # full list is the answer.
        if meta.trace_tids:
            cs.cgsim_tids = [int(t) for t in meta.trace_tids]
            continue
        gid = int(meta.graph_id)
        ctid = int(meta.compiled_tensor_id)
        resolved: list[int] = []
        for lid in meta.used_by_launch_ids:
            try:
                lid_int = int(lid)
            except (TypeError, ValueError):
                continue
            for nid in lid_to_gpu_nodes.get((gid, lid_int), []):
                cg = resolve_tid_for_node(
                    trace, nid, gid, lid_int, ctid,
                    tensor_metas, lid_ctid_order,
                )
                if cg is not None and cg not in resolved:
                    resolved.append(int(cg))
        if resolved:
            cs.cgsim_tids = resolved


def write_neutral_schedule(
    path: str | Path,
    schedule: NeutralSchedule,
    *,
    trace=None,
) -> None:
    if trace is not None:
        resolve_neutral_cgsim_tids(schedule, trace)
        n_pf = sum(1 for p in schedule.prefetches if p.cgsim_tid >= 0)
        n_ev = sum(1 for e in schedule.evicts if e.cgsim_tid >= 0)
        n_cs = sum(1 for c in schedule.cold_starts if c.cgsim_tids)
        print(
            f"[neutral] resolved cgsim_tids: "
            f"prefetch={n_pf}/{len(schedule.prefetches)} "
            f"evict={n_ev}/{len(schedule.evicts)} "
            f"cold_start={n_cs}/{len(schedule.cold_starts)}",
            flush=True,
        )
    doc: dict[str, Any] = {
        "format_version": schedule.format_version,
        "graph_order": list(schedule.graph_order),
        "compilation_hashes": {
            str(gid): h for gid, h in schedule.compilation_hashes.items()
        },
        "tensors": [
            {
                "uid": t.uid,
                "graph_id": t.graph_id,
                "compiled_tensor_id": t.compiled_tensor_id,
                "graph_input_name": t.graph_input_name,
                "size_bytes": t.size_bytes,
                "dtype": t.dtype,
                "used_by_launch_ids": list(t.used_by_launch_ids),
                "shape": list(t.shape) if t.shape else [],
                "graph_input_idx": t.graph_input_idx,
                "trace_tids": [int(x) for x in t.trace_tids],
            }
            for t in schedule.tensors
        ],
        "prefetches": [
            {
                "tensor_uid": p.tensor_uid,
                "issue_launch_id": p.issue_launch_id,
                "wait_launch_id": p.wait_launch_id,
                "transfer_start_ns": p.transfer_start_ns,
                "transfer_end_ns": p.transfer_end_ns,
                "reason": p.reason,
                "issue_node_id": p.issue_node_id,
                "wait_node_id": p.wait_node_id,
                "trusted_async": p.trusted_async,
                "issue_graph_id": p.issue_graph_id,
                "cross_iter": p.cross_iter,
                "iter_mask": list(p.iter_mask),
                "cgsim_tid": int(p.cgsim_tid),
                "gate_evict_node_ids": [int(n) for n in p.gate_evict_node_ids],
            }
            for p in schedule.prefetches
        ],
        "evicts": [
            {
                "tensor_uid": e.tensor_uid,
                "issue_launch_id": e.issue_launch_id,
                "transfer_start_ns": e.transfer_start_ns,
                "transfer_end_ns": e.transfer_end_ns,
                "reason": e.reason,
                "issue_node_id": e.issue_node_id,
                "iter_mask": list(e.iter_mask),
                "cgsim_tid": int(e.cgsim_tid),
            }
            for e in schedule.evicts
        ],
        "cold_starts": [
            {
                "tensor_uid": c.tensor_uid,
                "anchor_launch_id": c.anchor_launch_id,
                "reason": c.reason,
                "cgsim_tids": [int(t) for t in c.cgsim_tids],
            }
            for c in schedule.cold_starts
        ],
        "meta": dict(schedule.meta),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)


def load_neutral_schedule(path: str | Path) -> NeutralSchedule:
    with open(path) as f:
        doc = json.load(f)
    fmt = str(doc.get("format_version", ""))
    if fmt and fmt != FORMAT_VERSION:
        # Forward compatibility: keep loading as long as required fields
        # are present. Log a warning via print for CLI visibility.
        print(
            f"[neutral] warning: loading schedule with format_version={fmt}, "
            f"loader expects {FORMAT_VERSION}"
        )
    return NeutralSchedule(
        graph_order=list(doc.get("graph_order", [])),
        compilation_hashes={
            int(gid): str(h) for gid, h in doc.get("compilation_hashes", {}).items()
        },
        tensors=[
            NeutralTensor(
                uid=int(t["uid"]),
                graph_id=int(t["graph_id"]),
                compiled_tensor_id=int(t["compiled_tensor_id"]),
                graph_input_name=str(t.get("graph_input_name", "")),
                size_bytes=int(t.get("size_bytes", 0)),
                dtype=str(t.get("dtype", "")),
                used_by_launch_ids=[int(x) for x in t.get("used_by_launch_ids", [])],
                shape=list(t.get("shape") or []),
                graph_input_idx=(
                    int(t["graph_input_idx"])
                    if t.get("graph_input_idx") is not None else None
                ),
                trace_tids=[int(x) for x in t.get("trace_tids", [])],
            )
            for t in doc.get("tensors", [])
        ],
        prefetches=[
            NeutralPrefetch(
                tensor_uid=int(p["tensor_uid"]),
                issue_launch_id=int(p.get("issue_launch_id", -1)),
                wait_launch_id=int(p.get("wait_launch_id", -1)),
                transfer_start_ns=int(p.get("transfer_start_ns", 0)),
                transfer_end_ns=int(p.get("transfer_end_ns", 0)),
                reason=str(p.get("reason", "")),
                issue_node_id=int(p.get("issue_node_id", -1)),
                wait_node_id=int(p.get("wait_node_id", -1)),
                trusted_async=bool(p.get("trusted_async", False)),
                issue_graph_id=int(p.get("issue_graph_id", -1)),
                cross_iter=bool(p.get("cross_iter", False)),
                iter_mask=[int(x) for x in p.get("iter_mask", [])],
                cgsim_tid=int(p.get("cgsim_tid", -1)),
                gate_evict_node_ids=[
                    int(n) for n in p.get("gate_evict_node_ids", [])
                ],
            )
            for p in doc.get("prefetches", [])
        ],
        evicts=[
            NeutralEvict(
                tensor_uid=int(e["tensor_uid"]),
                issue_launch_id=int(e.get("issue_launch_id", -1)),
                transfer_start_ns=int(e.get("transfer_start_ns", 0)),
                transfer_end_ns=int(e.get("transfer_end_ns", 0)),
                reason=str(e.get("reason", "")),
                issue_node_id=int(e.get("issue_node_id", -1)),
                iter_mask=[int(x) for x in e.get("iter_mask", [])],
                cgsim_tid=int(e.get("cgsim_tid", -1)),
            )
            for e in doc.get("evicts", [])
        ],
        cold_starts=[
            NeutralColdStart(
                tensor_uid=int(c["tensor_uid"]),
                anchor_launch_id=int(c.get("anchor_launch_id", 0)),
                reason=str(c.get("reason", "")),
                cgsim_tids=[int(t) for t in c.get("cgsim_tids", [])],
            )
            for c in doc.get("cold_starts", [])
        ],
        meta=dict(doc.get("meta", {})),
        format_version=fmt or FORMAT_VERSION,
    )


# ---------------------------------------------------------------------------
# Conversion to PyTorch jit_sim_prune_schedule.json
# ---------------------------------------------------------------------------

_GPU_KINDS = ("gpu_stream", "gpu", "gpu_runtime")


def _build_gpu_node_index(trace) -> tuple[
    dict[int, tuple[int, int, int]], dict[int, int],
]:
    """node_id -> (graph_id, launch_id, iter_idx) for GPU compute nodes,
    plus per-graph multiplicity (wrapper invocations in the trace).

    Each wrapper invocation executes every launch_id exactly once, so the
    k-th occurrence of a (gid, lid) pair ordered by start_ns belongs to the
    graph's k-th iteration.
    """
    per_key: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for nid, node in trace.node_map.items():
        args = node.args or {}
        if str(args.get("resource_kind") or "") not in _GPU_KINDS:
            continue
        gid_raw = args.get("compiled_graph_id")
        lid_raw = args.get("compiled_launch_id")
        if gid_raw is None or lid_raw is None:
            continue
        try:
            gid, lid = int(gid_raw), int(lid_raw)
        except (TypeError, ValueError):
            continue
        if lid < 0:
            continue
        start = int(args.get("start_ns") or 0)
        per_key.setdefault((gid, lid), []).append((start, int(nid)))
    occ_counts: dict[int, list[int]] = {}
    for (gid, _lid), occ in per_key.items():
        occ_counts.setdefault(gid, []).append(len(occ))
    # Mode, not max: a launch backed by several GPU trace nodes per
    # execution (kernel + memcpy sub-events) would otherwise inflate the
    # graph's iteration count and missize every cyclic iter mask.
    multiplicity = {
        gid: Counter(counts).most_common(1)[0][0]
        for gid, counts in occ_counts.items()
    }
    # Iter index = occurrence index normalized by the launch's nodes-per-
    # iteration: a 4-iter graph whose lid has 24 GPU nodes maps occurrence
    # 0..5 → iter 0, 6..11 → iter 1, … Raw occurrence indices would put
    # garbage iters (9, 20, …) into the cyclic masks.
    index: dict[int, tuple[int, int, int]] = {}
    for (gid, lid), occ in per_key.items():
        occ.sort()
        mult = max(1, multiplicity.get(gid, 1))
        per_iter = max(1, len(occ) // mult)
        for idx, (_s, nid) in enumerate(occ):
            index[nid] = (gid, lid, min(idx // per_iter, mult - 1))
    return index, multiplicity


def _nodes_section(
    trace,
    node_starts: list[int] | None,
    node_ends: list[int] | None,
) -> list[dict[str, Any]]:
    if node_starts is None or node_ends is None:
        node_starts, node_ends = [], []
        t_cursor = 0
        for _nid, node in trace.node_map.items():
            node_starts.append(t_cursor)
            t_cursor += int(getattr(node, "compute_time_micros", 0) * 1_000)
            node_ends.append(t_cursor)
    return [
        {
            "idx": int(nid),
            "name": str(getattr(node, "name", "")),
            "resource_kind": str(node.args.get("resource_kind") or ""),
            "start_ns": int(node_starts[i]),
            "end_ns": int(node_ends[i]),
        }
        for i, (nid, node) in enumerate(trace.node_map.items())
    ]


def remap_neutral_to_compile_space(
    neutral: NeutralSchedule,
    tid_to_sidecar: dict[int, Any],
) -> tuple[int, int]:
    """Rewrite tensor identities from trace-space to compile-space.

    Order-axis emitters (``ct_milp_overlap._emit_neutral`` family) fill
    ``compiled_tensor_id`` with the cg-sim trace tid and
    ``graph_input_name`` with the profiler name. The PyTorch wrapper
    resolves ops only by sidecar ``compiled_tensor_id`` /
    ``graph_input_name``, so each tensor is rewritten via
    ``tid_to_sidecar`` (see ``tid_resolve.map_trace_tids_to_sidecar``).

    Unmappable tensors are marked with ``compiled_tensor_id=-1`` and an
    empty ``graph_input_name``; converters must skip their ops (the
    weights stay CUDA-resident). Returns (n_unmapped, unmapped_bytes).
    """
    n_unmapped = 0
    unmapped_bytes = 0
    for t in neutral.tensors:
        tid = int(t.trace_tids[0]) if t.trace_tids else int(t.compiled_tensor_id)
        side = tid_to_sidecar.get(tid)
        if side is None:
            t.compiled_tensor_id = -1
            t.graph_input_name = ""
            n_unmapped += 1
            unmapped_bytes += int(t.size_bytes)
            continue
        t.graph_id = int(side.graph_id)
        t.compiled_tensor_id = int(side.compiled_tensor_id)
        t.graph_input_name = str(side.graph_input_name)
        t.graph_input_idx = side.entry.get("graph_input_idx")
    return n_unmapped, unmapped_bytes


def neutral_to_pytorch_anchored(
    neutral: NeutralSchedule,
    *,
    trace,
    node_starts: list[int] | None = None,
    node_ends: list[int] | None = None,
    compilation_hashes: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Convert an order-axis NeutralSchedule into the PyTorch runtime JSON.

    ``neutral_to_pytorch`` trusts the launch-id fields, but order-axis
    emitters (``ct_milp_overlap._emit_neutral`` and derivatives) set
    ``issue_launch_id == wait_launch_id`` and carry the real issuer only in
    ``issue_node_id`` — converting those directly degrades every same-graph
    async prefetch to a sync H2D at its consumer. This converter:

      * recovers (graph, launch, iter) anchors from exact trace node ids,
        classifying each prefetch as same-iter async, cross-iter wrap, or
        cross-graph async;
      * merges the per-iteration entries the order axis emits (one per
        consumer event) into one op per anchor pair;
      * normalizes per-tensor iter masks — the PyTorch wrapper applies ONE
        h2d mask and ONE evict mask per tensor (``set_iter_mask``), so
        per-op masks are unioned per tensor and dropped when they cover
        every iteration.
    """
    node_index, multiplicity = _build_gpu_node_index(trace)
    uid_to_tensor = neutral.tensor_by_uid()
    hashes = {
        int(g): str(h)
        for g, h in (compilation_hashes or neutral.compilation_hashes or {}).items()
    }

    # (gid, lid) -> first-occurrence node id, for cold-start anchoring.
    first_node: dict[tuple[int, int], int] = {}
    for nid, (gid, lid, it) in node_index.items():
        if it == 0:
            first_node[(gid, lid)] = nid

    def _mapped(t: NeutralTensor | None) -> bool:
        return (
            t is not None
            and int(t.compiled_tensor_id) >= 0
            and bool(t.graph_input_name)
        )

    skipped_ops = 0
    skipped_bytes = 0

    # ---- prefetches: derive anchors, merge per-iter duplicates ----
    pf_groups: dict[tuple, dict[str, Any]] = {}
    for p in neutral.prefetches:
        t = uid_to_tensor.get(p.tensor_uid)
        if not _mapped(t):
            skipped_ops += 1
            skipped_bytes += int(t.size_bytes) if t is not None else 0
            continue
        cons = node_index.get(int(p.wait_node_id))
        if cons is not None:
            cgid, clid, citer = cons
        else:
            cgid, clid, citer = int(t.graph_id), int(p.wait_launch_id), 0
        issuer = node_index.get(int(p.issue_node_id))
        if issuer is None or int(p.issue_node_id) == int(p.wait_node_id):
            igid, ilid, iiter = cgid, clid, citer  # sync demand-load
        else:
            igid, ilid, iiter = issuer
        cross_graph = igid != cgid
        cross_iter = not cross_graph and iiter < citer
        is_async = cross_graph or cross_iter or ilid != clid
        after_lid = ilid if is_async else -1
        key = (
            p.tensor_uid, after_lid, clid,
            igid if cross_graph else -1, cross_iter,
        )
        g = pf_groups.get(key)
        if g is None:
            g = pf_groups[key] = {
                "tensor": t,
                "trusted": bool(p.trusted_async),
                "iters": set(),
                "duration_ns": max(
                    int(p.transfer_end_ns - p.transfer_start_ns), 0,
                ),
                "reason": p.reason,
                "before_node": int(p.wait_node_id),
                "issue_node": int(p.issue_node_id),
            }
        g["iters"].add(citer)
        if not cross_graph:
            # The fire boundary is gated by the same per-tensor h2d mask
            # (in the consumer graph's iter domain); for cross-iter ops
            # the fire iter precedes the wait iter and must be in the
            # mask too. (Cross-graph fires are dispatched unmasked.)
            g["iters"].add(iiter)
        g["trusted"] = g["trusted"] and bool(p.trusted_async)

    # ---- evicts: anchor at the issuing consumer's launch, merge ----
    ev_groups: dict[tuple, dict[str, Any]] = {}
    for e in neutral.evicts:
        t = uid_to_tensor.get(e.tensor_uid)
        if not _mapped(t):
            skipped_ops += 1
            continue
        anchor = node_index.get(int(e.issue_node_id))
        if anchor is not None:
            agid, alid, aiter = anchor
        else:
            agid, alid, aiter = int(t.graph_id), int(e.issue_launch_id), 0
        key = (e.tensor_uid, alid)
        g = ev_groups.get(key)
        if g is None:
            g = ev_groups[key] = {
                "tensor": t,
                "graph_id": agid,
                "iters": set(),
                "duration_ns": max(
                    int(e.transfer_end_ns - e.transfer_start_ns), 0,
                ),
                "reason": e.reason,
                "after_node": int(e.issue_node_id),
            }
        g["iters"].add(aiter)

    # ---- lifetime release ----
    # The order-axis model frees a streamed tensor after its last consumer
    # event; cg-sim's executor releases at consumer-retire natively, but
    # the PyTorch runtime needs an explicit evict or the tensor stays
    # resident and the realized peak reverts to baseline. Anchor at the
    # tensor's last consumer launch on its graph's last iteration; the
    # reload next pipeline call is the tensor's own (cyclic) initial
    # prefetch. Hybrid cold tensors reload via cold_start, whose sync
    # wrapper-top load would serialize — leave them resident.
    cold_uids = {c.tensor_uid for c in neutral.cold_starts}
    for uid in {k[0] for k in pf_groups} - cold_uids:
        t = uid_to_tensor[uid]
        lids = [int(x) for x in t.used_by_launch_ids]
        if not lids:
            continue
        key = (uid, max(lids))
        last_iter = multiplicity.get(int(t.graph_id), 1) - 1
        g = ev_groups.get(key)
        if g is None:
            ev_groups[key] = {
                "tensor": t,
                "graph_id": int(t.graph_id),
                "iters": {last_iter},
                "duration_ns": 0,
                "reason": "lifetime_release",
                "after_node": -1,
            }
        else:
            g["iters"].add(last_iter)

    # ---- per-tensor mask normalization ----
    # The wrapper installs one h2d mask + one evict mask per tensor; ops of
    # the same tensor must agree. Union per tensor; a union covering every
    # iteration of the tensor's graph collapses to the fire-every-iter
    # (empty mask) semantic.
    def _tensor_masks(groups: dict[tuple, dict[str, Any]]) -> dict[int, list[int]]:
        union: dict[int, set[int]] = {}
        for (uid, *_k), g in groups.items():
            union.setdefault(uid, set()).update(g["iters"])
        masks: dict[int, list[int]] = {}
        for uid, iters in union.items():
            t = uid_to_tensor[uid]
            mult = multiplicity.get(int(t.graph_id), 1)
            masks[uid] = [] if iters >= set(range(mult)) else sorted(iters)
        return masks

    pf_masks = _tensor_masks(pf_groups)
    ev_masks = _tensor_masks(ev_groups)
    # A masked h2d also masks its WAIT, so a partial h2d mask is only safe
    # for tensors whose residency the schedule itself cycles. Never-evicted
    # tensors keep the fire-every-iter semantic (resident fires are O(1)
    # skips) so any externally-emptied storage is re-covered by the wait's
    # sync fallback.
    evicted_uids = {uid for (uid, *_k) in ev_groups}
    pf_masks = {
        uid: (m if uid in evicted_uids else [])
        for uid, m in pf_masks.items()
    }

    io_operations: list[dict[str, Any]] = []
    for (uid, after_lid, before_lid, issue_gid, cross_iter), g in sorted(
        pf_groups.items(), key=lambda kv: (kv[0][2], kv[0][0]),
    ):
        t = g["tensor"]
        io_operations.append({
            "type": "vram_prefetch_h2d",
            "tensor_name": t.graph_input_name,
            "tensor_kind": "WEIGHT",
            "before_node": g["before_node"],
            "after_node": -1,
            "duration_ns": g["duration_ns"],
            "size_bytes": int(t.size_bytes),
            "reason": g["reason"],
            "before_launch_id": int(before_lid),
            "after_launch_id": int(after_lid),
            "compiled_tensor_id": int(t.compiled_tensor_id),
            "compiled_graph_id": int(t.graph_id),
            "compilation_hash": hashes.get(int(t.graph_id), ""),
            "compiled_graph_input_name": t.graph_input_name,
            "trusted_async": bool(g["trusted"]),
            "issue_compiled_graph_id": int(issue_gid),
            "issue_node_id": g["issue_node"],
            "wait_node_id": g["before_node"],
            "cross_iter": bool(cross_iter),
            "iter_mask": pf_masks.get(uid, []),
        })
    for (uid, after_lid), g in sorted(
        ev_groups.items(), key=lambda kv: (kv[0][1], kv[0][0]),
    ):
        t = g["tensor"]
        io_operations.append({
            "type": "vram_evict_d2h",
            "tensor_name": t.graph_input_name,
            "tensor_kind": "WEIGHT",
            "after_node": g["after_node"],
            "duration_ns": g["duration_ns"],
            "size_bytes": int(t.size_bytes),
            "reason": g["reason"],
            "after_launch_id": int(after_lid),
            "compiled_tensor_id": int(t.compiled_tensor_id),
            "compiled_graph_id": int(g["graph_id"]),
            "compilation_hash": hashes.get(int(g["graph_id"]), ""),
            "compiled_graph_input_name": t.graph_input_name,
            "issue_node_id": g["after_node"],
            "evict_node_id": g["after_node"],
            "iter_mask": ev_masks.get(uid, []),
        })

    cold_start_prefetches: list[dict[str, Any]] = []
    for c in neutral.cold_starts:
        t = uid_to_tensor.get(c.tensor_uid)
        if not _mapped(t):
            skipped_ops += 1
            continue
        lid = int(c.anchor_launch_id)
        cold_start_prefetches.append({
            "tensor_name": t.graph_input_name,
            "tensor_kind": "WEIGHT",
            "reason": c.reason,
            "miss_ns": 0,
            "attach_before_node": int(
                first_node.get((int(t.graph_id), lid), -1)
            ),
            "eager_start": True,
            "before_launch_id": lid,
            "compiled_tensor_id": int(t.compiled_tensor_id),
            "compiled_graph_id": int(t.graph_id),
            "compilation_hash": hashes.get(int(t.graph_id), ""),
            "compiled_graph_input_name": t.graph_input_name,
        })

    if skipped_ops:
        print(
            f"[neutral->pytorch] skipped {skipped_ops} ops on unmapped "
            f"tensors ({skipped_bytes / 1e6:.0f}MB of planned prefetch "
            f"volume stays CUDA-resident)",
            flush=True,
        )

    summary = dict(neutral.meta)
    summary["graph_multiplicity"] = {
        str(g): int(m) for g, m in sorted(multiplicity.items())
    }
    # Order-axis plans model residency across the whole pipeline; the
    # wrapper's per-graph-boundary cross-graph eviction would break them.
    summary["cross_graph_evict"] = False

    doc: dict[str, Any] = {
        "summary": summary,
        "nodes": _nodes_section(trace, node_starts, node_ends),
        "io_operations": io_operations,
        "spill_decisions": [],
        "cold_start_prefetches": cold_start_prefetches,
        "steady_state_resident": [],
    }
    if neutral.graph_order:
        doc["compilation_hash"] = hashes.get(int(neutral.graph_order[0]), "")
    return doc


def _build_launch_to_node(
    trace, graph_id: int,
) -> dict[int, int]:
    """Per-graph launch_id -> trace node_id. Filters nodes to the given graph."""
    result: dict[int, int] = {}
    for nid, node in trace.node_map.items():
        gid = node.args.get("compiled_graph_id")
        if gid is None or int(gid) != graph_id:
            continue
        lid = node.args.get("compiled_launch_id")
        if lid is None or int(lid) < 0:
            continue
        lid_i = int(lid)
        if lid_i in result:
            continue
        result[lid_i] = int(nid)
    return result


def neutral_to_pytorch(
    neutral: NeutralSchedule,
    *,
    trace,
    node_starts: list[int] | None = None,
    node_ends: list[int] | None = None,
) -> dict[str, Any]:
    """Convert a neutral schedule + trace into the PyTorch runtime JSON.

    The PyTorch ``load_io_schedule`` consumer expects:

      * ``io_operations`` with ``type`` in {``vram_prefetch_h2d``,
        ``vram_evict_d2h``, ``prefetch``} and per-op identity fields.
      * ``cold_start_prefetches`` with ``attach_before_node`` and
        ``before_launch_id``.
      * ``nodes`` section mirroring the profile timeline.
      * ``compilation_hash`` (top-level) for the first graph.

    The trace is used only to resolve per-graph ``launch_id`` -> ``node_id``
    for the ``before_node`` / ``after_node`` fields.
    """
    uid_to_tensor = neutral.tensor_by_uid()

    # launch_to_node per graph.
    launch_to_node_per_graph: dict[int, dict[int, int]] = {}
    for gid in neutral.graph_order:
        launch_to_node_per_graph[gid] = _build_launch_to_node(trace, gid)

    def _node_id(gid: int, lid: int) -> int:
        return launch_to_node_per_graph.get(gid, {}).get(int(lid), -1)

    io_operations: list[dict[str, Any]] = []

    for p in neutral.prefetches:
        t = uid_to_tensor.get(p.tensor_uid)
        if t is None:
            continue
        duration = max(int(p.transfer_end_ns - p.transfer_start_ns), 0)
        before_node = (
            int(p.wait_node_id)
            if int(p.wait_node_id) >= 0
            else _node_id(t.graph_id, p.wait_launch_id)
        )
        is_async = (
            p.issue_launch_id >= 0
            and (
                p.issue_graph_id >= 0
                or p.issue_launch_id != p.wait_launch_id
            )
        )
        after_launch_id = (
            int(p.issue_launch_id) if is_async else -1
        )
        io_operations.append({
            "type": "vram_prefetch_h2d",
            "tensor_name": t.graph_input_name,
            "tensor_kind": "WEIGHT",
            "before_node": int(before_node),
            "after_node": -1,
            "duration_ns": int(duration),
            "size_bytes": int(t.size_bytes),
            "reason": p.reason,
            "before_launch_id": int(p.wait_launch_id),
            "after_launch_id": after_launch_id,
            "compiled_tensor_id": int(t.compiled_tensor_id),
            "compiled_graph_id": int(t.graph_id),
            "compilation_hash": neutral.compilation_hashes.get(t.graph_id, ""),
            "compiled_graph_input_name": t.graph_input_name,
            "trusted_async": bool(p.trusted_async),
            "issue_compiled_graph_id": int(p.issue_graph_id),
            "issue_node_id": int(p.issue_node_id),
            "wait_node_id": int(p.wait_node_id),
            "cross_iter": bool(p.cross_iter),
            "iter_mask": list(p.iter_mask),
        })

    for e in neutral.evicts:
        t = uid_to_tensor.get(e.tensor_uid)
        if t is None:
            continue
        duration = max(int(e.transfer_end_ns - e.transfer_start_ns), 0)
        after_node = (
            int(e.issue_node_id)
            if int(e.issue_node_id) >= 0
            else _node_id(t.graph_id, e.issue_launch_id)
        )
        io_operations.append({
            "type": "vram_evict_d2h",
            "tensor_name": t.graph_input_name,
            "tensor_kind": "WEIGHT",
            "after_node": int(after_node),
            "duration_ns": int(duration),
            "size_bytes": int(t.size_bytes),
            "reason": e.reason,
            "after_launch_id": int(e.issue_launch_id),
            "compiled_tensor_id": int(t.compiled_tensor_id),
            "compiled_graph_id": int(t.graph_id),
            "compilation_hash": neutral.compilation_hashes.get(t.graph_id, ""),
            "compiled_graph_input_name": t.graph_input_name,
            "issue_node_id": int(e.issue_node_id),
            "evict_node_id": int(e.issue_node_id),
            "iter_mask": list(e.iter_mask),
        })

    cold_start_prefetches: list[dict[str, Any]] = []
    for c in neutral.cold_starts:
        t = uid_to_tensor.get(c.tensor_uid)
        if t is None:
            continue
        attach_node = _node_id(t.graph_id, c.anchor_launch_id)
        cold_start_prefetches.append({
            "tensor_name": t.graph_input_name,
            "tensor_kind": "WEIGHT",
            "reason": c.reason,
            "miss_ns": 0,
            "attach_before_node": int(attach_node),
            "eager_start": True,
            "before_launch_id": int(c.anchor_launch_id),
            "compiled_tensor_id": int(t.compiled_tensor_id),
            "compiled_graph_id": int(t.graph_id),
            "compilation_hash": neutral.compilation_hashes.get(t.graph_id, ""),
            "compiled_graph_input_name": t.graph_input_name,
        })

    nodes_section = _nodes_section(trace, node_starts, node_ends)

    first_hash = (
        neutral.compilation_hashes.get(neutral.graph_order[0], "")
        if neutral.graph_order else ""
    )

    doc: dict[str, Any] = {
        "summary": dict(neutral.meta),
        "nodes": nodes_section,
        "io_operations": io_operations,
        "spill_decisions": [],
        "cold_start_prefetches": cold_start_prefetches,
        "steady_state_resident": [],
    }
    if first_hash:
        doc["compilation_hash"] = first_hash
    return doc
