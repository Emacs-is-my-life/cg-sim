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


@dataclass
class NeutralColdStart:
    tensor_uid: int
    anchor_launch_id: int
    reason: str = ""


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
        tensors.append(NeutralTensor(
            uid=t.uid,
            graph_id=t.graph_id,
            compiled_tensor_id=t.compiled_tensor_id,
            graph_input_name=t.graph_input_name,
            size_bytes=t.size_bytes,
            dtype=t.dtype,
            used_by_launch_ids=[int(tl.tasks[pos].launch_id) for pos in t.uses],
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


def write_neutral_schedule(path: str | Path, schedule: NeutralSchedule) -> None:
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
            }
            for e in schedule.evicts
        ],
        "cold_starts": [
            {
                "tensor_uid": c.tensor_uid,
                "anchor_launch_id": c.anchor_launch_id,
                "reason": c.reason,
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
            )
            for e in doc.get("evicts", [])
        ],
        cold_starts=[
            NeutralColdStart(
                tensor_uid=int(c["tensor_uid"]),
                anchor_launch_id=int(c.get("anchor_launch_id", 0)),
                reason=str(c.get("reason", "")),
            )
            for c in doc.get("cold_starts", [])
        ],
        meta=dict(doc.get("meta", {})),
        format_version=fmt or FORMAT_VERSION,
    )


# ---------------------------------------------------------------------------
# Conversion to PyTorch jit_sim_prune_schedule.json
# ---------------------------------------------------------------------------


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

    # Node section: provide a default synthesized timeline if callers don't pass one.
    if node_starts is None or node_ends is None:
        node_starts = []
        node_ends = []
        t_cursor = 0
        for _nid, node in trace.node_map.items():
            node_starts.append(t_cursor)
            dur = int(getattr(node, "compute_time_micros", 0) * 1_000)
            t_cursor += dur
            node_ends.append(t_cursor)

    nodes_section: list[dict[str, Any]] = []
    for i, (nid, node) in enumerate(trace.node_map.items()):
        nodes_section.append({
            "idx": int(nid),
            "name": str(getattr(node, "name", "")),
            "resource_kind": str(node.args.get("resource_kind") or ""),
            "start_ns": int(node_starts[i]),
            "end_ns": int(node_ends[i]),
        })

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
