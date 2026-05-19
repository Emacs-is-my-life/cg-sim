"""Unified timeline across multiple compiled graphs.

Builds a single execution sequence (``tasks``) over all graphs in their
profile-observed order, with a per-tensor identity that is *scoped to a
graph* (``(graph_id, compiled_tensor_id)``) — SDXL-style pipelines do not
share weights across graphs, and collisions on ``compiled_launch_id``
(each graph counts from 0) make global aliasing unsafe.

The legacy ``build_compiled_tensor_problem`` silently uses a globally-shared
``launch_to_node`` map, which lets two graphs' launches collide. This
module avoids the collision by filtering nodes by ``compiled_graph_id``
first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sim.core.trace import Node, Trace

from .loader import MultiGraphSidecars
from .problem import (
    _DTYPE_BYTES,
    _GPU_RESOURCE_KINDS,
    _compute_wall_timeline,
    _is_gpu_node,
    _size_bytes_from_entry,
)


@dataclass
class GlobalTask:
    """One compute kernel on the unified pipeline timeline."""
    global_pos: int
    graph_id: int
    launch_id: int            # per-graph local launch_id (for op emission)
    node_id: int
    duration_ns: int
    start_ns: int             # cumulative across graphs
    end_ns: int
    used_tensors: list[int] = field(default_factory=list)  # indices into tensors[]


@dataclass
class GlobalTensor:
    """A unique (graph_id, compiled_tensor_id) tensor on the unified timeline.

    ``storage_group_id`` (when present in the sidecar) identifies an
    alias group: two ``(graph_id, compiled_tensor_id)`` entries with the
    same group id refer to the same physical storage at compile time
    and should be coalesced into a single schedulable unit by
    ``coalesce_by_storage`` before scheduling.  When the bundle producer
    didn't emit it (older bundles, or compile paths where storage isn't
    determinable), the value is ``None`` and the tensor is treated as a
    singleton group keyed by ``(graph_id, compiled_tensor_id)``.
    """
    uid: int                  # index in UnifiedTimeline.tensors
    graph_id: int
    compiled_tensor_id: int
    graph_input_name: str
    size_bytes: int
    dtype: str
    entry: dict[str, Any]     # original sidecar entry
    uses: list[int] = field(default_factory=list)  # task global_pos values
    storage_group_id: Any = None   # opaque, compared by equality


@dataclass
class UnifiedTimeline:
    """A concatenated view of all graphs for cross-graph scheduling."""
    tasks: list[GlobalTask]
    tensors: list[GlobalTensor]
    graph_order: list[int]
    per_graph_tasks: dict[int, list[int]]   # gid -> list of global_pos (in order)
    per_graph_launch_to_task: dict[int, dict[int, int]]  # gid -> (launch_id -> global_pos)
    per_graph_hash: dict[int, str]
    total_duration_ns: int
    cpu_per_launch_ns: int = 0
    # Per-pipeline-call iteration count for each graph. Defaults to 1
    # (single invocation per pipeline call). For SDXL diffusion: UNet
    # has multiplicity = num_inference_steps × cfg_factor; TEs and VAE
    # are 1. Required for sound issuer placement in multi-iter
    # consumers — a graph A can only issue cross-graph H2D for graph
    # B's tensors if multiplicity[A] >= multiplicity[B] (otherwise A's
    # one fire starves B's later iterations).
    graph_multiplicity: dict[int, int] = field(default_factory=dict)


def _graph_order_by_first_kernel(
    trace: Trace, node_to_graph_id: dict[int, int],
) -> list[int]:
    """Execution order = order of first GPU kernel per graph (by profile ``start_ns``)."""
    first_time: dict[int, int] = {}
    synth = 0
    synth_starts: dict[int, int] = {}
    for nid, node in trace.node_map.items():
        synth_starts[nid] = synth
        synth += int(node.compute_time_micros * 1_000)
    for nid, node in trace.node_map.items():
        gid = node_to_graph_id.get(nid, -1)
        if gid < 0 or not _is_gpu_node(node):
            continue
        raw = node.args.get("start_ns") or 0
        start = int(raw) if raw else synth_starts[nid]
        if gid not in first_time or start < first_time[gid]:
            first_time[gid] = start
    return sorted(first_time.keys(), key=lambda g: first_time[g])


def build_unified_timeline(
    trace: Trace,
    sidecars: MultiGraphSidecars,
    *,
    cpu_per_launch_ns: int = 0,
    graph_multiplicity: dict[int, int] | None = None,
    replicate_uses: bool = False,
) -> UnifiedTimeline:
    """Build a global timeline by concatenating per-graph kernel sequences.

    Graphs run sequentially in profile order. Each graph's kernels retain
    their per-graph ``launch_id`` so the wrapper filter can route emitted
    ops correctly. Tensors are scoped to the graph they live in (no
    cross-graph tensor sharing is assumed or attempted).

    When ``replicate_uses=False`` (default), each ``(graph_id, launch_id)``
    is recorded once — duplicates in the trace are dropped. This collapses
    multi-iter consumers (e.g., SDXL UNet running N times per pipeline)
    to a single pass; downstream LP schedulers see the compressed view
    and may underestimate H2D byte requirements.

    When ``replicate_uses=True``, every trace event is preserved. Tasks
    are sorted by ``start_ns`` so they appear in execution order, and a
    tensor's ``uses`` list contains every position the tensor is read
    (one entry per real GPU event, not per unique launch_id). The
    ``per_graph_launch_to_task`` map becomes ``dict[gid, dict[lid,
    list[int]]]`` because a single lid may map to multiple positions.
    Use this when a scheduler needs the true per-iter pattern.
    """
    # Per-node graph membership.
    node_to_graph_id: dict[int, int] = {}
    for nid, node in trace.node_map.items():
        gid = node.args.get("compiled_graph_id")
        if gid is None:
            continue
        node_to_graph_id[nid] = int(gid)

    graph_order = _graph_order_by_first_kernel(trace, node_to_graph_id)

    # Per-graph bucketing: only GPU nodes that carry a valid compiled_launch_id.
    # In replicate_uses=True mode, per_graph_lid maps lid → list[nid] (one
    # per occurrence). In dedup mode, per_graph_lid maps lid → first nid.
    per_graph_nodes: dict[int, list[int]] = {gid: [] for gid in graph_order}
    nid_to_lid: dict[int, int] = {}
    seen_lids: dict[int, set[int]] = {gid: set() for gid in graph_order}
    for nid, node in trace.node_map.items():
        if not _is_gpu_node(node):
            continue
        gid = node_to_graph_id.get(nid, -1)
        if gid not in per_graph_nodes:
            continue
        lid_raw = node.args.get("compiled_launch_id")
        if lid_raw is None:
            continue
        lid = int(lid_raw)
        if lid < 0:
            continue
        if not replicate_uses and lid in seen_lids[gid]:
            # Dedup mode: drop subsequent occurrences of the same lid.
            continue
        seen_lids[gid].add(lid)
        nid_to_lid[nid] = lid
        per_graph_nodes[gid].append(nid)

    # Sort each graph's nodes for deterministic execution order.
    # In dedup mode, sort by lid (one entry per lid). In replicate mode,
    # sort by (start_ns, lid) — primary by trace timestamp so multi-iter
    # copies appear in actual execution order; lid breaks ties when
    # timestamps are missing or equal.
    def _node_sort_key(nid: int) -> tuple:
        lid = nid_to_lid.get(nid, 0)
        if replicate_uses:
            raw = trace.node_map[nid].args.get("start_ns") or 0
            return (int(raw) if raw else 0, lid)
        return (lid,)

    for gid in graph_order:
        per_graph_nodes[gid].sort(key=_node_sort_key)

    # Compute per-kernel durations from compute_time_micros.
    def _dur_ns(nid: int) -> int:
        return int(trace.node_map[nid].compute_time_micros * 1_000)

    # Build global task list.
    tasks: list[GlobalTask] = []
    per_graph_tasks: dict[int, list[int]] = {gid: [] for gid in graph_order}
    # In replicate_uses=True mode, this maps lid → list[global_pos] (one
    # per occurrence). Dedup mode keeps the historical lid → int shape.
    per_graph_launch_to_task: dict[int, dict[int, Any]] = {
        gid: {} for gid in graph_order
    }
    per_graph_hash: dict[int, str] = {}

    # Compact GPU events back-to-back, removing the inter-event gaps
    # the profiler injects between record_function calls. Each event's
    # duration_ns from the trace is profiler-clean (the gaps are where
    # bookkeeping lives, not inside the events). Laying them out with
    # no gaps at their durations gives an iter wall ≈ max(GPU sum,
    # CPU sum) — close to real unprofiled wall — without needing any
    # calibration data.
    #
    # No CPU dispatch is added on top: cpu_per_launch_ns inflated wall
    # by ~70% on SDXL (586 ms vs ~350 ms reality) by treating CPU and
    # GPU as serial when in fact they overlap. This compaction pass
    # only sees GPU events anyway; CPU dispatch is implicit in the
    # gaps we just removed.
    global_pos = 0
    t_cursor = 0
    for gid in graph_order:
        lm = sidecars.launch_maps.get(gid, {})
        per_graph_hash[gid] = str(lm.get("compilation_hash", ""))
        for nid in per_graph_nodes[gid]:
            dur = _dur_ns(nid)
            start = t_cursor
            end = t_cursor + dur
            t_cursor = end
            lid = nid_to_lid.get(nid, 0)
            tasks.append(GlobalTask(
                global_pos=global_pos,
                graph_id=gid,
                launch_id=lid,
                node_id=nid,
                duration_ns=dur,
                start_ns=start,
                end_ns=end,
            ))
            per_graph_tasks[gid].append(global_pos)
            if replicate_uses:
                per_graph_launch_to_task[gid].setdefault(lid, []).append(global_pos)
            else:
                per_graph_launch_to_task[gid][lid] = global_pos
            global_pos += 1

    # Tensor table. Iterate sidecars in graph_order so per-graph tids stay
    # adjacent and tensors used earlier get smaller uids.
    tensors: list[GlobalTensor] = []
    for gid in graph_order:
        tmap = sidecars.tensor_maps.get(gid)
        if tmap is None:
            continue
        for entry in tmap.get("tensors", []):
            tid = int(entry["compiled_tensor_id"])
            size = _size_bytes_from_entry(entry)
            gname = str(entry.get("graph_input_name", f"g{gid}_tid{tid}"))
            uses: list[int] = []
            for ulid in entry.get("used_by_launch_ids", []):
                lid_int = int(ulid)
                hit = per_graph_launch_to_task[gid].get(lid_int)
                if hit is None:
                    continue
                if replicate_uses:
                    uses.extend(hit)
                else:
                    uses.append(hit)
            uses.sort()
            if not uses:
                continue
            uid = len(tensors)
            # Storage group: read straight from the sidecar.  Bundles
            # produced by the upstream-patched Inductor carry
            # ``storage_group_id`` per ctid (from compile-time
            # ``id(untyped_storage())``).  Bundles without it fall to
            # singleton groups keyed by ``(gid, ctid)`` in the
            # coalescer — same behaviour as no-coalescing.
            sgid = entry.get("storage_group_id", None)
            tensors.append(GlobalTensor(
                uid=uid,
                graph_id=gid,
                compiled_tensor_id=tid,
                graph_input_name=gname,
                size_bytes=size,
                dtype=str(entry.get("dtype", "")),
                entry=entry,
                uses=uses,
                storage_group_id=sgid,
            ))
            for pos in uses:
                tasks[pos].used_tensors.append(uid)

    # Default multiplicity = 1 for any graph not explicitly provided.
    mult: dict[int, int] = {gid: 1 for gid in graph_order}
    if graph_multiplicity:
        for gid, n in graph_multiplicity.items():
            if gid in mult:
                mult[gid] = max(1, int(n))

    total_duration_ns = max((t.end_ns for t in tasks), default=0)

    return UnifiedTimeline(
        tasks=tasks,
        tensors=tensors,
        graph_order=graph_order,
        per_graph_tasks=per_graph_tasks,
        per_graph_launch_to_task=per_graph_launch_to_task,
        per_graph_hash=per_graph_hash,
        total_duration_ns=total_duration_ns,
        cpu_per_launch_ns=cpu_per_launch_ns,
        graph_multiplicity=mult,
    )


def emit_h2d_op(
    tensor: GlobalTensor,
    task: GlobalTask,
    *,
    duration_ns: int,
    reason: str,
    after_launch_id: int = -1,
) -> dict[str, Any]:
    """Build a vram_prefetch_h2d op dict routed to the tensor's graph."""
    return {
        "type": "vram_prefetch_h2d",
        "tensor_name": tensor.graph_input_name,
        "tensor_kind": "WEIGHT",
        "before_node": int(task.node_id),
        "after_node": -1,
        "duration_ns": int(duration_ns),
        "size_bytes": int(tensor.size_bytes),
        "reason": reason,
        "before_launch_id": int(task.launch_id),
        "after_launch_id": int(after_launch_id),
        "compiled_tensor_id": int(tensor.compiled_tensor_id),
        "compiled_graph_id": int(tensor.graph_id),
        "compilation_hash": "",
        "compiled_graph_input_name": tensor.graph_input_name,
    }


def emit_d2h_op(
    tensor: GlobalTensor,
    task: GlobalTask,
    *,
    duration_ns: int,
    reason: str,
) -> dict[str, Any]:
    """Build a vram_evict_d2h op dict anchored in the tensor's graph."""
    return {
        "type": "vram_evict_d2h",
        "tensor_name": tensor.graph_input_name,
        "tensor_kind": "WEIGHT",
        "after_node": int(task.node_id),
        "duration_ns": int(duration_ns),
        "size_bytes": int(tensor.size_bytes),
        "reason": reason,
        "after_launch_id": int(task.launch_id),
        "compiled_tensor_id": int(tensor.compiled_tensor_id),
        "compiled_graph_id": int(tensor.graph_id),
        "compilation_hash": "",
        "compiled_graph_input_name": tensor.graph_input_name,
    }


def emit_cold_start(
    tensor: GlobalTensor,
    first_task: GlobalTask,
    *,
    reason: str,
) -> dict[str, Any]:
    """Build a cold_start_prefetch entry (fires once at module init)."""
    return {
        "tensor_name": tensor.graph_input_name,
        "tensor_kind": "WEIGHT",
        "reason": reason,
        "miss_ns": 0,
        "attach_before_node": int(first_task.node_id),
        "eager_start": True,
        "before_launch_id": int(first_task.launch_id),
        "compiled_tensor_id": int(tensor.compiled_tensor_id),
        "compiled_graph_id": int(tensor.graph_id),
        "compilation_hash": "",
        "compiled_graph_input_name": tensor.graph_input_name,
    }


def build_node_timeline(tl: UnifiedTimeline, trace: Trace) -> tuple[list[int], list[int]]:
    """Return (node_starts, node_ends) in trace.node_map iteration order.

    Compute nodes covered by the timeline use their assigned start/end; all
    other nodes are filled with cumulative compute_time_micros so the output
    JSON's ``nodes`` section stays consistent with the runtime schema.
    """
    task_by_node: dict[int, GlobalTask] = {
        t.node_id: t for t in tl.tasks
    }
    starts: list[int] = []
    ends: list[int] = []
    t = 0
    for nid, node in trace.node_map.items():
        if nid in task_by_node:
            tk = task_by_node[nid]
            starts.append(tk.start_ns)
            ends.append(tk.end_ns)
            t = tk.end_ns
            continue
        starts.append(t)
        dur = int(node.compute_time_micros * 1_000)
        t += dur
        ends.append(t)
    return starts, ends
