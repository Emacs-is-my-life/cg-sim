"""Build a compiled-tensor scheduling problem from a cg-sim ``Trace``.

Weight streaming schedulers plan H2D prefetches and D2H evictions against
PyTorch compile's ``compiled_tensor_id`` / ``compiled_launch_id`` namespace.
The glue from profile-level ids (cg-sim ``Trace``) to compile-level ids is:

* Each GPU ``Node`` carries ``args['compiled_launch_id']`` (populated by the
  PyTorch profiler via ws_launch markers; -1/None means "no launch").
* ``compiled_tensor_map_graph0.json`` sidecar provides
  ``(compiled_tensor_id, graph_input_name, used_by_launch_ids, shape, dtype)``.
* ``compiled_launch_map_graph0.json`` sidecar provides
  ``(compiled_launch_id, kernel_name)``.

This module produces a flat ``CompiledTensorProblem`` keyed by compiled
tensor id, with per-tid size in bytes, GPU node indices where the tid is
used, and a timeline (``node_starts`` / ``node_ends`` in ns).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sim.core.trace import Node, Trace


_DTYPE_BYTES = {
    "torch.float16": 2, "torch.bfloat16": 2,
    "torch.float32": 4, "torch.float64": 8,
    "torch.int8": 1, "torch.uint8": 1,
    "torch.int16": 2, "torch.int32": 4, "torch.int64": 8,
    "torch.bool": 1,
}


_GPU_RESOURCE_KINDS = frozenset({"gpu_stream", "gpu"})


def _is_gpu_node(node: Node) -> bool:
    rk = str(node.args.get("resource_kind") or "")
    if rk:
        return rk in _GPU_RESOURCE_KINDS
    device_type = str(node.args.get("device_type") or "").upper()
    if device_type in ("CUDA", "GPU"):
        return True
    return False


def _size_bytes_from_entry(entry: dict[str, Any]) -> int:
    bpe = _DTYPE_BYTES.get(entry["dtype"], 0)
    if bpe == 0:
        return 0
    numel = 1
    for s in entry["shape"]:
        numel *= int(s)
    return numel * bpe


@dataclass
class CompiledTensorProblem:
    """Flat view of the scheduling problem in compile-tensor space."""

    tid_to_size: dict[int, int] = field(default_factory=dict)
    tid_to_uses: dict[int, list[int]] = field(default_factory=dict)  # GPU node ids in exec order
    kernel_inputs: dict[int, list[int]] = field(default_factory=dict)  # GPU node id -> list of tids
    first_use_by_tid: dict[int, int] = field(default_factory=dict)
    tid_entry: dict[int, dict[str, Any]] = field(default_factory=dict)  # tensor_map entry per tid
    gpu_nodes_in_order: list[int] = field(default_factory=list)
    node_starts: list[int] = field(default_factory=list)
    node_ends: list[int] = field(default_factory=list)
    node_to_launch_id: dict[int, int] = field(default_factory=dict)
    iter_length: int = 0
    compilation_hash: str = ""


def infer_node_to_launch_id(
    trace: Trace, launch_map: dict[str, Any] | None
) -> tuple[dict[int, int], str]:
    """Map trace ``Node.id`` → ``compiled_launch_id``.

    Prefers per-node ``args['compiled_launch_id']`` set by the profiler. If
    those are absent but the launch_map sidecar has a matching
    GPU-node count, falls back to positional pairing.
    """
    compilation_hash = str((launch_map or {}).get("compilation_hash", ""))

    gpu_node_ids = [
        nid for nid, n in trace.node_map.items() if _is_gpu_node(n)
    ]
    annotated = {
        nid: int(trace.node_map[nid].args["compiled_launch_id"])
        for nid in gpu_node_ids
        if trace.node_map[nid].args.get("compiled_launch_id") is not None
        and int(trace.node_map[nid].args["compiled_launch_id"]) >= 0
    }
    if annotated:
        return annotated, compilation_hash

    if launch_map is None:
        return {}, compilation_hash

    launches = launch_map.get("launches", [])
    if len(launches) != len(gpu_node_ids):
        raise RuntimeError(
            f"[cg-sim graph_modifiers] GPU-node count {len(gpu_node_ids)} "
            f"!= launch count {len(launches)}; re-profile with ws_launch "
            f"markers so nodes carry compiled_launch_id."
        )
    return (
        {nid: int(launches[i]["compiled_launch_id"]) for i, nid in enumerate(gpu_node_ids)},
        compilation_hash,
    )


def _compute_wall_timeline(
    trace: Trace,
    node_to_launch_id: dict[int, int],
    cpu_per_launch_ns: int,
    profiler_overhead_per_event_ns: int = 0,
) -> tuple[list[int], list[int]]:
    """Compact node durations back-to-back per trace order.

    Per-event durations from the trace are profiler-clean — the
    profiler bookkeeping lives in the GAPS BETWEEN events, not inside
    them. So laying events back-to-back at their durations gives an
    iter wall close to the real unprofiled wall, no calibration data
    needed.

    Previously this added ``cpu_per_launch_ns`` between launches, which
    was a synthetic CPU-dispatch inflation that double-counted the
    profiler-injected gap and over-stated wall by ~70%. The
    ``cpu_per_launch_ns`` parameter is now ignored (kept in the signature
    for caller compatibility).

    ``profiler_overhead_per_event_ns`` is still honored to subtract any
    leakage that DID land inside CPU events (~0.3-0.5 µs / event in
    practice). GPU events don't get this subtraction.

    Returns (starts, ends) arrays indexed by the *position* in
    ``trace.node_map`` iteration (not by Node.id).
    """
    starts: list[int] = []
    ends: list[int] = []
    t = 0
    for nid, node in trace.node_map.items():
        starts.append(t)
        dur_ns = int(node.compute_time_micros * 1_000)
        if profiler_overhead_per_event_ns > 0 and not _is_gpu_node(node):
            dur_ns = max(0, dur_ns - profiler_overhead_per_event_ns)
        t += dur_ns
        ends.append(t)
    return starts, ends


def _cpu_event_count(trace: Trace) -> int:
    """Return the number of CPU-side nodes in the trace (non-GPU)."""
    return sum(1 for n in trace.node_map.values() if not _is_gpu_node(n))


def _solve_overhead_for_wall(
    trace: Trace, iter_wall_ns: int,
) -> int:
    """Back-solve ``profiler_overhead_per_event_ns`` to match a target wall.

    Given a target iter wall (e.g., measured from a non-profiled run),
    compute the per-CPU-event subtraction that brings the summed
    timeline down to that target. Returns 0 if the target is already
    above the raw sum.
    """
    num_cpu = _cpu_event_count(trace)
    if num_cpu == 0:
        return 0
    raw = sum(int(n.compute_time_micros * 1_000) for n in trace.node_map.values())
    delta = raw - int(iter_wall_ns)
    if delta <= 0:
        return 0
    return delta // num_cpu


@dataclass
class MultiGraphProblem:
    """Per-graph ``CompiledTensorProblem``s plus graph execution metadata.

    ``graph_order`` is the execution order of graph_ids (earliest first kernel
    first). ``per_graph`` maps graph_id → CompiledTensorProblem built from
    only that graph's nodes/tensors/sidecars.
    """
    per_graph: dict[int, CompiledTensorProblem] = field(default_factory=dict)
    graph_order: list[int] = field(default_factory=list)
    graph_start_ns: dict[int, int] = field(default_factory=dict)
    graph_end_ns: dict[int, int] = field(default_factory=dict)


def _graph_order_from_trace(
    trace: Trace, node_to_graph_id: dict[int, int],
) -> tuple[list[int], dict[int, int], dict[int, int]]:
    """Compute execution order of graphs + each graph's first/last kernel time.

    Order is determined by the FIRST kernel ``start_ns`` per graph. Uses
    the profile-recorded ``start_ns`` / ``end_ns`` from the node args;
    falls back to synthesized timeline if either is missing.
    """
    # Synthesize a fallback timeline from compute_time_micros.
    synth_starts: dict[int, int] = {}
    synth_ends: dict[int, int] = {}
    t = 0
    for nid, node in trace.node_map.items():
        synth_starts[nid] = t
        dur_ns = int(node.compute_time_micros * 1_000)
        t += dur_ns
        synth_ends[nid] = t

    first_time: dict[int, int] = {}
    last_time: dict[int, int] = {}
    for nid, node in trace.node_map.items():
        gid = node_to_graph_id.get(nid, -1)
        if gid < 0:
            continue
        start_raw = node.args.get("start_ns") or 0
        end_raw = node.args.get("end_ns") or 0
        start = int(start_raw) if start_raw else synth_starts[nid]
        end = int(end_raw) if end_raw else synth_ends[nid]
        if gid not in first_time or start < first_time[gid]:
            first_time[gid] = start
        if gid not in last_time or end > last_time[gid]:
            last_time[gid] = end
    order = sorted(first_time.keys(), key=lambda g: first_time[g])
    return order, first_time, last_time


def build_multi_graph_problem(
    trace: Trace,
    sidecars: "MultiGraphSidecars",  # forward ref
    *,
    cpu_per_launch_ns: int = 0,
    profiler_overhead_per_event_ns: int = 0,
    iter_wall_ns_override: int | None = None,
) -> MultiGraphProblem:
    """Build a ``MultiGraphProblem`` covering every graph in the bundle.

    Each per-graph sub-problem is a ``CompiledTensorProblem`` built from only
    that graph's nodes + tensors. This mirrors ``build_compiled_tensor_problem``
    but restricts the node set per graph.
    """
    from .loader import MultiGraphSidecars  # local import for type purity

    # Per-node graph membership: from cg-sim Node args['compiled_graph_id'].
    node_to_graph_id: dict[int, int] = {}
    for nid, node in trace.node_map.items():
        gid = node.args.get("compiled_graph_id")
        if gid is None:
            continue
        gid_int = int(gid)
        if gid_int < 0:
            continue
        node_to_graph_id[nid] = gid_int

    # Per-node launch_id via the first graph's sidecar (for infer_node_to_launch_id logic).
    # We do it per-graph below.
    mg = MultiGraphProblem()
    for gid, launch_map in sidecars.launch_maps.items():
        tensor_map = sidecars.tensor_maps.get(gid)
        if tensor_map is None:
            continue
        sub_prob = build_compiled_tensor_problem(
            trace, launch_map, tensor_map,
            cpu_per_launch_ns=cpu_per_launch_ns,
            profiler_overhead_per_event_ns=profiler_overhead_per_event_ns,
            iter_wall_ns_override=iter_wall_ns_override,
        )
        mg.per_graph[gid] = sub_prob

    # Merged node_to_launch_id across all graphs (for global launch_id lookup).
    node_to_launch_id: dict[int, int] = {}
    for gid, sub in mg.per_graph.items():
        node_to_launch_id.update(sub.node_to_launch_id)

    mg.graph_order, mg.graph_start_ns, mg.graph_end_ns = _graph_order_from_trace(
        trace, node_to_graph_id,
    )
    return mg


def build_compiled_tensor_problem(
    trace: Trace,
    launch_map: dict[str, Any] | None,
    tensor_map: dict[str, Any] | None,
    *,
    cpu_per_launch_ns: int = 0,
    profiler_overhead_per_event_ns: int = 0,
    iter_wall_ns_override: int | None = None,
) -> CompiledTensorProblem:
    """Build a compile-tensor-space problem from cg-sim Trace + sidecars.

    Skips tensor_map entries whose ``used_by_launch_ids`` reference launches
    absent from the launch_map (eg. unresolved non-compiled GPU kernels).
    The resulting problem's ``gpu_nodes_in_order`` includes only the node
    ids that actually participate in compiled launches.
    """
    if tensor_map is None:
        raise ValueError(
            "build_compiled_tensor_problem requires compiled_tensor_map sidecar"
        )
    if launch_map is None:
        raise ValueError(
            "build_compiled_tensor_problem requires compiled_launch_map sidecar"
        )

    node_to_launch_id, compilation_hash = infer_node_to_launch_id(trace, launch_map)

    # Resolve calibration: explicit override wins; otherwise use the
    # calibration constant as-is (0 = legacy behavior).
    if iter_wall_ns_override is not None and iter_wall_ns_override > 0:
        effective_overhead = _solve_overhead_for_wall(trace, iter_wall_ns_override)
    else:
        effective_overhead = int(profiler_overhead_per_event_ns)

    node_ids_in_order = list(trace.node_map.keys())
    starts_by_pos, ends_by_pos = _compute_wall_timeline(
        trace, node_to_launch_id, cpu_per_launch_ns,
        profiler_overhead_per_event_ns=effective_overhead,
    )
    pos_of_node: dict[int, int] = {nid: i for i, nid in enumerate(node_ids_in_order)}

    # launch_id -> first node id bearing that launch (earliest in order)
    launch_to_node: dict[int, int] = {}
    for nid in node_ids_in_order:
        lid = node_to_launch_id.get(nid, -1)
        if lid < 0 or lid in launch_to_node:
            continue
        launch_to_node[lid] = nid

    prob = CompiledTensorProblem()
    prob.compilation_hash = compilation_hash
    prob.node_to_launch_id = node_to_launch_id
    prob.node_starts = starts_by_pos
    prob.node_ends = ends_by_pos

    kernel_inputs: dict[int, list[int]] = {}
    for entry in tensor_map.get("tensors", []):
        tid = int(entry["compiled_tensor_id"])
        size = _size_bytes_from_entry(entry)
        prob.tid_to_size[tid] = size
        prob.tid_entry[tid] = entry
        launches = sorted(int(x) for x in entry.get("used_by_launch_ids", []))
        node_ids_for_tid: list[int] = []
        for lid in launches:
            n = launch_to_node.get(lid)
            if n is not None:
                node_ids_for_tid.append(n)
        # Sort node ids by their position in node_map (== execution order).
        node_ids_for_tid.sort(key=lambda nid: pos_of_node[nid])
        if not node_ids_for_tid:
            continue
        prob.tid_to_uses[tid] = node_ids_for_tid
        prob.first_use_by_tid[tid] = node_ids_for_tid[0]
        for nid in node_ids_for_tid:
            kernel_inputs.setdefault(nid, []).append(tid)

    prob.kernel_inputs = kernel_inputs
    prob.gpu_nodes_in_order = sorted(
        kernel_inputs.keys(), key=lambda nid: pos_of_node[nid]
    )
    if prob.gpu_nodes_in_order:
        last_node = prob.gpu_nodes_in_order[-1]
        prob.iter_length = ends_by_pos[pos_of_node[last_node]]
    return prob
