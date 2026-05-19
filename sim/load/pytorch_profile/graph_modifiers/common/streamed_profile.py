"""Per-launch streamed profile view of a cg-sim ``Trace``.

The default ``CompiledTensorProblem`` collapses the profile into a single 1D
timeline of node durations, which serializes everything onto one lane and
invents stall that does not exist on real hardware (CPU dispatch and GPU
execution happen concurrently; PCIe H2D runs on its own DMA engine).

``StreamedProfile`` preserves the information Kineto already captures:

* CPU-side dispatch start/end for each ``cudaLaunchKernel``
* GPU-side kernel start/end (ground-truth device timestamps)
* CUDA stream id on each GPU event
* ``correlation_id`` pairing CPU launch ↔ GPU kernel

It exposes one ``Launch`` per compile-level ``compiled_launch_id`` along with
the per-lane occupancy windows, so downstream schedulers can model explicit
compute / h2d / d2h / cpu lanes instead of one linear timeline.

No new profile data is collected; this module only re-reads the existing
``Trace`` with a richer schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sim.core.trace import Node, Trace

from .problem import (
    CompiledTensorProblem,
    _DTYPE_BYTES,
    _is_gpu_node,
    _size_bytes_from_entry,
    build_compiled_tensor_problem,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Launch:
    """One compile-level launch as a set of CPU + GPU events.

    A launch typically corresponds to a single ``cudaLaunchKernel`` CPU call
    paired with one GPU kernel via Kineto's ``correlation_id``. A launch may
    wrap multiple GPU kernels in the rare case where the compile sidecar
    groups them under one ``compiled_launch_id`` (e.g. cuBLAS split-k).
    """

    launch_id: int

    # GPU lane (device-side)
    gpu_exec_start_ns: int
    gpu_exec_end_ns: int
    gpu_stream_id: int

    # CPU lane (host-side dispatch)
    cpu_dispatch_start_ns: int = 0
    cpu_dispatch_end_ns: int = 0
    cpu_dispatch_valid: bool = False

    # Compile data
    kernel_name: str = ""
    input_tids: list[int] = field(default_factory=list)

    # Book-keeping: all trace node ids (GPU + CPU) that belong to this launch.
    gpu_node_ids: list[int] = field(default_factory=list)
    cpu_node_ids: list[int] = field(default_factory=list)


@dataclass
class StreamedProfile:
    """Per-launch + per-stream view of the profile.

    ``launches`` are sorted by ``gpu_exec_start_ns`` so that downstream
    simulators can walk them in execution order. ``iter_wall_ns`` is the
    observed wall-clock iteration length from Kineto — distinct from the
    sum-of-durations that ``CompiledTensorProblem.iter_length`` returns.
    """

    launches: list[Launch] = field(default_factory=list)
    tid_to_size: dict[int, int] = field(default_factory=dict)
    tid_to_uses: dict[int, list[int]] = field(default_factory=dict)  # tid -> launch_ids
    tid_entry: dict[int, dict[str, Any]] = field(default_factory=dict)

    iter_wall_ns: int = 0                # raw Kineto wall (max-min of GPU ts)
    iter_wall_calibrated_ns: int = 0     # post-profiler-overhead estimate
    gpu_busy_ns: int = 0                 # sum of GPU kernel durations (no gaps)
    cpu_dispatch_ns: int = 0             # sum of CPU launch-kernel durations
    profiler_overhead_per_event_ns: int = 0

    compilation_hash: str = ""

    # Underlying CompiledTensorProblem kept for compatibility with legacy
    # consumers (output writers, schedule JSON emission).
    ct_problem: CompiledTensorProblem | None = None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


_LAUNCH_CPU_OPS = (
    "cudaLaunchKernel",
    "cudaLaunchCooperativeKernel",
    "cudaMemcpyAsync",
    "cudaMemsetAsync",
    "cuLaunchKernel",            # Triton via driver API
    "cuLaunchKernelEx",
    "cuLaunchCooperativeKernel",
    "cuMemcpyAsync",
    "cuMemsetDAsync",
)


def _is_launch_cpu_op(node: Node) -> bool:
    op = str(node.args.get("op_name") or "")
    return any(op.startswith(prefix) for prefix in _LAUNCH_CPU_OPS)


def _pair_cpu_for_gpu(trace: Trace) -> dict[int, Node]:
    """Return {gpu_node_id: paired_cpu_node} via correlation_id.

    For each GPU node with a ``correlation_id``, picks the CPU node that
    shares that id AND matches a known launch op (``cudaLaunchKernel`` etc.).
    If multiple candidates exist, picks the one whose ``end_ns`` is closest
    to but not after the GPU event's ``start_ns`` (that is, the actual
    launching call — not an encompassing wrapper with the same corr id).
    """
    cpu_by_corr: dict[int, list[Node]] = {}
    for node in trace.node_map.values():
        if _is_gpu_node(node):
            continue
        corr = node.args.get("correlation_id") or 0
        if not corr:
            continue
        if not _is_launch_cpu_op(node):
            continue
        cpu_by_corr.setdefault(int(corr), []).append(node)

    paired: dict[int, Node] = {}
    for nid, node in trace.node_map.items():
        if not _is_gpu_node(node):
            continue
        corr = node.args.get("correlation_id") or 0
        if not corr:
            continue
        candidates = cpu_by_corr.get(int(corr), [])
        if not candidates:
            continue
        gpu_start = node.args.get("start_ns") or 0
        # Prefer the CPU op whose end is <= gpu_start and closest to it.
        pre = [c for c in candidates if (c.args.get("end_ns") or 0) <= gpu_start]
        if pre:
            best = max(pre, key=lambda c: c.args.get("end_ns") or 0)
        else:
            # Fall back to the earliest-starting candidate (best-effort).
            best = min(candidates, key=lambda c: c.args.get("start_ns") or 0)
        paired[nid] = best
    return paired


def build_streamed_profile(
    trace: Trace,
    launch_map: dict[str, Any],
    tensor_map: dict[str, Any],
    *,
    profiler_overhead_per_event_ns: int = 0,
    iter_wall_ns_override: int | None = None,
) -> StreamedProfile:
    """Construct a ``StreamedProfile`` from a cg-sim ``Trace`` + sidecars.

    The compile sidecars supply ``compiled_launch_id`` → kernel grouping and
    ``compiled_tensor_id`` → used_by_launch_ids. We join these against the
    per-node ``correlation_id`` pairing to form ``Launch`` records with both
    CPU and GPU timestamps.
    """
    ct_prob = build_compiled_tensor_problem(
        trace, launch_map, tensor_map,
        cpu_per_launch_ns=0,
        profiler_overhead_per_event_ns=profiler_overhead_per_event_ns,
        iter_wall_ns_override=iter_wall_ns_override,
    )

    cpu_pair = _pair_cpu_for_gpu(trace)

    # Group GPU nodes by compiled_launch_id.
    gpu_nodes_by_launch: dict[int, list[int]] = {}
    for gnid in ct_prob.gpu_nodes_in_order:
        lid = ct_prob.node_to_launch_id.get(gnid, -1)
        if lid < 0:
            continue
        gpu_nodes_by_launch.setdefault(lid, []).append(gnid)

    kernel_name_by_lid: dict[int, str] = {}
    for entry in (launch_map or {}).get("launches", []):
        lid = int(entry.get("compiled_launch_id", -1))
        kernel_name_by_lid[lid] = str(entry.get("kernel_name", ""))

    launches: list[Launch] = []
    cpu_dispatch_total = 0
    gpu_busy_total = 0
    min_gpu_start = 1 << 62
    max_gpu_end = 0

    for lid, gpu_nids in gpu_nodes_by_launch.items():
        # Aggregate GPU timing across this launch's kernels.
        starts = [trace.node_map[n].args.get("start_ns") or 0 for n in gpu_nids]
        ends = [trace.node_map[n].args.get("end_ns") or 0 for n in gpu_nids]
        gs = min(starts)
        ge = max(ends)
        sid = int(trace.node_map[gpu_nids[0]].args.get("stream_id") or 0)
        gpu_busy_total += sum(e - s for s, e in zip(starts, ends))
        min_gpu_start = min(min_gpu_start, gs)
        max_gpu_end = max(max_gpu_end, ge)

        # Aggregate CPU timing: the earliest start + latest end across the
        # paired CPU launch ops for all kernels in this launch.
        cpu_nids: list[int] = []
        cpu_starts: list[int] = []
        cpu_ends: list[int] = []
        for gnid in gpu_nids:
            cpu_node = cpu_pair.get(gnid)
            if cpu_node is None:
                continue
            cpu_nids.append(cpu_node.id)
            cpu_starts.append(cpu_node.args.get("start_ns") or 0)
            cpu_ends.append(cpu_node.args.get("end_ns") or 0)
        if cpu_nids:
            cs, ce = min(cpu_starts), max(cpu_ends)
            cpu_dispatch_total += sum(e - s for s, e in zip(cpu_starts, cpu_ends))
            cpu_valid = True
        else:
            cs, ce = 0, 0
            cpu_valid = False

        inputs = sorted(set(tid for n in gpu_nids for tid in ct_prob.kernel_inputs.get(n, [])))
        launches.append(Launch(
            launch_id=int(lid),
            gpu_exec_start_ns=int(gs),
            gpu_exec_end_ns=int(ge),
            gpu_stream_id=sid,
            cpu_dispatch_start_ns=int(cs),
            cpu_dispatch_end_ns=int(ce),
            cpu_dispatch_valid=cpu_valid,
            kernel_name=kernel_name_by_lid.get(int(lid), ""),
            input_tids=inputs,
            gpu_node_ids=gpu_nids,
            cpu_node_ids=cpu_nids,
        ))

    launches.sort(key=lambda L: L.gpu_exec_start_ns)

    # Re-map tid_to_uses to launch_id (instead of trace node id).
    tid_to_uses_launches: dict[int, list[int]] = {}
    for tid, node_ids in ct_prob.tid_to_uses.items():
        lids = sorted(set(
            ct_prob.node_to_launch_id.get(n, -1) for n in node_ids
        ))
        tid_to_uses_launches[tid] = [lid for lid in lids if lid >= 0]

    iter_wall_ns = max(0, max_gpu_end - min_gpu_start) if launches else 0

    # Back-solve the effective overhead that was applied (for diagnostics).
    if iter_wall_ns_override is not None and iter_wall_ns_override > 0:
        from .problem import _solve_overhead_for_wall
        effective_overhead = _solve_overhead_for_wall(trace, iter_wall_ns_override)
    else:
        effective_overhead = int(profiler_overhead_per_event_ns)

    return StreamedProfile(
        launches=launches,
        tid_to_size=dict(ct_prob.tid_to_size),
        tid_to_uses=tid_to_uses_launches,
        tid_entry=dict(ct_prob.tid_entry),
        iter_wall_ns=int(iter_wall_ns),
        iter_wall_calibrated_ns=int(ct_prob.iter_length),
        gpu_busy_ns=int(gpu_busy_total),
        cpu_dispatch_ns=int(cpu_dispatch_total),
        profiler_overhead_per_event_ns=effective_overhead,
        compilation_hash=ct_prob.compilation_hash,
        ct_problem=ct_prob,
    )
