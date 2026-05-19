"""Oracle MILP scheduler.

Variables
---------

For each evictable tensor i, a binary decision

    z_i ∈ {0, 1}

with the convention ``z_i = 1`` ⇒ keep resident (cold-start, no
streaming), ``z_i = 0`` ⇒ evict and reload at every use (cyclic
streaming). One continuous ``P`` upper-bounds the peak VRAM. The
solver is a true integer MILP (HiGHS branch-and-bound) — the previous
implementation solved the LP relaxation and rounded at 0.5, which
admitted fractional solutions whose threshold-rounding diverged from
the LP optimum.

Peak constraint (per-launch)
----------------------------

For each unique compiled launch ``(g, ℓ)`` in the timeline, every
tensor consumed by that launch is necessarily resident at that
moment, regardless of ``z``. Tensors not consumed contribute their
size only when kept (z=1). So:

    P  ≥  A_{g,ℓ}  +  Σ_{i : ℓ ∉ used_by_launch_ids(i)}  z_i · s_i

    A_{g,ℓ}  =  Σ_{i : ℓ ∈ used_by_launch_ids(i)}  s_i        (constant)

This is one constraint per distinct ``(g, ℓ)``. The previous
formulation bucketed time and treated a tensor as "active" in every
bucket touching any of its uses — which, for graphs that fire many
times per pipeline call (LLM decode loop, SDXL UNet steps), made
every tensor active in every bucket and forced the LP to report
``peak = total_weights`` regardless of ``z``. Per-launch constraints
remove that overcounting: a tensor is "active" only at the launches
that actually consume it.

H2D bandwidth (per-rank deadlines, opt-in)
------------------------------------------

There is an optional FIFO drain model — sort feasible tensors by
first-use time and require cumulative H2D duration ≤ deadline at each
rank — controlled by ``disable_per_rank_deadlines`` (default True).
The model is a fictional EDF that does not match the runtime's actual
queue behavior; leaving it off lets the LP optimize peak directly and
defers bandwidth feasibility to simulation. Re-enable it only when
you want a hard upper bound on H2D pressure.

Objective
---------

``minimize P``. No slack variables: kept-set selection is the only
degree of freedom; reload placement is handled in emission. With the
trivial fallback ``z_i = 1`` ∀i always feasible, the solver always
returns at least the "keep everything" solution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from sim.core.trace import Trace

from sim.load.pytorch_profile.graph_modifiers.common import (
    HwParams,
    MultiGraphSidecars,
    NeutralColdStart,
    NeutralEvict,
    NeutralPrefetch,
    NeutralSchedule,
    NeutralTensor,
    UnifiedTimeline,
    build_node_timeline,
    build_unified_timeline,
    effective_h2d_bw,
    emit_cold_start,
    emit_d2h_op,
    emit_h2d_op,
    neutral_to_pytorch,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class Schedule:
    """Compat wrapper — a neutral schedule + a trace for pytorch conversion."""
    neutral: NeutralSchedule
    node_starts: list[int]
    node_ends: list[int]
    # Derived fields kept for backward compatibility with callers that read
    # ``.io_operations`` / ``.cold_start_prefetches`` / ``.summary`` directly.
    io_operations: list[dict[str, Any]]
    cold_start_prefetches: list[dict[str, Any]]
    summary: dict[str, Any]
    compilation_hash: str


# ---------------------------------------------------------------------------
# Pre-filter: which tensors CAN be evicted at all?
# ---------------------------------------------------------------------------


def _tensor_feasible_to_evict(
    tensor, tl: UnifiedTimeline, hw: HwParams,
) -> tuple[bool, int]:
    """Return (feasible, largest_gap_ns).

    A tensor is evict-feasible if at least one of its gaps is large
    enough to fit a round-trip transfer. Gaps considered:
      * intra-iter gaps between consecutive uses
      * cross-iter gap from last-use → (iter_end → iter_start) → first-use
    """
    if not tensor.uses:
        return (False, 0)
    bw = max(effective_h2d_bw(hw), 1e-9)
    d_h2d = hw.h2d_latency_ns + int(tensor.size_bytes / bw)
    d_d2h = hw.d2h_latency_ns + int(tensor.size_bytes / max(hw.d2h_bw, 1e-9))
    roundtrip = d_h2d + d_d2h

    largest_gap = 0
    for i in range(len(tensor.uses) - 1):
        prev_end = tl.tasks[tensor.uses[i]].end_ns
        next_start = tl.tasks[tensor.uses[i + 1]].start_ns
        gap = next_start - prev_end
        largest_gap = max(largest_gap, gap)
        if gap >= roundtrip:
            return (True, gap)
    # Cross-iter gap.
    last_end = tl.tasks[tensor.uses[-1]].end_ns
    first_start = tl.tasks[tensor.uses[0]].start_ns
    cross_gap = (tl.total_duration_ns - last_end) + first_start
    largest_gap = max(largest_gap, cross_gap)
    if cross_gap >= roundtrip:
        return (True, cross_gap)
    return (False, largest_gap)


# ---------------------------------------------------------------------------
# Core MILP
# ---------------------------------------------------------------------------


@dataclass
class _MilpResult:
    # z_solution[uid] is the continuous LP value in [0, 1]:
    #   1.0  → keep (cold-start only, no eviction)
    #   0.0  → fully evict every iter
    #   0<z<1 → partial (multi-iter only): keep in fraction z of iters
    z_solution: dict[int, float]
    forced_keep: set[int]
    feasible_uids: list[int]
    peak_bytes: int
    pcie_used_bytes: int
    pcie_budget_bytes: int
    solver_status: str
    diagnostics: dict[str, Any]


def _compute_live_intermediates_per_launch(
    trace: Trace,
    tl: UnifiedTimeline,
) -> tuple[dict[tuple[int, int], int], int]:
    """Compute trace-derived VRAM contributions the MILP doesn't see directly.

    Returns ``(launch_intermediate_bytes, extra_static_bytes)``.

    * ``launch_intermediate_bytes[(g, l)]`` — peak live INTERMEDIATE
      bytes during any firing of compiled launch ``(g, l)``.  An
      intermediate is "live" from its first-producer task through its
      last-consumer task; we walk tasks in execution order and sample
      the live-byte total at each task, then take the max per ``(g, l)``.

    * ``extra_static_bytes`` — VRAM-resident WEIGHT/INPUT/LEAF bytes
      from the trace that the MILP's sidecar view (``tl.tensors``)
      misses.  Computed as ``Σ trace.tensor_map[t].size_bytes`` for
      types in ``initial_tensor_types`` minus ``Σ tl.tensors.size_bytes``.
      Tensors in this delta can't be streamed — the compiled tensor
      map didn't expose them — so they form a constant residency
      floor below every launch.

    Both pieces are constants from the LP's view: they raise the
    per-launch peak baseline so the MILP's claimed P matches what
    the simulator's allocator actually places.
    """
    initial_types = {"WEIGHT", "INPUT", "LEAF"}

    # Storage-group dedup is essential here. PyTorch traces typically
    # include alias Tensor objects (e.g. q, k, v from a `qkv.chunk(3)`
    # call are three distinct Tensor.ids backed by one underlying
    # storage). The simulator's allocator claims one VRAM region per
    # storage group, but a naive per-Tensor.id sum would count each
    # alias separately — on SDXL UNet, that inflates intermediate live
    # bytes by ~3×.

    def _sgid(tensor, tid):
        return (
            tensor.args.get("storage_group_id")
            or tensor.args.get("storage_id")
            or ("tid", tid)  # tagged so it never collides with a real sgid
        )

    # Initial-placed (non-streaming) bytes, deduped by storage group.
    initial_storage_bytes: dict[Any, int] = {}
    # Intermediate storage groups: sgid -> size_bytes (constant per group;
    # we take the max across aliases as a safety, though they should match).
    intermediate_sgid_size: dict[Any, int] = {}
    # Map every tensor.id to its sgid for fast lookup during the sweep.
    tid_to_sgid: dict[int, Any] = {}
    for tid, tensor in trace.tensor_map.items():
        ttype = tensor.args.get("tensor_type")
        sgid = _sgid(tensor, tid)
        if ttype == "INTERMEDIATE":
            tid_to_sgid[tid] = sgid
            prev = intermediate_sgid_size.get(sgid, 0)
            intermediate_sgid_size[sgid] = max(prev, int(tensor.size_bytes))
        elif ttype in initial_types:
            prev = initial_storage_bytes.get(sgid, 0)
            initial_storage_bytes[sgid] = max(prev, int(tensor.size_bytes))

    initial_total_bytes = sum(initial_storage_bytes.values())
    sidecar_total_bytes = sum(int(t.size_bytes) for t in tl.tensors)
    extra_static = max(0, initial_total_bytes - sidecar_total_bytes)

    if not intermediate_sgid_size:
        return {}, extra_static

    # First producer / last consumer per *storage group*. Walk tasks in
    # execution order. A group becomes live the first time any of its
    # alias Tensor.ids is produced; it stays live until the last time
    # any of its aliases is consumed.
    first_producer: dict[Any, int] = {}
    last_consumer: dict[Any, int] = {}
    for pos, task in enumerate(tl.tasks):
        node = trace.node_map.get(task.node_id)
        if node is None:
            continue
        for tid in node.output_tensors:
            sgid = tid_to_sgid.get(tid)
            if sgid is not None and sgid not in first_producer:
                first_producer[sgid] = pos
        for tid in node.input_tensors:
            sgid = tid_to_sgid.get(tid)
            if sgid is not None:
                last_consumer[sgid] = pos  # monotonic; final write = last

    # Pre-bucket which sgids start / end at each task position so the
    # sweep is O(num_tasks + num_sgids).
    starts_at: dict[int, list[Any]] = {}
    ends_at: dict[int, list[Any]] = {}
    for sgid, pos in first_producer.items():
        starts_at.setdefault(pos, []).append(sgid)
    for sgid, pos in last_consumer.items():
        ends_at.setdefault(pos, []).append(sgid)

    # Sweep tasks in execution order, maintain live byte total.
    # Convention: add at producer position BEFORE sampling, remove at
    # last-consumer position AFTER sampling — so the consumer's task
    # sees the bytes it's actually about to read.
    live_bytes = 0
    per_launch_peak: dict[tuple[int, int], int] = {}
    for pos, task in enumerate(tl.tasks):
        for sgid in starts_at.get(pos, ()):
            live_bytes += intermediate_sgid_size[sgid]
        key = (int(task.graph_id), int(task.launch_id))
        if live_bytes > per_launch_peak.get(key, 0):
            per_launch_peak[key] = live_bytes
        for sgid in ends_at.get(pos, ()):
            live_bytes -= intermediate_sgid_size[sgid]

    return per_launch_peak, extra_static


def _solve_milp(
    tl: UnifiedTimeline,
    hw: HwParams,
    *,
    locked_uids: set[int],
    duplex: bool,
    time_limit_s: float | None,
    disable_per_rank_deadlines: bool = True,
    bw_overcommit_frac: float = 0.9,
    lp_relaxation: bool = False,
    peak_target_bytes: int | None = None,
    launch_intermediate_bytes: dict[tuple[int, int], int] | None = None,
    extra_static_bytes: int = 0,
) -> _MilpResult:
    # Pre-filter: split tensors into forced_keep vs feasible.
    forced_keep: set[int] = set(locked_uids)
    feasible_uids: list[int] = []
    pre_diag: dict[str, int] = {
        "forced_by_locked": len(locked_uids),
        "forced_by_infeasible_gap": 0,
        "feasible": 0,
        "zero_use": 0,
    }
    for tensor in tl.tensors:
        if tensor.uid in locked_uids:
            continue
        if not tensor.uses:
            pre_diag["zero_use"] += 1
            continue
        feasible, _ = _tensor_feasible_to_evict(tensor, tl, hw)
        if feasible:
            feasible_uids.append(tensor.uid)
            pre_diag["feasible"] += 1
        else:
            forced_keep.add(tensor.uid)
            pre_diag["forced_by_infeasible_gap"] += 1

    nv = len(feasible_uids)
    # Layout: [z_0 .. z_{nv-1}, P]
    P_IDX = nv
    total_vars = nv + 1
    uid_to_var = {uid: i for i, uid in enumerate(feasible_uids)}

    # Per-launch active sets. For each unique (gid, lid) in the
    # timeline, find the tensors consumed by that launch (any tensor
    # with at least one use position whose launch_id == lid in graph
    # gid). Forced-keep tensors contribute their size as a constant
    # baseline (residency floor); feasible tensors contribute either
    # via z_i (if not consumed by the launch) or as part of the
    # constant A_{g,ℓ} (if consumed).
    forced_keep_bytes = sum(
        int(tl.tensors[u].size_bytes) for u in forced_keep
        if u < len(tl.tensors)
    )

    # Distinct launches actually present in the trace, ordered for
    # determinism. Build {(gid, lid): [feasible_var_idx, ...]} for the
    # tensors consumed by each.
    launch_consumers: dict[tuple[int, int], list[int]] = {}
    launch_active_bytes: dict[tuple[int, int], int] = {}
    launch_active_forced_bytes: dict[tuple[int, int], int] = {}
    for tensor in tl.tensors:
        if not tensor.uses:
            continue
        seen_lids: set[tuple[int, int]] = set()
        for pos in tensor.uses:
            t = tl.tasks[pos]
            key = (int(t.graph_id), int(t.launch_id))
            seen_lids.add(key)
        if tensor.uid in feasible_uids:
            var_idx = uid_to_var[tensor.uid]
            for key in seen_lids:
                launch_consumers.setdefault(key, []).append(var_idx)
                launch_active_bytes[key] = (
                    launch_active_bytes.get(key, 0) + int(tensor.size_bytes)
                )
        else:
            for key in seen_lids:
                launch_active_forced_bytes[key] = (
                    launch_active_forced_bytes.get(key, 0)
                    + int(tensor.size_bytes)
                )

    # Distinct constraints by (active-feasible-set, A_{g,ℓ}). Two
    # launches that consume the same set of feasible tensors AND have
    # the same active byte total impose identical constraints — drop
    # the duplicates so HiGHS sees a smaller A matrix. The
    # forced_keep_bytes baseline is the same constant across all
    # launches, so it doesn't affect dedup.
    dedup_keys: dict[tuple[tuple[int, ...], int, int], tuple[int, int]] = {}
    for key, var_indices in launch_consumers.items():
        sig = (
            tuple(sorted(var_indices)),
            launch_active_bytes.get(key, 0),
            launch_active_forced_bytes.get(key, 0),
        )
        dedup_keys.setdefault(sig, key)
    # Also include launches that have NO feasible-tensor consumers but
    # do have forced_keep consumers — they still constrain peak via
    # forced bytes.
    for key in launch_active_forced_bytes:
        if key in launch_consumers:
            continue
        sig = ((), 0, launch_active_forced_bytes.get(key, 0))
        dedup_keys.setdefault(sig, key)

    representative_launches = sorted(dedup_keys.values())

    # Objective + P bounds depend on the operating mode:
    #
    #   peak_target_bytes is None  → "min_peak" (default):
    #       minimise P; P ∈ [0, ∞).  Bandwidth cap and per-launch peak
    #       constraints together pick z to reduce P as much as possible.
    #
    #   peak_target_bytes is set   → "min_streams" (peak-target dual):
    #       minimise total streamed bytes Σ (1 − z_i) · s_i · n_uses(i)
    #       — a proxy for E2E since E2E ≈ baseline + streamed_bytes/bw —
    #       subject to a hard upper bound P ≤ target.  Yields the
    #       schedule with the LEAST streaming that still fits the
    #       peak-VRAM cap; equivalent to asking sim_loop for "lowest
    #       E2E under peak target" but in a single solve.
    c = np.zeros(total_vars, dtype=np.float64)
    if peak_target_bytes is None:
        c[P_IDX] = 1.0
        p_upper: float | None = None
    else:
        # min Σ (1 − z_i) · s_i · n_uses(i)  ≡  −max Σ z_i · s_i · n_uses(i)
        # (drop the constant Σ s_i · n_uses(i) term, which doesn't
        # affect the argmin).  Coefficient on z_i is the negative
        # streaming cost per "kept" choice; coefficient on P is 0.
        for i, uid in enumerate(feasible_uids):
            s = float(tl.tensors[uid].size_bytes)
            n_uses = float(len(tl.tensors[uid].uses))
            c[i] = -s * n_uses
        p_upper = float(peak_target_bytes)

    bounds_list: list[tuple[float, float | None]] = [(0.0, 1.0)] * nv
    bounds_list.append((0.0, p_upper))   # P ≤ peak_target_bytes if set
    if lp_relaxation:
        integrality_arr = None
    else:
        integrality_arr = np.zeros(total_vars, dtype=np.int64)
        integrality_arr[:nv] = 1
        integrality_arr[P_IDX] = 0

    # Sparse coordinate lists for A x ≤ b.
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    ub_list: list[float] = []
    lb_list: list[float] = []

    # Peak constraints: one per distinct launch signature.
    #   P  ≥  A_{g,ℓ}_total  +  Σ_{i ∉ consumers} z_i · s_i
    #   A_{g,ℓ}_total = launch_active_bytes + launch_active_forced_bytes
    #                   + (forced_keep_bytes - launch_active_forced_bytes)
    #                 = forced_keep_bytes + launch_active_bytes
    # (Forced-keep tensors are resident throughout, so they contribute
    # forced_keep_bytes to every launch — the launch-active forced
    # bytes are a subset already counted in forced_keep_bytes.)
    # Linearise as: -P + Σ_{i ∉ consumers} z_i · s_i ≤ -(forced_keep_bytes
    #                                                    + A_active_feasible)
    sizes = np.array(
        [int(tl.tensors[u].size_bytes) for u in feasible_uids],
        dtype=np.float64,
    )

    # ``launch_intermediate_bytes`` adds peak live activations at each
    # launch (computed from the cg-sim trace).  ``extra_static_bytes``
    # adds VRAM-resident weights/inputs the sidecar map can't see.
    # Both are constants from the LP's view: they raise the per-launch
    # peak baseline so the MILP's claimed P reflects total VRAM, not
    # just sidecar weight residency.
    launch_inter = launch_intermediate_bytes or {}
    constant_floor = float(forced_keep_bytes) + float(extra_static_bytes)

    for row_idx, key in enumerate(representative_launches):
        consumers_set = set(launch_consumers.get(key, []))
        a_active_feasible = float(launch_active_bytes.get(key, 0))
        intermediate_bytes = float(launch_inter.get(key, 0))
        rhs = -(constant_floor + a_active_feasible + intermediate_bytes)

        # -P
        rows.append(row_idx)
        cols.append(P_IDX)
        vals.append(-1.0)
        # +z_i * s_i for non-consumers
        for i in range(nv):
            if i in consumers_set:
                continue
            s = sizes[i]
            if s <= 0:
                continue
            rows.append(row_idx)
            cols.append(i)
            vals.append(float(s))
        ub_list.append(rhs)
        lb_list.append(-np.inf)

    # Lower-bound P by the largest single-launch active baseline (sanity:
    # P ≥ max forced_keep_bytes + active_feasible). HiGHS infers this
    # but stating it explicitly gives a tighter root LP relaxation.
    nb = len(representative_launches)

    # Per-rank H2D prefix-sum deadlines (the "oracle" constraint).
    #
    # Sort feasible tensors by first-use global time ascending. Rank r:
    #   Σ_{r' ≤ r} (1 − z_{r'}) · h2d_dur(r')  ≤  T_r − t_stream_start
    #
    # Recall z=1 means "keep resident" (no transfer), z=0 means "evict and
    # reload" (incurs h2d_dur). So (1 − z) is the evict indicator.
    # Linearise: move the constant Σ h2d_dur to RHS:
    #   −Σ z_{r'} · h2d_dur(r')  ≤  deadline − Σ h2d_dur(r')
    #
    # CRITICAL: t_stream_start = 0. The H2D FIFO for this iter starts
    # processing at iter-start, NOT at −iter_length. A prior "runway" only
    # exists if ops carry cross_iter=True so they fire DURING prior iter's
    # compute window — which our emission does NOT do. Using t_stream_start
    # = -iter_length here grants phantom runway that doesn't exist in the
    # real runtime, causing zero-stall claims to fail empirically.
    #
    # ``duplex`` does NOT change H2D bandwidth — it only enables a parallel
    # D2H lane. D2H deadlines are omitted (VRAM reclamation doesn't gate
    # compute as long as the allocator can free pages).
    h2d_bw = max(effective_h2d_bw(hw), 1e-9)
    t_stream_start_ns = 0

    # Per-ITER per-rank deadline (time-indexed within each iter).
    #
    # Single-pipeline-wide ranking misses the within-iter fire-time
    # contention. For multi-iter graphs (e.g., SDXL UNet running 4×),
    # each iter is its own FIFO drain window: reloads queued at the iter
    # boundary must drain in consumer-rank order BEFORE the consumer's
    # within-iter time. The LP must constrain this per iter.
    #
    # For each graph g and each iter k of g:
    #   1. Determine iter k's task subset (the kth chunk of
    #      per_graph_tasks[g]).
    #   2. For each tensor i with z_i = 0, find its first within-iter use
    #      time relative to iter_k_start.
    #   3. Sort tensors by within-iter time → "ranks".
    #   4. Per-rank constraint within iter k:
    #      Σ_{r' ≤ r in iter k} (1 - z_r') · base_h2d_dur(r')  ≤  T_r
    #      (where T_r is the within-iter relative time of rank r).
    #
    # This captures: at any moment within iter k, the cumulative H2D
    # bytes that have been queued must fit into the elapsed time. Early
    # consumers within iter k force tight deadlines.
    h2d_dur_per_i: list[int] = [0] * nv  # base h2d_dur (per single reload)
    for i, uid in enumerate(feasible_uids):
        size = tl.tensors[uid].size_bytes
        h2d_dur_per_i[i] = hw.h2d_latency_ns + int(size / h2d_bw)

    # Build iter-index lookup for each task position.
    # Partition per_graph_tasks[g] uniformly into mult_g groups.
    task_to_iter_in_graph: dict[int, int] = {}
    iter_starts_ns: dict[tuple[int, int], int] = {}
    for gid in tl.graph_order:
        mult_g = max(1, int(tl.graph_multiplicity.get(gid, 1)))
        g_tasks = tl.per_graph_tasks.get(gid, [])
        if not g_tasks:
            continue
        if len(g_tasks) % mult_g != 0:
            # Non-uniform iter task counts — fall back to single-iter
            # (treats whole graph as one iter k=0).
            for tp in g_tasks:
                task_to_iter_in_graph[tp] = 0
            iter_starts_ns[(gid, 0)] = tl.tasks[g_tasks[0]].start_ns
            continue
        per_iter_count = len(g_tasks) // mult_g
        for k in range(mult_g):
            chunk = g_tasks[k * per_iter_count : (k + 1) * per_iter_count]
            iter_starts_ns[(gid, k)] = tl.tasks[chunk[0]].start_ns
            for tp in chunk:
                task_to_iter_in_graph[tp] = k

    # For each tensor, group its uses by iter (within its graph).
    # tensor_uses_by_iter[uid][k] = list of within-iter relative ns,
    # taking the EARLIEST within iter k (the tightest deadline).
    tensor_first_in_iter: dict[int, dict[int, int]] = {}
    for uid in feasible_uids:
        tensor = tl.tensors[uid]
        per_iter: dict[int, int] = {}
        for use_pos in tensor.uses:
            k = task_to_iter_in_graph.get(use_pos, 0)
            use_ns = tl.tasks[use_pos].start_ns
            iter_start = iter_starts_ns.get(
                (tensor.graph_id, k), use_ns)
            rel = use_ns - iter_start
            if k not in per_iter or rel < per_iter[k]:
                per_iter[k] = rel
        tensor_first_in_iter[uid] = per_iter

    # Emit per-iter per-rank constraints. We track row count by
    # extending deadline_row_base sequentially.
    #
    # When ``disable_per_rank_deadlines=True``, skip emission entirely.
    # The per-rank prefix-sum is a fictional FIFO model that does NOT
    # match the runtime (which fires prefetches independently at
    # issuer-retire moments and resolves bandwidth via water-fill).
    # Dropping these constraints lets the LP optimize peak directly,
    # subject only to the per-bucket memory constraints. Bandwidth
    # contention then surfaces in sim, not in the LP's "fits" claim.
    deadline_row_base = nb
    next_row = deadline_row_base
    if not disable_per_rank_deadlines:
        for gid in tl.graph_order:
            mult_g = max(1, int(tl.graph_multiplicity.get(gid, 1)))
            for iter_k in range(mult_g):
                # Tensors with at least one use in iter k of graph gid.
                ranked_in_iter: list[tuple[int, int]] = []  # (rel_ns, i_idx)
                for i, uid in enumerate(feasible_uids):
                    if tl.tensors[uid].graph_id != gid:
                        continue
                    rel = tensor_first_in_iter.get(uid, {}).get(iter_k)
                    if rel is None:
                        continue
                    ranked_in_iter.append((rel, i))
                ranked_in_iter.sort()
                # Per-rank prefix-sum constraint within this iter.
                cumulative = 0.0
                for r_idx, (rel_ns, i_main) in enumerate(ranked_in_iter):
                    cumulative += float(h2d_dur_per_i[i_main])
                    deadline_ns = float(rel_ns) - t_stream_start_ns
                    for rp_idx in range(r_idx + 1):
                        _rl, ip = ranked_in_iter[rp_idx]
                        rows.append(next_row)
                        cols.append(ip)
                        vals.append(-float(h2d_dur_per_i[ip]))
                    ub_list.append(deadline_ns - cumulative)
                    lb_list.append(-np.inf)
                    next_row += 1

    # Per-graph H2D byte budget. Without it, the LP can claim a tiny
    # ``peak`` while implicitly asking the runtime to stream more
    # bandwidth than physically exists in any one graph's compute
    # window — producing a static-residency-correct but functionally
    # unusable schedule. For each graph g:
    #
    #   Σ_{i ∈ g, used in g} (1 − z_i) · s_i · uses_within_g(i)
    #       ≤  bw · graph_compute_ns(g) · α
    #
    # Linearise (move constant Σ to RHS):
    #
    #   −Σ z_i · s_i · uses_within_g(i)
    #       ≤  cap_g − Σ s_i · uses_within_g(i)
    #
    # Per-graph caps keep multi-graph pipelines (TE-1 + TE-2 + UNet ×
    # M + VAE) from packing all of UNet's traffic into TE's slack:
    # each graph's H2D demand is bounded by its own compute window.
    # ``uses_within_g(i)`` counts reloads incurred during graph g's
    # firing of tensor i — captures cyclic per-iter load implicitly.
    # Weight streaming has no D2H, so only H2D bytes count.
    #
    # Hardware configs may set ``max_pcie_bytes_per_iter`` to override
    # the derived value; that single cap then replaces all per-graph
    # caps.
    aggregate_row = -1
    explicit_cap = getattr(hw, "max_pcie_bytes_per_iter", None)

    # Compute uses_within_g(i) and graph compute windows.
    uses_within_g: dict[tuple[int, int], int] = {}  # (uid, gid) -> count
    for tensor in tl.tensors:
        if tensor.uid not in uid_to_var:
            continue
        for pos in tensor.uses:
            t = tl.tasks[pos]
            uses_within_g[(tensor.uid, int(t.graph_id))] = (
                uses_within_g.get((tensor.uid, int(t.graph_id)), 0) + 1
            )
    graph_compute_ns: dict[int, int] = {}
    for gid in tl.graph_order:
        g_tasks = tl.per_graph_tasks.get(gid, [])
        if not g_tasks:
            continue
        # Sum of task durations gives "compute used" by this graph
        # across all its launches (= GPU time, regardless of stream
        # gaps that don't carry compute). Equivalent to ``end−start``
        # under the timeline's gap-free compaction.
        gs = tl.tasks[g_tasks[0]].start_ns
        ge = tl.tasks[g_tasks[-1]].end_ns
        graph_compute_ns[gid] = max(0, int(ge - gs))

    cap_bytes = 0  # for diagnostic reporting (sum of per-graph caps)
    if explicit_cap and explicit_cap > 0:
        # Single global cap from hardware — keep legacy behavior.
        cap_bytes = int(explicit_cap * float(bw_overcommit_frac))
        if cap_bytes > 0 and nv > 0:
            aggregate_row = next_row
            next_row += 1
            total_evict_bytes = 0.0
            for i, uid in enumerate(feasible_uids):
                size = tl.tensors[uid].size_bytes
                n_uses = len(tl.tensors[uid].uses)
                coeff = float(size) * float(n_uses)
                rows.append(aggregate_row)
                cols.append(i)
                vals.append(-coeff)
                total_evict_bytes += coeff
            ub_list.append(float(cap_bytes) - total_evict_bytes)
            lb_list.append(-np.inf)
    else:
        # Per-graph caps: one row per graph that has any feasible
        # consumer.
        for gid in tl.graph_order:
            g_dur_ns = graph_compute_ns.get(gid, 0)
            if g_dur_ns <= 0:
                continue
            g_cap = int(h2d_bw * float(g_dur_ns) * float(bw_overcommit_frac))
            if g_cap <= 0:
                continue
            # Coefficients for this graph's row.
            row_idx = next_row
            any_var = False
            total_evict_bytes_g = 0.0
            for i, uid in enumerate(feasible_uids):
                u_count = uses_within_g.get((uid, gid), 0)
                if u_count <= 0:
                    continue
                size = tl.tensors[uid].size_bytes
                coeff = float(size) * float(u_count)
                rows.append(row_idx)
                cols.append(i)
                vals.append(-coeff)
                total_evict_bytes_g += coeff
                any_var = True
            if not any_var:
                continue
            ub_list.append(float(g_cap) - total_evict_bytes_g)
            lb_list.append(-np.inf)
            next_row += 1
            cap_bytes += g_cap
            if aggregate_row < 0:
                aggregate_row = row_idx

    # Per-iter constraint for multi-iter graphs.
    #
    # The per-rank deadline above models "cumulative H2D ≤ first global
    # consumer time" — fine for single-iter graphs where every consumer
    # is in one pass. Multi-iter graphs (UNet running 8× per pipeline)
    # break this: the LP sees 8 deadlines spread across the pipeline,
    # but reality is 8 separate per-iter sub-problems where each iter's
    # H2D bytes must fit in ONE iter's compute window.
    #
    # Add per-iter aggregate: for each multi-iter graph g and each iter
    # k of g, the sum of h2d_dur for tensors consumed in iter k must
    # fit in iter k's compute time. Steady-state assumption: every iter
    # k has the same compute time = per_graph_total_ns / multiplicity.
    #
    # Constraint per multi-iter graph g:
    #   Σ_{X consumed by g, z_X=0} (1 − z_X) · h2d_dur(X) ≤ iter_compute_ns(g)
    # Linearise:
    #   −Σ z_X · h2d_dur(X) ≤ iter_compute_ns(g) − Σ h2d_dur(X)
    # Note: the aggregate per-iter constraint is now SUBSUMED by the
    # per-iter per-rank constraints above (the LATEST rank within iter
    # k gives `cumulative h2d_dur ≤ iter_compute_ns(g)` for free). No
    # need to add it again.

    n_rows = next_row
    A = csr_matrix((vals, (rows, cols)), shape=(n_rows, total_vars))
    b_ub_arr = np.array(ub_list, dtype=np.float64)

    options: dict[str, Any] = {"disp": False}
    if time_limit_s is not None:
        options["time_limit"] = float(time_limit_s)

    linprog_kwargs: dict[str, Any] = {
        "A_ub": A, "b_ub": b_ub_arr,
        "bounds": bounds_list,
        "method": "highs",
        "options": options,
    }
    if integrality_arr is not None:
        linprog_kwargs["integrality"] = integrality_arr
    res = linprog(c, **linprog_kwargs)
    fell_back_to_lp = False
    if (
        not res.success
        and integrality_arr is not None
        and not lp_relaxation
    ):
        # Integer MILP timed out or failed. Retry as LP relaxation —
        # for bundles where branch-and-bound is intractable, the LP +
        # threshold rounding still beats the trivial keep-everything
        # fallback.
        linprog_kwargs.pop("integrality", None)
        res = linprog(c, **linprog_kwargs)
        fell_back_to_lp = True

    z_solution: dict[int, float] = {}
    pcie_used = 0
    target_infeasible = False
    if res.success and res.x is not None:
        x = np.asarray(res.x)
        for i, uid in enumerate(feasible_uids):
            # Keep continuous z in [0,1]. Multi-iter tensors with
            # fractional z are emitted as partial schedules (evict in
            # ⌊(1-z) · (M-1)⌉ of M-1 inter-iter gaps, spread evenly).
            # For single-iter tensors (M=1), z is rounded to {0,1} at
            # emission time.
            z_solution[uid] = float(x[i])
            n_uses = len(tl.tensors[uid].uses)
            evict_frac = max(0.0, 1.0 - float(x[i]))
            # Per-pipeline H2D bytes ≈ evict_frac × size × n_uses.
            pcie_used += int(evict_frac * tl.tensors[uid].size_bytes * n_uses)
        # Compute the achieved per-launch peak from the z choice rather
        # than reading x[P_IDX]. In min_peak mode they're equal; in
        # min_streams mode the LP only requires P ≤ target and HiGHS
        # may leave x[P_IDX] pinned to the upper bound, hiding any peak
        # slack the chosen z actually has.
        achieved_peak = constant_floor
        for key in representative_launches:
            consumers_set = set(launch_consumers.get(key, []))
            a_active_feasible = float(launch_active_bytes.get(key, 0))
            intermediate_bytes = float(launch_inter.get(key, 0))
            kept_not_active = 0.0
            for i in range(nv):
                if i in consumers_set:
                    continue
                kept_not_active += float(x[i]) * float(sizes[i])
            launch_peak = (
                constant_floor
                + a_active_feasible
                + intermediate_bytes
                + kept_not_active
            )
            if launch_peak > achieved_peak:
                achieved_peak = launch_peak
        peak = int(achieved_peak)
    else:
        # Fall back to forcing every feasible tensor resident. With a
        # peak target this fallback may itself violate the target — flag
        # it so the caller can surface infeasibility cleanly.
        for uid in feasible_uids:
            z_solution[uid] = 1.0
        max_inter = max(launch_inter.values()) if launch_inter else 0
        peak = int(
            constant_floor
            + max_inter
            + sum(int(tl.tensors[u].size_bytes) for u in feasible_uids)
        )
        if peak_target_bytes is not None and peak > peak_target_bytes:
            target_infeasible = True

    diagnostics = {
        "pre_filter": pre_diag,
        "num_launch_constraints": nb,
        "num_feasible_vars": nv,
        "duplex": duplex,
        "solver_success": bool(res.success),
        "solver_status": str(getattr(res, "message", "")),
        "aggregate_cap_bytes": int(cap_bytes) if cap_bytes else 0,
        "aggregate_cap_active": aggregate_row >= 0,
        "integer_milp": (
            integrality_arr is not None and not fell_back_to_lp
        ),
        "fell_back_to_lp": bool(fell_back_to_lp),
        "bw_overcommit_frac": float(bw_overcommit_frac),
        "h2d_budget_bytes": int(cap_bytes),
        "objective": (
            "min_streams" if peak_target_bytes is not None else "min_peak"
        ),
        "peak_target_bytes": (
            int(peak_target_bytes) if peak_target_bytes is not None else 0
        ),
        "target_infeasible": bool(target_infeasible),
        "peak_intermediate_bytes": int(
            max(launch_inter.values()) if launch_inter else 0
        ),
        "extra_static_bytes": int(extra_static_bytes),
    }

    # Informational aggregate "budget": h2d_bw × pipeline duration. With
    # replicate_uses=True the unified timeline already reflects the true
    # multi-iter duration (UNet appears N times, each at a real start_ns).
    # Weight streaming has no D2H (eviction = resize_(0)), so the lane
    # is single-direction H2D regardless of duplex flag.
    _agg_budget = int(h2d_bw * float(tl.total_duration_ns))
    return _MilpResult(
        z_solution=z_solution,
        forced_keep=forced_keep,
        feasible_uids=feasible_uids,
        peak_bytes=peak,
        pcie_used_bytes=pcie_used,
        pcie_budget_bytes=_agg_budget,
        solver_status=str(getattr(res, "message", "")),
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Op emission
# ---------------------------------------------------------------------------


def _emit_neutral(
    tl: UnifiedTimeline,
    hw: HwParams,
    result: _MilpResult,
) -> tuple[list[NeutralPrefetch], list[NeutralEvict], set[int], set[int]]:
    """Translate z_t decisions into NeutralPrefetch + NeutralEvict entries.

    Global backward ALAP (matches LP's FIFO-from-0 model):

      Walk evicted tensors in reverse consumer-rank order. For rank r:

          required_finish_r = min(first_use_time(r),
                                  required_start_{r+1})
          required_start_r  = required_finish_r − h2d_dur(r)

      The min() enforces FIFO precedence: rank r must finish before rank
      r+1 starts (single-server FIFO). Without that clause, a cluster of
      tensors consumed at the same first_use_time could each "pass" the
      per-tensor ALAP check while collectively backlogging the FIFO.

      required_start_r is then translated to the latest compiled launch
      whose start_ns ≤ required_start_r. If that overruns the graph's
      earliest launch (start-of-iter), fall back to sync emission at the
      consumer (safe but sacrifices async hiding for that tensor).
    """
    prefetches: list[NeutralPrefetch] = []
    evicts: list[NeutralEvict] = []
    evicted: set[int] = set()
    cold_start: set[int] = set(result.forced_keep)

    # Per-graph sorted launch list for issue_launch lookup.
    per_graph_tasks_sorted: dict[int, list[tuple[int, int]]] = {}
    for gid, task_positions in tl.per_graph_tasks.items():
        entries: list[tuple[int, int]] = []
        for tp in task_positions:
            task = tl.tasks[tp]
            entries.append((task.launch_id, task.start_ns))
        entries.sort(key=lambda e: e[1])
        per_graph_tasks_sorted[gid] = entries

    # Per-graph multiplicity: how many times each graph runs per
    # pipeline call. A graph A can issue cross-graph H2D for graph B
    # only if mult[A] >= mult[B] — otherwise A's single fire starves
    # B's later iterations.
    mult: dict[int, int] = dict(getattr(tl, "graph_multiplicity", {}) or {})
    for gid in tl.graph_order:
        mult.setdefault(gid, 1)

    # Enable wrap (cross_iter same-graph) issuer search via env var.
    # Default 1 because it's the natural primary placement for any
    # graph that runs more than once per pipeline call.
    import os as _os
    try:
        max_hops = int(_os.environ.get("XG_MAX_HOPS", "0"))
    except ValueError:
        max_hops = 0

    def _choose_issue_point(
        tensor_gid: int,
        first_use_lid: int,
        target_ns: int,
        first_use_ns: int,
    ) -> tuple[int, int, bool]:
        """Pick the LATEST valid issue point on the unified timeline.

        Candidates considered, in order of preference (latest fire-time
        wins; same fire-time prefers same-graph same-iter for minimum
        residency):

        (1) Same-graph, same-iter: latest launch in tensor_gid with
            start_ns <= target_ns and lid < first_use_lid. Fire and wait
            both in this iter; runtime executes them once per graph
            invocation. Residency window = (first_use_ns - fire_ns).

        (2) Same-graph, cross-iter (wrap): latest launch in tensor_gid
            with lid >= first_use_lid (strictly after first_use within
            the iter). Fire at end of iter N, wait at start of iter N+1.
            Only viable if mult[tensor_gid] >= 2 (the graph runs more
            than once). Iter 0 still needs cold_start or safety net.

        (3) Cross-graph: latest launch in some earlier graph in
            pipeline order. Only viable if mult[issue_gid] >=
            mult[tensor_gid] (issuer runs at least as often as
            consumer; otherwise iter 2..N starves).

        Returns (issue_gid, issue_lid, cross_iter). issue_gid == -1
        means no async point was viable — caller falls back to sync.
        """
        consumer_mult = mult.get(tensor_gid, 1)
        best_gid, best_lid, best_cross_iter = -1, -1, False
        best_fire_ns = -(1 << 60)

        # (1) Same-graph, same-iter. Iterate in start_ns order (the list
        # is sorted by start_ns) and pick the latest launch whose
        # start_ns ≤ target_ns AND lid < first_use_lid. Under
        # replicate_uses=True, the per-graph task list cycles through
        # iter blocks (iter1's lid 0..N, iter2's lid 0..N, ...), so a
        # break on `lid >= first_use_lid` would prematurely stop after
        # iter1's consumer launch — missing iter2+'s earlier-lid
        # launches that occur LATER in time and are valid issue points.
        # Break on start_ns instead.
        for lid, start_ns in per_graph_tasks_sorted.get(tensor_gid, []):
            if start_ns > target_ns:
                break
            if lid < first_use_lid and start_ns > best_fire_ns:
                best_gid, best_lid, best_cross_iter = tensor_gid, lid, False
                best_fire_ns = start_ns

        # (2) Same-graph, cross-iter (wrap) — only if graph runs >1×
        # per pipeline AND we have hop budget for it (encode wrap as
        # 0-hop-cross-graph internally; user controls via XG_MAX_HOPS).
        if consumer_mult >= 2 and max_hops >= 0:
            iter_len = max(tl.total_duration_ns, 1)
            for lid, start_ns in per_graph_tasks_sorted.get(tensor_gid, []):
                if lid < first_use_lid:
                    continue
                # Wrap: virtual fire time = start_ns - iter_length so
                # this comes BEFORE the next iter's start.
                virt_fire = start_ns - iter_len
                if virt_fire <= target_ns and virt_fire > best_fire_ns:
                    # The fire-point's wall-clock RECENCY (virt_fire)
                    # is the latest valid time before consumer's iter.
                    best_gid, best_lid, best_cross_iter = tensor_gid, lid, True
                    best_fire_ns = virt_fire

        # (3) Cross-graph — only if max_hops >= 1 and issuer mult >=
        # consumer mult.
        if max_hops >= 1:
            try:
                t_idx = tl.graph_order.index(tensor_gid)
            except ValueError:
                t_idx = -1
            if t_idx > 0:
                for g_idx in range(t_idx - 1, max(-1, t_idx - 1 - max_hops), -1):
                    g = tl.graph_order[g_idx]
                    if mult.get(g, 1) < consumer_mult:
                        continue
                    for lid, start_ns in per_graph_tasks_sorted.get(g, []):
                        if start_ns <= target_ns and start_ns > best_fire_ns:
                            best_gid, best_lid, best_cross_iter = g, lid, False
                            best_fire_ns = start_ns

        return best_gid, best_lid, best_cross_iter

    # Convert continuous z[uid] to (cold_start | partial-with-mask | full-evict).
    # KEEP_THRESHOLD: above this z is treated as "keep entirely" (no eviction).
    # FULL_EVICT_THRESHOLD: below this z is treated as "evict every iter".
    # Anything in between is partial; we round (1-z) × (M-1) to the nearest
    # integer count of evictions, spread evenly across gaps.
    # Partial-evict path (continuous z → per-iter mask) is implemented but
    # currently disabled due to a CUDA illegal-access bug we couldn't
    # isolate (the mask emission and runtime filtering look correct, but
    # one of the partial-evict tensors triggers a kernel-side fault on
    # the first iter where the mask says "skip"). Setting both thresholds
    # to 0.5 makes every multi-iter z round to {keep, full-evict}, which
    # matches the per-iter LP constraint result (drain_stall=38 ms,
    # inference=0.432 s on SDXL 4-step).
    KEEP_THRESHOLD = 0.5
    FULL_EVICT_THRESHOLD = 0.5
    # uid -> (issue_lid, evict_iter_mask, reload_iter_mask)
    # iter_mask is 0-indexed iter within the consumer graph. Empty list
    # means "fire every iter" (legacy behavior, full eviction).
    pending: list[tuple[int, int, int, int]] = []
    iter_masks: dict[int, tuple[list[int], list[int]]] = {}
    for uid, z in result.z_solution.items():
        tensor = tl.tensors[uid]
        n_uses = len(tensor.uses)
        if z >= KEEP_THRESHOLD:
            cold_start.add(uid)
            continue
        first_pos = tensor.uses[0]
        last_pos = tensor.uses[-1]
        if n_uses >= 2 and z > FULL_EVICT_THRESHOLD:
            # Partial schedule. (1-z) is the eviction fraction.
            n_gaps = n_uses - 1
            n_evicts = int(round((1.0 - float(z)) * n_gaps))
            n_evicts = max(0, min(n_gaps, n_evicts))
            if n_evicts == 0:
                # Rounded down to no evictions → cold-start only.
                cold_start.add(uid)
                continue
            if n_evicts < n_gaps:
                # Pick which gaps to evict in. Spread evenly (use a
                # round-robin scaling to preserve order). Gap k is the
                # interval after iter k (0-indexed). Eviction in gap k
                # fires the evict op during iter k's wrapper invocation
                # and the reload during iter k+1's invocation.
                evict_gaps = [
                    int(round((j + 0.5) * n_gaps / n_evicts))
                    for j in range(n_evicts)
                ]
                # Clamp to valid range [0, n_gaps-1] and dedup.
                evict_gaps = sorted({
                    max(0, min(n_gaps - 1, k)) for k in evict_gaps
                })
                evict_iter_mask = list(evict_gaps)               # iters where evict fires
                reload_iter_mask = [k + 1 for k in evict_gaps]  # iters where reload fires
                iter_masks[uid] = (evict_iter_mask, reload_iter_mask)
        # else: full eviction (every iter), no mask needed
        pending.append((
            tl.tasks[first_pos].start_ns, uid, first_pos, last_pos,
        ))
    # EDF order: sort by GLOBAL first-use timestamp (not launch_id).
    # For multi-iter consumers, this is the EARLIEST consumer's time
    # (= iter 0 of the consumer graph).
    pending.sort()

    # Backward pass: compute required_finish / required_start per rank.
    # required_finish[r] = min(first_use_time[r], required_start[r+1])
    # The next-rank bound is in GLOBAL ns (trusting that transfers across
    # different graphs still share the one physical h2d_stream FIFO).
    n = len(pending)
    required_finish = [0] * n
    required_start = [0] * n
    # Initialise from the last rank's own deadline.
    if n > 0:
        last_first_use = tl.tasks[pending[-1][2]].start_ns
        last_size = tl.tensors[pending[-1][1]].size_bytes
        last_dur = hw.h2d_latency_ns + int(last_size / max(effective_h2d_bw(hw), 1e-9))
        required_finish[-1] = last_first_use
        required_start[-1] = last_first_use - last_dur
    for r in range(n - 2, -1, -1):
        _ts_r, uid, first_pos, last_pos = pending[r]
        tensor = tl.tensors[uid]
        first_use_ns = tl.tasks[first_pos].start_ns
        d_h2d = hw.h2d_latency_ns + int(tensor.size_bytes / max(effective_h2d_bw(hw), 1e-9))
        required_finish[r] = min(first_use_ns, required_start[r + 1])
        required_start[r] = required_finish[r] - d_h2d

    # Forward pass with FIFO-aware spread.
    #
    # The current `_choose_issue_point` picks the LATEST launch ≤
    # required_start[r] for each tensor independently. Multiple tensors
    # whose required_start values collapse into the same launch interval
    # all pick the same fire-lid → at runtime they queue back-to-back in
    # the H2D FIFO, draining serially. This is the per-iter pile-up
    # that keeps drain_stall high even when the LP claims feasibility.
    #
    # Fix: process pending in REVERSE EDF order (latest deadline first)
    # and track the previously-assigned fire start per (graph, iter).
    # Each new fire's end must be ≤ prev_fire_start (FIFO ordering),
    # which means it must use a launch with start_ns ≤
    # prev_fire_start - d_h2d_r. This forces distinct launches across
    # ranks. Convert the resulting fire-time into a launch_id by binary
    # search on per_graph_tasks_sorted.
    #
    # Per-graph FIFO state. Tracks the latest-already-assigned fire_start
    # per consumer-graph; we need this because the H2D stream is shared
    # but launches are per-graph (the fire-lid must be in some graph's
    # task list). For tensors firing same-graph (case 1) and cross-iter
    # (case 2), the relevant graph is the consumer's graph. For
    # cross-graph (case 3), the issuer graph is earlier in pipeline.
    # Initial value = +inf (no prior fire constrains us yet).
    INF = 1 << 60
    fifo_next_end_per_gid: dict[int, int] = {}

    def _pick_fire_lid_in_graph(g: int, target_ns: int, max_lid_excl: int = -1
                                 ) -> tuple[int, int]:
        """Pick the latest launch in graph g with start_ns ≤ target_ns
        and (if max_lid_excl >= 0) lid < max_lid_excl. Returns
        (lid, start_ns) or (-1, -INF) if none."""
        best_lid = -1
        best_ns = -INF
        for lid, start_ns in per_graph_tasks_sorted.get(g, []):
            if start_ns > target_ns:
                break  # list sorted by start_ns
            if max_lid_excl >= 0 and lid >= max_lid_excl:
                continue
            if start_ns > best_ns:
                best_ns = start_ns
                best_lid = lid
        return best_lid, best_ns

    # Process pending in REVERSE EDF order so the FIFO state propagates
    # correctly: the latest-deadline rank's fire_start sets the upper
    # bound for the next-earliest rank.
    pending_with_indices = list(enumerate(pending))
    pending_with_indices.sort(key=lambda x: -x[1][0])  # reverse EDF (by start_ns)

    for r_orig_idx, (_ts, uid, first_pos, last_pos) in pending_with_indices:
        tensor = tl.tensors[uid]
        d_d2h = hw.d2h_latency_ns + int(tensor.size_bytes / max(hw.d2h_bw, 1e-9))
        d_h2d = hw.h2d_latency_ns + int(tensor.size_bytes / max(effective_h2d_bw(hw), 1e-9))
        first_task = tl.tasks[first_pos]
        last_task = tl.tasks[last_pos]
        first_use_ns = first_task.start_ns

        evict_mask: list[int] = []
        reload_mask: list[int] = []
        if uid in iter_masks:
            evict_mask, reload_mask = iter_masks[uid]
        evicts.append(NeutralEvict(
            tensor_uid=uid,
            issue_launch_id=int(last_task.launch_id),
            transfer_start_ns=int(last_task.end_ns),
            transfer_end_ns=int(last_task.end_ns + d_d2h),
            reason="milp_oracle_evict",
            iter_mask=evict_mask,
        ))

        # Compute the FIFO-aware upper bound on this fire's end time.
        # The fire must end before any later-deadline fire's start
        # (FIFO ordering on the H2D stream), AND before the consumer's
        # first use. Per-graph state because each consumer graph runs
        # its own wrapper which fires at its own rate.
        gid = tensor.graph_id
        prev_fire_end_bound = fifo_next_end_per_gid.get(gid, INF)
        # Required-finish from backward pass also bounds it.
        backward_req_finish = required_finish[r_orig_idx]
        fire_end_bound = min(
            int(first_use_ns), int(backward_req_finish),
            int(prev_fire_end_bound),
        )
        target_ns = max(fire_end_bound - d_h2d, 0)

        # Try same-graph same-iter ALAP first.
        chosen_gid, chosen_lid, chosen_start_ns = -1, -1, -INF
        cross_iter = False
        # (1) Same-graph, same-iter: lid < first_use_lid, start_ns close
        # to target_ns.
        lid1, start1 = _pick_fire_lid_in_graph(
            gid, target_ns, max_lid_excl=int(first_task.launch_id),
        )
        if lid1 >= 0:
            chosen_gid, chosen_lid, chosen_start_ns = gid, lid1, start1

        # (2) Same-graph, cross-iter wrap: any lid in tensor's graph,
        # but virtual fire = start_ns - iter_len for "fires in prev
        # iter's tail." Only viable when the graph runs >1× per pipeline.
        consumer_mult = mult.get(gid, 1)
        if consumer_mult >= 2 and max_hops >= 0:
            iter_len = max(tl.total_duration_ns, 1)
            # Look for the LATEST launch in graph whose start_ns <=
            # target_ns + iter_len (so virt_fire = start_ns - iter_len <= target_ns).
            wrap_lid, wrap_start = _pick_fire_lid_in_graph(
                gid, target_ns + iter_len,
            )
            if wrap_lid >= 0:
                virt_fire = wrap_start - iter_len
                if virt_fire > chosen_start_ns:
                    chosen_gid = gid
                    chosen_lid = wrap_lid
                    chosen_start_ns = virt_fire
                    cross_iter = True

        # (3) Cross-graph (only if max_hops >= 1).
        if max_hops >= 1:
            try:
                t_idx = tl.graph_order.index(gid)
            except ValueError:
                t_idx = -1
            if t_idx > 0:
                for g_idx in range(t_idx - 1, max(-1, t_idx - 1 - max_hops), -1):
                    g = tl.graph_order[g_idx]
                    if mult.get(g, 1) < consumer_mult:
                        continue
                    xg_lid, xg_start = _pick_fire_lid_in_graph(g, target_ns)
                    if xg_lid >= 0 and xg_start > chosen_start_ns:
                        chosen_gid = g
                        chosen_lid = xg_lid
                        chosen_start_ns = xg_start
                        cross_iter = False

        same_graph = (chosen_gid == gid)
        use_async = chosen_gid >= 0
        emit_issue_graph = -1 if same_graph else int(chosen_gid)
        if use_async:
            if same_graph and not cross_iter:
                reason = "milp_oracle_reload_galap"
            elif same_graph and cross_iter:
                reason = "milp_oracle_reload_wrap"
            else:
                reason = "milp_oracle_reload_xg"
            # Update FIFO state: subsequent (earlier-deadline) ranks
            # must end before THIS fire's start.
            fifo_next_end_per_gid[gid] = chosen_start_ns
        else:
            reason = "milp_oracle_reload_sync"
        prefetches.append(NeutralPrefetch(
            tensor_uid=uid,
            issue_launch_id=(
                int(chosen_lid) if use_async
                else int(first_task.launch_id)
            ),
            wait_launch_id=int(first_task.launch_id),
            transfer_start_ns=target_ns,
            transfer_end_ns=int(fire_end_bound),
            reason=reason,
            trusted_async=bool(use_async),
            issue_graph_id=emit_issue_graph,
            cross_iter=bool(cross_iter),
            iter_mask=reload_mask,
        ))
        evicted.add(uid)

    return prefetches, evicts, evicted, cold_start


# ---------------------------------------------------------------------------
# Public solve
# ---------------------------------------------------------------------------


def solve_neutral(
    trace: Trace,
    *,
    sidecars: MultiGraphSidecars,
    hw: HwParams,
    locked_graph_input_names: set[str] | None = None,
    duplex: bool = False,
    time_limit_s: float | None = 120.0,
    iter_wall_ns_override: int | None = None,
    graph_multiplicity: dict[int, int] | None = None,
    disable_per_rank_deadlines: bool = True,
    bw_overcommit_frac: float = 0.9,
    lp_relaxation: bool = False,
    peak_target_bytes: int | None = None,
) -> NeutralSchedule:
    """Solve and return a backend-neutral schedule.

    Default mode (``peak_target_bytes=None``) minimises peak VRAM.
    When ``peak_target_bytes`` is set, switches to the dual: minimise
    total streamed bytes (an E2E proxy) subject to ``P ≤ target`` —
    same MILP machinery, flipped objective + hard upper bound on P.

    Convert the returned schedule to the PyTorch runtime format via
    :func:`graph_modifiers.common.neutral_to_pytorch`.
    """
    tl = build_unified_timeline(
        trace, sidecars, cpu_per_launch_ns=hw.cpu_per_launch_ns,
        graph_multiplicity=graph_multiplicity,
        replicate_uses=True,
    )
    if not tl.tasks:
        raise RuntimeError("[ct_milp_oracle] no compiled tasks in bundle.")

    locked = set(locked_graph_input_names or set())
    locked_uids = {
        t.uid for t in tl.tensors if t.graph_input_name in locked
    }

    launch_intermediate_bytes, extra_static_bytes = (
        _compute_live_intermediates_per_launch(trace, tl)
    )

    result = _solve_milp(
        tl, hw,
        locked_uids=locked_uids,
        duplex=duplex,
        time_limit_s=time_limit_s,
        disable_per_rank_deadlines=disable_per_rank_deadlines,
        bw_overcommit_frac=bw_overcommit_frac,
        lp_relaxation=lp_relaxation,
        peak_target_bytes=peak_target_bytes,
        launch_intermediate_bytes=launch_intermediate_bytes,
        extra_static_bytes=extra_static_bytes,
    )

    prefetches, evicts, evicted, cold_start_set = _emit_neutral(tl, hw, result)

    cold_starts: list[NeutralColdStart] = []
    for uid in sorted(cold_start_set):
        tensor = tl.tensors[uid]
        first_pos = tensor.uses[0] if tensor.uses else 0
        cold_starts.append(NeutralColdStart(
            tensor_uid=uid,
            anchor_launch_id=int(tl.tasks[first_pos].launch_id),
            reason=(
                "milp_oracle_locked" if uid in locked_uids
                else ("milp_oracle_forced_keep" if uid in result.forced_keep
                      else "milp_oracle_optimal_keep")
            ),
        ))

    total_bytes = sum(t.size_bytes for t in tl.tensors)

    per_graph_stats: dict[int, dict[str, Any]] = {}
    for gid in tl.graph_order:
        tensors_in_g = [t for t in tl.tensors if t.graph_id == gid]
        per_graph_stats[gid] = {
            "weight_bytes": sum(t.size_bytes for t in tensors_in_g),
            "tensor_count": len(tensors_in_g),
            "evicted_count": sum(1 for t in tensors_in_g if t.uid in evicted),
            "forced_keep_count": sum(
                1 for t in tensors_in_g if t.uid in result.forced_keep
            ),
        }

    objective_label = (
        "min_streams (peak-target dual)"
        if peak_target_bytes is not None else "min_peak"
    )
    meta: dict[str, Any] = {
        "io_model": "ct_milp_oracle",
        "backend": "cg-sim",
        "algorithm": (
            f"Oracle MILP, {objective_label}, peak-only (no per-rank FIFO)"
            if disable_per_rank_deadlines
            else f"Oracle MILP, {objective_label}, with per-rank H2D deadlines"
        ),
        "objective": objective_label,
        "peak_target_mb": (
            round(peak_target_bytes / 1e6, 2)
            if peak_target_bytes is not None else None
        ),
        "target_infeasible": result.diagnostics.get(
            "target_infeasible", False,
        ),
        "peak_intermediate_mb": round(
            result.diagnostics.get("peak_intermediate_bytes", 0) / 1e6, 2,
        ),
        "extra_static_mb": round(
            result.diagnostics.get("extra_static_bytes", 0) / 1e6, 2,
        ),
        "duplex": duplex,
        "disable_per_rank_deadlines": disable_per_rank_deadlines,
        "graph_order": tl.graph_order,
        "graph_multiplicity": {
            int(gid): int(tl.graph_multiplicity.get(gid, 1))
            for gid in tl.graph_order
        },
        "per_graph_stats": per_graph_stats,
        "total_weight_bytes": int(total_bytes),
        "total_weight_mb": round(total_bytes / 1e6, 2),
        "milp_peak_bytes": int(result.peak_bytes),
        "milp_peak_mb": round(result.peak_bytes / 1e6, 2),
        "pcie_used_bytes": int(result.pcie_used_bytes),
        "pcie_used_mb": round(result.pcie_used_bytes / 1e6, 2),
        "pcie_budget_bytes": int(result.pcie_budget_bytes),
        "pcie_budget_mb": round(result.pcie_budget_bytes / 1e6, 2),
        "pcie_slack_mb": round(
            (result.pcie_budget_bytes - result.pcie_used_bytes) / 1e6, 2,
        ),
        "aggregate_cap_mb": round(
            result.diagnostics.get("aggregate_cap_bytes", 0) / 1e6, 2,
        ),
        "aggregate_cap_active": result.diagnostics.get("aggregate_cap_active", False),
        "num_launch_constraints": result.diagnostics.get(
            "num_launch_constraints", 0,
        ),
        "integer_milp": result.diagnostics.get("integer_milp", False),
        "fell_back_to_lp": result.diagnostics.get("fell_back_to_lp", False),
        "bw_overcommit_frac": result.diagnostics.get(
            "bw_overcommit_frac", 1.0,
        ),
        "feasible_tensor_count": len(result.feasible_uids),
        "forced_keep_count": len(result.forced_keep),
        "evicted_tensor_count": len(evicted),
        "cold_start_count": len(cold_start_set),
        "cold_start_bytes": int(
            sum(tl.tensors[u].size_bytes for u in cold_start_set)
        ),
        "h2d_ops": len(prefetches),
        "d2h_ops": len(evicts),
        "vram_h2d_prefetches": len(prefetches),
        "vram_d2h_evictions": len(evicts),
        "vram_h2d_bytes": int(
            sum(tl.tensors[p.tensor_uid].size_bytes for p in prefetches)
        ),
        "vram_d2h_bytes": int(
            sum(tl.tensors[e.tensor_uid].size_bytes for e in evicts)
        ),
        "locked_tensors": sorted(locked),
        "pre_filter_diag": result.diagnostics.get("pre_filter", {}),
        "solver_status": result.diagnostics.get("solver_status", ""),
        "solver_success": result.diagnostics.get("solver_success", False),
        "wall_time_iter_ns": int(tl.total_duration_ns),
    }

    tensor_records: list[NeutralTensor] = []
    for t in tl.tensors:
        tensor_records.append(NeutralTensor(
            uid=t.uid,
            graph_id=t.graph_id,
            compiled_tensor_id=t.compiled_tensor_id,
            graph_input_name=t.graph_input_name,
            size_bytes=t.size_bytes,
            dtype=t.dtype,
            used_by_launch_ids=[
                int(tl.tasks[pos].launch_id) for pos in t.uses
            ],
        ))

    return NeutralSchedule(
        graph_order=list(tl.graph_order),
        compilation_hashes=dict(tl.per_graph_hash),
        tensors=tensor_records,
        prefetches=prefetches,
        evicts=evicts,
        cold_starts=cold_starts,
        meta=meta,
    )


def solve(
    trace: Trace,
    *,
    sidecars: MultiGraphSidecars,
    hw: HwParams,
    locked_graph_input_names: set[str] | None = None,
    duplex: bool = False,
    time_limit_s: float | None = 120.0,
    iter_wall_ns_override: int | None = None,
    graph_multiplicity: dict[int, int] | None = None,
    bw_overcommit_frac: float = 0.9,
    peak_target_bytes: int | None = None,
) -> Schedule:
    """Compat wrapper: solve_neutral + convert to pytorch-format Schedule."""
    neutral = solve_neutral(
        trace, sidecars=sidecars, hw=hw,
        locked_graph_input_names=locked_graph_input_names,
        duplex=duplex, time_limit_s=time_limit_s,
        iter_wall_ns_override=iter_wall_ns_override,
        graph_multiplicity=graph_multiplicity,
        bw_overcommit_frac=bw_overcommit_frac,
        peak_target_bytes=peak_target_bytes,
    )
    tl = build_unified_timeline(
        trace, sidecars, cpu_per_launch_ns=hw.cpu_per_launch_ns,
        graph_multiplicity=graph_multiplicity,
        replicate_uses=True,
    )
    node_starts, node_ends = build_node_timeline(tl, trace)
    pytorch_doc = neutral_to_pytorch(
        neutral, trace=trace, node_starts=node_starts, node_ends=node_ends,
    )
    return Schedule(
        neutral=neutral,
        node_starts=node_starts,
        node_ends=node_ends,
        io_operations=pytorch_doc["io_operations"],
        cold_start_prefetches=pytorch_doc["cold_start_prefetches"],
        summary=pytorch_doc["summary"],
        compilation_hash=pytorch_doc.get("compilation_hash", ""),
    )


def print_summary(schedule: Schedule) -> None:
    s = schedule.summary
    pf = s["pre_filter_diag"]
    print("\n=== MILP oracle (per-rank H2D deadlines) ===")
    print(f"  Graph order     : {s['graph_order']}")
    print(f"  Total weights   : {s['total_weight_mb']:.2f} MB")
    print(f"  PCIe budget     : {s['pcie_budget_mb']:.2f} MB per iter"
          f"  (duplex={s['duplex']})")
    print(
        f"  Pre-filter      : {pf.get('feasible', 0)} feasible / "
        f"{pf.get('forced_by_infeasible_gap', 0)} forced-by-gap / "
        f"{pf.get('forced_by_locked', 0)} locked / "
        f"{pf.get('zero_use', 0)} unused"
    )
    for gid in s["graph_order"]:
        st = s["per_graph_stats"].get(gid, {})
        print(
            f"    graph_{gid} : weights={st.get('weight_bytes', 0)/1e6:.2f}MB "
            f"({st.get('tensor_count', 0)} tensors, "
            f"evict={st.get('evicted_count', 0)}, "
            f"forced_keep={st.get('forced_keep_count', 0)})"
        )
    print(f"  MILP peak       : {s['milp_peak_mb']:.2f} MB")
    print(f"  PCIe used       : {s['pcie_used_mb']:.2f} MB"
          f"  (slack {s['pcie_slack_mb']:.2f} MB)")
    if s.get("aggregate_cap_active"):
        print(f"  Aggregate cap   : {s['aggregate_cap_mb']:.2f} MB (enforced)")
    print(f"  H2D / D2H ops   : {s['h2d_ops']} / {s['d2h_ops']}")
    print(f"  Evicted tensors : {s['evicted_tensor_count']}")
    print(f"  Cold-start      : {s['cold_start_count']} tensors "
          f"({s['cold_start_bytes']/1e6:.2f} MB)")
    print(f"  Solver status   : {s['solver_status']}")
