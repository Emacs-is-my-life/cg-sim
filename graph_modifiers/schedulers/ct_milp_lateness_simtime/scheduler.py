"""Pool-first MILP weight-streaming scheduler with lateness objective.

For each cuda-resident WEIGHT/LEAF/INPUT trace tensor (one entry per
physical storage thanks to the loader's `(device, storage_id)` dedup),
decide:

  c_t ∈ {0, 1}:     1 = cold (resident from layout)
                    0 = streamed (JIT prefetch before first consumer +
                                  per-feasible-gap evict+refetch)
  e_{t,k} ∈ {0, 1}: 1 = evict after consumer k and refetch before k+1
                    coupled to (1 − c_t) on feasible gaps

Objective: minimize L_max + Λ·s_P
  - L_max ≥ 0 — max consumer lateness (∝ end-to-end stall, ns)
  - s_P ≥ 0   — peak VRAM overrun slack (bytes); priced at Λ = 1e6
                ns / byte so the LP fits cap whenever feasible

PCIe lateness model (h2d_streams = 1, duplex with d2h):
  At sample point T (a gpu compute consumer's trace_start_ns):
    cumulative_h2d_required_by(T) = Σ_{t: any_consumer(t) ≤ T} δ_t · (1 − c_t)
                                 +  Σ_{(t,k): consumer_{k+1}(t) ≤ T}
                                                 δ_t · e_{t,k}
    cumulative_h2d_required_by(T)  ≤  T + L_max
  δ_t = h2d_latency_ns + size_t / h2d_bw   — per-tid H2D transfer time
  D2H runs concurrent with H2D under duplex, so evicts don't enter the
  cumulative-h2d budget (matches jit_sim_prune / current LP semantics).

Peak VRAM model (sampled at every gpu compute consumer's T):
  Σ_{t: alive(t, T)} size_t  +  forced_const  ≤  cap·(1−margin) + s_P
  alive(t, T) classifies T into one of pre-arc / arc_0 / dead-zone /
  arc_{k+1} / post regions of t's consumer pattern, contributing either
  "size unconditional" (in arc) or "size · c_t" (in dead zone / pre /
  post — alive only if cold).

Schedule emission: every NeutralPrefetch / NeutralEvict / NeutralColdStart
carries the cgsim_tid directly, so the injector's pre_resolved fast path
fires and shape-disambiguation / synth_gates / coverage_repair don't run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

try:
    import highspy  # type: ignore
    _HIGHSPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HIGHSPY_AVAILABLE = False

from graph_modifiers.common import (
    HwParams,
    NeutralColdStart,
    NeutralEvict,
    NeutralPrefetch,
    NeutralSchedule,
    NeutralTensor,
)
from graph_modifiers.common.hw import effective_h2d_bw
from sim.core.trace import Trace


_GPU_RESOURCE_KINDS = ("gpu_stream", "gpu", "gpu_runtime")
_POOL_TENSOR_TYPES = ("WEIGHT", "LEAF", "INPUT")
PEAK_SLACK_PENALTY = 1.0e6

# Numerical scaling. The model is solved in MB (bytes / S) and ms (ns / S)
# rather than raw bytes/ns. Raw units give HiGHS a ~1e16 objective dynamic
# range (sizes ~7e8, RHS ~2e10, slack penalty 1e6) → its simplex fails with
# model_status Unknown / "Solve error" on some models/caps even though the LP
# is provably feasible. Because bytes and ns are scaled by the SAME factor,
# every ratio coefficient (e.g. bytes/ns PCIe rate) is invariant and all
# objective weights stay identical, so the optimum is unchanged — only the
# conditioning improves (Matrix/Cost ~[1e-2,1e6], RHS ~[1,1e4]). P, s_P and
# the per-window slacks are converted back to bytes/ns at decode, so all
# downstream code (emit, diagnostics, sim) is untouched.
MODEL_SCALE = 1.0e6


def _default_solver_threads() -> int:
    return max(1, os.cpu_count() or 1)


def _normalize_solver_threads(solver_threads: int | None) -> int:
    if solver_threads is None:
        return _default_solver_threads()
    solver_threads = int(solver_threads)
    if solver_threads < 1:
        raise ValueError("solver_threads must be >= 1")
    return solver_threads


def _set_highs_threads(h: Any, solver_threads: int) -> None:
    h.setOptionValue("threads", int(solver_threads))


def _load_baseline_sim_times(
    baseline_sim_result_path: str,
) -> dict[int, tuple[int, int]]:
    """Parse a baseline sim_result.json for per-trace-node sim times.

    The sim emits ``COMPUTE_JOB`` chrome-trace events whose
    ``args.Payload.id`` is the trace node_id and whose ``ts`` / ``dur``
    are sim wall-clock in microseconds. We extract gpu compute events
    (Hardware.name starts with 'gpu') and build:

      {trace_node_id: (sim_start_ns, sim_end_ns)}

    These are the *actual* deadlines a prefetch must beat in sim —
    not the trace's profiler wall-clock (which includes idle gaps
    sim collapses) and not the packed GPU-cumulative (which ignores
    CPU interleaving). Run sim once with a neutral schedule, parse
    its output, feed the times here, and the LP plans on the ground
    truth.

    The baseline run should ideally be cold-all (no streaming) to
    avoid feedback. In practice any prior run works — sim times
    perturb only slightly with different schedules.
    """
    import json
    with open(baseline_sim_result_path) as f:
        d = json.load(f)
    sim_times: dict[int, tuple[int, int]] = {}
    for e in d.get("traceEvents", ()):
        if e.get("name") != "COMPUTE_JOB" or e.get("ph") != "X":
            continue
        args = e.get("args") or {}
        hw = args.get("Hardware") or {}
        if not str(hw.get("name", "")).startswith("gpu"):
            continue
        payload = args.get("Payload") or {}
        nid = payload.get("id")
        if not isinstance(nid, int):
            continue
        ts = float(e.get("ts") or 0)
        dur = float(e.get("dur") or 0)
        # Chrome-trace ts/dur are in microseconds; convert to ns to
        # match the rest of the scheduler.
        sim_start_ns = int(ts * 1000)
        sim_end_ns = int((ts + dur) * 1000)
        sim_times[int(nid)] = (sim_start_ns, sim_end_ns)
    return sim_times


# ---------------------------------------------------------------------------
# Pool + per-tid consumer pattern
# ---------------------------------------------------------------------------


@dataclass
class _PoolTensor:
    """One cgsim_tid in the optimization domain.

    ``consumers`` is the trace-time-ordered list of GPU node reads. For
    multi-iter workloads (UNet ×N steps, LLM decoding ×N tokens) a single
    physical storage is read N times → N entries here. Aux/aten gpu ops
    that read the storage are also entries (the LP treats them no
    differently from compiled kernels).
    """
    cgsim_tid: int
    size_bytes: int
    name: str
    dtype: str
    # (node_id, lp_start_ns, lp_end_ns) sorted by lp_start_ns. lp_*_ns
    # is SIM-TIME when ``sim_times`` was provided, else trace-time.
    # The LP plans gap_feasibility, c_feasibility, peak samples, and
    # per-window lateness rows on this axis.
    consumers: list[tuple[int, int, int]]
    tau_h2d_ns: int                  # δ_t — per-tid H2D transfer time
    tau_d2h_ns: int                  # per-tid D2H transfer time
    gap_feasibility: list[bool]
    c_feasibility: bool
    consumer_graph_ids: list[int] = field(default_factory=list)
    consumer_launch_ids: list[int] = field(default_factory=list)
    # Parallel arrays in TRACE wall-clock ns (same order as
    # ``consumers``). Used only by emit when populating the schedule
    # JSON's transfer_start_ns / transfer_end_ns fields — the
    # injector keys those to trace times. Equal to ``consumers``
    # when sim_times isn't provided.
    consumer_trace_starts: list[int] = field(default_factory=list)
    consumer_trace_ends: list[int] = field(default_factory=list)


def _build_pool(
    trace: Trace, hw: HwParams,
    sim_times: dict[int, tuple[int, int]] | None = None,
) -> dict[int, _PoolTensor]:
    """Build the pool. When ``sim_times`` is provided, the LP's
    timing axis uses sim wall-clock from a baseline run (option #1:
    ground-truth deadlines from a prior sim). Otherwise falls back
    to trace_start_ns (identical to the original ct_milp_lateness).
    """
    bw_h2d = max(effective_h2d_bw(hw), 1e-9)
    bw_d2h = max(float(hw.d2h_bw), 1e-9)

    def _lp_time(nid: int, trace_t: int) -> int:
        """Pick the LP's time axis: sim_time if available, else trace."""
        if sim_times is None:
            return trace_t
        st = sim_times.get(nid)
        if st is None:
            # Node not in baseline sim — fall back to trace time.
            return trace_t
        return st[0]  # sim_start_ns

    def _lp_time_end(nid: int, trace_e: int) -> int:
        if sim_times is None:
            return trace_e
        st = sim_times.get(nid)
        if st is None:
            return trace_e
        return st[1]  # sim_end_ns

    candidate_tids: set[int] = set()
    for tid, t in trace.tensor_map.items():
        if int(t.size_bytes) <= 0:
            continue
        ttype = (t.args or {}).get("tensor_type")
        if ttype not in _POOL_TENSOR_TYPES:
            continue
        device = str((t.args or {}).get("device", "")).lower()
        if not device.startswith("cuda"):
            continue
        candidate_tids.add(int(tid))

    # consumers_by_tid stores both LP-axis (sim or trace) and trace
    # times so emit can correctly populate transfer_start_ns/end_ns.
    # Tuple: (lp_start, lp_end, nid, gid, lid, trace_start, trace_end).
    consumers_by_tid: dict[int, list[tuple[int, int, int, int, int, int, int]]] = {}
    for nid, node in trace.node_map.items():
        rk = str((node.args or {}).get("resource_kind") or "")
        if rk not in _GPU_RESOURCE_KINDS:
            continue
        nid_i = int(nid)
        start_ns = int((node.args or {}).get("start_ns") or 0)
        end_ns = int((node.args or {}).get("end_ns") or start_ns)
        if start_ns <= 0:
            continue
        lp_s = _lp_time(nid_i, start_ns)
        lp_e = _lp_time_end(nid_i, end_ns)
        gid_raw = (node.args or {}).get("compiled_graph_id")
        lid_raw = (node.args or {}).get("compiled_launch_id")
        try:
            gid_i = int(gid_raw) if gid_raw is not None else -1
        except (TypeError, ValueError):
            gid_i = -1
        try:
            lid_i = int(lid_raw) if lid_raw is not None else -1
        except (TypeError, ValueError):
            lid_i = -1
        for raw_tid in (node.input_tensors or []):
            t = int(raw_tid)
            if t not in candidate_tids:
                continue
            consumers_by_tid.setdefault(t, []).append(
                (lp_s, lp_e, nid_i, gid_i, lid_i, start_ns, end_ns)
            )

    # Per-graph + global indices over gpu nodes in LP-axis.
    #
    # IMPORTANT: feasibility checks now use GLOBAL gpu starts (any
    # graph), not per-graph. PCIe is a global resource; a prefetch
    # for a tid first consumed in graph G_3 can be issued from any
    # earlier gpu node in graphs G_0..G_2. The same applies to
    # cross-iter refetches in the unified timeline. Restricting to
    # the consumer's own graph (the old behavior) made c_feasibility
    # / gap_feasibility falsely reject tids whose graph happens to
    # start late but have plenty of earlier cross-graph slack.
    graph_first_gpu_ns: dict[int, int] = {}  # kept for diagnostic
    sorted_gpu_starts_by_graph: dict[int, list[int]] = {}
    all_gpu_starts: list[int] = []
    for nid, node in trace.node_map.items():
        rk = str((node.args or {}).get("resource_kind") or "")
        if rk not in _GPU_RESOURCE_KINDS:
            continue
        nid_i = int(nid)
        start_ns = int((node.args or {}).get("start_ns") or 0)
        if start_ns <= 0:
            continue
        lp_s = _lp_time(nid_i, start_ns)
        gid_raw = (node.args or {}).get("compiled_graph_id")
        try:
            gid_i = int(gid_raw) if gid_raw is not None else -1
        except (TypeError, ValueError):
            gid_i = -1
        cur = graph_first_gpu_ns.get(gid_i)
        if cur is None or lp_s < cur:
            graph_first_gpu_ns[gid_i] = lp_s
        sorted_gpu_starts_by_graph.setdefault(gid_i, []).append(lp_s)
        all_gpu_starts.append(lp_s)
    for g in sorted_gpu_starts_by_graph:
        sorted_gpu_starts_by_graph[g].sort()
    all_gpu_starts.sort()

    pool: dict[int, _PoolTensor] = {}
    for tid, raw in consumers_by_tid.items():
        raw.sort(key=lambda r: r[0])
        tensor = trace.tensor_map[tid]
        size = int(tensor.size_bytes)
        tau_h2d = int(hw.h2d_latency_ns) + int(size / bw_h2d)
        tau_d2h = int(hw.d2h_latency_ns) + int(size / bw_d2h)
        # raw rows now include trace timestamps too:
        # (lp_start, lp_end, nid, gid, lid, trace_start, trace_end).
        # `consumers` carries the LP axis (sim_time when available);
        # `consumer_trace_starts/ends` carry trace wall-clock for emit.
        consumers = [
            (int(nid), int(lps), int(lpe))
            for (lps, lpe, nid, _g, _l, _ts, _te) in raw
        ]
        graph_ids = [int(g) for (_lps, _lpe, _nid, g, _l, _ts, _te) in raw]
        launch_ids = [int(l) for (_lps, _lpe, _nid, _g, l, _ts, _te) in raw]
        consumer_trace_starts = [
            int(ts) for (_lps, _lpe, _nid, _g, _l, ts, _te) in raw
        ]
        consumer_trace_ends = [
            int(te) for (_lps, _lpe, _nid, _g, _l, _ts, te) in raw
        ]
        # Per-gap feasibility check — must mirror what emit's
        # `_pick_issuer_node` + `re_ts <= ck_end` short-circuit will
        # accept, otherwise the LP picks e=1 for gaps that emit can't
        # actually realise (issuer ends up sync, injector demotes).
        #
        # The constraints: for gap k between consumer_k and
        # consumer_{k+1}, the refetch needs a gpu issuer X in
        # consumer_{k+1}'s graph with
        #     consumer_k.end  <  X.trace_start  ≤  consumer_{k+1}.start − τ_h2d
        #
        # Lower bound (X > consumer_k.end): issuer must fire AFTER
        # the evict frees the source pages — otherwise the refetch H2D
        # claims dst pages while the old pages are still resident,
        # doubling VRAM. Emit enforces this via `re_ts ≤ ck_end →
        # sync fallback`.
        #
        # Upper bound (X ≤ consumer_{k+1}.start − τ_h2d): the
        # transfer must finish before consumer_{k+1} dispatches.
        #
        # If no gpu node in consumer_{k+1}'s graph falls in this
        # window, no async issuer exists → e_var dropped, tid stays
        # alive across the gap.
        # GLOBAL issuer search (any graph). PCIe is global; a refetch
        # can be issued from any earlier gpu node, not just one in
        # consumer_{k+1}'s own graph. Without this, an early consumer
        # in a late-starting graph (e.g. UNet first kernel in G_3
        # when G_0..G_2 already ran) wrongly looks c-infeasible
        # because its graph's first node provides no slack.
        gap_feas: list[bool] = []
        for i in range(len(consumers) - 1):
            ck_end = consumers[i][2]
            ckp1_start = consumers[i + 1][1]
            target = ckp1_start - tau_h2d
            if target <= ck_end:
                gap_feas.append(False)
                continue
            import bisect
            idx = bisect.bisect_right(all_gpu_starts, ck_end)
            issuer_ok = (
                idx < len(all_gpu_starts)
                and all_gpu_starts[idx] <= target
            )
            gap_feas.append(issuer_ok)
        # c_feasibility: must exist a gpu node ANYWHERE that fires
        # before consumer_0 with cum-time gap ≥ τ_h2d. Equivalent to
        # consumer_0.lp_start ≥ τ_h2d + earliest_gpu_lp_start (≈ 0
        # when sim_time origin or trace_t origin is small). Allows
        # cross-graph issuers.
        origin_for_c = all_gpu_starts[0] if all_gpu_starts else 0
        c_feas = (consumers[0][1] - origin_for_c) >= tau_h2d
        name_raw = getattr(tensor, "name", None) or ""
        dtype_raw = str((tensor.args or {}).get("dtype") or "")
        pool[tid] = _PoolTensor(
            cgsim_tid=int(tid),
            size_bytes=size,
            name=str(name_raw),
            dtype=dtype_raw,
            consumers=consumers,
            tau_h2d_ns=tau_h2d,
            tau_d2h_ns=tau_d2h,
            gap_feasibility=gap_feas,
            c_feasibility=c_feas,
            consumer_graph_ids=graph_ids,
            consumer_launch_ids=launch_ids,
            consumer_trace_starts=consumer_trace_starts,
            consumer_trace_ends=consumer_trace_ends,
        )
    return pool


def _build_intermediate_residencies(
    trace: Trace,
    sim_times: dict[int, tuple[int, int]] | None = None,
    axis_fix: bool = True,
) -> list[tuple[int, int, int]]:
    """List of (start_ns, end_ns, size_bytes) for cuda INTERMEDIATEs.

    Residency window in the LP's time axis (sim_t if available, else
    trace_t): [producer.start, last_consumer.end]. The LP adds the
    sum of intermediates alive at each sample T into that sample's
    peak-row const_addons — these aren't scheduler variables, their
    lifetimes are fixed by the trace's producer/consumer graph.

    Without this, the LP plans weight residency up to cap and sim
    OOMs when intermediate working set materialises (esp. SDXL kv
    cache and similar large transients).
    """
    # AXIS-MIX BUG FIX (``axis_fix``, default on; set False only for the
    # ablation that shows the bug's cost). When sim_times is given, the old
    # code fell back to a node's *trace* ns whenever it lacked a sim_time.
    # But sim-axis and trace-axis are offset per-node (same node can be
    # 67ms in sim, 469ms in trace), so a sim-time producer paired with a
    # trace-fallback consumer (e.g. a cpu_thread consumer with no sim_time)
    # yields a ~400ms PHANTOM residency window. ~160/219 sdxl intermediates
    # were inflated this way, blowing the modeled activation floor to
    # 1941MB vs ~192MB actual. Fix: when sim_times is active, use sim-time
    # endpoints ONLY — skip nodes with no sim_time rather than mixing axes.
    # (Recovers ~200MB, ≈ the sim truth; small residual covered by margin.)
    _fix_axis = (sim_times is not None) and axis_fix

    def _t(nid: int, fallback: int) -> int:
        if sim_times is None:
            return fallback
        st = sim_times.get(int(nid))
        return int(st[0]) if st is not None else fallback

    def _t_end(nid: int, fallback: int) -> int:
        if sim_times is None:
            return fallback
        st = sim_times.get(int(nid))
        return int(st[1]) if st is not None else fallback

    producer_of: dict[int, int] = {}
    last_consumer_end: dict[int, int] = {}
    for nid, node in trace.node_map.items():
        start_ns = int((node.args or {}).get("start_ns") or 0)
        end_ns = int((node.args or {}).get("end_ns") or start_ns)
        if start_ns <= 0:
            continue
        # Under the axis-fix, a node with no sim_time can't contribute a
        # sim-axis endpoint — skip it (no trace-ns fallback that would mix
        # axes). The producer pass also skips, so a tid produced only by
        # no-sim-time nodes is dropped (its residency is unmodelable here).
        if _fix_axis and sim_times.get(int(nid)) is None:
            continue
        for tid in (node.output_tensors or []):
            t = int(tid)
            if t not in producer_of:
                producer_of[t] = nid
        for tid in (node.input_tensors or []):
            t = int(tid)
            lp_end = _t_end(nid, end_ns)
            if lp_end > last_consumer_end.get(t, 0):
                last_consumer_end[t] = lp_end

    out: list[tuple[int, int, int]] = []
    for tid, t in trace.tensor_map.items():
        if (t.args or {}).get("tensor_type") != "INTERMEDIATE":
            continue
        device = str((t.args or {}).get("device", "")).lower()
        if not device.startswith("cuda"):
            continue
        size = int(t.size_bytes)
        if size <= 0:
            continue
        prod_nid = producer_of.get(int(tid))
        if prod_nid is None:
            continue
        prod_start_trace = int(
            (trace.node_map[prod_nid].args or {}).get("start_ns") or 0
        )
        prod_start = _t(prod_nid, prod_start_trace)
        end_t = last_consumer_end.get(int(tid), prod_start)
        if prod_start <= 0 or end_t <= prod_start:
            continue
        out.append((prod_start, end_t, size))
    out.sort(key=lambda x: x[0])
    return out


def _derive_backpressure_edges(
    trace: Trace,
    result: "_LPResult",
    hw: HwParams,
    *,
    sim_times: dict[int, tuple[int, int]] | None = None,
    lateness_threshold_ns: int = 100_000,
    audit: bool = False,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Derive synthetic GPU→CPU control edges from per-window lateness.

    For each window i where L_window_i > threshold:
      - gpu_anchor = last gpu_runtime node with start_ns ≤ window end
      - delta = bw_h2d × L_window_i  (in ns; equivalent CPU work the
        allocator should make CPU wait for under real-PyTorch
        backpressure)
      - cpu_anchor = first cpu_leaf node with start_ns ≥
        gpu_anchor.start_ns + delta
      - add edge (gpu_anchor.nid, cpu_anchor.nid)

    Edge timing uses the LP's time axis (sim_t if sim_times given).
    """
    L_windows = result.per_window_lateness_ns
    windows = result.window_bounds_ns
    if not L_windows or not windows:
        return [], {"reason": "no per-window data from LP"}

    bw_Bpns = max(effective_h2d_bw(hw), 1e-18)  # bytes/ns
    # delta in ns: bytes-of-lateness / bytes-per-ns = L (ns). So delta = L.
    # That is: 1 ns of lateness costs 1 ns of CPU wait. Adjust below
    # via safety factor if desired.

    # Build node timelines in LP time axis.
    def _lp_t(nid: int, fallback: int) -> int:
        if sim_times is None:
            return fallback
        st = sim_times.get(int(nid))
        return int(st[0]) if st is not None else fallback

    # Two indexes per node: lp_t for lateness-threshold matching;
    # trace_t for anchor ordering (cycle avoidance — cpu_node we pick
    # must be "future" in trace order vs gpu_node, otherwise gpu→cpu
    # closes a cycle through the existing cpu_leaf→submit→gpu_runtime
    # chain).
    gpu_events: list[tuple[int, int, int]] = []     # (lp_start, trace_start, nid)
    cpu_events: list[tuple[int, int, int]] = []     # (lp_start, trace_start, nid)
    for nid, node in trace.node_map.items():
        rk = str((node.args or {}).get("resource_kind") or "")
        role = str((node.args or {}).get("runtime_role") or "")
        ts = int((node.args or {}).get("start_ns") or 0)
        if ts <= 0:
            continue
        lp_t = _lp_t(int(nid), ts)
        if rk in _GPU_RESOURCE_KINDS:
            gpu_events.append((lp_t, ts, int(nid)))
        elif role == "cpu_leaf":
            cpu_events.append((lp_t, ts, int(nid)))
    gpu_events.sort()
    cpu_events.sort()

    if not gpu_events or not cpu_events:
        return [], {"reason": "no gpu or cpu events"}

    gpu_ts = [t for t, _, _ in gpu_events]
    gpu_traces = [tr for _, tr, _ in gpu_events]
    gpu_nids = [n for _, _, n in gpu_events]
    cpu_ts = [t for t, _, _ in cpu_events]
    cpu_traces = [tr for _, tr, _ in cpu_events]
    cpu_nids = [n for _, _, n in cpu_events]

    import bisect
    edges: list[tuple[int, int]] = []
    n_skip_below_threshold = 0
    n_skip_no_gpu = 0
    n_skip_no_cpu = 0
    n_skip_dup = 0
    seen: set[tuple[int, int]] = set()
    for i, L_i in enumerate(L_windows):
        if L_i <= lateness_threshold_ns:
            n_skip_below_threshold += 1
            continue
        s_i, e_i = windows[i]
        # Last gpu node whose start_ns ≤ window end
        idx = bisect.bisect_right(gpu_ts, e_i) - 1
        if idx < 0:
            n_skip_no_gpu += 1
            continue
        gpu_nid = gpu_nids[idx]
        gpu_t = gpu_ts[idx]
        gpu_trace = gpu_traces[idx]
        delta = int(L_i)  # ns
        cpu_target_t = gpu_t + delta
        # First cpu node with lp_start ≥ cpu_target_t AND
        # trace_start > gpu_trace (cycle avoidance — cpu must be in
        # the "future" relative to gpu in profile wall-clock order,
        # so the existing cpu→submit→gpu chain doesn't close on us).
        c_idx = bisect.bisect_left(cpu_ts, cpu_target_t)
        while c_idx < len(cpu_nids) and cpu_traces[c_idx] <= gpu_trace:
            c_idx += 1
        if c_idx >= len(cpu_nids):
            n_skip_no_cpu += 1
            continue
        cpu_nid = cpu_nids[c_idx]
        pair = (gpu_nid, cpu_nid)
        if pair in seen:
            n_skip_dup += 1
            continue
        seen.add(pair)
        edges.append(pair)

    diag = {
        "n_windows_total": len(L_windows),
        "n_windows_above_threshold": len(L_windows) - n_skip_below_threshold,
        "n_edges_emitted": len(edges),
        "n_skip_no_gpu": n_skip_no_gpu,
        "n_skip_no_cpu": n_skip_no_cpu,
        "n_skip_dup": n_skip_dup,
        "lateness_threshold_ns": lateness_threshold_ns,
        "total_predicted_stall_ms": sum(
            L_i for L_i in L_windows if L_i > lateness_threshold_ns
        ) / 1e6,
    }
    if audit:
        print(
            f"[ct_milp_lateness:backpressure] derived {len(edges)} edges "
            f"({diag['n_windows_above_threshold']}/{diag['n_windows_total']} "
            f"windows above {lateness_threshold_ns/1e3:.0f}us threshold; "
            f"total predicted stall = "
            f"{diag['total_predicted_stall_ms']:.1f}ms)"
        )
    return edges, diag


def _build_gpu_consumer_timeline(
    trace: Trace,
    sim_times: dict[int, tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """All gpu compute nodes sorted by the LP's time axis.

    Uses sim_time_start when ``sim_times`` provided, else trace_start_ns.
    """
    out: list[tuple[int, int]] = []
    for nid, node in trace.node_map.items():
        rk = str((node.args or {}).get("resource_kind") or "")
        if rk not in _GPU_RESOURCE_KINDS:
            continue
        nid_i = int(nid)
        start_ns = int((node.args or {}).get("start_ns") or 0)
        if start_ns <= 0:
            continue
        if sim_times is not None:
            st = sim_times.get(nid_i)
            if st is not None:
                start_ns = int(st[0])
        out.append((nid_i, int(start_ns)))
    out.sort(key=lambda x: x[1])
    return out


# ---------------------------------------------------------------------------
# LP
# ---------------------------------------------------------------------------


@dataclass
class _LPResult:
    c_solution: dict[int, float]                       # cgsim_tid → c value
    e_solution: dict[tuple[int, int], float]           # (cgsim_tid, k) → e value
    forced_cold: set[int]                              # cgsim_tids forced to c=1
    feasible_tids: list[int]                           # tids that entered LP as vars
    peak_bytes: int                                    # LP-modeled peak VRAM
    lateness_ns: int                                   # LP's L_max in ns
    peak_overrun_bytes: int                            # LP's s_P
    target_infeasible: bool
    solver_status: str
    diagnostics: dict[str, Any]
    # Per-window lateness slack (ns). Length = NUM_LATENESS_WINDOWS.
    # Used for backpressure-edge derivation in solve_neutral.
    per_window_lateness_ns: list[int] = field(default_factory=list)
    # Window bounds in the LP's time axis (sim_t if available, else
    # trace_t). Length = NUM_LATENESS_WINDOWS, (s_i, e_i) per window.
    window_bounds_ns: list[tuple[int, int]] = field(default_factory=list)


def _select_sample_points(
    gpu_consumers: list[tuple[int, int]],
    max_samples: int = 256,
) -> list[tuple[int, int]]:
    """Pick a representative subset of gpu consumer timeline points.

    The LP's peak and lateness constraints scale with #samples × #tids.
    For 10k+ gpu nodes the dense matrix gets unwieldy. Sampling uniformly
    across the timeline keeps constraint count tractable while still
    catching the moments where cumulative PCIe load peaks and where vram
    residency is densest.
    """
    if len(gpu_consumers) <= max_samples:
        return list(gpu_consumers)
    step = len(gpu_consumers) / max_samples
    picked: list[tuple[int, int]] = []
    for i in range(max_samples):
        idx = int(i * step)
        picked.append(gpu_consumers[idx])
    # Always include the very last sample so the cumulative lateness at
    # the end of the run is constrained.
    if picked[-1] != gpu_consumers[-1]:
        picked.append(gpu_consumers[-1])
    return picked


def _stream_cold_tensors_to_cover_overrun(
    pool: dict[int, _PoolTensor],
    feasible_tids: list[int],
    c_solution: dict[int, float],
    e_solution: dict[tuple[int, int], float],
    target_adj_bytes: float,
    peak_fn: "Any",
) -> dict[str, int]:
    """Flip cold tensors to streamed until the TRUE modeled peak fits.

    ``peak_fn(c_solution, e_solution)`` recomputes the real max-over-samples
    alive-set peak for the current assignment. We flip cold→streamed in
    priority order and *recompute* the true peak rather than assuming each
    flipped byte removes a byte from peak (the old code subtracted
    ``streamed_bytes`` from ``P`` directly — a flipped tensor still occupies
    ``size`` at its consumers and in-flight, so that under-credited the peak
    and silently shipped plans that overran in sim).

    Selection policy:
      1. tensors with fewer consumers first;
      2. then tensors whose first use is farthest in the future;
      3. then larger tensors, to avoid excessive tiny picks.

    Only c-feasible tensors are candidates because c-infeasible and
    forced-cold tensors cannot be safely initial-prefetched. Converges to
    the stream-everything-feasible plan in the limit; if even that overruns
    (cap below the streaming floor) the residual is reported honestly so
    the caller sets ``target_infeasible=True``.
    """
    candidates: list[tuple[int, int, int, int, int]] = []
    for tid in feasible_tids:
        pt = pool[tid]
        if not pt.c_feasibility:
            continue
        if float(c_solution.get(tid, 1.0)) < 0.5:
            continue
        first_use = int(pt.consumers[0][1]) if pt.consumers else 0
        candidates.append((
            len(pt.consumers),
            -first_use,
            -int(pt.size_bytes),
            int(tid),
            int(pt.size_bytes),
        ))
    candidates.sort()

    streamed_bytes = 0
    streamed_count = 0
    cur_peak = float(peak_fn(c_solution, e_solution))
    idx = 0
    n = len(candidates)
    # Flip in byte-sized batches (estimated from the current overrun), then
    # recompute the true peak; repeat while still over. Each round's batch
    # is an over-estimate of the reduction, so a few rounds converge.
    while cur_peak > target_adj_bytes and idx < n:
        overrun = cur_peak - target_adj_bytes
        batch_bytes = 0.0
        while idx < n and batch_bytes < overrun:
            _n_uses, _neg_first_use, _neg_size, tid, size = candidates[idx]
            idx += 1
            if float(c_solution.get(tid, 1.0)) < 0.5:
                continue
            pt = pool[tid]
            c_solution[tid] = 0.0
            for k in range(len(pt.consumers) - 1):
                if pt.gap_feasibility[k]:
                    e_solution[(tid, k)] = 1.0
            streamed_bytes += size
            streamed_count += 1
            batch_bytes += size
        cur_peak = float(peak_fn(c_solution, e_solution))

    # peak_fn returns model units (MB); streamed_bytes is raw bytes (for
    # display). Caller multiplies the *_model fields by MODEL_SCALE.
    return {
        "streamed_count": streamed_count,
        "streamed_bytes": streamed_bytes,
        "final_peak_model": float(cur_peak),
        "residual_overrun_model": max(0.0, float(cur_peak - target_adj_bytes)),
    }


def _solve_lp_highspy(
    *,
    total_vars: int,
    c_obj: np.ndarray,
    bounds_list: list[tuple[float, float | None]],
    rows: list[int],
    cols: list[int],
    vals: list[float],
    ub_list: list[float],
    time_limit_s: float | None,
    solver_threads: int,
    audit: bool,
) -> tuple[np.ndarray | None, bool, str]:
    """LP relaxation via highspy directly (no scipy wrapper).

    Why: scipy.linprog's HiGHS wrapper hides the time-limit-feasible
    primal — it returns ``success=False, x=None`` when HiGHS reports
    "Time limit reached, primal_status is Feasible". Going through
    highspy directly lets us read ``h.getSolution().col_value`` even
    on time-limit. Also avoids a non-deterministic heap-corruption
    bug we observed in scipy/HiGHS at long iteration counts on dense
    problems.

    Returns (x, has_primal, message). ``has_primal`` is True if we
    have ANY feasible point (optimal or time-limited feasible), False
    if no primal was found at all.
    """
    inf = highspy.kHighsInf

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    # Conditioning is handled by building the model in MB/ms (see
    # MODEL_SCALE), so no HiGHS user_bound_scale / user_objective_scale
    # hacks are needed.
    _set_highs_threads(h, solver_threads)
    if time_limit_s is not None:
        h.setOptionValue("time_limit", float(time_limit_s))

    lo_arr = [float(b[0]) for b in bounds_list]
    hi_arr = [
        float(b[1]) if b[1] is not None else inf for b in bounds_list
    ]
    obj_arr = [float(c_obj[i]) for i in range(total_vars)]
    h.addVars(total_vars, lo_arr, hi_arr)
    h.changeColsCost(total_vars, list(range(total_vars)), obj_arr)

    row_data: dict[int, list[tuple[int, float]]] = {}
    for r, c, v in zip(rows, cols, vals):
        row_data.setdefault(int(r), []).append((int(c), float(v)))
    n_rows = len(ub_list)
    for r in range(n_rows):
        entries = row_data.get(r)
        if not entries:
            continue
        col_idx = [e[0] for e in entries]
        coef = [e[1] for e in entries]
        h.addRow(-inf, float(ub_list[r]), len(col_idx), col_idx, coef)

    h.run()
    status = h.getModelStatus()
    status_str = h.modelStatusToString(status)
    info = h.getInfo()
    # HighsSolutionStatus.kSolutionStatusFeasible = 2.
    primal_status = int(getattr(info, "primal_solution_status", 0))
    primal_feasible = primal_status == 2
    if audit:
        print(
            f"[ct_milp_lateness:solver] highspy LP done: "
            f"model_status={status_str!r} primal_status={primal_status}"
        )
    if status == highspy.HighsModelStatus.kOptimal:
        sol = h.getSolution()
        x = np.asarray(list(sol.col_value), dtype=np.float64)
        return (x, True, f"LP optimal ({status_str})")
    # Accept ANY model status as long as a feasible primal exists.
    # HiGHS sometimes reports "TimeLimit" or "Unknown" while still
    # having a feasible incumbent — scipy's wrapper drops these but
    # we can read them via highspy directly.
    if primal_feasible:
        sol = h.getSolution()
        x = np.asarray(list(sol.col_value), dtype=np.float64)
        return (x, True, f"LP {status_str} with feasible primal")
    return (None, False, f"LP solver returned {status_str} (no primal)")


def _build_highspy_model(
    *,
    total_vars: int,
    c_obj: np.ndarray,
    bounds_list: list[tuple[float, float | None]],
    rows: list[int],
    cols: list[int],
    vals: list[float],
    ub_list: list[float],
    time_limit_s: float | None,
    solver_threads: int,
) -> Any:
    """Build a continuous highspy model; caller may add integrality."""
    inf = highspy.kHighsInf

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    # Conditioning is handled by building the model in MB/ms (see
    # MODEL_SCALE), so no HiGHS user_bound_scale / user_objective_scale
    # hacks are needed.
    _set_highs_threads(h, solver_threads)
    if time_limit_s is not None:
        h.setOptionValue("time_limit", float(time_limit_s))

    lo_arr = [float(b[0]) for b in bounds_list]
    hi_arr = [
        float(b[1]) if b[1] is not None else inf for b in bounds_list
    ]
    obj_arr = [float(c_obj[i]) for i in range(total_vars)]
    h.addVars(total_vars, lo_arr, hi_arr)
    h.changeColsCost(total_vars, list(range(total_vars)), obj_arr)

    row_data: dict[int, list[tuple[int, float]]] = {}
    for r, c, v in zip(rows, cols, vals):
        row_data.setdefault(int(r), []).append((int(c), float(v)))
    n_rows = len(ub_list)
    for r in range(n_rows):
        entries = row_data.get(r)
        if not entries:
            continue
        col_idx = [e[0] for e in entries]
        coef = [e[1] for e in entries]
        h.addRow(-inf, float(ub_list[r]), len(col_idx), col_idx, coef)

    return h


def _solve_integer_highspy_model(
    h: Any,
    *,
    total_vars: int,
    integrality_arr: np.ndarray,
    warm_start: np.ndarray | None,
    label: str,
) -> tuple[np.ndarray | None, bool, str, str, bool]:
    """Run a highspy model as a MILP, optionally with an incumbent."""
    int_indices = [i for i in range(total_vars) if integrality_arr[i] == 1]
    h.changeColsIntegrality(
        len(int_indices),
        int_indices,
        [highspy.HighsVarType.kInteger] * len(int_indices),
    )
    if warm_start is not None:
        sol = highspy.HighsSolution()
        sol.col_value = list(warm_start)
        h.setSolution(sol)

    h.run()
    status = h.getModelStatus()
    status_str = h.modelStatusToString(status)
    if status == highspy.HighsModelStatus.kOptimal:
        final = np.asarray(list(h.getSolution().col_value), dtype=np.float64)
        return (final, True, f"{label} MILP optimal", status_str, False)

    info = h.getInfo()
    primal_status = int(getattr(info, "primal_solution_status", 0))
    primal_feasible = primal_status == 2
    if primal_feasible:
        final = np.asarray(list(h.getSolution().col_value), dtype=np.float64)
        binary_mask = np.asarray(integrality_arr) == 1
        bvals = final[binary_mask]
        is_integer = bool(np.all(
            (bvals < 0.01) | (bvals > 0.99)
        ))
        if is_integer:
            return (
                final, True,
                f"{label} MILP {status_str} (returning incumbent)",
                status_str, False,
            )
    if warm_start is not None and status == highspy.HighsModelStatus.kTimeLimit:
        return (
            warm_start, True,
            f"{label} MILP {status_str} (returning warm-start)",
            status_str, True,
        )

    return (
        None, False,
        f"{label} MILP returned {status_str}",
        status_str, True,
    )


def _solve_two_phase_highspy(
    *,
    total_vars: int,
    c_obj: np.ndarray,
    bounds_list: list[tuple[float, float | None]],
    rows: list[int],
    cols: list[int],
    vals: list[float],
    ub_list: list[float],
    integrality_arr: np.ndarray,
    time_limit_s: float | None,
    solver_threads: int,
    audit: bool,
    phase1_time_limit_s: float | None = None,
    feasible_fallback: np.ndarray | None = None,
) -> tuple[np.ndarray | None, bool, str, str, bool]:
    """Two-phase solve via highspy.

    Phase 1: build the LP and solve it as a relaxation (all binaries
    continuous in [0, 1]). Fast (~1 s on sd3med 14g) — gives a
    near-integer point. Pass ``phase1_time_limit_s`` (defaults to
    ``time_limit_s``) to bound this phase separately — useful when
    the LP relaxation is large enough that it can't solve in a few
    seconds, in which case the warm-start it provides isn't worth
    much and we'd rather give the full budget to phase 2.

    Phase 2: round the Phase-1 solution to integer-feasible (binary
    vars rounded at 0.5; continuous vars kept as-is from the LP),
    flip the binaries to integer, pass the rounded values as
    warm-start via ``Highs.setSolution()``, and run MILP. A good
    warm-start sets a tight initial incumbent, which prunes the
    branch-and-bound tree aggressively. On problems where the LP
    relaxation is already 99 %+ integer (typical for this LP), MILP
    converges to the proven optimum in a small fraction of the
    cold-start budget.

    If phase 1 does not prove LP optimality, skip the warm start and
    run the MILP cold. Phase 1 is only a performance hint; it should
    not gate correctness.

    Returns (x, success, message, status_str, lp_only) where
    ``lp_only=True`` means Phase 2 didn't complete with a proven
    integer solution and the LP relaxation was used as the final
    plan (rounding at emit time, same as the legacy fallback).
    """
    _phase1_limit = (
        float(phase1_time_limit_s)
        if phase1_time_limit_s is not None
        else (float(time_limit_s) if time_limit_s is not None else None)
    )
    h = _build_highspy_model(
        total_vars=total_vars,
        c_obj=c_obj,
        bounds_list=bounds_list,
        rows=rows,
        cols=cols,
        vals=vals,
        ub_list=ub_list,
        time_limit_s=_phase1_limit,
        solver_threads=solver_threads,
    )
    if audit and _phase1_limit is not None:
        print(
            f"[ct_milp_lateness:solver] phase1 LP time_limit={_phase1_limit:.1f}s "
            f"(phase2 MILP gets full time_limit_s={time_limit_s})"
        )

    # ---- Phase 1: LP relaxation (no integrality yet) ----
    h.run()
    status = h.getModelStatus()
    if status != highspy.HighsModelStatus.kOptimal:
        msg = h.modelStatusToString(status)
        seeded = feasible_fallback is not None
        if audit:
            print(
                f"[ct_milp_lateness:solver] phase1 LP not optimal "
                f"({msg}); running MILP "
                f"{'seeded with stream-everything feasible incumbent' if seeded else 'cold'}"
            )
        h_cold = _build_highspy_model(
            total_vars=total_vars,
            c_obj=c_obj,
            bounds_list=bounds_list,
            rows=rows,
            cols=cols,
            vals=vals,
            ub_list=ub_list,
            time_limit_s=time_limit_s,
            solver_threads=solver_threads,
        )
        return _solve_integer_highspy_model(
            h_cold,
            total_vars=total_vars,
            integrality_arr=integrality_arr,
            warm_start=feasible_fallback,
            label=(
                f"feasible-seed after phase1 {msg}" if seeded
                else f"cold-start after phase1 {msg}"
            ),
        )
    lp_sol = h.getSolution()
    x_lp = np.asarray(list(lp_sol.col_value), dtype=np.float64)
    if audit:
        # c-value distribution of the LP relaxation — useful for
        # diagnosing how tight the relaxation is.
        binary_mask = np.asarray(integrality_arr) == 1
        bvals = x_lp[binary_mask]
        n_zero = int(np.sum(bvals < 0.01))
        n_one = int(np.sum(bvals > 0.99))
        n_frac = int(np.sum((bvals >= 0.01) & (bvals <= 0.99)))
        print(
            f"[ct_milp_lateness:audit] phase 1 LP relaxation: "
            f"binaries ≈0: {n_zero}, ≈1: {n_one}, fractional: {n_frac}"
        )

    # ---- Phase 2: round, set integrality, warm-start, solve MILP ----
    x_warm = x_lp.copy()
    int_indices = [i for i in range(total_vars) if integrality_arr[i] == 1]
    for i in int_indices:
        x_warm[i] = 1.0 if x_lp[i] >= 0.5 else 0.0

    return _solve_integer_highspy_model(
        h,
        total_vars=total_vars,
        integrality_arr=integrality_arr,
        warm_start=x_warm,
        label="phase2",
    )


def _solve_milp(
    pool: dict[int, _PoolTensor],
    trace: Trace,
    hw: HwParams,
    *,
    sim_times: dict[int, tuple[int, int]] | None = None,
    peak_target_bytes: int | None,
    extra_static_bytes: int,
    safety_margin_frac: float,
    max_peak_samples: int,
    time_limit_s: float | None,
    solver_threads: int | None,
    lp_relaxation: bool,
    arc_queue_factor: float = 1.0,
    lateness_peak_coupling: bool = False,
    relax_cinfeasible: bool = False,
    intermediate_axis_fix: bool = True,
    audit: bool,
    phase1_time_limit_s: float | None = None,
) -> _LPResult:
    """Build and solve the lateness MILP.

    Variables:
        c_t           binary per feasible pool tid
        e_{t,k}       binary per feasible cross-iter gap
        P             continuous ≥ 0 — modeled peak (driven by per-sample rows)
        L_window_i    continuous ≥ 0 per timeline window — per-window
                      stall slack; total ns of stall = Σ L_window_i
        s_P           continuous ≥ 0 — peak overrun slack (bytes)
    """
    solver_threads = _normalize_solver_threads(solver_threads)
    if audit:
        print(f"[ct_milp_lateness:audit] solver_threads={solver_threads}")

    # ``relax_cinfeasible`` (param, default off): let the LP stream
    # c-infeasible tids instead of pinning them resident, trading a startup
    # stall (priced by the lateness rows) for VRAM at tight caps. When on,
    # this also keeps the would-be `forced_cold` tids in the LP as
    # streamable vars (c∈{0,1}, no e-vars since they have no feasible gap):
    # streaming them frees VRAM OUTSIDE their consumer span (a big win for
    # short-span tids like embedding/lm_head). See the bounds section.
    _relax_cinfeasible = relax_cinfeasible

    # ---- 1. Feasibility filter ----
    feasible_tids: list[int] = []
    forced_cold: set[int] = set()
    for tid, pt in pool.items():
        # Forced cold iff initial prefetch cannot fit (consumer_0 too
        # early) AND no cross-iter gap admits a refetch either: the LP
        # has no choice but to keep the tid resident from layout.
        if not pt.c_feasibility and not any(pt.gap_feasibility):
            if not _relax_cinfeasible:
                forced_cold.add(tid)
                continue
        feasible_tids.append(tid)

    nv = len(feasible_tids)
    if audit:
        forced_bytes = sum(pool[t].size_bytes for t in forced_cold)
        c_feas_false = [t for t in feasible_tids if not pool[t].c_feasibility]
        c_feas_false_bytes = sum(pool[t].size_bytes for t in c_feas_false)
        print(
            f"[ct_milp_lateness:audit] pool size={len(pool)} tensors "
            f"forced_cold={len(forced_cold)} ({forced_bytes/1e6:.1f}MB) "
            f"feasible_lp_vars={nv} c_feas_false_bound_cold={len(c_feas_false)} "
            f"({c_feas_false_bytes/1e6:.1f}MB)"
        )

    # ---- 2. Variable layout ----
    c_var_idx: dict[int, int] = {}
    for col, tid in enumerate(feasible_tids):
        c_var_idx[tid] = col

    e_var_idx: dict[tuple[int, int], int] = {}
    col = nv
    for tid in feasible_tids:
        pt = pool[tid]
        for k in range(len(pt.consumers) - 1):
            # Only feasible gaps get e variables; infeasible ones are
            # implicit (e ≡ 0).
            if pt.gap_feasibility[k]:
                e_var_idx[(tid, k)] = col
                col += 1
    n_e = col - nv

    P_IDX = col
    col += 1
    # Per-window stall slacks: one continuous L_i ≥ 0 per timeline
    # window. Replaces the single L_max — a single global slack let
    # the LP "average" PCIe load across iters, hiding per-iter
    # saturation that sim couldn't actually realize. Per-window
    # forces the LP to keep PCIe density within each window's
    # available time.
    NUM_LATENESS_WINDOWS = 20
    L_WINDOW_IDX_BASE = col
    col += NUM_LATENESS_WINDOWS
    S_PEAK_IDX = col
    col += 1
    total_vars = col

    # ---- 3. Variable bounds + integrality ----
    # c-infeasible tids: pinned c=1 by default, or streamable (0,1) when
    # `relax_cinfeasible` is set (read above at the feasibility filter).
    # forced_cold tids, when relaxed, are also in feasible_tids here and
    # get (0,1) via the same `not c_feasibility` branch.
    bounds_list: list[tuple[float, float | None]] = []
    for tid in feasible_tids:
        pt = pool[tid]
        if not pt.c_feasibility and not _relax_cinfeasible:
            # No GPU node fires early enough that a runtime prefetch
            # could deliver this tid in time. Pin c=1 → load at layout
            # phase, before runtime PCIe contention starts. The c+e<=1
            # coupling below is SUPPRESSED for these tids so e_{t,k}
            # can still be 1 on feasible gaps — yielding the hybrid
            # `cold-at-layout + evict-between-consumers + refetch`
            # pattern. See the coupling section and the emit path.
            bounds_list.append((1.0, 1.0))
        else:
            bounds_list.append((0.0, 1.0))
    bounds_list.extend([(0.0, 1.0)] * n_e)
    bounds_list.append((0.0, None))                          # P
    bounds_list.extend(
        [(0.0, None)] * NUM_LATENESS_WINDOWS
    )                                                        # L_window_i
    bounds_list.append(
        (0.0, None) if peak_target_bytes is not None else (0.0, 0.0)
    )                                                        # s_P

    integrality_arr = None
    if not lp_relaxation:
        integrality_arr = np.zeros(total_vars, dtype=np.int64)
        integrality_arr[: nv + n_e] = 1

    # ---- 4. Objective ----
    #
    # Primary: L_max (max consumer lateness in ns).
    # Hard penalty: Λ · s_P (peak overrun slack in bytes; 1e6 ns/byte).
    # Tiebreaker: cumulative H2D time per byte streamed +
    #             per byte refetched, priced at (1/bw_h2d) ns/byte.
    #
    # Without the tiebreaker, when the cap admits an L_max=0 plan, the
    # LP picks an arbitrary feasible point — often heavy streaming
    # (more PCIe, more queue contention, worse sim e2e) — because cold
    # has zero direct reward. The 1/bw weight is the actual PCIe time
    # saved per cold byte, so the tiebreaker exactly prices streaming
    # at its physical cost.
    # ε scales the per-byte streaming cost. Using 1.0 matches the
    # current ct_milp_multistream "minimize streaming bytes" scale,
    # which empirically lets the LP find tight cap-binding plans.
    # Smaller ε leaves L_max as the dominant signal (LP arbitrary at
    # equal-lateness plans); larger ε pushes the LP to MAX cold
    # subject to peak, mirroring the multistream objective.
    epsilon_per_byte = 1.0  # ns / byte
    c_obj = np.zeros(total_vars, dtype=np.float64)
    # Per-window slack: cost 1 ns of objective per ns of stall in
    # each window. Sum across windows = total wall-clock extension
    # (stalls cascade physically). With independent slacks the LP
    # can't shift PCIe load across windows to "hide" per-iter
    # saturation.
    # Objective weights are UNCHANGED from the byte/ns formulation — because
    # bytes→MB and ns→ms both divide by MODEL_SCALE, every objective term
    # (lateness, streaming, peak-slack) shrinks by the same factor, so the
    # argmin is identical. We just express size in MB so the cost vector
    # sits in [1, 1e6] instead of [1, 1e8].
    for i in range(NUM_LATENESS_WINDOWS):
        c_obj[L_WINDOW_IDX_BASE + i] = 1.0          # per ms of stall
    if peak_target_bytes is not None:
        c_obj[S_PEAK_IDX] = PEAK_SLACK_PENALTY      # per MB over cap
    for tid in feasible_tids:
        size_mb = pool[tid].size_bytes / MODEL_SCALE
        c_obj[c_var_idx[tid]] = -float(size_mb) * epsilon_per_byte
    for (tid, k), col in e_var_idx.items():
        size_mb = pool[tid].size_bytes / MODEL_SCALE
        c_obj[col] = float(size_mb) * epsilon_per_byte

    # ---- 5. Coupling: forbid cold-start + runtime refetch hybrid ----
    #
    # The injector's coverage_repair pass does not treat cold-start
    # residency as a gate for consumers before a later runtime refetch.
    # For *c-feasible* tids the LP could otherwise spuriously pick
    # `c=1, e=1` and the injector would silently demote the tid back
    # to cuda-resident, adding peak the LP didn't see. Forbid that
    # combination on those tids:
    #
    #   c_t + e_{t,k} <= 1   (only when c_t is a free [0,1] variable)
    #
    # For *c-infeasible* tids the c bound is pinned to (1, 1) — they
    # are going to be cold-resident no matter what. Allowing e_{t,k}=1
    # for them unlocks the hybrid `cold-at-layout + evict-between-
    # consumers + refetch` pattern, which is the right plan for
    # widely-spaced multi-iter weights and recovers tight-budget
    # feasibility (this set is exactly the population whose dead-zone
    # residency the original c+e<=1 forced into peak). Coverage_repair
    # still demotes these tids in sim, so the LP's dead-zone savings
    # are accounting for VRAM that sim won't actually free — bump
    # ``safety_margin_frac`` if the gap matters for your cap.
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    ub_list: list[float] = []
    row = 0
    for (tid, k), e_col in e_var_idx.items():
        if not pool[tid].c_feasibility:
            # c pinned to 1 — hybrid allowed, no coupling row needed.
            continue
        rows.append(row)
        cols.append(c_var_idx[tid])
        vals.append(1.0)
        rows.append(row)
        cols.append(e_col)
        vals.append(1.0)
        ub_list.append(1.0)
        row += 1

    # ---- 6. Sample grid (shared by peak & lateness rows) ----
    #
    # Event-aligned sampling. The LP's peak P is a max over the
    # set of sampled time points; between any two adjacent samples
    # the alive set for every pool tid is piecewise constant. So
    # peak only ever transitions at one of:
    #   - GPU consumer start (consumer fires → "currently consumed"
    #     window),
    #   - consumer end+1 (consumer done → gap residency may go
    #     dead under hybrid),
    #   - arc_start = next_consumer.start − τ · arc_queue_factor
    #     (refetch arrival window opens → tid must be alive).
    # Sampling at every union of those events makes the LP's
    # modeled peak its true peak under the residency model.
    # The legacy uniform-256-stride over GPU consumers left ~30
    # events between samples on llama3b — enough room for ~300 MB
    # of hidden transient peak on a 5g cap. That was the dominant
    # contributor to LP-vs-sim divergence on tight budgets.
    gpu_consumers = _build_gpu_consumer_timeline(trace, sim_times)
    if not gpu_consumers:
        raise RuntimeError(
            "[ct_milp_lateness] no gpu consumer events in trace; "
            "cannot build LP sample grid."
        )

    # Precompute per-tid arc_tau and consumer_nids_set for the
    # per-sample LP build inner loop (consumed below) — saves a
    # per-iteration linear scan of pt.consumers for the
    # is_currently_consumed check.
    _pt_arc_tau: dict[int, int] = {
        tid: int(round(pool[tid].tau_h2d_ns * float(arc_queue_factor)))
        for tid in feasible_tids
    }
    _pt_consumer_nids_set: dict[int, frozenset[int]] = {
        tid: frozenset(c[0] for c in pool[tid].consumers)
        for tid in feasible_tids
    }
    _pt_first_start: dict[int, int] = {
        tid: pool[tid].consumers[0][1] for tid in feasible_tids
    }
    _pt_last_end: dict[int, int] = {
        tid: pool[tid].consumers[-1][2] for tid in feasible_tids
    }

    seen_t: set[int] = set()
    samples: list[tuple[int, int]] = []
    for nid_e, t_e in gpu_consumers:
        if t_e in seen_t:
            continue
        samples.append((int(nid_e), int(t_e)))
        seen_t.add(int(t_e))
    n_gpu_samples = len(samples)

    # arc_start (refetch arrival window) events for every pool tid:
    # the moment when a streaming tid must be alive because its
    # prefetch is in-flight. These are the binding peak moments the
    # legacy 256-uniform-stride grid systematically missed.
    # Collect arc_start times deduped by time, recording the LARGEST
    # tid size whose arc opens at that time. Size drives the thinning
    # policy below: big tids dominate peak, so their arc moments are
    # the ones worth keeping when the grid must be capped.
    arc_size_by_t: dict[int, int] = {}
    for _tid in feasible_tids:
        _tau = _pt_arc_tau[_tid]
        _size = int(pool[_tid].size_bytes)
        _consumers = pool[_tid].consumers
        for _c_nid, _c_start, _c_end in _consumers:
            _arc_start = max(0, int(_c_start) - _tau)
            if _arc_start in seen_t:
                continue
            prev = arc_size_by_t.get(_arc_start)
            if prev is None or _size > prev:
                arc_size_by_t[_arc_start] = _size
    for _arc_start in arc_size_by_t:
        samples.append((-1, _arc_start))
        seen_t.add(_arc_start)
    n_arc_samples = len(arc_size_by_t)
    n_post_consumer_samples = 0

    # Big intermediate producer events (large transient intermediates
    # whose producer doesn't already coincide with a sampled event).
    BIG_INTERM_BYTES = 64 * 1024 * 1024  # 64 MB threshold
    n_big_interm_samples = 0
    for nid, node in trace.node_map.items():
        start_ns_trace = int((node.args or {}).get("start_ns") or 0)
        if start_ns_trace <= 0:
            continue
        sample_t = start_ns_trace
        if sim_times is not None:
            st = sim_times.get(int(nid))
            if st is not None:
                sample_t = int(st[0])
        has_big = False
        for tid in (node.output_tensors or []):
            t = trace.tensor_map.get(int(tid))
            if t is None:
                continue
            if (t.args or {}).get("tensor_type") != "INTERMEDIATE":
                continue
            dev = str((t.args or {}).get("device", "")).lower()
            if not dev.startswith("cuda"):
                continue
            if int(t.size_bytes) >= BIG_INTERM_BYTES:
                has_big = True
                break
        if not has_big:
            continue
        if sample_t in seen_t:
            continue
        samples.append((int(nid), sample_t))
        seen_t.add(sample_t)
        n_big_interm_samples += 1

    samples.sort(key=lambda x: x[1])

    # Safety cap for very large traces. The arc_start events (one per
    # distinct consumer time of every streaming tid) DOMINATE the grid
    # on multi-iter workloads — e.g. llama8b: ~22k arc vs ~9k consumer
    # events, sd3med: ~69k arc. The legacy guard kept ALL arc samples
    # and thinned only consumers, so --max-peak-samples had no effect
    # on the binding term and the LP relaxation itself timed out
    # (falling back to stream-everything). We now thin arc samples too:
    #   - keep the arcs of the LARGEST tids (they dominate peak), plus
    #   - a uniform time-spread of the rest (preserves coverage of
    #     binding moments across the whole timeline).
    # Consumer / big-interm samples are thinned uniformly as before.
    # max_peak_samples ≤ 0 disables the cap.
    n_arc_kept = n_arc_samples
    if max_peak_samples > 0:
        arc_samples = [s for s in samples if s[0] == -1]
        other_samples = [s for s in samples if s[0] != -1]
        # Arc budget: a small multiple of max_peak_samples is plenty —
        # adjacent arcs constrain nearly the same alive set.
        arc_budget = max_peak_samples * 2
        if len(arc_samples) > arc_budget:
            n_by_size = arc_budget // 2
            by_size = sorted(
                arc_samples,
                key=lambda s: arc_size_by_t.get(s[1], 0),
                reverse=True,
            )
            keep_size = by_size[:n_by_size]
            keep_size_t = {s[1] for s in keep_size}
            rest = [s for s in arc_samples if s[1] not in keep_size_t]
            n_stride = arc_budget - len(keep_size)
            if rest and n_stride > 0:
                step = len(rest) / float(n_stride)
                keep_stride = [rest[int(i * step)] for i in range(n_stride)]
            else:
                keep_stride = []
            arc_samples = keep_size + keep_stride
        n_arc_kept = len(arc_samples)
        if len(other_samples) > max_peak_samples:
            step = len(other_samples) / float(max_peak_samples)
            other_samples = [
                other_samples[int(i * step)] for i in range(max_peak_samples)
            ]
        samples = sorted(arc_samples + other_samples, key=lambda x: x[1])

    if audit:
        print(
            f"[ct_milp_lateness:audit] event-aligned sample grid: "
            f"gpu_consumer={n_gpu_samples} "
            f"arc_start={n_arc_samples}→{n_arc_kept} "
            f"post_consumer={n_post_consumer_samples} "
            f"big_interm={n_big_interm_samples} "
            f"total={len(samples)} "
            f"(legacy uniform stride would have been "
            f"{min(len(gpu_consumers), max_peak_samples)})",
            flush=True,
        )

    # ---- 7. Peak VRAM rows (one per sample point) ----
    #
    # At each sample T_i: P ≥ const_i + Σ var_coef_i · vars
    #   const_i = forced_cold bytes + extras + Σ size_t (alive unconditional)
    #   var_coef_i[c_t] = size_t when t's contribution at T_i is "size · c"
    #
    # Region classification per (t, T_i):
    #   T_i < first_consumer:        pre-arc (alive iff c=1) OR arc_0 (always)
    #   T_i > last_consumer.end:     post (alive iff c=1)
    #   T_i in gap (k, k+1):
    #       T_i ≥ arc_{k+1}_start:   always alive
    #       gap_feasible & T_i below: dead zone, alive iff c=1
    #       gap infeasible:          always alive (no evict can fit)
    #   T_i within a consumer's [start, end]: always alive (currently consumed)
    forced_cold_bytes = sum(pool[t].size_bytes for t in forced_cold)
    # Model is in MB: constant_floor (and every size/const below) is bytes/S.
    constant_floor = (float(forced_cold_bytes) + float(extra_static_bytes)) / MODEL_SCALE

    # Intermediates: fixed (non-variable) VRAM residency per sample.
    _intermediates = _build_intermediate_residencies(
        trace, sim_times, axis_fix=intermediate_axis_fix,
    )
    if audit and _intermediates:
        sizes_alive = []
        for _, t_l in samples:
            s = sum(sz for s_, e_, sz in _intermediates if s_ <= t_l <= e_)
            sizes_alive.append(s)
        print(
            f"[ct_milp_lateness:audit] intermediates: "
            f"n={len(_intermediates)} per-sample residency MB "
            f"min={min(sizes_alive)/1e6:.1f} "
            f"max={max(sizes_alive)/1e6:.1f} "
            f"mean={sum(sizes_alive)/len(sizes_alive)/1e6:.1f}"
        )

    # Option-(b) lateness→peak coupling. When window i admits L_window_i
    # ns of late streaming, those bytes (L · bw) effectively need to be
    # cold-resident (the injector demotes late tids back to cold). Couple
    # this into the per-sample peak rows so the LP can't escape peak by
    # accepting lateness.
    _bw_h2d_Bpns = max(effective_h2d_bw(hw), 1e-18)
    _bytes_per_ns_late = _bw_h2d_Bpns
    _timeline_start_w = min(c[1] for pt in pool.values() for c in pt.consumers)
    _timeline_end_w = max(c[1] for pt in pool.values() for c in pt.consumers)
    _window_length_w = max(1, (_timeline_end_w - _timeline_start_w) / NUM_LATENESS_WINDOWS)

    def _sample_window_idx(t_l_val: float) -> int:
        idx = int((t_l_val - _timeline_start_w) / _window_length_w)
        return max(0, min(NUM_LATENESS_WINDOWS - 1, idx))

    # Lateness→peak coupling is OFF by default (mis-calibrated, see the
    # per-sample gate below). Re-enable via arg or MILP_ENABLE_LP_COUPLING=1.
    _coupling_on = (
        lateness_peak_coupling
        or os.environ.get("MILP_ENABLE_LP_COUPLING") == "1"
    )

    if audit:
        print(
            f"[ct_milp_lateness:audit] lateness→peak coupling: "
            f"{'ON' if _coupling_on else 'OFF (default)'} — "
            f"bw={_bw_h2d_Bpns:.1f}GB/s → "
            f"{_bytes_per_ns_late*1e6/1e6:.1f}MB peak per 1ms window-lateness"
        )
        if arc_queue_factor != 1.0:
            print(
                f"[ct_milp_lateness:audit] arc widening: τ × "
                f"{arc_queue_factor} (models PCIe queue residency)"
            )

    # Per-sample peak terms, saved so we can recompute the TRUE modeled
    # alive-set peak for any (c, e) assignment after the solve — used by
    # the honest overrun-repair below. Each entry is
    # (const_bytes, [(var_col, coef), ...]); peak(x) = max over samples of
    # const + Σ coef·x[var_col]. This is the exact same arithmetic the
    # per-sample peak rows encode (minus the lateness→peak coupling term,
    # which is a modeling addon, not physical VRAM), so the recomputed
    # peak is consistent with the constraints by construction.
    peak_sample_terms: list[tuple[float, list[tuple[int, float]]]] = []

    n_samples_total = len(samples)
    audit_progress_step = max(1, n_samples_total // 10)
    for _sample_idx, (nid_sample, t_l) in enumerate(samples):
        if audit and _sample_idx > 0 and _sample_idx % audit_progress_step == 0:
            print(
                f"[ct_milp_lateness:audit] LP build progress: "
                f"{_sample_idx}/{n_samples_total} samples processed",
                flush=True,
            )
        const_addons = constant_floor
        # Intermediate residency at this sample (bytes → MB).
        for s_, e_, sz_ in _intermediates:
            if s_ > t_l:
                break
            if t_l <= e_:
                const_addons += sz_ / MODEL_SCALE
        var_coefs: dict[int, float] = {}
        for tid in feasible_tids:
            pt = pool[tid]
            size = pt.size_bytes / MODEL_SCALE  # MB; every use below is MB
            # Currently consumed at this exact node? O(1) via precomputed
            # frozenset of consumer node ids.
            if nid_sample in _pt_consumer_nids_set[tid]:
                const_addons += size
                continue
            first_start = _pt_first_start[tid]
            last_end = _pt_last_end[tid]

            # PCIe-queue-residency widening. With h2d_streams=1 (sim's
            # default), prefetches serialize on the PCIe channel. When
            # N issuers fire near the same time, all N tids' dst VRAM
            # is claimed at issuer-fire but transfers run one at a
            # time — so each tid stays resident for an extended window
            # of up to N×τ_h2d rather than just τ_h2d. The LP's
            # original arc model (width = τ) understates this. We
            # widen by `arc_queue_factor` to match. Set to 1.0 to
            # disable, larger to model deeper queue serialization.
            arc_tau = _pt_arc_tau[tid]

            if t_l < first_start:
                arc_0_start = max(0, first_start - arc_tau)
                if t_l >= arc_0_start:
                    const_addons += size
                else:
                    var_coefs[c_var_idx[tid]] = (
                        var_coefs.get(c_var_idx[tid], 0.0) + size
                    )
                continue
            if t_l > last_end:
                var_coefs[c_var_idx[tid]] = (
                    var_coefs.get(c_var_idx[tid], 0.0) + size
                )
                continue
            k_in = None
            for k in range(len(pt.consumers) - 1):
                if pt.consumers[k][2] < t_l < pt.consumers[k + 1][1]:
                    k_in = k
                    break
            if k_in is None:
                # Sample sits on a consumer's [start, end] boundary
                # without matching node id — treat as alive.
                const_addons += size
                continue
            arc_kp1_start = pt.consumers[k_in + 1][1] - arc_tau
            if t_l >= arc_kp1_start:
                const_addons += size
            elif pt.gap_feasibility[k_in] and (tid, k_in) in e_var_idx:
                # Dead zone of a feasible gap (HYBRID): residency is
                # `size · (1 − e_{t,k_in})`, independent of c_t.
                # Encode the constant `size` and the −size coefficient
                # on the e variable. Captures all four (c, e) patterns
                # correctly:
                #   c=0,e=0  initial prefetched, no evict → resident
                #   c=0,e=1  prefetched then evicted → gone
                #   c=1,e=0  cold, no evict → resident
                #   c=1,e=1  cold then evicted → gone
                const_addons += size
                e_col = e_var_idx[(tid, k_in)]
                var_coefs[e_col] = (
                    var_coefs.get(e_col, 0.0) - size
                )
            else:
                # Infeasible gap — no evict can fit, tensor stays.
                const_addons += size

        # Save this sample's peak terms for post-solve true-peak recompute.
        peak_sample_terms.append((
            float(const_addons),
            [(vc, cf) for vc, cf in var_coefs.items() if abs(cf) >= 1e-9],
        ))

        # P ≥ const_addons + Σ var_coef · c + α · L_window_i
        #   ⇒  Σ var_coef · c − P + α · L_window_i ≤ −const_addons
        rows.append(row)
        cols.append(P_IDX)
        vals.append(-1.0)
        for var_col, coef in var_coefs.items():
            if abs(coef) < 1e-9:
                continue
            rows.append(row)
            cols.append(var_col)
            vals.append(float(coef))
        # Lateness→peak coupling (option-b): add bw·L_window to this
        # sample's peak. OFF by default — it's mis-calibrated: it injects
        # ~bw·L of *phantom* peak into every sample of a high-stall window
        # (conflating stall time with resident bytes), over-predicting the
        # modeled peak by ~2 GiB and making the LP under-fill VRAM. Measured
        # to lose to the no-coupling LP on all of sdxl/sd3/llama3b/llama8b at
        # tight budgets (exp_results/0601_calibrate/nocoupling_4models). Its
        # stated purpose — stop the LP escaping peak by accepting lateness —
        # is already served by the lateness objective. Re-enable for A/B via
        # the `lateness_peak_coupling=True` arg or MILP_ENABLE_LP_COUPLING=1.
        # (`_coupling_on` is computed once before this loop.)
        if _coupling_on:
            _wi = _sample_window_idx(t_l)
            rows.append(row)
            cols.append(L_WINDOW_IDX_BASE + _wi)
            vals.append(float(_bytes_per_ns_late))
        ub_list.append(-float(const_addons))
        row += 1

    # ---- 7a. Cold-floor cut (DROPPED under hybrid) ----
    #
    # Pre-hybrid this row enforced `Σ size_t · c_t + const ≤ P`
    # because cold tids were alive at every sample by definition.
    # Under hybrid, a cold tid (c=1) can be evicted mid-run
    # (e_{t,k}=1), so its peak contribution at dead-zone samples is
    # `size · (1 − e_{t,k})`, not unconditional `size`. Keeping the
    # cut would over-constrain — it'd force P ≥ Σ size · c even for
    # cold tids that the LP planned to evict in their gaps.
    #
    # The per-sample peak rows above still enforce P at every binding
    # sample; the cut was only a relaxation-tightener, not a soundness
    # constraint.

    # ---- 7b. Soft peak cap: P − s_P ≤ target·(1 − margin) ----  (MB)
    if peak_target_bytes is not None:
        target_adj_mb = max(
            0.0,
            float(peak_target_bytes) * (1.0 - float(safety_margin_frac)),
        ) / MODEL_SCALE
        rows.append(row)
        cols.append(P_IDX)
        vals.append(1.0)
        rows.append(row)
        cols.append(S_PEAK_IDX)
        vals.append(-1.0)
        ub_list.append(target_adj_mb)
        row += 1
        if audit:
            print(
                f"[ct_milp_lateness:audit] peak cap: target={peak_target_bytes/1e6:.1f}MB "
                f"margin={safety_margin_frac*100:.1f}% → P ≤ {target_adj_mb:.1f}MB"
            )

    # ---- 8. Per-window lateness rows ----
    #
    # The timeline is divided into NUM_LATENESS_WINDOWS equal
    # wall-clock spans. For each window i with bounds [s_i, e_i]:
    #
    #   Σ_{t : first_consumer(t).start ∈ [s_i, e_i]}  δ_t · (1 − c_t)
    # + Σ_{(t,k) feasible : consumer_{k+1}.start ∈ [s_i, e_i]}
    #                                                 δ_t · e_{t,k}
    #   ≤ (e_i − s_i) + L_window_i
    #
    # The PCIe budget per window is the WALL-CLOCK duration of the
    # window — *not* GPU compute time. Rationale: the PCIe DMA
    # engine is a separate subsystem from CPU and GPU; it runs in
    # parallel with both compute AND idle time. The wall-clock
    # duration is the time available for PCIe to do work, whatever
    # else is happening on the host. This matches the "reuse
    # distance" interpretation — for each tensor's gap, the
    # wall-clock between consumer_k.end and consumer_{k+1}.start is
    # the available PCIe time for that tensor's evict+refetch, and
    # the per-window form aggregates these across tensors whose
    # deadlines fall in the same window.
    #
    # (An earlier version used Σ gpu duration_ns in the window as
    # the budget, on the theory that "the PCIe queue can only do
    # work during compute." That undercounted by 10×+ on llama
    # traces where avg gpu duration is ~ms but wall-clock spans
    # are ~hundreds of ms — leading the LP to think every plan
    # stalls heavily even when sim doesn't.)
    #
    # Expansion: (1 − c_t) = 1 − c_t, move constant to RHS:
    #   Σ (−δ_t · c_t) + Σ (δ_t · e_{t,k}) − L_window_i
    #     ≤ (e_i − s_i) − Σ δ_t
    timeline_start = min(c[1] for pt in pool.values() for c in pt.consumers)
    timeline_end = max(c[1] for pt in pool.values() for c in pt.consumers)
    window_length = (timeline_end - timeline_start) / NUM_LATENESS_WINDOWS

    if audit:
        print(
            f"[ct_milp_lateness:audit] lateness windows: "
            f"N={NUM_LATENESS_WINDOWS} wall_clock={window_length/1e6:.1f}ms each; "
            f"timeline=[{timeline_start/1e6:.1f}, {timeline_end/1e6:.1f}]ms"
        )

    for i in range(NUM_LATENESS_WINDOWS):
        s_i = timeline_start + i * window_length
        e_i = timeline_start + (i + 1) * window_length
        is_last = (i == NUM_LATENESS_WINDOWS - 1)

        # Coefficients/RHS are in ms (ns / S); the window-membership
        # comparisons stay in ns (deadlines, s_i, e_i are ns).
        const_lhs = 0.0  # ms
        rows.append(row)
        cols.append(L_WINDOW_IDX_BASE + i)
        vals.append(-1.0)
        for tid in feasible_tids:
            pt = pool[tid]
            delta_ms = pt.tau_h2d_ns / MODEL_SCALE
            first_dl = pt.consumers[0][1]
            in_window_first = (
                s_i <= first_dl < e_i if not is_last
                else s_i <= first_dl <= e_i
            )
            if in_window_first:
                const_lhs += delta_ms
                rows.append(row)
                cols.append(c_var_idx[tid])
                vals.append(-float(delta_ms))
            for k in range(len(pt.consumers) - 1):
                if (tid, k) not in e_var_idx:
                    continue
                dl = pt.consumers[k + 1][1]
                in_window = (
                    s_i <= dl < e_i if not is_last
                    else s_i <= dl <= e_i
                )
                if in_window:
                    rows.append(row)
                    cols.append(e_var_idx[(tid, k)])
                    vals.append(float(delta_ms))
        ub_list.append(window_length / MODEL_SCALE - const_lhs)
        row += 1

    nb = row

    # ---- 9. Solve ----
    # ---- 9. Solve (two-phase: LP relaxation → MILP with warm-start) ----
    #
    # Phase 1: solve the LP relaxation (all binaries continuous in
    #          [0, 1]). Typically lands near-integer; gives a strong
    #          starting point.
    # Phase 2: round Phase-1 solution to a feasible integer assignment,
    #          flip binaries to integer, pass the rounded values as
    #          a warm-start via highspy's setSolution(), then run MILP.
    #          A good warm-start = tight initial incumbent = aggressive
    #          branch-pruning = MILP converges fast.
    #
    # Rounding safety for c+e≥1: the LP relaxation always satisfies
    # the continuous form. With 0.5-thresholds (c ≥ 0.5 ⇒ rounded 1,
    # else 0; same for e), rounding preserves c+e≥1 — proved by case
    # analysis on the LP's continuous c+e value (≥1 means at least one
    # is ≥0.5).
    fell_back = False
    used_two_phase = False
    res_x: np.ndarray | None = None
    res_success = False
    res_message = ""

    # Feasibility-aware warm-start: the stream-everything assignment
    # (c=0 for c-feasible, c=1 for c-infeasible, e=1 on feasible gaps of
    # streamed tids) is a known sim-feasible plan with a low residency
    # floor. Continuous vars are set to safe over-estimates so the point
    # satisfies every row (slacks have no upper bound). When phase-1 LP
    # doesn't reach Optimal (e.g. numerical "Solve error" on tight caps),
    # seeding phase-2 with THIS instead of running cold guarantees the
    # MILP returns a feasible under-floor incumbent rather than a garbage
    # high-peak one that only the overrun-repair could (dishonestly) mask.
    feasible_warm_start = np.zeros(total_vars, dtype=np.float64)
    for tid in feasible_tids:
        pt = pool[tid]
        c_val = 0.0 if pt.c_feasibility else 1.0
        feasible_warm_start[c_var_idx[tid]] = c_val
        for k in range(len(pt.consumers) - 1):
            if (tid, k) in e_var_idx:
                feasible_warm_start[e_var_idx[(tid, k)]] = (
                    1.0 if c_val < 0.5 else 0.0
                )
    # Continuous slacks (MB / ms) set to safe over-estimates so the seed
    # is a valid feasible point (slacks have no upper bound). constant_floor
    # is already MB; sizes are bytes → MB.
    _total_feasible_mb = float(
        sum(pool[t].size_bytes for t in feasible_tids)
    ) / MODEL_SCALE + float(constant_floor)
    feasible_warm_start[P_IDX] = _total_feasible_mb
    if peak_target_bytes is not None:
        feasible_warm_start[S_PEAK_IDX] = _total_feasible_mb
    _big_L_ms = float(sum(pool[t].tau_h2d_ns for t in feasible_tids)) / MODEL_SCALE
    for i in range(NUM_LATENESS_WINDOWS):
        feasible_warm_start[L_WINDOW_IDX_BASE + i] = _big_L_ms

    if _HIGHSPY_AVAILABLE and integrality_arr is not None and not lp_relaxation:
        used_two_phase = True
        res_x, res_success, res_message, milp_status_str, lp_only = (
            _solve_two_phase_highspy(
                total_vars=total_vars,
                c_obj=c_obj,
                bounds_list=bounds_list,
                rows=rows,
                cols=cols,
                vals=vals,
                ub_list=ub_list,
                integrality_arr=integrality_arr,
                time_limit_s=time_limit_s,
                solver_threads=solver_threads,
                audit=audit,
                phase1_time_limit_s=phase1_time_limit_s,
                feasible_fallback=feasible_warm_start,
            )
        )
        # If MILP couldn't even start (rare), fall through to scipy fallback.
        if not res_success:
            if audit:
                print(
                    f"[ct_milp_lateness:solver] two-phase highspy failed: "
                    f"{res_message!r} — falling back to scipy linprog."
                )
            res_x = None  # trigger scipy path below
        else:
            fell_back = lp_only

    # Direct highspy LP-relaxation path. Used when the caller
    # explicitly requested --lp-relaxation, OR when the MILP path
    # failed and we need a feasible LP primal. Reads the feasible
    # primal even on time-limit (scipy's wrapper drops it).
    if res_x is None and _HIGHSPY_AVAILABLE and lp_relaxation:
        if audit:
            print("[ct_milp_lateness:solver] using direct highspy LP")
        res_x, lp_has_primal, lp_msg = _solve_lp_highspy(
            total_vars=total_vars,
            c_obj=c_obj,
            bounds_list=bounds_list,
            rows=rows,
            cols=cols,
            vals=vals,
            ub_list=ub_list,
            time_limit_s=time_limit_s,
            solver_threads=solver_threads,
            audit=audit,
        )
        res_message = lp_msg
        res_success = bool(lp_has_primal)
        fell_back = True
        if audit:
            print(
                f"[ct_milp_lateness:solver] direct LP: "
                f"has_primal={lp_has_primal} status={lp_msg!r}"
            )

    # If the two-phase MILP returned nothing and highspy is available,
    # try the direct LP-relaxation path before falling to scipy. This
    # is the post-MILP recovery: we get a feasible LP primal that the
    # scipy wrapper would otherwise hide on time-limit.
    if res_x is None and _HIGHSPY_AVAILABLE and not lp_relaxation:
        if audit:
            print(
                "[ct_milp_lateness:solver] MILP gave no primal, "
                "trying direct highspy LP relaxation"
            )
        lp_x, lp_has, lp_msg = _solve_lp_highspy(
            total_vars=total_vars,
            c_obj=c_obj,
            bounds_list=bounds_list,
            rows=rows,
            cols=cols,
            vals=vals,
            ub_list=ub_list,
            time_limit_s=time_limit_s,
            solver_threads=solver_threads,
            audit=audit,
        )
        if lp_has:
            res_x = lp_x
            res_success = True
            res_message = lp_msg
            fell_back = True

    if res_x is None:
        # Scipy fallback: when highspy isn't available, or when
        # two-phase reported a fatal model error.
        A = csr_matrix((vals, (rows, cols)), shape=(nb, total_vars))
        b_ub_arr = np.array(ub_list, dtype=np.float64)
        options: dict[str, Any] = {"disp": False}
        if time_limit_s is not None:
            options["time_limit"] = float(time_limit_s)
        options["threads"] = int(solver_threads)
        kwargs: dict[str, Any] = {
            "A_ub": A,
            "b_ub": b_ub_arr,
            "bounds": bounds_list,
            "method": "highs",
            "options": options,
        }
        if integrality_arr is not None:
            kwargs["integrality"] = integrality_arr
        res = linprog(c_obj, **kwargs)
        if not res.success and integrality_arr is not None and not lp_relaxation:
            if audit:
                print(
                    f"[ct_milp_lateness:solver] scipy MILP failed: "
                    f"status={getattr(res, 'message', '')!r} — "
                    f"falling back to LP relaxation."
                )
            kwargs.pop("integrality", None)
            res = linprog(c_obj, **kwargs)
            fell_back = True
        # Accept any feasible primal even if the solver reports
        # success=False due to time-limit. With option-b coupling +
        # large pool, scipy's HiGHS sometimes returns "TimeLimit
        # reached but primal_status is Feasible" — the feasible plan
        # is strictly better than the all-cold hard-fallback below.
        message_str = str(getattr(res, "message", ""))
        has_x = res.x is not None
        primal_feasible = (
            "primal_status is Feasible" in message_str
            or "primal_status is feasible" in message_str
        )
        accepted_via_timeout = (not res.success) and has_x and primal_feasible
        if audit and accepted_via_timeout:
            print(
                f"[ct_milp_lateness:solver] accepting timeout-feasible "
                f"primal (has_x={has_x} status={message_str!r})"
            )
        res_x = np.asarray(res.x) if has_x and (res.success or accepted_via_timeout) else None
        res_success = bool(res.success) or accepted_via_timeout
        res_message = message_str
        if audit and not res_success:
            print(
                f"[ct_milp_lateness:solver] no usable primal: "
                f"has_x={has_x} success={res.success} "
                f"primal_feasible_in_msg={primal_feasible} "
                f"message={message_str!r}"
            )

    if audit:
        tag = "highspy-two-phase" if used_two_phase else "scipy-linprog"
        print(
            f"[ct_milp_lateness:solver] backend={tag} success={res_success} "
            f"fell_back={fell_back} status={res_message!r}"
        )
        if res_x is not None and fell_back:
            cvals = [float(res_x[c_var_idx[t]]) for t in feasible_tids]
            n_zero = sum(1 for v in cvals if v < 0.01)
            n_one = sum(1 for v in cvals if v > 0.99)
            n_frac = sum(1 for v in cvals if 0.01 <= v <= 0.99)
            print(
                f"[ct_milp_lateness:audit] c-value distribution (LP relaxation): "
                f"≈0: {n_zero}, ≈1: {n_one}, fractional: {n_frac}"
            )

    # Build a shim object for the decode block below.
    class _Res:
        pass
    res = _Res()
    res.success = res_success
    res.x = res_x
    res.message = res_message

    # ---- 10. Decode ----
    c_solution: dict[int, float] = {}
    e_solution: dict[tuple[int, int], float] = {}
    target_infeasible = False
    peak_overrun_bytes = 0
    peak_bytes = 0
    lateness_ns = 0
    overrun_repair_diag: dict[str, int] = {}
    per_window_lateness_ns: list[int] = [0] * NUM_LATENESS_WINDOWS
    window_bounds_ns: list[tuple[int, int]] = [
        (
            int(timeline_start + i * window_length),
            int(timeline_start + (i + 1) * window_length),
        )
        for i in range(NUM_LATENESS_WINDOWS)
    ]

    # True modeled peak for any (c, e) assignment, recomputed over the
    # saved per-sample terms. Consistent-by-construction with the peak
    # constraint rows; used to report an honest peak and to drive the
    # iterative overrun-repair below.
    _c_col_to_tid = {col: tid for tid, col in c_var_idx.items()}
    _e_col_to_key = {col: key for key, col in e_var_idx.items()}

    def _alive_peak(c_sol: dict[int, float],
                    e_sol: dict[tuple[int, int], float]) -> float:
        pk = 0.0
        for const_s, terms in peak_sample_terms:
            v = const_s
            for col, coef in terms:
                tid_c = _c_col_to_tid.get(col)
                if tid_c is not None:
                    v += coef * float(c_sol.get(tid_c, 0.0))
                else:
                    key_e = _e_col_to_key.get(col)
                    if key_e is not None:
                        v += coef * float(e_sol.get(key_e, 0.0))
            if v > pk:
                pk = v
        return pk

    if res.success and res.x is not None:
        x = np.asarray(res.x)
        for tid in feasible_tids:
            c_solution[tid] = float(x[c_var_idx[tid]])
            pt = pool[tid]
            for k in range(len(pt.consumers) - 1):
                if (tid, k) in e_var_idx:
                    e_solution[(tid, k)] = float(x[e_var_idx[(tid, k)]])
                else:
                    e_solution[(tid, k)] = 0.0
        peak_bytes = int(_alive_peak(c_solution, e_solution) * MODEL_SCALE)
        # Slacks are in ms (model units); convert back to ns. Total
        # lateness = sum across windows (cascading stalls add up).
        window_slacks_ms = [
            float(x[L_WINDOW_IDX_BASE + i])
            for i in range(NUM_LATENESS_WINDOWS)
        ]
        lateness_ns = int(sum(window_slacks_ms) * MODEL_SCALE)
        per_window_lateness_ns = [int(v * MODEL_SCALE) for v in window_slacks_ms]
        if audit:
            nonzero = [
                (i, v) for i, v in enumerate(window_slacks_ms) if v > 1e-3
            ]
            print(
                f"[ct_milp_lateness:audit] per-window stalls (ms): "
                f"total={sum(window_slacks_ms):.2f}, "
                f"nonzero windows: "
                f"{[(i, round(v, 2)) for i, v in nonzero]}"
            )
        if peak_target_bytes is not None:
            # Repair works in model units (MB): _alive_peak and target_adj_mb
            # are both MB. Convert the results back to bytes for reporting.
            target_adj_mb = max(
                0.0,
                float(peak_target_bytes) * (1.0 - float(safety_margin_frac)),
            ) / MODEL_SCALE
            # The TRUE peak is peak_bytes (recomputed above), not the
            # solver's s_P slack. Repair against it so the reported number
            # is what sim will actually hit — not the old under-credit.
            if peak_bytes > target_adj_mb * MODEL_SCALE + 1:
                overrun_repair_diag = _stream_cold_tensors_to_cover_overrun(
                    pool,
                    feasible_tids,
                    c_solution,
                    e_solution,
                    target_adj_mb,
                    _alive_peak,
                )
                peak_bytes = int(
                    overrun_repair_diag["final_peak_model"] * MODEL_SCALE
                )
                peak_overrun_bytes = int(
                    overrun_repair_diag["residual_overrun_model"] * MODEL_SCALE
                )
                target_infeasible = peak_overrun_bytes > 1
                if audit:
                    print(
                        f"[ct_milp_lateness:solver] true modeled peak "
                        f"exceeds cap; streamed "
                        f"{overrun_repair_diag['streamed_count']} cold tensors "
                        f"({overrun_repair_diag['streamed_bytes']/1e6:.1f}MB) "
                        f"selected by fewest consumers, then farthest first "
                        f"use → true peak now {peak_bytes/1e6:.1f}MB, "
                        f"residual_overrun={peak_overrun_bytes/1e6:.1f}MB "
                        f"(target_infeasible={target_infeasible})"
                    )
            else:
                peak_overrun_bytes = 0
                target_infeasible = False
    if not (res.success and res.x is not None):
        # Hard-fallback: stream every async-feasible tid (c=0),
        # cold-load every c_feasibility=False tid (c=1, layout time).
        # Evict at every feasible gap only for streamed tids (e=1).
        #
        # Rationale: streaming c=0 in runtime requires a gpu_runtime
        # issuer. For c_feasibility=False tids none exists; routing
        # them via cpu_leaf issuers would storm PCIe at sim_t≈0 and
        # block GPU work. Cold-start instead: load at layout (no
        # runtime PCIe contention), evict immediately after first
        # consumer (e=1 in gap 0 where feasible), refetch later via
        # gpu_runtime issuer for subsequent consumers.
        for tid in feasible_tids:
            pt = pool[tid]
            if not pt.c_feasibility:
                c_solution[tid] = 1.0
            else:
                c_solution[tid] = 0.0
            for k in range(len(pt.consumers) - 1):
                if (
                    c_solution[tid] < 0.5
                    and (tid, k) in e_var_idx
                    and pt.gap_feasibility[k]
                ):
                    e_solution[(tid, k)] = 1.0
                else:
                    e_solution[(tid, k)] = 0.0
        # Honest peak: recompute the true max-over-samples alive set for
        # the stream-everything assignment (model units MB → bytes).
        peak_bytes = int(_alive_peak(c_solution, e_solution) * MODEL_SCALE)
        lateness_ns = 0
        if peak_target_bytes is not None:
            target_adj = max(
                0.0,
                float(peak_target_bytes) * (1.0 - float(safety_margin_frac)),
            )
            peak_overrun_bytes = max(0, int(peak_bytes - target_adj))
            target_infeasible = peak_overrun_bytes > 1
        if audit:
            n_stream = sum(1 for t in feasible_tids if c_solution[t] < 0.5)
            n_cold = sum(1 for t in feasible_tids if c_solution[t] >= 0.5)
            print(
                f"[ct_milp_lateness:solver] hard-fallback: stream-where-feasible "
                f"(c=0 for {n_stream} async-feasible tids, c=1 for {n_cold} "
                f"bound-cold). true_peak={peak_bytes/1e6:.1f}MB "
                f"target_infeasible={target_infeasible}"
            )

    diagnostics = {
        "pool_size": len(pool),
        "forced_cold_count": len(forced_cold),
        "forced_cold_bytes": forced_cold_bytes,
        "feasible_var_count": nv,
        "e_var_count": n_e,
        "n_samples": len(samples),
        "solver_success": bool(res.success),
        "solver_status": str(getattr(res, "message", "")),
        "solver_threads": int(solver_threads),
        "fell_back_to_lp": bool(fell_back),
        "target_infeasible": bool(target_infeasible),
        "peak_overrun_bytes": int(peak_overrun_bytes),
        "lateness_ns": int(lateness_ns),
        "lp_relaxation": bool(lp_relaxation),
        "overrun_repair": overrun_repair_diag,
    }

    return _LPResult(
        c_solution=c_solution,
        e_solution=e_solution,
        forced_cold=forced_cold,
        feasible_tids=feasible_tids,
        peak_bytes=int(peak_bytes),
        lateness_ns=int(lateness_ns),
        peak_overrun_bytes=int(peak_overrun_bytes),
        target_infeasible=bool(target_infeasible),
        solver_status=str(getattr(res, "message", "")),
        diagnostics=diagnostics,
        per_window_lateness_ns=per_window_lateness_ns,
        window_bounds_ns=window_bounds_ns,
    )


# ---------------------------------------------------------------------------
# Emit: NeutralSchedule with cgsim_tid pre-resolved
# ---------------------------------------------------------------------------


def _emit_neutral(
    pool: dict[int, _PoolTensor],
    result: _LPResult,
    trace: Trace,
    hw: HwParams,
    sim_times: dict[int, tuple[int, int]] | None = None,
) -> NeutralSchedule:
    """Build a NeutralSchedule with cgsim_tids pre-resolved on every entry.

    The schedule emits:
      - NeutralTensor per pool tid with ``trace_tids = [cgsim_tid]`` so
        the injector's pre-resolution path picks it up without invoking
        the shape-disambiguation resolver.
      - NeutralColdStart per cold tid (c ≥ 0.5 or forced).
      - NeutralPrefetch (initial) per streamed tid, anchored at consumer
        node 0 with issuer placed by ``_pick_issuer_node``.
      - NeutralPrefetch (refetch) + NeutralEvict per feasible cross-iter
        gap where the streamed tid evicts and refetches.

    Per-iter handling: each consumer in ``pt.consumers`` is a separate
    trace node, so a refetch + evict around gap k targets the SPECIFIC
    iter's node_id (no iter_mask expansion needed; the injector keys
    ``evict_after_node`` and ``arrival.consumer_node_id`` by node_id).
    """
    KEEP_THRESHOLD = 0.5

    # Global gpu-node index in LP axis. We allow CROSS-GRAPH issuers
    # — PCIe is a global resource, so an early gpu node from graph
    # G_0 can issue a prefetch whose consumer is in G_3. The
    # injector handles cross-graph via NeutralPrefetch.issue_graph_id.
    # Each entry: (lp_start_ns, trace_start_ns, node_id, graph_id).
    all_gpu_nodes_lp: list[tuple[int, int, int, int]] = []
    for nid, node in trace.node_map.items():
        rk = str((node.args or {}).get("resource_kind") or "")
        if rk not in _GPU_RESOURCE_KINDS:
            continue
        nid_i = int(nid)
        start_ns = int((node.args or {}).get("start_ns") or 0)
        if start_ns <= 0:
            continue
        lp_start = start_ns
        if sim_times is not None:
            st = sim_times.get(nid_i)
            if st is not None:
                lp_start = int(st[0])
        gid_raw = (node.args or {}).get("compiled_graph_id")
        try:
            gid_i = int(gid_raw) if gid_raw is not None else -1
        except (TypeError, ValueError):
            gid_i = -1
        all_gpu_nodes_lp.append((lp_start, int(start_ns), nid_i, gid_i))
    all_gpu_nodes_lp.sort(key=lambda x: x[0])

    def _pick_issuer_node(
        consumer_gid: int, consumer_lp_start: int, tau_h2d_ns: int,
        earliest_allowed_lp: int = -1,
    ) -> tuple[int, int, int]:
        """Latest gpu node ANYWHERE whose LP-axis start ≤
        consumer_lp_start − τ_h2d (and > earliest_allowed_lp for
        post-evict refetches). Cross-graph allowed: the returned
        ``issuer_graph_id`` may differ from consumer_gid.

        Returns (issuer_node_id, issuer_lp_start, issuer_graph_id),
        or (-1, -1, -1) if none.
        """
        target = consumer_lp_start - tau_h2d_ns
        best_lp = -1
        best_nid = -1
        best_gid = -1
        import bisect
        idx = bisect.bisect_right(
            [n[0] for n in all_gpu_nodes_lp], target,
        ) - 1
        # Walk back from the latest candidate ≤ target until one
        # satisfies the earliest_allowed_lp lower bound (and prefers
        # the same graph as consumer when equally late, to match the
        # old behavior on workloads where in-graph issuer existed).
        while idx >= 0:
            lp_s, _trace_s, nid, gid = all_gpu_nodes_lp[idx]
            if lp_s <= earliest_allowed_lp:
                break
            if lp_s > best_lp or (
                lp_s == best_lp and gid == consumer_gid
            ):
                best_lp = lp_s
                best_nid = nid
                best_gid = gid
                # Prefer in-graph issuer when at the latest valid
                # lp position; if we already have it, stop.
                if gid == consumer_gid:
                    break
            idx -= 1
        if best_nid < 0:
            return -1, -1, -1
        return best_nid, best_lp, best_gid

    # Build NeutralTensor entries. The injector reads:
    #   - uid           — used as the cross-reference key from prefetch/evict
    #   - trace_tids    — pre-resolved cgsim_tids; non-empty bypasses
    #                     shape-disambiguation in the resolver
    #   - graph_id, compiled_tensor_id, compiled_graph_input_name,
    #     graph_input_name, size_bytes — written to schedule JSON for
    #     compile-side metadata; the injector's fast path doesn't use
    #     them for decisions.
    neutral_tensors: list[NeutralTensor] = []
    uid_by_tid: dict[int, int] = {}
    for tid in sorted(pool.keys()):
        pt = pool[tid]
        primary_gid = pt.consumer_graph_ids[0] if pt.consumer_graph_ids else -1
        uid = len(neutral_tensors)
        uid_by_tid[tid] = uid
        # Pull shape + dtype from the trace tensor. The sim-side injector
        # at sim/load/pytorch_profile/graph_modifiers/inject_schedule does
        # NOT have a pre-resolved cgsim_tid fast path: it always calls
        # _resolve_tid_for_node, which does shape/byte disambiguation. With
        # shape=[] and dtype="" that resolver returns None for every
        # prefetch (target_numel=1, no candidates pass _byte_match), and
        # every prefetch is silently dropped — the entire pool then runs
        # cuda-resident and the cap is missed.
        ttensor = trace.tensor_map.get(int(tid))
        t_shape: list[Any] = []
        t_dtype: str = pt.dtype or ""
        if ttensor is not None:
            ta = ttensor.args or {}
            raw_shape = ta.get("shape")
            if isinstance(raw_shape, (list, tuple)):
                t_shape = list(raw_shape)
            if not t_dtype:
                t_dtype = str(ta.get("dtype") or "")
        neutral_tensors.append(NeutralTensor(
            uid=uid,
            graph_id=int(primary_gid),
            compiled_tensor_id=int(tid),
            graph_input_name=pt.name or f"cgtid_{tid}",
            size_bytes=int(pt.size_bytes),
            dtype=t_dtype,
            used_by_launch_ids=sorted(set(pt.consumer_launch_ids)),
            shape=t_shape,
            graph_input_idx=None,
            storage_group_id=int(tid),
            trace_tids=[int(tid)],
        ))

    prefetches: list[NeutralPrefetch] = []
    evicts: list[NeutralEvict] = []
    cold_starts: list[NeutralColdStart] = []

    for tid, pt in pool.items():
        uid = uid_by_tid[tid]
        cv = result.c_solution.get(tid, None)
        is_forced = tid in result.forced_cold or cv is None
        is_cold = is_forced or float(cv) >= KEEP_THRESHOLD

        # Emit cold-start or initial prefetch, depending on c_t.
        # c-feasible cold tids must not get runtime refetches — the
        # MILP's c+e<=1 coupling rules that out, and the injector
        # would silently demote them anyway.
        # c-infeasible cold tids (pinned c=1) ARE allowed to emit
        # per-gap evict+refetch when the LP picks e_{t,k}=1 — that's
        # the hybrid pattern this scheduler now supports for the
        # cold-pinned set.
        if is_cold:
            cold_starts.append(NeutralColdStart(
                tensor_uid=uid,
                anchor_launch_id=(
                    max(0, int(pt.consumer_launch_ids[0]))
                    if pt.consumer_launch_ids else 0
                ),
                reason=(
                    "lateness_forced_cold" if is_forced
                    else "lateness_optimal_cold"
                ),
                cgsim_tids=[int(tid)],
            ))
        else:
            consumer_0 = pt.consumers[0]
            c0_nid, c0_lp_start, _c0_lp_end = consumer_0
            c0_gid = pt.consumer_graph_ids[0]
            c0_lid = pt.consumer_launch_ids[0]
            issue_nid, _issue_lp, issue_gid = _pick_issuer_node(
                c0_gid, c0_lp_start, pt.tau_h2d_ns,
            )
            if issue_nid < 0:
                issue_nid = c0_nid
                issue_gid = c0_gid
            c0_trace_start = (
                pt.consumer_trace_starts[0]
                if pt.consumer_trace_starts else int(c0_lp_start)
            )
            # issue_graph_id: -1 if same graph as consumer, else the
            # issuer's actual graph (cross-graph prefetch).
            issue_graph_id_field = (
                -1 if int(issue_gid) == int(c0_gid) else int(issue_gid)
            )
            prefetches.append(NeutralPrefetch(
                tensor_uid=uid,
                issue_launch_id=max(0, int(c0_lid)),
                wait_launch_id=max(0, int(c0_lid)),
                transfer_start_ns=int(max(0, c0_trace_start - pt.tau_h2d_ns)),
                transfer_end_ns=int(c0_trace_start),
                reason="lateness_initial",
                issue_node_id=int(issue_nid),
                wait_node_id=int(c0_nid),
                cgsim_tid=int(tid),
                trusted_async=(issue_nid != c0_nid),
                issue_graph_id=issue_graph_id_field,
                iter_mask=[],
            ))

        # Skip per-gap emit only for c-feasible cold tids — the
        # MILP's coupling guarantees e=0 for them. c-infeasible cold
        # tids (forced or pinned) fall through to the per-gap loop
        # so their hybrid evict+refetch can be emitted when e=1.
        if is_cold and pt.c_feasibility:
            continue

        # Per-gap evict + refetch. For c=0 (streamed) tids this is the
        # standard prefetch+evict cycle. For c=1 c-pinned tids it's
        # the hybrid `cold-at-layout + mid-run evict + refetch`
        # pattern (see Coupling section in _solve_milp).
        for k in range(len(pt.consumers) - 1):
            if not pt.gap_feasibility[k]:
                continue
            ev = result.e_solution.get((tid, k), 0.0)
            if float(ev) < KEEP_THRESHOLD:
                continue
            consumer_k = pt.consumers[k]
            consumer_kp1 = pt.consumers[k + 1]
            ck_nid, _ck_lp_start, ck_lp_end = consumer_k
            ckp1_nid, ckp1_lp_start, _ckp1_lp_end = consumer_kp1
            kp1_gid = pt.consumer_graph_ids[k + 1]
            kp1_lid = pt.consumer_launch_ids[k + 1]
            # Trace times for emit's transfer fields.
            ck_trace_end = (
                pt.consumer_trace_ends[k]
                if pt.consumer_trace_ends else int(ck_lp_end)
            )
            ckp1_trace_start = (
                pt.consumer_trace_starts[k + 1]
                if pt.consumer_trace_starts else int(ckp1_lp_start)
            )
            evict_reason = (
                "lateness_hybrid_gap_evict" if is_cold
                else "lateness_gap_evict"
            )
            refetch_reason = (
                "lateness_hybrid_gap_refetch" if is_cold
                else "lateness_gap_refetch"
            )
            evicts.append(NeutralEvict(
                tensor_uid=uid,
                issue_launch_id=max(0, int(pt.consumer_launch_ids[k])),
                transfer_start_ns=int(ck_trace_end),
                transfer_end_ns=int(ck_trace_end + pt.tau_d2h_ns),
                reason=evict_reason,
                issue_node_id=int(ck_nid),
                iter_mask=[],
                cgsim_tid=int(tid),
            ))
            # Refetch issuer: search globally; must fire AFTER
            # consumer_k.end (evict releases pages first).
            re_nid, re_lp_ts, re_gid = _pick_issuer_node(
                kp1_gid, ckp1_lp_start, pt.tau_h2d_ns,
                earliest_allowed_lp=ck_lp_end,
            )
            if re_nid < 0:
                re_nid = ckp1_nid
                re_lp_ts = ckp1_lp_start
                re_gid = kp1_gid
            refetch_graph_id_field = (
                -1 if int(re_gid) == int(kp1_gid) else int(re_gid)
            )
            prefetches.append(NeutralPrefetch(
                tensor_uid=uid,
                issue_launch_id=max(0, int(pt.consumer_launch_ids[k + 1])),
                wait_launch_id=max(0, int(kp1_lid)),
                transfer_start_ns=int(max(
                    ck_trace_end + 1, ckp1_trace_start - pt.tau_h2d_ns,
                )),
                transfer_end_ns=int(ckp1_trace_start),
                reason=refetch_reason,
                issue_node_id=int(re_nid),
                wait_node_id=int(ckp1_nid),
                cgsim_tid=int(tid),
                trusted_async=(re_nid != ckp1_nid),
                issue_graph_id=refetch_graph_id_field,
                iter_mask=[],
            ))

    graph_ids_seen: set[int] = set()
    for pt in pool.values():
        for g in pt.consumer_graph_ids:
            if g >= 0:
                graph_ids_seen.add(g)
    graph_order = sorted(graph_ids_seen)

    return NeutralSchedule(
        graph_order=graph_order,
        compilation_hashes={int(g): "" for g in graph_order},
        tensors=neutral_tensors,
        prefetches=prefetches,
        evicts=evicts,
        cold_starts=cold_starts,
        meta={},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def solve_neutral(
    trace: Trace,
    *,
    hw: HwParams,
    baseline_sim_result_path: str | None = None,
    peak_target_bytes: int | None = None,
    safety_margin_frac: float = 0.07,
    max_peak_samples: int = 256,
    time_limit_s: float | None = 240.0,
    phase1_time_limit_s: float | None = None,
    solver_threads: int | None = None,
    lp_relaxation: bool = False,
    backpressure_edges: bool = False,
    backpressure_lateness_threshold_ns: int = 100_000,  # 100 us
    arc_queue_factor: float = 1.0,
    lateness_peak_coupling: bool = False,
    relax_cinfeasible: bool = False,
    intermediate_axis_fix: bool = True,
    audit: bool = False,
    sidecars: Any = None,                 # accepted but ignored
    **_legacy_kwargs: Any,
) -> NeutralSchedule:
    """Build a pool-first lateness MILP schedule from the runtime trace.

    Inputs:
      ``trace``               — Trace with deduped-by-storage_id
                                cgsim Tensors.
      ``hw``                  — HwParams (h2d/d2h bandwidth, latencies).
      ``peak_target_bytes``   — VRAM cap. None ⇒ no peak constraint.
      ``safety_margin_frac``  — Pad below cap (default 5%) to absorb
                                modeling gaps the LP can't see directly
                                (intermediate activations spikes, page
                                fragmentation).
      ``max_peak_samples``    — How many gpu-consumer points to sample
                                for peak/lateness rows. ~256 is a sweet
                                spot for sd3-med scale (10k events).
      ``time_limit_s``        — HiGHS time limit. None ⇒ no limit.
      ``solver_threads``      — CPU threads for HiGHS. None ⇒ all
                                detected CPU cores.
      ``lp_relaxation``       — Skip integrality, solve continuous LP
                                (debug aid).
      ``relax_cinfeasible``   — Let the LP stream c-infeasible tids
                                (default False) instead of pinning them
                                resident, trading a startup stall for VRAM.
                                Makes tight budgets feasible where they'd
                                otherwise be target_infeasible (e.g.
                                llama8b@6gib). Opt-in: changes the feasible
                                set / decisions.
      ``intermediate_axis_fix`` — Use sim-time-only endpoints for
                                intermediate (activation) residency windows
                                instead of mixing sim/trace axes (default
                                True). The old mix inflated the modeled
                                activation floor ~10x on diffusion (sdxl
                                1941MB vs ~192MB actual). Set False only for
                                the ablation that shows the bug's cost.
      ``lateness_peak_coupling`` — Re-enable the lateness→peak coupling
                                term (default False). Mis-calibrated: it
                                adds bw·window_lateness to each peak row,
                                over-predicting modeled peak by ~2 GiB and
                                making the LP under-fill VRAM. Off wins on
                                every model at tight budgets; leave False
                                except for A/B. Also honored via env
                                MILP_ENABLE_LP_COUPLING=1.
      ``audit``               — Print pool/LP/solver diagnostics.
      ``sidecars``            — Accepted for interface parity with
                                ct_milp_multistream's main.py; ignored.

    Returns:
      A NeutralSchedule with cgsim_tid pre-resolved on every entry.
    """
    # Load baseline sim_result to get per-node sim wall-clock. The
    # LP plans on these times instead of trace_start_ns — matches sim's
    # actual back-to-back execution rate (no profiler idle gaps,
    # accounts for any CPU+GPU interleaving the baseline run observed).
    sim_times: dict[int, tuple[int, int]] | None = None
    if baseline_sim_result_path is not None:
        sim_times = _load_baseline_sim_times(baseline_sim_result_path)
        gpu_trace_nids = [
            int(nid) for nid, n in trace.node_map.items()
            if str((n.args or {}).get("resource_kind") or "")
               in _GPU_RESOURCE_KINDS
            and int((n.args or {}).get("start_ns") or 0) > 0
        ]
        n_total = len(gpu_trace_nids)
        n_matched = sum(1 for nid in gpu_trace_nids if nid in sim_times)
        if n_matched < n_total:
            coverage = (n_matched / n_total) if n_total else 1.0
            raise RuntimeError(
                "[ct_milp_lateness] baseline sim_result does not match "
                "this trace closely enough: loaded sim times for "
                f"{n_matched} / {n_total} gpu trace nodes "
                f"({coverage:.1%}) from {baseline_sim_result_path}. "
                "Use the matching workload's baseline sim_result, or omit "
                "--baseline-sim-result to use trace times."
            )
        if audit:
            print(
                f"[ct_milp_lateness:audit] loaded baseline sim times "
                f"for {n_matched} / {n_total} gpu trace nodes "
                f"from {baseline_sim_result_path}"
            )

    pool = _build_pool(trace, hw, sim_times)
    if not pool:
        raise RuntimeError(
            "[ct_milp_lateness] pool is empty — no cuda WEIGHT/LEAF/INPUT "
            "tensors with gpu consumers found in trace."
        )

    # Layout-time vram residency that the LP doesn't model explicitly:
    # cuda WEIGHT/LEAF/INPUT tensors with no gpu consumers. Sim still
    # allocates them at layout.
    extra_static_bytes = 0
    pool_tids = set(pool.keys())
    for tid, t in trace.tensor_map.items():
        if int(t.size_bytes) <= 0:
            continue
        if (t.args or {}).get("tensor_type") not in _POOL_TENSOR_TYPES:
            continue
        if not str((t.args or {}).get("device", "")).lower().startswith("cuda"):
            continue
        if int(tid) in pool_tids:
            continue
        extra_static_bytes += int(t.size_bytes)
    if audit and extra_static_bytes:
        print(
            f"[ct_milp_lateness:audit] no-consumer cuda layout overhead "
            f"= {extra_static_bytes/1e6:.1f}MB"
        )

    # Unschedulable tensor mass: tensors in the compile-side sidecar that
    # couldn't be matched to trace tids. The MILP can't plan for these,
    # but they still consume VRAM at runtime. Add to extra_static_bytes
    # so the MILP knows the true available budget.
    if sidecars is not None:
        try:
            from graph_modifiers.common import build_unified_timeline
            tl = build_unified_timeline(
                trace, sidecars, cpu_per_launch_ns=hw.cpu_per_launch_ns,
            )
            unschedulable_bytes = sum(
                t.size_bytes for t in tl.tensors if not t.trace_tids
            )
            if unschedulable_bytes > 0:
                extra_static_bytes += unschedulable_bytes
                if audit:
                    n_unsch = sum(1 for t in tl.tensors if not t.trace_tids)
                    print(
                        f"[ct_milp_lateness:audit] unschedulable sidecar tensors: "
                        f"{n_unsch} ({unschedulable_bytes/1e6:.1f}MB) — "
                        f"added to static overhead"
                    )
        except Exception as e:
            if audit:
                print(f"[ct_milp_lateness:audit] sidecars provided but timeline "
                      f"build failed: {e}")

    result = _solve_milp(
        pool, trace, hw,
        sim_times=sim_times,
        peak_target_bytes=peak_target_bytes,
        extra_static_bytes=extra_static_bytes,
        safety_margin_frac=safety_margin_frac,
        max_peak_samples=max_peak_samples,
        time_limit_s=time_limit_s,
        phase1_time_limit_s=phase1_time_limit_s,
        solver_threads=solver_threads,
        lp_relaxation=lp_relaxation,
        arc_queue_factor=arc_queue_factor,
        lateness_peak_coupling=lateness_peak_coupling,
        relax_cinfeasible=relax_cinfeasible,
        intermediate_axis_fix=intermediate_axis_fix,
        audit=audit,
    )

    neutral = _emit_neutral(pool, result, trace, hw, sim_times)

    n_cold = len(neutral.cold_starts)
    n_pf = len(neutral.prefetches)
    n_ev = len(neutral.evicts)
    streamed_bytes = sum(
        pool[t].size_bytes for t in result.feasible_tids
        if result.c_solution.get(t, 1.0) < 0.5
    )
    cold_bytes = sum(
        pool[t].size_bytes for t in pool
        if result.c_solution.get(t, 1.0) >= 0.5 or t in result.forced_cold
    )
    pcie_h2d_bytes = streamed_bytes + sum(
        pool[t].size_bytes
        for (t, k), v in result.e_solution.items() if v >= 0.5
    )

    # ---- Backpressure edges: derive synthetic GPU→CPU control edges
    # from the LP's per-window lateness. For each window with stall
    # above threshold, find the gpu_runtime node at window's end
    # (in LP time axis) and the cpu_leaf node that, without the edge,
    # would fire during the stall. Adding (gpu, cpu) as a dependency
    # in the trace caps sim's CPU race-ahead, modeling the real
    # CachingAllocator's wait-on-stream-event under tight memory.
    bp_edges: list[tuple[int, int]] = []
    bp_diag: dict[str, Any] = {"enabled": bool(backpressure_edges)}
    if backpressure_edges and result.per_window_lateness_ns:
        bp_edges, bp_diag_extra = _derive_backpressure_edges(
            trace, result, hw, sim_times=sim_times,
            lateness_threshold_ns=backpressure_lateness_threshold_ns,
            audit=audit,
        )
        bp_diag.update(bp_diag_extra)

    neutral.meta = {
        "io_model": "ct_milp_lateness_simtime",
        "backpressure_edges": bp_edges,
        "backpressure_diagnostics": bp_diag,
        "graph_order": neutral.graph_order,
        "milp_peak_mb": round(result.peak_bytes / 1e6, 2),
        "milp_lateness_ms": round(result.lateness_ns / 1e6, 3),
        "milp_lateness_ns": int(result.lateness_ns),
        "pcie_used_mb": round(pcie_h2d_bytes / 1e6, 2),
        "cold_bytes_mb": round(cold_bytes / 1e6, 2),
        "streamed_bytes_mb": round(streamed_bytes / 1e6, 2),
        "extras_static_mb": round(extra_static_bytes / 1e6, 2),
        "peak_overrun_mb": round(result.peak_overrun_bytes / 1e6, 2),
        "target_infeasible": result.target_infeasible,
        "n_cold_starts": n_cold,
        "n_prefetches": n_pf,
        "n_evicts": n_ev,
        "diagnostics": result.diagnostics,
    }
    return neutral


def print_summary(neutral: NeutralSchedule) -> None:
    """One-line schedule summary (mirrors ct_milp_multistream)."""
    print(
        f"Variant: {neutral.meta.get('io_model')} "
        f"| peak: {neutral.meta.get('milp_peak_mb')} MB "
        f"| lateness: {neutral.meta.get('milp_lateness_ms')} ms "
        f"| PCIe H2D used: {neutral.meta.get('pcie_used_mb')} MB "
        f"| cold: {neutral.meta.get('cold_bytes_mb')} MB "
        f"| streamed: {neutral.meta.get('streamed_bytes_mb')} MB "
        f"| prefetches: {neutral.meta.get('n_prefetches')} "
        f"| evicts: {neutral.meta.get('n_evicts')} "
        f"| cold_start: {neutral.meta.get('n_cold_starts')}"
    )
