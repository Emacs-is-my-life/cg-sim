"""Pool-first MILP weight-streaming scheduler with peak-VRAM objective.

Inverse of ``ct_milp_lateness``: instead of minimizing stall subject to a
peak cap, this variant minimizes peak VRAM subject to a *hard* zero-stall
constraint. PCIe load in every timeline window must fit inside the
window's wall-clock duration; no slack is allowed.

For each cuda-resident WEIGHT/LEAF/INPUT trace tensor (one entry per
physical storage thanks to the loader's `(device, storage_id)` dedup),
decide:

  c_t ∈ {0, 1}:     1 = cold (resident from layout)
                    0 = streamed (JIT prefetch before first consumer +
                                  per-feasible-gap evict+refetch)
  e_{t,k} ∈ {0, 1}: 1 = evict after consumer k and refetch before k+1
                    coupled to (1 − c_t) on feasible gaps

Objective: minimize P + ε·(streamed_bytes + refetched_bytes)
  - P ≥ 0 — modeled peak VRAM (bytes)
  - ε = 1e-6 — small tiebreaker so the LP prefers fewer streaming
    bytes among equal-peak plans; never overrides peak

PCIe constraint (hard, cumulative-by-T / prefix-sum):
  At each sample T_j on the gpu compute timeline:
    Σ_{t : first_consumer(t).start ≤ T_j}  δ_t · (1 − c_t)
  + Σ_{(t,k) feas : consumer_{k+1}.start ≤ T_j}  δ_t · e_{t,k}
    ≤  T_j
  δ_t = h2d_latency_ns + size_t / h2d_bw

This is the strict serial-queue invariant — by time T_j, cumulative
H2D work whose deadlines have arrived must fit in T_j ns of queue
throughput (h2d_streams=1, runs from sim t=0). It catches deadline
clustering that an averaged per-window budget can't see: if a burst
of deadlines lands at T_j, the LP must keep prior cumulative work
small enough that the queue can deliver the burst on time.

Cold-all (c=1, e=0) trivially satisfies every row (0 ≤ T_j), so the
LP is always feasible.

Peak VRAM model (sampled at every gpu compute consumer's T):
  Σ_{t: alive(t, T)} size_t  +  forced_const  ≤  P
  alive(t, T) classifies T into one of pre-arc / arc_0 / dead-zone /
  arc_{k+1} / post regions of t's consumer pattern, contributing either
  "size unconditional" (in arc) or "size · c_t" (in dead zone / pre /
  post — alive only if cold).

Caveat: peak is sampled at ~256 of typically ~10k gpu events. The LP
minimizes the *modeled* peak; sim's actual peak may exceed it. Treat
``milp_peak_mb`` as a lower bound; verify in sim.

Schedule emission: every NeutralPrefetch / NeutralEvict / NeutralColdStart
carries the cgsim_tid directly, so the injector's pre_resolved fast path
fires and shape-disambiguation / synth_gates / coverage_repair don't run.
"""

from __future__ import annotations

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
# ε in the objective: per-byte tiebreaker on streaming. Must be small
# enough that 1 byte of peak improvement beats any tiebreaker shuffle
# (P scales ~1e10; 1e-6 means a tiebreaker can shift ~10kB of streaming
# per byte of peak, far below resolution).
EPSILON_PER_BYTE = 1.0e-6


# ---------------------------------------------------------------------------
# Pool + per-tid consumer pattern
# ---------------------------------------------------------------------------


@dataclass
class _PoolTensor:
    """One cgsim_tid in the optimization domain.

    See ct_milp_lateness for the consumer-pattern semantics; this scheduler
    uses the identical pool structure.
    """
    cgsim_tid: int
    size_bytes: int
    name: str
    dtype: str
    consumers: list[tuple[int, int, int]]
    tau_h2d_ns: int
    tau_d2h_ns: int
    gap_feasibility: list[bool]
    c_feasibility: bool
    consumer_graph_ids: list[int] = field(default_factory=list)
    consumer_launch_ids: list[int] = field(default_factory=list)


def _build_pool(trace: Trace, hw: HwParams) -> dict[int, _PoolTensor]:
    bw_h2d = max(effective_h2d_bw(hw), 1e-9)
    bw_d2h = max(float(hw.d2h_bw), 1e-9)

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

    consumers_by_tid: dict[int, list[tuple[int, int, int, int, int]]] = {}
    for nid, node in trace.node_map.items():
        rk = str((node.args or {}).get("resource_kind") or "")
        if rk not in _GPU_RESOURCE_KINDS:
            continue
        start_ns = int((node.args or {}).get("start_ns") or 0)
        end_ns = int((node.args or {}).get("end_ns") or start_ns)
        if start_ns <= 0:
            continue
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
                (start_ns, end_ns, int(nid), gid_i, lid_i)
            )

    # Build global gpu-event ordering and cum-time. The feasibility
    # checks below must use cum-time (the actual sim-time available
    # between issuer and consumer), not trace-time (which can be
    # 5-10× looser on profiled traces with idle gaps).
    gpu_events_sorted: list[tuple[int, int, int]] = []  # (start_ns, nid, dur)
    for nid, node in trace.node_map.items():
        rk = str((node.args or {}).get("resource_kind") or "")
        if rk not in _GPU_RESOURCE_KINDS:
            continue
        start_ns = int((node.args or {}).get("start_ns") or 0)
        end_ns = int((node.args or {}).get("end_ns") or start_ns)
        if start_ns <= 0:
            continue
        gpu_events_sorted.append((start_ns, int(nid), max(0, end_ns - start_ns)))
    gpu_events_sorted.sort(key=lambda x: x[0])
    global_event_idx_of_nid: dict[int, int] = {
        nid: i for i, (_s, nid, _d) in enumerate(gpu_events_sorted)
    }
    global_g_cum_start: list[int] = [0]
    for _s, _nid, d in gpu_events_sorted:
        global_g_cum_start.append(global_g_cum_start[-1] + d)
    # global_g_cum_start[i] = sim time at start of event i.

    # Per-graph gpu event lists (gid → sorted list of (trace_start, nid)),
    # for the in-graph issuer search.
    nodes_by_graph: dict[int, list[tuple[int, int]]] = {}
    for start_ns, nid, _d in gpu_events_sorted:
        node = trace.node_map.get(nid)
        gid_raw = (node.args or {}).get("compiled_graph_id") if node else None
        try:
            gid_i = int(gid_raw) if gid_raw is not None else -1
        except (TypeError, ValueError):
            gid_i = -1
        nodes_by_graph.setdefault(gid_i, []).append((int(start_ns), int(nid)))

    def _has_cum_feasible_issuer(
        consumer_gid: int, consumer_nid: int, tau_h2d_ns: int,
        earliest_allowed_trace_ns: int = -1,
    ) -> bool:
        """True iff consumer_gid contains a gpu node whose cum-time start
        satisfies G_cum(issuer.idx) ≤ G_cum(consumer.idx) − τ AND whose
        trace_start is > earliest_allowed_trace_ns (so for evict+refetch
        gaps the issuer fires after the prior consumer ended)."""
        c_idx = global_event_idx_of_nid.get(consumer_nid)
        if c_idx is None:
            return False
        target_g_cum = global_g_cum_start[c_idx] - tau_h2d_ns
        if target_g_cum < 0:
            return False
        for ts, nid in nodes_by_graph.get(consumer_gid, ()):
            if ts <= earliest_allowed_trace_ns:
                continue
            if ts >= int(global_g_cum_start[c_idx]):  # not really useful guard
                break
            i = global_event_idx_of_nid.get(nid)
            if i is None:
                continue
            if global_g_cum_start[i] <= target_g_cum:
                return True
        return False

    pool: dict[int, _PoolTensor] = {}
    for tid, raw in consumers_by_tid.items():
        raw.sort(key=lambda r: r[0])
        tensor = trace.tensor_map[tid]
        size = int(tensor.size_bytes)
        tau_h2d = int(hw.h2d_latency_ns) + int(size / bw_h2d)
        tau_d2h = int(hw.d2h_latency_ns) + int(size / bw_d2h)
        consumers = [(int(nid), int(s), int(e)) for s, e, nid, _, _ in raw]
        graph_ids = [int(g) for _, _, _, g, _ in raw]
        launch_ids = [int(l) for _, _, _, _, l in raw]
        # Per-gap feasibility in cum-time. The gap is feasible iff
        #   (a) consumer_k.end < consumer_{k+1}.start in trace order
        #       (evict can fire after consumer_k ends)
        #   (b) some gpu node in consumer_{k+1}'s graph with
        #       trace_start > consumer_k.end has cum-time gap to
        #       consumer_{k+1} ≥ τ_h2d (refetch can finish in sim time)
        gap_feas: list[bool] = []
        for i in range(len(consumers) - 1):
            ck_end = consumers[i][2]
            ckp1_start = consumers[i + 1][1]
            ckp1_nid = consumers[i + 1][0]
            next_gid = graph_ids[i + 1]
            if ckp1_start <= ck_end:
                gap_feas.append(False)
                continue
            gap_feas.append(_has_cum_feasible_issuer(
                next_gid, ckp1_nid, tau_h2d,
                earliest_allowed_trace_ns=ck_end,
            ))
        # c_feasibility in cum-time: initial prefetch needs some gpu node
        # in consumer_0's graph whose cum-time start ≤ G_cum(c_0.idx) − τ.
        consumer_0_gid = graph_ids[0] if graph_ids else -1
        consumer_0_nid = consumers[0][0]
        c_feas = _has_cum_feasible_issuer(
            consumer_0_gid, consumer_0_nid, tau_h2d,
        )
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
        )
    return pool


def _build_gpu_consumer_timeline(
    trace: Trace,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for nid, node in trace.node_map.items():
        rk = str((node.args or {}).get("resource_kind") or "")
        if rk not in _GPU_RESOURCE_KINDS:
            continue
        start_ns = int((node.args or {}).get("start_ns") or 0)
        if start_ns <= 0:
            continue
        out.append((int(nid), int(start_ns)))
    out.sort(key=lambda x: x[1])
    return out


# ---------------------------------------------------------------------------
# LP
# ---------------------------------------------------------------------------


@dataclass
class _LPResult:
    c_solution: dict[int, float]
    e_solution: dict[tuple[int, int], float]
    forced_cold: set[int]
    feasible_tids: list[int]
    peak_bytes: int
    makespan_ns: int
    g_total_ns: int
    solver_status: str
    diagnostics: dict[str, Any]


def _gpu_total_duration_ns(trace: Trace) -> int:
    """Sum of gpu compute node durations.

    The sim's SimpleGPU serializes all gpu compute (max_concurrent_jobs=1),
    so the sum of per-node (end_ns − start_ns) is the lower bound on
    gpu-side runtime in sim. This is the makespan floor: no streaming
    plan can complete faster than running every gpu kernel back-to-back.
    """
    total = 0
    for _nid, node in trace.node_map.items():
        rk = str((node.args or {}).get("resource_kind") or "")
        if rk not in _GPU_RESOURCE_KINDS:
            continue
        start_ns = int((node.args or {}).get("start_ns") or 0)
        end_ns = int((node.args or {}).get("end_ns") or start_ns)
        if start_ns <= 0:
            continue
        dur = end_ns - start_ns
        if dur > 0:
            total += dur
    return int(total)


def _select_sample_points(
    gpu_consumers: list[tuple[int, int]],
    max_samples: int = 256,
) -> list[tuple[int, int]]:
    if len(gpu_consumers) <= max_samples:
        return list(gpu_consumers)
    step = len(gpu_consumers) / max_samples
    picked: list[tuple[int, int]] = []
    for i in range(max_samples):
        idx = int(i * step)
        picked.append(gpu_consumers[idx])
    if picked[-1] != gpu_consumers[-1]:
        picked.append(gpu_consumers[-1])
    return picked


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
    audit: bool,
) -> tuple[np.ndarray | None, bool, str, str, bool]:
    """Two-phase solve via highspy. Mirrors ct_milp_lateness."""
    inf = highspy.kHighsInf

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
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
    if status != highspy.HighsModelStatus.kOptimal:
        msg = h.modelStatusToString(status)
        return (
            None, False,
            f"phase1 LP not optimal: {msg}",
            msg, True,
        )
    lp_sol = h.getSolution()
    x_lp = np.asarray(list(lp_sol.col_value), dtype=np.float64)
    if audit:
        binary_mask = np.asarray(integrality_arr) == 1
        bvals = x_lp[binary_mask]
        n_zero = int(np.sum(bvals < 0.01))
        n_one = int(np.sum(bvals > 0.99))
        n_frac = int(np.sum((bvals >= 0.01) & (bvals <= 0.99)))
        print(
            f"[ct_milp_peak:audit] phase 1 LP relaxation: "
            f"binaries ≈0: {n_zero}, ≈1: {n_one}, fractional: {n_frac}"
        )

    x_warm = x_lp.copy()
    int_indices = [i for i in range(total_vars) if integrality_arr[i] == 1]
    for i in int_indices:
        x_warm[i] = 1.0 if x_lp[i] >= 0.5 else 0.0

    h.changeColsIntegrality(
        len(int_indices),
        int_indices,
        [highspy.HighsVarType.kInteger] * len(int_indices),
    )
    sol = highspy.HighsSolution()
    sol.col_value = list(x_warm)
    h.setSolution(sol)
    h.run()

    status = h.getModelStatus()
    status_str = h.modelStatusToString(status)
    if status == highspy.HighsModelStatus.kOptimal:
        final = np.asarray(list(h.getSolution().col_value), dtype=np.float64)
        return (final, True, "phase2 MILP optimal", status_str, False)
    if status == highspy.HighsModelStatus.kTimeLimit:
        final = np.asarray(list(h.getSolution().col_value), dtype=np.float64)
        binary_mask = np.asarray(integrality_arr) == 1
        bvals = final[binary_mask]
        is_integer = bool(np.all(
            (bvals < 0.01) | (bvals > 0.99)
        ))
        if is_integer:
            return (
                final, True,
                "phase2 MILP time-limited (returning incumbent)",
                status_str, False,
            )
        return (
            x_warm, True,
            "phase2 MILP time-limited (returning warm-start)",
            status_str, True,
        )
    return (
        None, False,
        f"phase2 MILP returned {status_str}",
        status_str, True,
    )


def _solve_milp(
    pool: dict[int, _PoolTensor],
    trace: Trace,
    hw: HwParams,
    *,
    extra_static_bytes: int,
    g_total_ns: int,
    makespan_target_ns: int | None,
    max_peak_samples: int,
    time_limit_s: float | None,
    lp_relaxation: bool,
    audit: bool,
) -> _LPResult:
    """Build and solve the peak-minimization MILP.

    Variables:
        c_t           binary per feasible pool tid
        e_{t,k}       binary per feasible cross-iter gap
        P             continuous ≥ 0 — modeled peak (driven by per-sample rows)
        M             continuous ≥ G_total — modeled makespan (ns),
                      bounded above by makespan_target_ns when supplied.
                      Lower-bound rows: M ≥ G_total (via lower bound) and
                      M ≥ H_total = Σ δ_t·(1−c_t) + Σ δ_t·e_{t,k}.
    """
    # ---- 1. Feasibility filter ----
    feasible_tids: list[int] = []
    forced_cold: set[int] = set()
    for tid, pt in pool.items():
        if not pt.c_feasibility and not any(pt.gap_feasibility):
            forced_cold.add(tid)
            continue
        feasible_tids.append(tid)

    nv = len(feasible_tids)
    if audit:
        forced_bytes = sum(pool[t].size_bytes for t in forced_cold)
        c_feas_false = [t for t in feasible_tids if not pool[t].c_feasibility]
        c_feas_false_bytes = sum(pool[t].size_bytes for t in c_feas_false)
        print(
            f"[ct_milp_peak:audit] pool size={len(pool)} tensors "
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
            if pt.gap_feasibility[k]:
                e_var_idx[(tid, k)] = col
                col += 1
    n_e = col - nv

    P_IDX = col
    col += 1
    M_IDX = col
    col += 1

    # ---- 2b. Build gpu-event order, G_cum, and per-tid deadline event indices ----
    #
    # The cumulative PCIe budget is *gpu-cumulative time*, not trace time:
    # in sim, the PCIe queue runs against gpu compute progress (G_cum),
    # not the trace's wall clock. Trace timeline is typically 5-10×
    # G_total because the captured profile contains gpu-idle stretches
    # the sim collapses away.
    #
    # G_cum_start[i] = sum of durations of gpu events 0..i-1 (the sim
    # time at which event i starts, assuming no stall). Prefetch for a
    # deadline at event i must complete by G_cum_start[i].
    gpu_event_dur: list[tuple[int, int, int]] = []  # (start_ns, nid, dur)
    for nid, node in trace.node_map.items():
        rk = str((node.args or {}).get("resource_kind") or "")
        if rk not in _GPU_RESOURCE_KINDS:
            continue
        start_ns = int((node.args or {}).get("start_ns") or 0)
        end_ns = int((node.args or {}).get("end_ns") or start_ns)
        if start_ns <= 0:
            continue
        gpu_event_dur.append((start_ns, int(nid), max(0, end_ns - start_ns)))
    gpu_event_dur.sort(key=lambda x: x[0])
    event_idx_of_nid: dict[int, int] = {
        nid: i for i, (_s, nid, _d) in enumerate(gpu_event_dur)
    }
    g_cum_start: list[int] = [0]
    for _s, _nid, d in gpu_event_dur:
        g_cum_start.append(g_cum_start[-1] + d)
    # g_cum_start[i] = sim time at the start of gpu event i. The final
    # entry g_cum_start[N] == G_total (sim time at end of last event).

    # Group deadlines by event idx. Both initial prefetches (keyed on c_t)
    # and refetches (keyed on e_{t,k}) share the same PCIe queue.
    deadlines_by_event: dict[int, dict[str, list[Any]]] = {}
    for tid in feasible_tids:
        pt = pool[tid]
        first_nid = pt.consumers[0][0]
        d = event_idx_of_nid.get(first_nid)
        if d is None:
            continue
        deadlines_by_event.setdefault(
            d, {"tids": [], "refetches": []}
        )["tids"].append(tid)
    for (tid, k), _col in e_var_idx.items():
        pt = pool[tid]
        cons_kp1_nid = pt.consumers[k + 1][0]
        d = event_idx_of_nid.get(cons_kp1_nid)
        if d is None:
            continue
        deadlines_by_event.setdefault(
            d, {"tids": [], "refetches": []}
        )["refetches"].append((tid, k))

    sorted_deadline_events = sorted(deadlines_by_event.keys())

    # One H_used aux var per unique deadline event. Sparse running-sum
    # formulation: each step adds only the new deadlines arriving at
    # that event, vs. dense cumulative rows which would be O(prior
    # deadlines) wide.
    h_used_idx: dict[int, int] = {}
    for d in sorted_deadline_events:
        h_used_idx[d] = col
        col += 1
    total_vars = col

    if audit:
        print(
            f"[ct_milp_peak:audit] cumulative-by-G_cum: "
            f"deadline_events={len(sorted_deadline_events)} of "
            f"{len(gpu_event_dur)} gpu events; G_total={g_total_ns/1e6:.1f}ms"
        )

    # ---- 3. Variable bounds + integrality ----
    bounds_list: list[tuple[float, float | None]] = []
    for tid in feasible_tids:
        pt = pool[tid]
        if not pt.c_feasibility:
            bounds_list.append((1.0, 1.0))
        else:
            bounds_list.append((0.0, 1.0))
    bounds_list.extend([(0.0, 1.0)] * n_e)
    bounds_list.append((0.0, None))                          # P
    # M bounds: lower = G_total (gpu compute can't go faster than itself),
    # upper = makespan_target_ns when supplied, else unbounded. With the
    # H2D lower-bound row below this gives M ≥ max(G_total, H_total) —
    # the 2-machine flow-shop perfect-overlap makespan bound.
    bounds_list.append(
        (float(g_total_ns), float(makespan_target_ns))
        if makespan_target_ns is not None
        else (float(g_total_ns), None)
    )                                                        # M
    # H_used[d]: continuous ≥ 0, monotonically increasing in d. Cap rows
    # below enforce H_used[d] ≤ g_cum_start[d].
    bounds_list.extend([(0.0, None)] * len(sorted_deadline_events))

    integrality_arr = None
    if not lp_relaxation:
        integrality_arr = np.zeros(total_vars, dtype=np.int64)
        integrality_arr[: nv + n_e] = 1

    # ---- 4. Objective ----
    #
    # Primary: P (peak VRAM in bytes).
    # Tiebreaker: ε · (streamed_bytes + refetched_bytes).
    #   Cold reward:  c_obj[c_t]    = −ε · size_t
    #   Refetch cost: c_obj[e_{t,k}] = +ε · size_t
    # With ε = 1e-6 (bytes / byte): peak resolution dominates. Without
    # the tiebreaker the LP picks an arbitrary feasible point among
    # equal-peak plans (often heavier streaming than necessary).
    c_obj = np.zeros(total_vars, dtype=np.float64)
    c_obj[P_IDX] = 1.0
    for tid in feasible_tids:
        size = pool[tid].size_bytes
        c_obj[c_var_idx[tid]] = -float(size) * EPSILON_PER_BYTE
    for (tid, k), col in e_var_idx.items():
        size = pool[tid].size_bytes
        c_obj[col] = float(size) * EPSILON_PER_BYTE

    # ---- 5. Symmetric coupling: c + e = 1 per feasible gap ----
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    ub_list: list[float] = []
    row = 0

    for tid in feasible_tids:
        pt = pool[tid]
        for k, feas in enumerate(pt.gap_feasibility):
            if not feas:
                continue
            ci = c_var_idx[tid]
            ei = e_var_idx[(tid, k)]
            rows.extend([row, row])
            cols.extend([ci, ei])
            vals.extend([1.0, 1.0])
            ub_list.append(1.0)
            row += 1
            rows.extend([row, row])
            cols.extend([ci, ei])
            vals.extend([-1.0, -1.0])
            ub_list.append(-1.0)
            row += 1

    # ---- 6. Sample grid ----
    gpu_consumers = _build_gpu_consumer_timeline(trace)
    if not gpu_consumers:
        raise RuntimeError(
            "[ct_milp_peak] no gpu consumer events in trace; "
            "cannot build LP sample grid."
        )
    samples = _select_sample_points(gpu_consumers, max_samples=max_peak_samples)
    if audit:
        print(
            f"[ct_milp_peak:audit] gpu_consumer_events={len(gpu_consumers)} "
            f"sampled_points={len(samples)}"
        )

    # ---- 7. Peak VRAM rows (one per sample point) ----
    forced_cold_bytes = sum(pool[t].size_bytes for t in forced_cold)
    constant_floor = float(forced_cold_bytes) + float(extra_static_bytes)

    for nid_sample, t_l in samples:
        const_addons = constant_floor
        var_coefs: dict[int, float] = {}
        for tid in feasible_tids:
            pt = pool[tid]
            size = pt.size_bytes
            is_currently_consumed = any(c[0] == nid_sample for c in pt.consumers)
            if is_currently_consumed:
                const_addons += size
                continue
            first_start = pt.consumers[0][1]
            last_end = pt.consumers[-1][2]
            tau = pt.tau_h2d_ns

            if t_l < first_start:
                arc_0_start = max(0, first_start - tau)
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
                const_addons += size
                continue
            arc_kp1_start = pt.consumers[k_in + 1][1] - tau
            if t_l >= arc_kp1_start:
                const_addons += size
            elif pt.gap_feasibility[k_in]:
                var_coefs[c_var_idx[tid]] = (
                    var_coefs.get(c_var_idx[tid], 0.0) + size
                )
            else:
                const_addons += size

        rows.append(row)
        cols.append(P_IDX)
        vals.append(-1.0)
        for var_col, coef in var_coefs.items():
            if abs(coef) < 1e-9:
                continue
            rows.append(row)
            cols.append(var_col)
            vals.append(float(coef))
        ub_list.append(-float(const_addons))
        row += 1

    # ---- 7a. Global cold-floor cut ----
    # Σ_t size_t · c_t  +  forced_cold_bytes  +  extras  ≤  P
    # Tightens the LP relaxation: cold tids are alive everywhere, so
    # P must dominate their sum. Without this, fractional c_t can
    # leak into per-sample rows ambiguously.
    rows.append(row)
    cols.append(P_IDX)
    vals.append(-1.0)
    for tid in feasible_tids:
        size = pool[tid].size_bytes
        rows.append(row)
        cols.append(c_var_idx[tid])
        vals.append(float(size))
    ub_list.append(-(float(forced_cold_bytes) + float(extra_static_bytes)))
    row += 1

    # ---- 7b. Makespan lower-bound row: M ≥ H_total ----
    #
    # H_total = Σ_t δ_t · (1 − c_t) + Σ_{(t,k) feas} δ_t · e_{t,k}
    #         = total serial-queue PCIe time the plan requires.
    #
    # M ≥ H_total combined with the M lower bound (M ≥ G_total via
    # variable bounds) gives the 2-machine flow-shop makespan floor:
    #   M  ≥  max(G_total, H_total)
    #
    # With the cumulative-by-T rows enforcing zero LP-stall (every
    # individual prefetch deadline met), max(G, H) is a tight
    # estimate of the actual sim makespan — neither queue idles.
    # Without cumulative-by-T, head-of-line blocking can stretch sim
    # makespan above max(G, H).
    #
    # Expansion: H_total - M ≤ 0
    #          = Σ δ_t · (1−c_t) + Σ δ_t · e_{t,k} − M  ≤ 0
    #          = −Σ δ_t · c_t + Σ δ_t · e_{t,k} − M  ≤  −Σ δ_t
    rows.append(row)
    cols.append(M_IDX)
    vals.append(-1.0)
    h_total_const = 0.0
    for tid in feasible_tids:
        pt = pool[tid]
        delta = pt.tau_h2d_ns
        h_total_const += delta
        rows.append(row)
        cols.append(c_var_idx[tid])
        vals.append(-float(delta))
    for (tid, k), col in e_var_idx.items():
        pt = pool[tid]
        rows.append(row)
        cols.append(col)
        vals.append(float(pt.tau_h2d_ns))
    ub_list.append(-h_total_const)
    row += 1

    if audit:
        cap_str = (
            f"{makespan_target_ns/1e6:.1f}ms cap"
            if makespan_target_ns is not None else "no cap"
        )
        print(
            f"[ct_milp_peak:audit] makespan: G_total={g_total_ns/1e6:.1f}ms "
            f"target={cap_str}"
        )

    # ---- 8. Cumulative-by-G_cum PCIe rows (HARD: no stall slack) ----
    #
    # Sparse running-sum formulation with auxiliary H_used[d] variables.
    # For each unique deadline event d (sorted by gpu-event order):
    #
    #   (a) Recurrence row:
    #       H_used[d]  ≥  H_used[d_prev]
    #                  +  Σ_{t : first_consumer(t) at d}  δ_t · (1 − c_t)
    #                  +  Σ_{(t,k) feas : consumer_{k+1} at d}  δ_t · e_{t,k}
    #
    #   (b) Cap row:
    #       H_used[d]  ≤  g_cum_start[d]
    #
    # Together these enforce the serial-queue invariant in *gpu-time*:
    # by the moment gpu event d would start in a zero-stall sim
    # (sim_time = g_cum_start[d] = Σ durations of events 0..d−1), the
    # cumulative PCIe work whose deadlines have arrived must fit in
    # that elapsed sim time.
    #
    # Why gpu-time, not trace-time:
    # In sim, gpu compute proceeds back-to-back (no idle gaps) when
    # data is ready. The PCIe queue runs in parallel from sim t=0.
    # The deadline for a prefetch is *when the consuming gpu event
    # would fire if zero-stall* — i.e., g_cum_start[d], not the
    # consumer's trace_start_ns. Trace timeline (typically 5-10× G_total
    # on profiling-captured traces) gives the LP ~10× more headroom
    # than sim actually has, so cumulative-by-trace plans look fine
    # to the LP but stall heavily in sim.
    #
    # Why aux vars (vs direct cumulative rows):
    # Direct cumulative `Σ_{deadline ≤ d} δ ≤ g_cum_start[d]` would
    # produce O(N_events^2) nonzeros at full sampling. The running-sum
    # H_used variable lets each recurrence row touch only the *new*
    # deadlines at event d (typically 1-3 terms) plus the prior
    # H_used[d_prev], so total nz ≈ O(N_deadline_events) — tractable
    # even at 16k events.
    #
    # D2H evictions run concurrent with H2D under duplex, so eviction
    # transfers don't enter this budget.
    #
    # Cold-all (c=1, e=0): each recurrence becomes H_used[d] ≥ H_used[d_prev]
    # + 0, and cap H_used[d] ≤ g_cum_start[d] is trivially satisfied at
    # H_used[d] = 0. So the LP is always feasible.
    #
    # Edge case at d=0 (first gpu event is a deadline): g_cum_start[0]=0.
    # Cap forces H_used[0] = 0 → all tids first-consumed at event 0
    # must be cold (c=1) AND no refetch can be due there. This matches
    # sim semantics: prefetch deadline at sim t=0 is impossible to meet.
    #
    # Expansion of recurrence to ≤ form:
    #   −H_used[d] + H_used[d_prev]
    #   + Σ_{first[t]=d} (−δ_t · c_t)
    #   + Σ_{(t,k):cons_kp1=d} δ_t · e_{t,k}
    #   ≤  −Σ_{first[t]=d} δ_t
    prev_h_idx: int | None = None
    for d in sorted_deadline_events:
        dls = deadlines_by_event[d]
        # Recurrence row
        rows.append(row)
        cols.append(h_used_idx[d])
        vals.append(-1.0)
        if prev_h_idx is not None:
            rows.append(row)
            cols.append(prev_h_idx)
            vals.append(1.0)
        const_terms = 0
        for tid in dls["tids"]:
            delta = pool[tid].tau_h2d_ns
            rows.append(row)
            cols.append(c_var_idx[tid])
            vals.append(-float(delta))
            const_terms += delta
        for (tid, k) in dls["refetches"]:
            delta = pool[tid].tau_h2d_ns
            rows.append(row)
            cols.append(e_var_idx[(tid, k)])
            vals.append(float(delta))
        ub_list.append(-float(const_terms))
        row += 1

        # Cap row: H_used[d] ≤ g_cum_start[d]
        rows.append(row)
        cols.append(h_used_idx[d])
        vals.append(1.0)
        ub_list.append(float(g_cum_start[d]))
        row += 1

        prev_h_idx = h_used_idx[d]

    if audit and sorted_deadline_events:
        first_d = sorted_deadline_events[0]
        n_forced_at_0 = (
            len(deadlines_by_event[first_d]["tids"])
            + len(deadlines_by_event[first_d]["refetches"])
            if first_d == 0 else 0
        )
        if n_forced_at_0:
            print(
                f"[ct_milp_peak:audit] {n_forced_at_0} deadlines at "
                f"gpu event 0 (G_cum=0) — these tids forced cold by "
                f"cumulative-by-G_cum cap."
            )

    nb = row

    # ---- 9. Solve (two-phase: LP relaxation → MILP with warm-start) ----
    fell_back = False
    used_two_phase = False
    res_x: np.ndarray | None = None
    res_success = False
    res_message = ""

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
                audit=audit,
            )
        )
        if not res_success:
            if audit:
                print(
                    f"[ct_milp_peak:solver] two-phase highspy failed: "
                    f"{res_message!r} — falling back to scipy linprog."
                )
            res_x = None
        else:
            fell_back = lp_only

    if res_x is None:
        A = csr_matrix((vals, (rows, cols)), shape=(nb, total_vars))
        b_ub_arr = np.array(ub_list, dtype=np.float64)
        options: dict[str, Any] = {"disp": False}
        if time_limit_s is not None:
            options["time_limit"] = float(time_limit_s)
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
                    f"[ct_milp_peak:solver] scipy MILP failed: "
                    f"status={getattr(res, 'message', '')!r} — "
                    f"falling back to LP relaxation."
                )
            kwargs.pop("integrality", None)
            res = linprog(c_obj, **kwargs)
            fell_back = True
        res_x = np.asarray(res.x) if res.success and res.x is not None else None
        res_success = bool(res.success)
        res_message = str(getattr(res, "message", ""))

    if audit:
        tag = "highspy-two-phase" if used_two_phase else "scipy-linprog"
        print(
            f"[ct_milp_peak:solver] backend={tag} success={res_success} "
            f"fell_back={fell_back} status={res_message!r}"
        )

    class _Res:
        pass
    res = _Res()
    res.success = res_success
    res.x = res_x
    res.message = res_message

    # ---- 10. Decode ----
    c_solution: dict[int, float] = {}
    e_solution: dict[tuple[int, int], float] = {}
    peak_bytes = 0
    makespan_ns = 0

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
        peak_bytes = int(float(x[P_IDX]))
        makespan_ns = int(float(x[M_IDX]))
    else:
        # Hard fallback: cold-start everything feasible. This is the
        # zero-stall feasible point; only the peak suffers.
        for tid in feasible_tids:
            c_solution[tid] = 1.0
            pt = pool[tid]
            for k in range(len(pt.consumers) - 1):
                e_solution[(tid, k)] = 0.0
        peak_bytes = int(
            constant_floor + sum(pool[t].size_bytes for t in feasible_tids)
        )
        # Cold-all has H_total = 0, so makespan floor is just G_total.
        makespan_ns = int(g_total_ns)

    diagnostics = {
        "pool_size": len(pool),
        "forced_cold_count": len(forced_cold),
        "forced_cold_bytes": forced_cold_bytes,
        "feasible_var_count": nv,
        "e_var_count": n_e,
        "n_samples": len(samples),
        "g_total_ns": int(g_total_ns),
        "makespan_target_ns": (
            int(makespan_target_ns) if makespan_target_ns is not None else None
        ),
        "lp_makespan_ns": int(makespan_ns),
        "solver_success": bool(res.success),
        "solver_status": str(getattr(res, "message", "")),
        "fell_back_to_lp": bool(fell_back),
        "lp_relaxation": bool(lp_relaxation),
    }

    return _LPResult(
        c_solution=c_solution,
        e_solution=e_solution,
        forced_cold=forced_cold,
        feasible_tids=feasible_tids,
        peak_bytes=int(peak_bytes),
        makespan_ns=int(makespan_ns),
        g_total_ns=int(g_total_ns),
        solver_status=str(getattr(res, "message", "")),
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Emit: NeutralSchedule with cgsim_tid pre-resolved
# ---------------------------------------------------------------------------


def _emit_neutral(
    pool: dict[int, _PoolTensor],
    result: _LPResult,
    trace: Trace,
    hw: HwParams,
) -> NeutralSchedule:
    KEEP_THRESHOLD = 0.5

    # Build cum-time issuer lookup. Sim's PCIe queue can only start
    # serving a prefetch when its ISSUER gpu node fires. In sim time
    # (where gpu events pack back-to-back), the time available from
    # the issuer to the consumer is:
    #   G_cum(consumer.idx) − G_cum(issuer.idx)
    # NOT (consumer.trace_start_ns − issuer.trace_start_ns), which can
    # be 5-10× larger on profiled traces because the trace includes
    # idle gaps the sim collapses away.
    #
    # The original _pick_issuer_node walked the trace-time gap, which
    # picked issuers that fired ~simultaneously with the consumer in
    # sim time, leaving the PCIe queue no cushion. Result: prefetches
    # arrived late, gpu stalled, sim runtime balloned by ~G_total.
    #
    # Walk per-graph gpu events sorted by trace_start_ns (which equals
    # gpu-event order), build per-graph G_cum, then pick the latest
    # issuer with G_cum(issuer.idx) ≤ G_cum(consumer.idx) − τ_h2d.
    gpu_events_sorted: list[tuple[int, int, int]] = []  # (start_ns, nid, dur)
    for nid, node in trace.node_map.items():
        rk = str((node.args or {}).get("resource_kind") or "")
        if rk not in _GPU_RESOURCE_KINDS:
            continue
        start_ns = int((node.args or {}).get("start_ns") or 0)
        end_ns = int((node.args or {}).get("end_ns") or start_ns)
        if start_ns <= 0:
            continue
        gpu_events_sorted.append((start_ns, int(nid), max(0, end_ns - start_ns)))
    gpu_events_sorted.sort(key=lambda x: x[0])
    global_event_idx_of_nid: dict[int, int] = {
        nid: i for i, (_s, nid, _d) in enumerate(gpu_events_sorted)
    }
    global_g_cum: list[int] = [0]
    for _s, _nid, d in gpu_events_sorted:
        global_g_cum.append(global_g_cum[-1] + d)
    # global_g_cum[i] = sim time at start of event i (assuming zero-stall
    # gpu queue packing).

    # Per-graph sub-list for the issuer search. Still constrain to the
    # consumer's graph (the injector's compile-side metadata uses
    # issue_graph_id), so an issuer from another graph wouldn't link.
    nodes_by_graph_trace: dict[int, list[tuple[int, int]]] = {}
    for start_ns, nid, _d in gpu_events_sorted:
        node = trace.node_map.get(nid)
        gid_raw = (node.args or {}).get("compiled_graph_id") if node else None
        try:
            gid_i = int(gid_raw) if gid_raw is not None else -1
        except (TypeError, ValueError):
            gid_i = -1
        nodes_by_graph_trace.setdefault(gid_i, []).append(
            (int(start_ns), int(nid))
        )

    def _pick_issuer_node(
        consumer_gid: int, consumer_start_ns: int, tau_h2d_ns: int,
    ) -> tuple[int, int]:
        """Latest gpu node in consumer_gid with cum-time gap ≥ τ_h2d.

        Returns (issuer_node_id, issuer_trace_start_ns). −1 sentinel
        when no in-graph predecessor offers enough cum-time cushion.
        """
        # Find consumer's index in the trace's gpu-event ordering.
        consumer_node_id = None
        for ts, nid in nodes_by_graph_trace.get(consumer_gid, ()):
            if ts == consumer_start_ns:
                consumer_node_id = nid
                break
        if consumer_node_id is None:
            return -1, -1
        c_idx = global_event_idx_of_nid.get(consumer_node_id)
        if c_idx is None:
            return -1, -1
        # target_g_cum: latest cum-time a valid issuer can start at and
        # still have its prefetch finish by the consumer's cum-time.
        target_g_cum = global_g_cum[c_idx] - tau_h2d_ns
        best_ns = -1
        best_nid = -1
        for ts, nid in nodes_by_graph_trace.get(consumer_gid, ()):
            if ts >= consumer_start_ns:
                break
            i = global_event_idx_of_nid.get(nid)
            if i is None:
                continue
            if global_g_cum[i] <= target_g_cum:
                if ts > best_ns:
                    best_ns = ts
                    best_nid = nid
        if best_nid < 0:
            return -1, -1
        return best_nid, best_ns

    neutral_tensors: list[NeutralTensor] = []
    uid_by_tid: dict[int, int] = {}
    for tid in sorted(pool.keys()):
        pt = pool[tid]
        primary_gid = pt.consumer_graph_ids[0] if pt.consumer_graph_ids else -1
        uid = len(neutral_tensors)
        uid_by_tid[tid] = uid
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

        if is_cold:
            cold_starts.append(NeutralColdStart(
                tensor_uid=uid,
                anchor_launch_id=(
                    max(0, int(pt.consumer_launch_ids[0]))
                    if pt.consumer_launch_ids else 0
                ),
                reason=(
                    "peak_forced_cold" if is_forced
                    else "peak_optimal_cold"
                ),
                cgsim_tids=[int(tid)],
            ))
        else:
            consumer_0 = pt.consumers[0]
            c0_nid, c0_start, _c0_end = consumer_0
            c0_gid = pt.consumer_graph_ids[0]
            c0_lid = pt.consumer_launch_ids[0]
            issue_nid, _ = _pick_issuer_node(
                c0_gid, c0_start, pt.tau_h2d_ns,
            )
            if issue_nid < 0:
                issue_nid = c0_nid
            prefetches.append(NeutralPrefetch(
                tensor_uid=uid,
                issue_launch_id=max(0, int(c0_lid)),
                wait_launch_id=max(0, int(c0_lid)),
                transfer_start_ns=int(max(0, c0_start - pt.tau_h2d_ns)),
                transfer_end_ns=int(c0_start),
                reason="peak_initial",
                issue_node_id=int(issue_nid),
                wait_node_id=int(c0_nid),
                cgsim_tid=int(tid),
                trusted_async=(issue_nid != c0_nid),
                issue_graph_id=-1,
                iter_mask=[],
            ))

        for k in range(len(pt.consumers) - 1):
            if not pt.gap_feasibility[k]:
                continue
            ev = result.e_solution.get((tid, k), 0.0)
            if float(ev) < KEEP_THRESHOLD:
                continue
            consumer_k = pt.consumers[k]
            consumer_kp1 = pt.consumers[k + 1]
            ck_nid, _ck_start, ck_end = consumer_k
            ckp1_nid, ckp1_start, _ckp1_end = consumer_kp1
            kp1_gid = pt.consumer_graph_ids[k + 1]
            kp1_lid = pt.consumer_launch_ids[k + 1]
            evict_reason = (
                "peak_hybrid_gap_evict" if is_cold
                else "peak_gap_evict"
            )
            refetch_reason = (
                "peak_hybrid_gap_refetch" if is_cold
                else "peak_gap_refetch"
            )
            evicts.append(NeutralEvict(
                tensor_uid=uid,
                issue_launch_id=max(0, int(pt.consumer_launch_ids[k])),
                transfer_start_ns=int(ck_end),
                transfer_end_ns=int(ck_end + pt.tau_d2h_ns),
                reason=evict_reason,
                issue_node_id=int(ck_nid),
                iter_mask=[],
                cgsim_tid=int(tid),
            ))
            re_nid, re_ts = _pick_issuer_node(
                kp1_gid, ckp1_start, pt.tau_h2d_ns,
            )
            if re_nid < 0 or re_ts <= ck_end:
                re_nid = ckp1_nid
                re_ts = ckp1_start
            prefetches.append(NeutralPrefetch(
                tensor_uid=uid,
                issue_launch_id=max(0, int(pt.consumer_launch_ids[k + 1])),
                wait_launch_id=max(0, int(kp1_lid)),
                transfer_start_ns=int(max(
                    ck_end + 1, ckp1_start - pt.tau_h2d_ns,
                )),
                transfer_end_ns=int(ckp1_start),
                reason=refetch_reason,
                issue_node_id=int(re_nid),
                wait_node_id=int(ckp1_nid),
                cgsim_tid=int(tid),
                trusted_async=(re_nid != ckp1_nid),
                issue_graph_id=-1,
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
    makespan_target_s: float | None = None,
    max_peak_samples: int = 256,
    time_limit_s: float | None = 240.0,
    lp_relaxation: bool = False,
    audit: bool = False,
    sidecars: Any = None,                 # accepted but ignored
    **_legacy_kwargs: Any,
) -> NeutralSchedule:
    """Build a pool-first peak-VRAM MILP schedule with hard zero-stall.

    Inputs:
      ``trace``               — Trace with deduped-by-storage_id cgsim Tensors.
      ``hw``                  — HwParams (h2d/d2h bandwidth, latencies).
      ``makespan_target_s``   — Optional hard upper bound on modeled
                                makespan (seconds). M is bounded below by
                                max(G_total, H_total); supplying this
                                caps the LP so plans whose total serial
                                PCIe time exceeds the target are rejected.
                                None ⇒ unbounded (LP picks the smallest
                                peak with zero LP-stall, runtime free).
      ``max_peak_samples``    — How many gpu-consumer points to sample
                                for peak rows. ~256 is a sweet spot for
                                sd3-med scale (10k events).
      ``time_limit_s``        — HiGHS time limit. None ⇒ no limit.
      ``lp_relaxation``       — Skip integrality, solve continuous LP
                                (debug aid).
      ``audit``               — Print pool/LP/solver diagnostics.
      ``sidecars``            — Accepted for interface parity; ignored.

    Returns:
      A NeutralSchedule with cgsim_tid pre-resolved on every entry.

    Note: peak rows sample ~256 of typically ~10k gpu events, so
    ``milp_peak_mb`` is a LOWER bound on sim's actual peak. Verify in
    sim before treating as ground truth. The makespan bound is also
    a *lower* bound (M ≥ max(G,H), the perfect-overlap floor); sim's
    actual runtime can run ~1.1–1.3× higher due to per-event sync
    overhead. Set targets with headroom.
    """
    pool = _build_pool(trace, hw)
    if not pool:
        raise RuntimeError(
            "[ct_milp_peak] pool is empty — no cuda WEIGHT/LEAF/INPUT "
            "tensors with gpu consumers found in trace."
        )

    g_total_ns = _gpu_total_duration_ns(trace)
    makespan_target_ns: int | None = None
    if makespan_target_s is not None:
        makespan_target_ns = int(round(float(makespan_target_s) * 1e9))
        if makespan_target_ns < g_total_ns:
            raise RuntimeError(
                f"[ct_milp_peak] makespan_target_s={makespan_target_s:.3f}s "
                f"({makespan_target_ns/1e6:.1f}ms) is below the gpu compute "
                f"floor G_total={g_total_ns/1e6:.1f}ms. No streaming plan "
                f"can run faster than gpu kernels run sequentially. Raise "
                f"the target above {g_total_ns/1e6:.1f}ms."
            )

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
            f"[ct_milp_peak:audit] no-consumer cuda layout overhead "
            f"= {extra_static_bytes/1e6:.1f}MB"
        )

    result = _solve_milp(
        pool, trace, hw,
        extra_static_bytes=extra_static_bytes,
        g_total_ns=g_total_ns,
        makespan_target_ns=makespan_target_ns,
        max_peak_samples=max_peak_samples,
        time_limit_s=time_limit_s,
        lp_relaxation=lp_relaxation,
        audit=audit,
    )

    neutral = _emit_neutral(pool, result, trace, hw)

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

    neutral.meta = {
        "io_model": "ct_milp_peak",
        "graph_order": neutral.graph_order,
        "milp_peak_mb": round(result.peak_bytes / 1e6, 2),
        "pcie_used_mb": round(pcie_h2d_bytes / 1e6, 2),
        "cold_bytes_mb": round(cold_bytes / 1e6, 2),
        "streamed_bytes_mb": round(streamed_bytes / 1e6, 2),
        "extras_static_mb": round(extra_static_bytes / 1e6, 2),
        "g_total_ms": round(result.g_total_ns / 1e6, 2),
        "lp_makespan_ms": round(result.makespan_ns / 1e6, 2),
        "makespan_target_ms": (
            round(result.diagnostics["makespan_target_ns"] / 1e6, 2)
            if result.diagnostics.get("makespan_target_ns") is not None
            else None
        ),
        "n_cold_starts": n_cold,
        "n_prefetches": n_pf,
        "n_evicts": n_ev,
        "diagnostics": result.diagnostics,
    }
    return neutral


def print_summary(neutral: NeutralSchedule) -> None:
    """One-line schedule summary."""
    print(
        f"Variant: {neutral.meta.get('io_model')} "
        f"| peak: {neutral.meta.get('milp_peak_mb')} MB "
        f"| makespan: {neutral.meta.get('lp_makespan_ms')} ms "
        f"(G_total={neutral.meta.get('g_total_ms')} ms"
        f", target={neutral.meta.get('makespan_target_ms')} ms) "
        f"| PCIe H2D used: {neutral.meta.get('pcie_used_mb')} MB "
        f"| cold: {neutral.meta.get('cold_bytes_mb')} MB "
        f"| streamed: {neutral.meta.get('streamed_bytes_mb')} MB "
        f"| prefetches: {neutral.meta.get('n_prefetches')} "
        f"| evicts: {neutral.meta.get('n_evicts')} "
        f"| cold_start: {neutral.meta.get('n_cold_starts')}"
    )
