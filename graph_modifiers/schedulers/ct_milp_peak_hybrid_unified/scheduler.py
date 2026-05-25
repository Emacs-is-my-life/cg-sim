"""Pool-first MILP — peak-VRAM objective, hybrid + soft-slack + UNIFIED structure.

Operates on ``UnifiedTimeline.tensors`` (coalesced by storage) at
compiled-launch granularity. Deadlines and issuer searches happen on
``tl.tasks`` positions (one per compiled kernel), but the *cumulative
time axis* is the trace's full-gpu cumulative — sum of all gpu node
durations (compiled kernels + aux/aten ops), compacted back-to-back.

Why this mix:

  * Trace timestamps include profiler idle gaps that sim collapses;
    using them overcounts queue budget.
  * The unified timeline's task start_ns/end_ns excludes aux ops
    (compiled-kernel-only). On workloads where aux is significant
    (sdxl-turbo: 247 ms of aux vs 47 ms of compiled kernel), using
    just unified-task time massively undercounts sim's actual queue
    capacity at each consumer.
  * The right answer is: sum of *all* gpu work durations (kernels +
    aux) compacted — equivalent to sim's wall-clock when no PCIe
    stall happens. We map each compiled-launch task to its
    corresponding G_cum position via tl.tasks[pos].node_id.
  * Structural benefit of unified: consumer events are compiled
    launches, not aux ops, so aux time between launches becomes
    *available queue budget* (no artificial deadline to schedule
    around).
  * Side benefit: tl.tasks[].node_id / .launch_id for emit identity —
    the injector keys on these exactly.

Inherits from ``ct_milp_peak_hybrid``:

  * Hybrid c-e independence: any (c, e_{t,k}) combination, including
    cold-at-layout + mid-run evict + refetch.
  * Soft cumulative-by-U_cum: caps H_used[d] − M ≤ u_cum_start[d] −
    U_total. M's upper bound is ``makespan_target_s``; higher target
    relaxes cumulative caps, allowing more streaming and lower peak.
  * Cum-time c_feasibility and gap_feasibility (now in unified time
    instead of gpu-trace time).

Caveats unchanged from hybrid:

  * Injector compatibility (coverage_repair) for c=1, e=1 plans.
  * LP assumes EDF queue order; sim uses water-fill.
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
    MultiGraphSidecars,
    NeutralColdStart,
    NeutralEvict,
    NeutralPrefetch,
    NeutralSchedule,
    NeutralTensor,
    UnifiedTimeline,
    build_unified_timeline,
    coalesce_by_storage,
    effective_h2d_bw,
)
from sim.core.trace import Trace


EPSILON_PER_BYTE = 1.0e-6


# ---------------------------------------------------------------------------
# Pool: one entry per coalesced-by-storage unified-timeline tensor
# ---------------------------------------------------------------------------


@dataclass
class _PoolTensor:
    """One unified-timeline storage group in the optimization domain.

    Fields are unified-timeline native — positions index ``tl.tasks``,
    timestamps are unified-compacted (no trace idle gaps).
    """
    uid: int
    size_bytes: int
    name: str
    dtype: str
    uses: list[tuple[int, int, int]]
    tau_h2d_ns: int
    tau_d2h_ns: int
    gap_feasibility: list[bool]
    c_feasibility: bool
    trace_tids: list[int]
    use_launch_ids: list[int] = field(default_factory=list)
    use_graph_ids: list[int] = field(default_factory=list)
    use_node_ids: list[int] = field(default_factory=list)
    primary_graph_id: int = -1
    storage_group_id: Any = None


_GPU_RESOURCE_KINDS = ("gpu_stream", "gpu", "gpu_runtime")


def _build_full_gcum_at_task(
    trace: Trace, tl: UnifiedTimeline,
) -> tuple[list[int], list[int], int]:
    """Compact-all-gpu cumulative time, indexed by tl.tasks position.

    Sum durations of every gpu trace node (compiled kernels + aux/aten)
    in trace-time order, building a per-node G_cum_start array. Then,
    for each tl.task position pos (which references a specific
    compiled-kernel trace_node_id), look up that node's G_cum_start.

    Returns:
      g_cum_start_at_task[pos] = sim time at the START of tl.tasks[pos]
                                 (assuming back-to-back gpu execution,
                                  no PCIe stall, no idle gaps).
      g_cum_end_at_task[pos] = sim time at the END of tl.tasks[pos].
      g_total_ns = sum of all gpu node durations.
    """
    gpu_events: list[tuple[int, int, int]] = []  # (start_ns, nid, dur)
    for nid, node in trace.node_map.items():
        rk = str((node.args or {}).get("resource_kind") or "")
        if rk not in _GPU_RESOURCE_KINDS:
            continue
        start_ns = int((node.args or {}).get("start_ns") or 0)
        end_ns = int((node.args or {}).get("end_ns") or start_ns)
        if start_ns <= 0:
            continue
        gpu_events.append((start_ns, int(nid), max(0, end_ns - start_ns)))
    gpu_events.sort(key=lambda x: x[0])

    g_cum_start_at_nid: dict[int, int] = {}
    g_cum_end_at_nid: dict[int, int] = {}
    cum = 0
    for _s, nid, d in gpu_events:
        g_cum_start_at_nid[nid] = cum
        cum += d
        g_cum_end_at_nid[nid] = cum
    g_total_ns = cum

    g_cum_start_at_task: list[int] = []
    g_cum_end_at_task: list[int] = []
    for task in tl.tasks:
        nid = int(task.node_id)
        # Default to task.start_ns / end_ns if the trace doesn't have
        # this node (shouldn't happen, but defensive).
        s = g_cum_start_at_nid.get(nid, int(task.start_ns))
        e = g_cum_end_at_nid.get(nid, int(task.end_ns))
        g_cum_start_at_task.append(s)
        g_cum_end_at_task.append(e)
    return g_cum_start_at_task, g_cum_end_at_task, g_total_ns


def _build_pool(
    tl: UnifiedTimeline, hw: HwParams,
    g_cum_start_at_task: list[int],
    g_cum_end_at_task: list[int],
) -> dict[int, _PoolTensor]:
    bw_h2d = max(effective_h2d_bw(hw), 1e-9)
    bw_d2h = max(float(hw.d2h_bw), 1e-9)

    per_graph_tasks: dict[int, list[tuple[int, int]]] = {}
    for pos, task in enumerate(tl.tasks):
        per_graph_tasks.setdefault(int(task.graph_id), []).append(
            (int(g_cum_start_at_task[pos]), int(pos))
        )
    for g in per_graph_tasks:
        per_graph_tasks[g].sort(key=lambda x: x[0])

    pool: dict[int, _PoolTensor] = {}
    for representative, _members in coalesce_by_storage(tl.tensors):
        if representative.size_bytes <= 0:
            continue
        if not representative.uses:
            continue
        if not representative.trace_tids:
            continue
        size = int(representative.size_bytes)
        tau_h2d = int(hw.h2d_latency_ns) + int(size / bw_h2d)
        tau_d2h = int(hw.d2h_latency_ns) + int(size / bw_d2h)
        # Sort uses by G_cum_start_at_task (compacted-all-gpu time),
        # not tl's synthetic kernel-only start_ns.
        uses_sorted = sorted(
            representative.uses, key=lambda p: g_cum_start_at_task[p]
        )
        # Each use entry: (task_pos, g_cum_start_at_pos, g_cum_end_at_pos).
        uses: list[tuple[int, int, int]] = [
            (int(p), int(g_cum_start_at_task[p]), int(g_cum_end_at_task[p]))
            for p in uses_sorted
        ]
        use_graph_ids = [int(tl.tasks[p].graph_id) for p in uses_sorted]
        use_launch_ids = [int(tl.tasks[p].launch_id) for p in uses_sorted]
        use_node_ids = [int(tl.tasks[p].node_id) for p in uses_sorted]

        gap_feas: list[bool] = []
        for k in range(len(uses) - 1):
            (_pk, _sk, ek) = uses[k]
            (_pk1, sk1, _ek1) = uses[k + 1]
            target = sk1 - tau_h2d
            if target <= ek:
                gap_feas.append(False)
                continue
            gid_next = use_graph_ids[k + 1]
            tasks_in_graph = per_graph_tasks.get(gid_next, ())
            import bisect
            keys = [t[0] for t in tasks_in_graph]
            idx = bisect.bisect_right(keys, ek)
            issuer_ok = (
                idx < len(tasks_in_graph) and tasks_in_graph[idx][0] <= target
            )
            gap_feas.append(issuer_ok)

        c_first_start = uses[0][1]
        gid_first = use_graph_ids[0]
        target_c = c_first_start - tau_h2d
        c_feas = False
        if target_c > 0:
            for ts, _pos in per_graph_tasks.get(gid_first, ()):
                if ts >= c_first_start:
                    break
                if ts <= target_c:
                    c_feas = True
                    break

        pool[int(representative.uid)] = _PoolTensor(
            uid=int(representative.uid),
            size_bytes=size,
            name=str(representative.graph_input_name or ""),
            dtype=str(representative.dtype or ""),
            uses=uses,
            tau_h2d_ns=tau_h2d,
            tau_d2h_ns=tau_d2h,
            gap_feasibility=gap_feas,
            c_feasibility=c_feas,
            trace_tids=[int(t) for t in representative.trace_tids],
            use_launch_ids=use_launch_ids,
            use_graph_ids=use_graph_ids,
            use_node_ids=use_node_ids,
            primary_graph_id=use_graph_ids[0],
            storage_group_id=representative.storage_group_id,
        )
    return pool


# ---------------------------------------------------------------------------
# LP
# ---------------------------------------------------------------------------


@dataclass
class _LPResult:
    c_solution: dict[int, float]
    e_solution: dict[tuple[int, int], float]
    forced_cold: set[int]
    feasible_uids: list[int]
    peak_bytes: int
    makespan_ns: int
    u_total_ns: int
    solver_status: str
    diagnostics: dict[str, Any]


def _select_sample_positions(
    tl: UnifiedTimeline, max_samples: int = 256,
) -> list[int]:
    n = len(tl.tasks)
    if n <= max_samples:
        return list(range(n))
    step = n / max_samples
    picked: list[int] = []
    for i in range(max_samples):
        picked.append(int(i * step))
    if picked[-1] != n - 1:
        picked.append(n - 1)
    return picked


def _solve_two_phase_highspy(
    *, total_vars, c_obj, bounds_list, rows, cols, vals, ub_list,
    integrality_arr, time_limit_s, audit,
):
    inf = highspy.kHighsInf
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    if time_limit_s is not None:
        h.setOptionValue("time_limit", float(time_limit_s))
    lo_arr = [float(b[0]) for b in bounds_list]
    hi_arr = [float(b[1]) if b[1] is not None else inf for b in bounds_list]
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
        return None, False, f"phase1 LP not optimal: {msg}", msg, True
    lp_sol = h.getSolution()
    x_lp = np.asarray(list(lp_sol.col_value), dtype=np.float64)
    if audit:
        binary_mask = np.asarray(integrality_arr) == 1
        bvals = x_lp[binary_mask]
        n_zero = int(np.sum(bvals < 0.01))
        n_one = int(np.sum(bvals > 0.99))
        n_frac = int(np.sum((bvals >= 0.01) & (bvals <= 0.99)))
        print(
            f"[ct_milp_peak_hybrid_unified:audit] phase 1 LP relaxation: "
            f"binaries ≈0: {n_zero}, ≈1: {n_one}, fractional: {n_frac}"
        )
    x_warm = x_lp.copy()
    int_indices = [i for i in range(total_vars) if integrality_arr[i] == 1]
    for i in int_indices:
        x_warm[i] = 1.0 if x_lp[i] >= 0.5 else 0.0
    h.changeColsIntegrality(
        len(int_indices), int_indices,
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
        return final, True, "phase2 MILP optimal", status_str, False
    if status == highspy.HighsModelStatus.kTimeLimit:
        final = np.asarray(list(h.getSolution().col_value), dtype=np.float64)
        binary_mask = np.asarray(integrality_arr) == 1
        bvals = final[binary_mask]
        is_integer = bool(np.all((bvals < 0.01) | (bvals > 0.99)))
        if is_integer:
            return (final, True,
                    "phase2 MILP time-limited (returning incumbent)",
                    status_str, False)
        return (x_warm, True,
                "phase2 MILP time-limited (returning warm-start)",
                status_str, True)
    return None, False, f"phase2 MILP returned {status_str}", status_str, True


def _solve_milp(
    pool: dict[int, _PoolTensor],
    tl: UnifiedTimeline,
    hw: HwParams,
    *,
    g_cum_start_at_task: list[int],
    extra_static_bytes: int,
    u_total_ns: int,
    makespan_target_ns: int | None,
    max_peak_samples: int,
    time_limit_s: float | None,
    lp_relaxation: bool,
    audit: bool,
) -> _LPResult:
    feasible_uids: list[int] = []
    forced_cold: set[int] = set()
    for uid, pt in pool.items():
        if not pt.uses:
            forced_cold.add(uid)
            continue
        feasible_uids.append(uid)
    nv = len(feasible_uids)
    if audit:
        forced_bytes = sum(pool[u].size_bytes for u in forced_cold)
        c_feas_false = [u for u in feasible_uids if not pool[u].c_feasibility]
        c_feas_false_bytes = sum(pool[u].size_bytes for u in c_feas_false)
        print(
            f"[ct_milp_peak_hybrid_unified:audit] pool size={len(pool)} "
            f"forced_cold={len(forced_cold)} ({forced_bytes/1e6:.1f}MB) "
            f"feasible_lp_vars={nv} c_feas_false={len(c_feas_false)} "
            f"({c_feas_false_bytes/1e6:.1f}MB)"
        )

    c_var_idx: dict[int, int] = {}
    for col, uid in enumerate(feasible_uids):
        c_var_idx[uid] = col

    e_var_idx: dict[tuple[int, int], int] = {}
    col = nv
    for uid in feasible_uids:
        pt = pool[uid]
        for k in range(len(pt.uses) - 1):
            if pt.gap_feasibility[k]:
                e_var_idx[(uid, k)] = col
                col += 1
    n_e = col - nv

    P_IDX = col
    col += 1
    M_IDX = col
    col += 1

    deadlines_by_pos: dict[int, dict[str, list[Any]]] = {}
    for uid in feasible_uids:
        pt = pool[uid]
        first_pos = pt.uses[0][0]
        deadlines_by_pos.setdefault(
            first_pos, {"uids": [], "refetches": []}
        )["uids"].append(uid)
    for (uid, k), _ in e_var_idx.items():
        pt = pool[uid]
        kp1_pos = pt.uses[k + 1][0]
        deadlines_by_pos.setdefault(
            kp1_pos, {"uids": [], "refetches": []}
        )["refetches"].append((uid, k))

    sorted_deadline_positions = sorted(deadlines_by_pos.keys())
    h_used_idx: dict[int, int] = {}
    for pos in sorted_deadline_positions:
        h_used_idx[pos] = col
        col += 1
    total_vars = col

    if audit:
        print(
            f"[ct_milp_peak_hybrid_unified:audit] tl.tasks={len(tl.tasks)} "
            f"deadline_positions={len(sorted_deadline_positions)} "
            f"U_total={u_total_ns/1e6:.1f}ms"
        )

    bounds_list: list[tuple[float, float | None]] = []
    for uid in feasible_uids:
        bounds_list.append((0.0, 1.0))
    bounds_list.extend([(0.0, 1.0)] * n_e)
    bounds_list.append((0.0, None))
    m_upper = (
        float(makespan_target_ns) if makespan_target_ns is not None
        else float(u_total_ns)
    )
    bounds_list.append((float(u_total_ns), m_upper))
    bounds_list.extend([(0.0, None)] * len(sorted_deadline_positions))

    integrality_arr = None
    if not lp_relaxation:
        integrality_arr = np.zeros(total_vars, dtype=np.int64)
        integrality_arr[: nv + n_e] = 1

    c_obj = np.zeros(total_vars, dtype=np.float64)
    c_obj[P_IDX] = 1.0
    c_obj[M_IDX] = 1.0e-12
    for uid in feasible_uids:
        size = pool[uid].size_bytes
        c_obj[c_var_idx[uid]] = -float(size) * EPSILON_PER_BYTE
    for (uid, k), col_e in e_var_idx.items():
        size = pool[uid].size_bytes
        c_obj[col_e] = float(size) * EPSILON_PER_BYTE

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    ub_list: list[float] = []
    row = 0

    sample_positions = _select_sample_positions(tl, max_samples=max_peak_samples)
    if audit:
        print(
            f"[ct_milp_peak_hybrid_unified:audit] peak samples = "
            f"{len(sample_positions)}"
        )

    forced_cold_bytes = sum(pool[u].size_bytes for u in forced_cold)
    constant_floor = float(forced_cold_bytes) + float(extra_static_bytes)

    for sample_pos in sample_positions:
        sample_start = int(g_cum_start_at_task[sample_pos])
        const_addons = constant_floor
        var_coefs: dict[int, float] = {}
        for uid in feasible_uids:
            pt = pool[uid]
            size = pt.size_bytes
            is_currently_consumed = any(
                p == sample_pos for (p, _s, _e) in pt.uses
            )
            if is_currently_consumed:
                const_addons += size
                continue
            first_start = pt.uses[0][1]
            last_end = pt.uses[-1][2]
            tau = pt.tau_h2d_ns
            if sample_start < first_start:
                arc_0_start = max(0, first_start - tau)
                if sample_start >= arc_0_start:
                    const_addons += size
                else:
                    var_coefs[c_var_idx[uid]] = (
                        var_coefs.get(c_var_idx[uid], 0.0) + size
                    )
                continue
            if sample_start > last_end:
                var_coefs[c_var_idx[uid]] = (
                    var_coefs.get(c_var_idx[uid], 0.0) + size
                )
                continue
            k_in = None
            for k in range(len(pt.uses) - 1):
                if pt.uses[k][2] < sample_start < pt.uses[k + 1][1]:
                    k_in = k
                    break
            if k_in is None:
                const_addons += size
                continue
            arc_kp1_start = pt.uses[k_in + 1][1] - tau
            if sample_start >= arc_kp1_start:
                const_addons += size
            elif pt.gap_feasibility[k_in] and (uid, k_in) in e_var_idx:
                const_addons += size
                e_col = e_var_idx[(uid, k_in)]
                var_coefs[e_col] = var_coefs.get(e_col, 0.0) - size
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

    prev_h_idx: int | None = None
    for pos in sorted_deadline_positions:
        # Use compacted-all-gpu cumulative time, not tl's kernel-only.
        u_cum_start = int(g_cum_start_at_task[pos])
        dls = deadlines_by_pos[pos]
        rows.append(row)
        cols.append(h_used_idx[pos])
        vals.append(-1.0)
        if prev_h_idx is not None:
            rows.append(row)
            cols.append(prev_h_idx)
            vals.append(1.0)
        const_terms = 0
        for uid in dls["uids"]:
            delta = pool[uid].tau_h2d_ns
            rows.append(row)
            cols.append(c_var_idx[uid])
            vals.append(-float(delta))
            const_terms += delta
        for (uid, k) in dls["refetches"]:
            delta = pool[uid].tau_h2d_ns
            rows.append(row)
            cols.append(e_var_idx[(uid, k)])
            vals.append(float(delta))
        ub_list.append(-float(const_terms))
        row += 1

        rows.append(row)
        cols.append(h_used_idx[pos])
        vals.append(1.0)
        rows.append(row)
        cols.append(M_IDX)
        vals.append(-1.0)
        ub_list.append(float(u_cum_start) - float(u_total_ns))
        row += 1

        prev_h_idx = h_used_idx[pos]

    if audit:
        cap_str = (
            f"{makespan_target_ns/1e6:.1f}ms cap"
            if makespan_target_ns is not None else "strict (U_total)"
        )
        print(
            f"[ct_milp_peak_hybrid_unified:audit] makespan: "
            f"U_total={u_total_ns/1e6:.1f}ms target={cap_str}"
        )

    nb = row

    fell_back = False
    used_two_phase = False
    res_x: np.ndarray | None = None
    res_success = False
    res_message = ""

    if _HIGHSPY_AVAILABLE and integrality_arr is not None and not lp_relaxation:
        used_two_phase = True
        res_x, res_success, res_message, _ms, lp_only = (
            _solve_two_phase_highspy(
                total_vars=total_vars, c_obj=c_obj, bounds_list=bounds_list,
                rows=rows, cols=cols, vals=vals, ub_list=ub_list,
                integrality_arr=integrality_arr,
                time_limit_s=time_limit_s, audit=audit,
            )
        )
        if not res_success:
            if audit:
                print(
                    f"[ct_milp_peak_hybrid_unified:solver] two-phase failed: "
                    f"{res_message!r} — falling back to scipy."
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
            "A_ub": A, "b_ub": b_ub_arr, "bounds": bounds_list,
            "method": "highs", "options": options,
        }
        if integrality_arr is not None:
            kwargs["integrality"] = integrality_arr
        res = linprog(c_obj, **kwargs)
        if not res.success and integrality_arr is not None and not lp_relaxation:
            kwargs.pop("integrality", None)
            res = linprog(c_obj, **kwargs)
            fell_back = True
        res_x = np.asarray(res.x) if res.success and res.x is not None else None
        res_success = bool(res.success)
        res_message = str(getattr(res, "message", ""))

    if audit:
        tag = "highspy-two-phase" if used_two_phase else "scipy-linprog"
        print(
            f"[ct_milp_peak_hybrid_unified:solver] backend={tag} "
            f"success={res_success} fell_back={fell_back} status={res_message!r}"
        )

    c_solution: dict[int, float] = {}
    e_solution: dict[tuple[int, int], float] = {}
    peak_bytes = 0
    makespan_ns = u_total_ns
    if res_success and res_x is not None:
        x = np.asarray(res_x)
        for uid in feasible_uids:
            c_solution[uid] = float(x[c_var_idx[uid]])
            pt = pool[uid]
            for k in range(len(pt.uses) - 1):
                if (uid, k) in e_var_idx:
                    e_solution[(uid, k)] = float(x[e_var_idx[(uid, k)]])
                else:
                    e_solution[(uid, k)] = 0.0
        peak_bytes = int(float(x[P_IDX]))
        makespan_ns = int(float(x[M_IDX]))
    else:
        for uid in feasible_uids:
            c_solution[uid] = 1.0
            pt = pool[uid]
            for k in range(len(pt.uses) - 1):
                e_solution[(uid, k)] = 0.0
        peak_bytes = int(
            constant_floor + sum(pool[u].size_bytes for u in feasible_uids)
        )

    diagnostics = {
        "pool_size": len(pool),
        "forced_cold_count": len(forced_cold),
        "forced_cold_bytes": forced_cold_bytes,
        "feasible_var_count": nv,
        "e_var_count": n_e,
        "n_samples": len(sample_positions),
        "n_deadline_positions": len(sorted_deadline_positions),
        "u_total_ns": int(u_total_ns),
        "makespan_target_ns": (
            int(makespan_target_ns) if makespan_target_ns is not None else None
        ),
        "lp_makespan_ns": int(makespan_ns),
        "solver_success": bool(res_success),
        "solver_status": res_message,
        "fell_back_to_lp": bool(fell_back),
        "lp_relaxation": bool(lp_relaxation),
    }

    return _LPResult(
        c_solution=c_solution,
        e_solution=e_solution,
        forced_cold=forced_cold,
        feasible_uids=feasible_uids,
        peak_bytes=int(peak_bytes),
        makespan_ns=int(makespan_ns),
        u_total_ns=int(u_total_ns),
        solver_status=res_message,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------


def _emit_neutral(
    pool: dict[int, _PoolTensor],
    result: _LPResult,
    tl: UnifiedTimeline,
    hw: HwParams,
    g_cum_start_at_task: list[int],
) -> NeutralSchedule:
    KEEP_THRESHOLD = 0.5

    def _pick_issuer_task(
        consumer_graph_id: int, consumer_start_ns: int, tau_h2d_ns: int,
        earliest_allowed_ns: int = -1,
    ) -> int:
        target = consumer_start_ns - tau_h2d_ns
        best_ns = -1
        best_pos = -1
        for ts, pos in per_graph_tasks.get(consumer_graph_id, ()):
            if ts >= consumer_start_ns:
                break
            if ts <= earliest_allowed_ns:
                continue
            if ts <= target and ts > best_ns:
                best_ns = ts
                best_pos = pos
        return best_pos

    neutral_tensors: list[NeutralTensor] = []
    uid_to_neutral_uid: dict[int, int] = {}
    for uid in sorted(pool.keys()):
        pt = pool[uid]
        n_uid = len(neutral_tensors)
        uid_to_neutral_uid[uid] = n_uid
        neutral_tensors.append(NeutralTensor(
            uid=n_uid,
            graph_id=int(pt.primary_graph_id),
            compiled_tensor_id=int(uid),
            graph_input_name=pt.name or f"uid_{uid}",
            size_bytes=int(pt.size_bytes),
            dtype=pt.dtype,
            used_by_launch_ids=sorted(set(pt.use_launch_ids)),
            shape=[],
            graph_input_idx=None,
            storage_group_id=pt.storage_group_id,
            trace_tids=list(pt.trace_tids),
        ))

    prefetches: list[NeutralPrefetch] = []
    evicts: list[NeutralEvict] = []
    cold_starts: list[NeutralColdStart] = []

    for uid, pt in pool.items():
        n_uid = uid_to_neutral_uid[uid]
        cv = result.c_solution.get(uid, None)
        is_forced = uid in result.forced_cold or cv is None
        is_cold = is_forced or float(cv) >= KEEP_THRESHOLD

        if is_cold:
            cold_starts.append(NeutralColdStart(
                tensor_uid=n_uid,
                anchor_launch_id=(
                    max(0, int(pt.use_launch_ids[0]))
                    if pt.use_launch_ids else 0
                ),
                reason=(
                    "peak_hybrid_unified_forced_cold" if is_forced
                    else "peak_hybrid_unified_cold"
                ),
                cgsim_tids=list(pt.trace_tids),
            ))
        else:
            first_pos, first_start, _first_end = pt.uses[0]
            first_gid = pt.use_graph_ids[0]
            first_lid = pt.use_launch_ids[0]
            first_nid = pt.use_node_ids[0]
            issue_pos = _pick_issuer_task(first_gid, first_start, pt.tau_h2d_ns)
            if issue_pos < 0:
                issue_pos = first_pos
            issue_task = tl.tasks[issue_pos]
            prefetches.append(NeutralPrefetch(
                tensor_uid=n_uid,
                issue_launch_id=max(0, int(issue_task.launch_id)),
                wait_launch_id=max(0, int(first_lid)),
                transfer_start_ns=int(max(0, first_start - pt.tau_h2d_ns)),
                transfer_end_ns=int(first_start),
                reason="peak_hybrid_unified_initial",
                issue_node_id=int(issue_task.node_id),
                wait_node_id=int(first_nid),
                cgsim_tid=int(pt.trace_tids[0]) if pt.trace_tids else -1,
                trusted_async=(issue_pos != first_pos),
                issue_graph_id=(
                    -1 if int(issue_task.graph_id) == int(first_gid)
                    else int(issue_task.graph_id)
                ),
                iter_mask=[],
            ))

        for k in range(len(pt.uses) - 1):
            if not pt.gap_feasibility[k]:
                continue
            ev = result.e_solution.get((uid, k), 0.0)
            if float(ev) < KEEP_THRESHOLD:
                continue
            k_pos, _k_start, k_end = pt.uses[k]
            kp1_pos, kp1_start, _kp1_end = pt.uses[k + 1]
            k_task = tl.tasks[k_pos]
            kp1_gid = pt.use_graph_ids[k + 1]
            kp1_lid = pt.use_launch_ids[k + 1]
            kp1_nid = pt.use_node_ids[k + 1]
            evicts.append(NeutralEvict(
                tensor_uid=n_uid,
                issue_launch_id=max(0, int(k_task.launch_id)),
                transfer_start_ns=int(k_end),
                transfer_end_ns=int(k_end + pt.tau_d2h_ns),
                reason=(
                    "peak_hybrid_unified_hybrid_evict" if is_cold
                    else "peak_hybrid_unified_gap_evict"
                ),
                issue_node_id=int(k_task.node_id),
                iter_mask=[],
                cgsim_tid=int(pt.trace_tids[0]) if pt.trace_tids else -1,
            ))
            re_pos = _pick_issuer_task(
                kp1_gid, kp1_start, pt.tau_h2d_ns,
                earliest_allowed_ns=k_end,
            )
            if re_pos < 0:
                re_pos = kp1_pos
            re_task = tl.tasks[re_pos]
            prefetches.append(NeutralPrefetch(
                tensor_uid=n_uid,
                issue_launch_id=max(0, int(re_task.launch_id)),
                wait_launch_id=max(0, int(kp1_lid)),
                transfer_start_ns=int(max(k_end + 1, kp1_start - pt.tau_h2d_ns)),
                transfer_end_ns=int(kp1_start),
                reason=(
                    "peak_hybrid_unified_hybrid_refetch" if is_cold
                    else "peak_hybrid_unified_gap_refetch"
                ),
                issue_node_id=int(re_task.node_id),
                wait_node_id=int(kp1_nid),
                cgsim_tid=int(pt.trace_tids[0]) if pt.trace_tids else -1,
                trusted_async=(re_pos != kp1_pos),
                issue_graph_id=(
                    -1 if int(re_task.graph_id) == int(kp1_gid)
                    else int(re_task.graph_id)
                ),
                iter_mask=[],
            ))

    graph_ids_seen = sorted({
        int(g) for pt in pool.values() for g in pt.use_graph_ids if g >= 0
    })
    return NeutralSchedule(
        graph_order=graph_ids_seen,
        compilation_hashes={int(g): "" for g in graph_ids_seen},
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
    sidecars: MultiGraphSidecars,
    hw: HwParams,
    makespan_target_s: float | None = None,
    max_peak_samples: int = 256,
    time_limit_s: float | None = 240.0,
    lp_relaxation: bool = False,
    audit: bool = False,
    **_legacy_kwargs: Any,
) -> NeutralSchedule:
    """Build a unified-timeline-native MILP schedule."""
    if not sidecars.launch_maps:
        raise RuntimeError(
            "[ct_milp_peak_hybrid_unified] no compile sidecars in bundle"
        )
    tl = build_unified_timeline(
        trace, sidecars, cpu_per_launch_ns=hw.cpu_per_launch_ns,
    )
    # The "compacted-all-gpu" cumulative time at each compiled-launch
    # task position. Sums *every* gpu node (kernel + aux) in trace
    # order, so the budget at each consumer reflects sim's actual
    # back-to-back queue capacity — not just the compiled-kernel
    # time. This is the budget axis the LP uses; tl.tasks itself is
    # used only for STRUCTURE (consumer identity, issuer search).
    g_cum_start_at_task, g_cum_end_at_task, u_total_ns = (
        _build_full_gcum_at_task(trace, tl)
    )

    makespan_target_ns: int | None = None
    if makespan_target_s is not None:
        makespan_target_ns = int(round(float(makespan_target_s) * 1e9))
        if makespan_target_ns < u_total_ns:
            raise RuntimeError(
                f"[ct_milp_peak_hybrid_unified] makespan_target_s="
                f"{makespan_target_s:.3f}s ({makespan_target_ns/1e6:.1f}ms) "
                f"is below the compacted-gpu floor U_total="
                f"{u_total_ns/1e6:.1f}ms."
            )

    pool = _build_pool(tl, hw, g_cum_start_at_task, g_cum_end_at_task)
    if not pool:
        raise RuntimeError(
            "[ct_milp_peak_hybrid_unified] pool is empty — no "
            "unified-timeline tensors with valid storage / trace tids."
        )

    pool_trace_tids = set()
    for pt in pool.values():
        pool_trace_tids.update(pt.trace_tids)
    extra_static_bytes = 0
    for tid, t in trace.tensor_map.items():
        if int(t.size_bytes) <= 0:
            continue
        if (t.args or {}).get("tensor_type") not in ("WEIGHT", "LEAF", "INPUT"):
            continue
        if not str((t.args or {}).get("device", "")).lower().startswith("cuda"):
            continue
        if int(tid) in pool_trace_tids:
            continue
        extra_static_bytes += int(t.size_bytes)
    if audit and extra_static_bytes:
        print(
            f"[ct_milp_peak_hybrid_unified:audit] extras static = "
            f"{extra_static_bytes/1e6:.1f}MB"
        )

    result = _solve_milp(
        pool, tl, hw,
        g_cum_start_at_task=g_cum_start_at_task,
        extra_static_bytes=extra_static_bytes,
        u_total_ns=u_total_ns,
        makespan_target_ns=makespan_target_ns,
        max_peak_samples=max_peak_samples,
        time_limit_s=time_limit_s,
        lp_relaxation=lp_relaxation,
        audit=audit,
    )

    neutral = _emit_neutral(pool, result, tl, hw, g_cum_start_at_task)

    n_cold = len(neutral.cold_starts)
    n_pf = len(neutral.prefetches)
    n_ev = len(neutral.evicts)
    streamed_bytes = sum(
        pool[u].size_bytes for u in result.feasible_uids
        if result.c_solution.get(u, 1.0) < 0.5
    )
    cold_bytes = sum(
        pool[u].size_bytes for u in pool
        if result.c_solution.get(u, 1.0) >= 0.5 or u in result.forced_cold
    )
    pcie_h2d_bytes = streamed_bytes + sum(
        pool[u].size_bytes
        for (u, k), v in result.e_solution.items() if v >= 0.5
    )

    neutral.meta = {
        "io_model": "ct_milp_peak_hybrid_unified",
        "graph_order": neutral.graph_order,
        "milp_peak_mb": round(result.peak_bytes / 1e6, 2),
        "pcie_used_mb": round(pcie_h2d_bytes / 1e6, 2),
        "cold_bytes_mb": round(cold_bytes / 1e6, 2),
        "streamed_bytes_mb": round(streamed_bytes / 1e6, 2),
        "extras_static_mb": round(extra_static_bytes / 1e6, 2),
        "u_total_ms": round(result.u_total_ns / 1e6, 2),
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
    print(
        f"Variant: {neutral.meta.get('io_model')} "
        f"| peak: {neutral.meta.get('milp_peak_mb')} MB "
        f"| makespan: {neutral.meta.get('lp_makespan_ms')} ms "
        f"(U_total={neutral.meta.get('u_total_ms')} ms"
        f", target={neutral.meta.get('makespan_target_ms')} ms) "
        f"| PCIe used: {neutral.meta.get('pcie_used_mb')} MB "
        f"| cold: {neutral.meta.get('cold_bytes_mb')} MB "
        f"| streamed: {neutral.meta.get('streamed_bytes_mb')} MB "
        f"| prefetches: {neutral.meta.get('n_prefetches')} "
        f"| evicts: {neutral.meta.get('n_evicts')} "
        f"| cold_start: {neutral.meta.get('n_cold_starts')}"
    )
