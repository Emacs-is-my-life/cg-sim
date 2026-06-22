"""Order-axis residency MILP — EXACT peak, NO Belady seed.

A measurement variant of ``ct_milp_orderax``. Two deliberate changes,
both aimed at isolating the *pure MILP* solution quality:

1. **Exact incremental residency recurrence** replaces the sampled peak
   rows. On the consumer-order axis residency is piecewise-constant with
   breakpoints only at consumer events, so we track a running VRAM level
   ``R_q`` at every distinct breakpoint and bound ``P >= R_q``. This is
   exact (no sampling, no inter-position overflow) and stays sparse: each
   tensor contributes O(1) + O(#evictable gaps) delta terms, so the whole
   chain carries O(M + n_e) nonzeros — the same complexity class as the
   binary e-vars that already dominate. Because the peak is exact by
   construction, the lazy peak-row rounds and the exact-sweep violation
   feedback of ``ct_milp_orderax`` are gone.

2. **No Belady seed / no seed fallback.** ``ct_milp_orderax`` warm-starts
   the MILP with a greedy-Belady incumbent and, if the LP plan is
   exact-infeasible, *ships* that incumbent. Here we hand the solver
   nothing but its own phase-1 LP relaxation (the intrinsic two-phase
   warm start), and whatever the MILP returns is what we emit. This lets
   us measure how good the formulation is on its own, with no external
   caching heuristic propping it up.

Everything else — pool build, c/e classes, the c-coupling rows, the
bucketed single-server EDF channel model, intermediates overlay, and the
NeutralSchedule emit — is inherited unchanged from ct_milp_overlap.

Pairs with the same sim-side faithful executor knobs:
  DAV_PACED_PREFETCH_MB=<B>   bound claimed-but-unconsumed bytes
  DAV_PF_WAIT_ON_FULL=1       claim-miss waits for planned evicts
"""

from __future__ import annotations

import bisect
import os
from collections import defaultdict
from typing import Any

import numpy as np

from graph_modifiers.schedulers.ct_milp_overlap.scheduler import (
    MODEL_SCALE,
    _LPResult,
    _build_intermediate_residencies,
    _build_pool,
    _emit_neutral,
    _load_baseline_sim_times,
    _solve_two_phase_highspy,
    print_summary,  # re-export for harnesses                     # noqa: F401
)
from graph_modifiers.common.hw import effective_h2d_bw
from graph_modifiers.common import HwParams, NeutralSchedule
from sim.core.trace import Trace

PEAK_SLACK_PENALTY = 1e6  # objective per MB of peak overrun (hard-ish)


def _paced_budget_bytes() -> int:
    return int(float(os.environ.get("DAV_PACED_PREFETCH_MB", "130")) * 1e6)


def order_axis(pool, tids):
    """Global consumer-event order. Returns (events, P_t) where
    events[p] = (start_ns, nid, tid, k) sorted by baseline start and
    P_t[tid] = ascending positions of the tid's consumers."""
    events: list[tuple[int, int, int, int]] = []
    for tid in tids:
        for k, (nid, s, _e) in enumerate(pool[tid].consumers):
            events.append((int(s), int(nid), tid, k))
    events.sort()
    pos_of = {(tid, k): p for p, (_s, _n, tid, k) in enumerate(events)}
    P_t = {
        tid: [pos_of[(tid, k)] for k in range(len(pool[tid].consumers))]
        for tid in tids
    }
    return events, P_t


def _solve_orderax_exact(
    pool,
    *,
    hw: HwParams,
    peak_target_bytes: int | None,
    extra_static_bytes: int,
    safety_margin_frac: float,
    max_peak_samples: int,
    time_limit_s: float | None,
    phase1_time_limit_s: float | None,
    solver_threads: int,
    relax_cinfeasible: bool,
    lookahead_ns: int,
    intermediates: list[tuple[int, int, int]],
    audit: bool,
) -> _LPResult:
    B_pool = _paced_budget_bytes()
    bw = max(float(effective_h2d_bw(hw)), 1e-9)          # bytes / ns

    # Release credit: the real PyTorch executor only realizes lifetime
    # release for STREAMED tensors; cold tensors stay resident forever.
    # cg-sim's DAV releases at retire natively, so legacy harnesses keep
    # the credit unless this is set.
    cold_no_release = os.environ.get("MILP_COLD_NO_RELEASE") == "1"

    tids = sorted(t for t, pt in pool.items() if pt.consumers)

    # ---- order axis: global consumer-event positions ----
    events, P_t = order_axis(pool, tids)
    M = len(events)
    event_start_ns = [e[0] for e in events]

    # ---- intermediates overlay, mapped time → position ----
    interm_pos: list[tuple[int, int, float]] = []   # (p_start, p_end, MB)
    for s_, e_, sz_ in intermediates:
        ps = bisect.bisect_left(event_start_ns, int(s_))
        pe = bisect.bisect_right(event_start_ns, int(e_))
        if pe > ps:
            interm_pos.append((ps, pe, sz_ / MODEL_SCALE))
    interm_pos.sort()

    # ---- classes ----
    # big (> pool budget B): refetch claims can't be bounded by the paced
    # pool → no e vars (load once, stay). If also c-infeasible: pinned cold.
    forced_cold: set[int] = set()
    for tid in tids:
        pt = pool[tid]
        if not pt.c_feasibility and (
                pt.size_bytes > B_pool or not relax_cinfeasible):
            forced_cold.add(tid)

    # ---- variables: c, e, P, channel C_b, L, s_P, then R_q (recurrence) ----
    c_idx: dict[int, int] = {}
    col = 0
    for tid in tids:
        c_idx[tid] = col
        col += 1
    nv = col
    e_idx: dict[tuple[int, int], int] = {}
    for tid in tids:
        pt = pool[tid]
        if pt.size_bytes > B_pool or tid in forced_cold:
            continue  # no refetches for the big / pinned class
        for k in range(len(pt.consumers) - 1):
            e_idx[(tid, k)] = col
            col += 1
    n_e = col - nv
    P_IDX = col; col += 1
    NB = max(16, min(512, int(max_peak_samples)))
    C_BASE = col; col += NB
    L_IDX = col; col += 1
    SP_IDX = col; col += 1

    # ---- constant floor (paced pool + one inverted claim + static) ----
    _max_streamed = max(
        (pool[t].size_bytes for t in tids
         if t not in forced_cold and pool[t].size_bytes <= B_pool),
        default=0,
    )
    constant_floor = (
        float(extra_static_bytes) + float(B_pool) + float(_max_streamed)
    ) / MODEL_SCALE

    # ---- EXACT incremental residency: build per-coordinate deltas ----
    # Per tid (positions ps = P_t[tid], size s in MB):
    #   constant base block  [ps[0], ps[-1]+1):  +s @ ps[0],  -s @ ps[-1]+1
    #   cold pre-use         [0, ps[0]):         +s·c @ 0,    -s·c @ ps[0]
    #   cold post-use (opt)  [ps[-1]+1, M+1):    +s·c @ ps[-1]+1, -s·c @ M+1
    #   evicted gap k        [ps[k]+1, ps[k+1]): -s·e @ ps[k]+1, +s·e @ ps[k+1]
    # (Verified delta-for-delta against the order-space _exact_sweep below.)
    END_ALL = M + 1
    dconst: dict[int, float] = defaultdict(float)
    dterm: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for tid in tids:
        s = pool[tid].size_bytes / MODEL_SCALE
        ps = P_t[tid]
        p0, pl = ps[0], ps[-1]
        ci = c_idx[tid]
        # streamed-lifetime base block (constant — present for c=0 and c=1)
        dconst[p0] += s
        dconst[pl + 1] -= s
        # cold pre-use rectangle s·c over [0, p0)
        if p0 > 0:
            dterm[0][ci] += s
            dterm[p0][ci] -= s
        # cold post-use rectangle s·c over [pl+1, M+1) (only when credited)
        if cold_no_release:
            dterm[pl + 1][ci] += s
            dterm[END_ALL][ci] -= s
        # evicted-gap rectangles −s·e over [ps[k]+1, ps[k+1])
        for k in range(len(ps) - 1):
            ei = e_idx.get((tid, k))
            if ei is None:
                continue
            gs, ge = ps[k] + 1, ps[k + 1]
            if ge <= gs:
                continue
            dterm[gs][ei] -= s
            dterm[ge][ei] += s
    for ps_, pe_, mb in interm_pos:
        dconst[ps_] += mb
        dconst[pe_] -= mb

    coords = sorted(set(dconst) | set(dterm))
    R_BASE = col; col += len(coords)
    total_vars = col

    # ---- bounds / integrality / objective ----
    bounds: list[tuple[float, float | None]] = []
    for tid in tids:
        bounds.append((1.0, 1.0) if tid in forced_cold else (0.0, 1.0))
    bounds.extend([(0.0, 1.0)] * n_e)
    bounds.append((0.0, None))                     # P
    bounds.extend([(0.0, None)] * NB)              # C_b
    bounds.append((0.0, None))                     # L
    bounds.append((0.0, None) if peak_target_bytes is not None
                  else (0.0, 0.0))                 # s_P
    bounds.extend([(0.0, None)] * len(coords))     # R_q

    integrality = np.zeros(total_vars, dtype=np.int64)
    integrality[: nv + n_e] = 1                    # only c, e are binary

    eps = 1.0 / bw                                  # ns per byte (physical)
    c_obj = np.zeros(total_vars, dtype=np.float64)
    c_obj[L_IDX] = 1.0
    if peak_target_bytes is not None:
        c_obj[SP_IDX] = PEAK_SLACK_PENALTY
    for tid in tids:
        c_obj[c_idx[tid]] = -(pool[tid].size_bytes / MODEL_SCALE) * eps
    for (tid, _k), ci in e_idx.items():
        c_obj[ci] = (pool[tid].size_bytes / MODEL_SCALE) * eps

    rows: list[int] = []
    cols_: list[int] = []
    vals: list[float] = []
    ub: list[float] = []
    row = 0

    # ---- coupling: c-feasible cold tids don't hybrid-refetch ----
    for (tid, k), ci in e_idx.items():
        if pool[tid].c_feasibility:
            rows.append(row); cols_.append(c_idx[tid]); vals.append(1.0)
            rows.append(row); cols_.append(ci); vals.append(1.0)
            ub.append(1.0)
            row += 1

    # ---- EXACT residency recurrence + peak bound ----
    # Chain (driven to equality by the downward pressure on P):
    #   R_q >= R_{q-1} + Δ_q     ⟺  R_{q-1} − R_q + Σ coef·var ≤ −const_q
    # Peak:
    #   P  >= R_q                ⟺  R_q − P ≤ 0
    # By induction R_q ≥ const_floor + (true cumulative level), so P ≥ the
    # true peak in EVERY feasible solution — the cap is honest, no sampling.
    for i, q in enumerate(coords):
        Rq = R_BASE + i
        rows.append(row); cols_.append(Rq); vals.append(-1.0)
        base = 0.0
        if i > 0:
            rows.append(row); cols_.append(R_BASE + i - 1); vals.append(1.0)
        else:
            base = constant_floor
        for ci, cf in dterm.get(q, {}).items():
            if abs(cf) >= 1e-12:
                rows.append(row); cols_.append(ci); vals.append(float(cf))
        ub.append(-float(dconst.get(q, 0.0)) - base)
        row += 1
        rows.append(row); cols_.append(Rq); vals.append(1.0)
        rows.append(row); cols_.append(P_IDX); vals.append(-1.0)
        ub.append(0.0)
        row += 1

    # ---- soft cap ----
    target_adj_mb = 0.0
    if peak_target_bytes is not None:
        target_adj_mb = max(
            0.0, float(peak_target_bytes) * (1.0 - float(safety_margin_frac))
        ) / MODEL_SCALE
        rows.append(row); cols_.append(P_IDX); vals.append(1.0)
        rows.append(row); cols_.append(SP_IDX); vals.append(-1.0)
        ub.append(max(0.0, target_adj_mb))
        row += 1

    # ---- channel rows (bucketed single-server EDF, time axis) ----
    t0 = min(pt.consumers[0][1] for pt in pool.values() if pt.consumers)
    t_end = max(pt.consumers[-1][2] for pt in pool.values() if pt.consumers)
    W_ns = max(1, int(lookahead_ns))
    bucket_w = max(1.0, (t_end - t0) / NB)
    b_const = [0.0] * NB
    b_terms: list[list[tuple[int, float]]] = [[] for _ in range(NB)]

    def _bucket(d_ns: float) -> int:
        return max(0, min(NB - 1, int((d_ns - t0) / bucket_w)))

    for tid in tids:
        pt = pool[tid]
        p_ms = pt.tau_h2d_ns / MODEL_SCALE
        b = _bucket(pt.consumers[0][1])
        b_const[b] += p_ms                          # p·(1−c) = p − p·c
        b_terms[b].append((c_idx[tid], -p_ms))
        for k in range(len(pt.consumers) - 1):
            ei = e_idx.get((tid, k))
            if ei is None:
                continue
            b_terms[_bucket(pt.consumers[k + 1][1])].append((ei, p_ms))

    t0_ms = t0 / MODEL_SCALE
    for b in range(NB):
        Cb = C_BASE + b
        left = t0 + b * bucket_w
        r_b = max(t0_ms, left / MODEL_SCALE - W_ns / MODEL_SCALE)
        T_b = (t0 + (b + 1) * bucket_w) / MODEL_SCALE
        # accumulation: C_b ≥ C_{b−1} + W_b
        rows.append(row); cols_.append(Cb); vals.append(-1.0)
        base = 0.0
        if b > 0:
            rows.append(row); cols_.append(C_BASE + b - 1); vals.append(1.0)
        else:
            base = t0_ms
        for ci, cf in b_terms[b]:
            rows.append(row); cols_.append(ci); vals.append(float(cf))
        ub.append(-float(b_const[b]) - base)
        row += 1
        # release floor: C_b ≥ r_b + W_b
        rows.append(row); cols_.append(Cb); vals.append(-1.0)
        for ci, cf in b_terms[b]:
            rows.append(row); cols_.append(ci); vals.append(float(cf))
        ub.append(-float(b_const[b]) - float(r_b))
        row += 1
        # lateness: C_b − L ≤ T_b
        rows.append(row); cols_.append(Cb); vals.append(1.0)
        rows.append(row); cols_.append(L_IDX); vals.append(-1.0)
        ub.append(float(T_b))
        row += 1

    # ---- exact position-space sweep (reporting only — no lazy feedback) ----
    def _exact_sweep(c_sol, e_sol) -> list[tuple[int, float]]:
        deltas: list[tuple[int, float]] = []
        for tid in tids:
            size = pool[tid].size_bytes / MODEL_SCALE
            ps = P_t[tid]
            cold = float(c_sol.get(tid, 0.0)) >= 0.5
            start = 0 if cold else ps[0]
            cur = start
            for k in range(len(ps) - 1):
                if float(e_sol.get((tid, k), 0.0)) >= 0.5:
                    gs, ge = ps[k] + 1, ps[k + 1]
                    if ge <= gs:
                        continue
                    if gs > cur:
                        deltas.append((cur, size))
                        deltas.append((gs, -size))
                    cur = max(cur, ge)
            end_p = M + 1 if (cold_no_release and cold) else ps[-1] + 1
            if end_p > cur:
                deltas.append((cur, size))
                deltas.append((end_p, -size))
        for ips, ipe, mb in interm_pos:
            deltas.append((ips, mb))
            deltas.append((ipe, -mb))
        deltas.sort(key=lambda x: x[0])
        out: list[tuple[int, float]] = []
        acc = 0.0
        i = 0
        nd = len(deltas)
        while i < nd:
            p0 = deltas[i][0]
            while i < nd and deltas[i][0] == p0:
                acc += deltas[i][1]
                i += 1
            out.append((p0, acc + constant_floor))
        return out

    def _exact_peak(c_sol, e_sol) -> float:
        sw = _exact_sweep(c_sol, e_sol)
        return max((v for _p, v in sw), default=constant_floor)

    # ---- solve: PURE MILP, no Belady incumbent (feasible_fallback=None) ----
    if audit:
        print(f"[ct_milp_orderax_exact] M={M} coords={len(coords)} "
              f"vars={total_vars} (c={nv} e={n_e} R={len(coords)}) "
              f"rows={row} forced_cold={len(forced_cold)} "
              f"floor={constant_floor:.0f}MB target={target_adj_mb:.0f}MB "
              f"NO-SEED (pure MILP)", flush=True)
    x, success, message, _status, lp_only = _solve_two_phase_highspy(
        total_vars=total_vars,
        c_obj=c_obj,
        bounds_list=bounds,
        rows=rows,
        cols=cols_,
        vals=vals,
        ub_list=ub,
        integrality_arr=integrality,
        time_limit_s=time_limit_s,
        solver_threads=solver_threads,
        audit=audit,
        phase1_time_limit_s=phase1_time_limit_s,
        feasible_fallback=None,          # pure MILP — no external seed
    )

    c_sol: dict[int, float] = {}
    e_sol: dict[tuple[int, int], float] = {}
    if success and x is not None:
        xa = np.asarray(x)
        for t in tids:
            c_sol[t] = float(xa[c_idx[t]])
        for key, ci in e_idx.items():
            e_sol[key] = float(xa[ci])
        lateness_ns = int(float(xa[L_IDX]) * MODEL_SCALE)
    else:
        # No seed to fall back on — emit the trivial stream-everything plan
        # (c=forced_cold only, no evicts) and report it honestly.
        for t in tids:
            c_sol[t] = 1.0 if t in forced_cold else 0.0
        for key in e_idx:
            e_sol[key] = 0.0
        lateness_ns = 0
    for t in tids:
        for k in range(len(pool[t].consumers) - 1):
            e_sol.setdefault((t, k), 0.0)

    peak_mb = _exact_peak(c_sol, e_sol)
    overrun_mb = (max(0.0, peak_mb - target_adj_mb)
                  if peak_target_bytes is not None else 0.0)
    target_infeasible = overrun_mb > 1.0

    diagnostics = {
        "pool_size": len(pool),
        "order_events": M,
        "recurrence_coords": len(coords),
        "forced_cold_count": len(forced_cold),
        "e_var_count": n_e,
        "total_vars": total_vars,
        "n_rows": row,
        "solver_success": bool(success),
        "solver_status": str(message),
        "fell_back_to_lp": bool(lp_only),
        "exact_peak": True,
        "seeded": False,
        "paced_pool_mb": B_pool / 1e6,
    }

    return _LPResult(
        c_solution=c_sol,
        e_solution=e_sol,
        forced_cold=set(forced_cold),
        feasible_tids=list(tids),
        peak_bytes=int(peak_mb * MODEL_SCALE),
        lateness_ns=int(lateness_ns),
        peak_overrun_bytes=int(overrun_mb * MODEL_SCALE),
        target_infeasible=target_infeasible,
        solver_status=str(message),
        diagnostics=diagnostics,
    )


def solve_neutral(
    trace: Trace,
    *,
    hw: HwParams,
    baseline_sim_result_path: str | None = None,
    peak_target_bytes: int | None = None,
    safety_margin_frac: float = 0.0,
    max_peak_samples: int = 256,
    time_limit_s: float | None = 150.0,
    phase1_time_limit_s: float | None = 45.0,
    solver_threads: int | None = None,
    relax_cinfeasible: bool = True,
    intermediate_axis_fix: bool = True,
    lookahead_ns: int = 5_000_000,
    audit: bool = False,
    sidecars: Any = None,
    schedulable_tids: set[int] | None = None,
    **_legacy_kwargs: Any,
) -> NeutralSchedule:
    """Exact-peak, no-seed order-axis solve → NeutralSchedule.

    Same I/O shape as ct_milp_orderax. A single MILP solve: the exact
    residency recurrence makes the peak honest by construction, so there
    are no lazy rounds and no Belady fallback. Whatever the solver returns
    is emitted — this measures the formulation's own solution quality.
    """
    # Emit semantics: lifetime release, evicts on every chosen gap, pinned
    # tags for the cold c-infeasible class.
    os.environ["MILP_GMODE"] = "1"
    os.environ["MILP_EVAR_ALL_GAPS"] = "1"
    os.environ["MILP_CINFEAS_INFLIGHT"] = "1"

    sim_times = (
        _load_baseline_sim_times(baseline_sim_result_path)
        if baseline_sim_result_path else None
    )
    pool = _build_pool(trace, hw, sim_times=sim_times)
    intermediates = _build_intermediate_residencies(
        trace, sim_times=sim_times, axis_fix=bool(intermediate_axis_fix))

    extra_static_bytes = 0
    if schedulable_tids is not None:
        dropped = {t: pt for t, pt in pool.items() if t not in schedulable_tids}
        if dropped:
            pool = {t: pt for t, pt in pool.items() if t in schedulable_tids}
            dropped_bytes = sum(pt.size_bytes for pt in dropped.values())
            extra_static_bytes += int(dropped_bytes)
            if audit:
                print(
                    f"[ct_milp_orderax_exact] {len(dropped)} pool tensors "
                    f"({dropped_bytes / 1e6:.0f}MB) not executor-schedulable "
                    f"— pinned as static floor",
                    flush=True,
                )
    if sidecars is not None and getattr(sidecars, "launch_maps", None):
        try:
            from graph_modifiers.common import build_unified_timeline
            tl = build_unified_timeline(
                trace, sidecars, cpu_per_launch_ns=hw.cpu_per_launch_ns)
            extra_static_bytes += sum(
                t.size_bytes for t in tl.tensors if not t.trace_tids)
        except Exception:
            pass

    threads = max(1, int(solver_threads or os.cpu_count() or 1))
    result = _solve_orderax_exact(
        pool,
        hw=hw,
        peak_target_bytes=peak_target_bytes,
        extra_static_bytes=extra_static_bytes,
        safety_margin_frac=safety_margin_frac,
        max_peak_samples=max_peak_samples,
        time_limit_s=time_limit_s,
        phase1_time_limit_s=phase1_time_limit_s,
        solver_threads=threads,
        relax_cinfeasible=relax_cinfeasible,
        lookahead_ns=lookahead_ns,
        intermediates=intermediates,
        audit=audit,
    )

    neutral = _emit_neutral(
        pool, result, trace, hw, sim_times,
        lookahead_ns=lookahead_ns,
        cold_budget_bytes=(int(peak_target_bytes)
                           if peak_target_bytes is not None else None),
    )

    streamed_bytes = sum(
        pool[t].size_bytes for t in result.feasible_tids
        if result.c_solution.get(t, 1.0) < 0.5
    )
    cold_bytes = sum(
        pool[t].size_bytes for t in pool
        if result.c_solution.get(t, 1.0) >= 0.5 or t in result.forced_cold
    )
    pcie = streamed_bytes + sum(
        pool[t].size_bytes
        for (t, _k), v in result.e_solution.items() if v >= 0.5
    )
    neutral.meta = {
        "io_model": "ct_milp_orderax_exact",
        "graph_order": neutral.graph_order,
        "milp_peak_mb": round(result.peak_bytes / 1e6, 2),
        "milp_lateness_ms": round(result.lateness_ns / 1e6, 3),
        "milp_lateness_ns": int(result.lateness_ns),
        "pcie_used_mb": round(pcie / 1e6, 2),
        "cold_bytes_mb": round(cold_bytes / 1e6, 2),
        "streamed_bytes_mb": round(streamed_bytes / 1e6, 2),
        "extras_static_mb": round(extra_static_bytes / 1e6, 2),
        "peak_overrun_mb": round(result.peak_overrun_bytes / 1e6, 2),
        "target_infeasible": result.target_infeasible,
        "n_cold_starts": len(neutral.cold_starts),
        "n_prefetches": len(neutral.prefetches),
        "n_evicts": len(neutral.evicts),
        "diagnostics": result.diagnostics,
    }
    return neutral
