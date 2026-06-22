"""Parametric-knapsack residency scheduler — MILP-free Path 1.

Macro decision: which tensors stay *resident* (cold) vs are *streamed* per use,
under a VRAM peak budget. We rank items by benefit density

    rho_i = (H2D bytes saved per period by keeping i resident) / size_i
          = uses_i                              # weights used uses_i times/period

and keep the highest-density items that fit the budget (continuous-knapsack /
Lagrangian rule: resident iff rho_i >= lambda for the price lambda that exhausts
the budget). Two structural consequences we exploit:

  * Monotone-by-construction. As the budget shrinks (e.g. KV cache grows over a
    long-context decode), lambda only rises and items leave the resident set in
    increasing-density order and never return — a nested family. So the
    budget->residency map is a step function with a few breakpoints; we never
    re-solve per budget (see ``parametric_curve``).
  * In the bandwidth-bound regime (tight cap, single H2D channel) minimising
    streamed volume == minimising makespan, and for uniform-density weights the
    knapsack reduces to "pack the most resident bytes into the budget". This is
    why a greedy selection ties the order-axis MILP here (DESIGN.md).

KV age-bands drop in as additional items whose density is their attention
read-frequency (hot window ~ 1, evicted-old ~ 0); the same ranking then trades
weight residency against KV residency jointly under one budget.

I/O matches ``ct_milp_orderax.solve_neutral`` so it is a drop-in for the
validation harness. No solver, no channel model: timing is delegated to the
order-driven executor, exactly as for the MILP's shipped seed plan.
"""

from __future__ import annotations

import os
from typing import Any

from graph_modifiers.schedulers.ct_milp_overlap.scheduler import (
    _build_intermediate_residencies,
    _build_pool,
    _emit_neutral,
    _load_baseline_sim_times,
    _LPResult,
    print_summary,  # re-export for harnesses                       # noqa: F401
)
from graph_modifiers.common import HwParams, NeutralSchedule
from sim.core.trace import Trace


def _paced_budget_bytes() -> int:
    """In-flight (claimed-but-unconsumed) pool the executor paces to; carried
    as a constant peak floor, same constant the orderax model uses."""
    return int(float(os.environ.get("DAV_PACED_PREFETCH_MB", "130")) * 1e6)


def _max_intermediate_overlay(intermediates: list[tuple[int, int, int]]) -> int:
    """Peak concurrent intermediate (activation/KV) bytes, by a sweep over the
    (start, end, size) overlay. A constant floor — conservative but honest;
    the knapsack has no time axis to credit their decay."""
    if not intermediates:
        return 0
    deltas: list[tuple[int, int]] = []
    for s, e, sz in intermediates:
        deltas.append((int(s), int(sz)))
        deltas.append((int(e), -int(sz)))
    deltas.sort()
    cur = peak = 0
    for _t, d in deltas:
        cur += d
        peak = max(peak, cur)
    return peak


def _density(pt) -> float:
    """Benefit density rho_i. uses = number of consumer events => H2D reloads
    avoided per period by residency, per byte. Generalises to KV bands via
    read-frequency once those items exist in the pool."""
    return float(len(pt.consumers))


def select_resident(
    pool: dict[int, Any],
    *,
    budget_bytes: float,
    relax_cinfeasible: bool,
    paced_pool_bytes: int,
) -> tuple[set[int], set[int]]:
    """Parametric-knapsack resident-set selection.

    Returns (resident, forced_cold). ``forced_cold`` are tensors that *cannot*
    be streamed (c-infeasible and too big to bound by the paced pool, or relax
    disabled) — pinned resident first, then the budget is filled greedily by
    density. Deterministic and monotone in ``budget_bytes``.
    """
    tids = [t for t, pt in pool.items() if pt.consumers]

    forced_cold = {
        t for t in tids
        if not pool[t].c_feasibility
        and (pool[t].size_bytes > paced_pool_bytes or not relax_cinfeasible)
    }

    resident = set(forced_cold)
    used = float(sum(pool[t].size_bytes for t in forced_cold))

    # Greedy by (density desc, size desc): keep the most-reloaded, largest
    # tensors resident. For uniform-density weights this maximises resident
    # bytes packed into the budget == minimises streamed volume.
    candidates = sorted(
        (t for t in tids if t not in forced_cold),
        key=lambda t: (_density(pool[t]), pool[t].size_bytes),
        reverse=True,
    )
    for t in candidates:
        sz = float(pool[t].size_bytes)
        if used + sz <= budget_bytes:
            resident.add(t)
            used += sz
    return resident, forced_cold


def _build_result(
    pool: dict[int, Any],
    resident: set[int],
    forced_cold: set[int],
    *,
    constant_floor_bytes: int,
) -> _LPResult:
    """Hand-build the _LPResult the emitter consumes. c=1 resident (cold), c=0
    streamed; streamed multi-use tensors are evicted in every inter-use gap."""
    c_solution: dict[int, float] = {}
    e_solution: dict[tuple[int, int], float] = {}
    for t, pt in pool.items():
        cold = t in resident
        c_solution[t] = 1.0 if cold else 0.0
        if not cold:
            for k in range(len(pt.consumers) - 1):
                e_solution[(t, k)] = 1.0  # evict between uses, refetch next
    resident_bytes = sum(pool[t].size_bytes for t in resident)
    peak_bytes = int(resident_bytes + constant_floor_bytes)
    return _LPResult(
        c_solution=c_solution,
        e_solution=e_solution,
        forced_cold=set(forced_cold),
        feasible_tids=list(pool.keys()),
        peak_bytes=peak_bytes,
        lateness_ns=0,
        peak_overrun_bytes=0,
        target_infeasible=False,
        solver_status="knapsack",
        diagnostics={},
    )


def parametric_curve(
    pool: dict[int, Any],
    *,
    relax_cinfeasible: bool = True,
    paced_pool_bytes: int | None = None,
) -> list[tuple[int, frozenset[int]]]:
    """The budget->resident-set breakpoint table (Path 1's headline object).

    Sweeps the Lagrangian price implicitly by adding items in density order and
    recording, at each cumulative-size breakpoint, the resident set. The result
    is a *nested* family: entry i's set is a superset of entry i+1's. There are
    at most |streamable| breakpoints, independent of decode length — so a
    growing-KV decode never re-solves; it walks down this curve as the budget
    shrinks. Returns [(budget_floor_bytes, resident_set), ...] descending.
    """
    B = paced_pool_bytes if paced_pool_bytes is not None else _paced_budget_bytes()
    tids = [t for t, pt in pool.items() if pt.consumers]
    forced_cold = {
        t for t in tids
        if not pool[t].c_feasibility
        and (pool[t].size_bytes > B or not relax_cinfeasible)
    }
    forced_bytes = sum(pool[t].size_bytes for t in forced_cold)
    order = sorted(
        (t for t in tids if t not in forced_cold),
        key=lambda t: (_density(pool[t]), pool[t].size_bytes),
        reverse=True,
    )
    curve: list[tuple[int, frozenset[int]]] = []
    resident = set(forced_cold)
    used = forced_bytes
    curve.append((int(used), frozenset(resident)))
    for t in order:
        resident.add(t)
        used += pool[t].size_bytes
        curve.append((int(used), frozenset(resident)))
    curve.reverse()  # descending budget: largest resident set needs most room
    return curve


def solve_neutral(
    trace: Trace,
    *,
    hw: HwParams,
    baseline_sim_result_path: str | None = None,
    peak_target_bytes: int | None = None,
    safety_margin_frac: float = 0.0,
    relax_cinfeasible: bool = True,
    intermediate_axis_fix: bool = True,
    lookahead_ns: int = 5_000_000,
    audit: bool = False,
    sidecars: Any = None,
    schedulable_tids: set[int] | None = None,
    **_legacy_kwargs: Any,
) -> NeutralSchedule:
    """Parametric-knapsack solve -> NeutralSchedule (orderax-compatible I/O).

    No MILP: the resident set is the budget-fitted density ranking; timing is
    left to the order-driven executor (same as the MILP's shipped seed plan).
    Ignores solver knobs (time_limit_s, max_peak_samples, ...) via _legacy.
    """
    # Same emit semantics as orderax: lifetime release, evict on chosen gaps,
    # pinned tags for the cold c-infeasible class.
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
            extra_static_bytes += int(sum(pt.size_bytes for pt in dropped.values()))
            if audit:
                print(f"[ct_knapsack] {len(dropped)} pool tensors "
                      f"({extra_static_bytes / 1e6:.0f}MB) not schedulable — "
                      f"pinned as static floor", flush=True)

    B = _paced_budget_bytes()
    interm_floor = _max_intermediate_overlay(intermediates)
    # In-flight headroom: a streamed weight is resident (in flight) at peak on
    # top of the paced pool, so reserve the largest streamable tensor — the
    # orderax floor's "one inverted claim" term. Without it the resident set
    # fills to the cap and the first streamed prefetch OOMs.
    max_streamed = max((pt.size_bytes for pt in pool.values()
                        if pt.size_bytes <= B), default=0)
    constant_floor = B + interm_floor + extra_static_bytes + max_streamed

    if peak_target_bytes is not None:
        budget = (float(peak_target_bytes) * (1.0 - float(safety_margin_frac))
                  - constant_floor)
    else:
        budget = float("inf")

    resident, forced_cold = select_resident(
        pool, budget_bytes=budget, relax_cinfeasible=relax_cinfeasible,
        paced_pool_bytes=B)

    result = _build_result(
        pool, resident, forced_cold, constant_floor_bytes=constant_floor)

    if audit:
        res_mb = sum(pool[t].size_bytes for t in resident) / 1e6
        tot_mb = sum(pt.size_bytes for pt in pool.values()) / 1e6
        print(f"[ct_knapsack] budget={budget / 1e6:.0f}MB "
              f"floor={constant_floor / 1e6:.0f}MB (B={B/1e6:.0f} "
              f"interm={interm_floor/1e6:.0f}) resident={res_mb:.0f}MB/"
              f"{tot_mb:.0f}MB ({len(resident)}/{len(pool)} tids) "
              f"forced_cold={len(forced_cold)} modeled_peak="
              f"{result.peak_bytes/1e6:.0f}MB", flush=True)

    neutral = _emit_neutral(
        pool, result, trace, hw, sim_times,
        lookahead_ns=lookahead_ns,
        cold_budget_bytes=(int(peak_target_bytes)
                           if peak_target_bytes is not None else None),
    )

    streamed_bytes = sum(pool[t].size_bytes for t in pool if t not in resident)
    cold_bytes = sum(pool[t].size_bytes for t in resident)
    neutral.meta = {
        "io_model": "ct_knapsack",
        "graph_order": neutral.graph_order,
        "milp_peak_mb": round(result.peak_bytes / 1e6, 2),
        "milp_lateness_ms": 0.0,
        "milp_lateness_ns": 0,
        "cold_bytes_mb": round(cold_bytes / 1e6, 2),
        "streamed_bytes_mb": round(streamed_bytes / 1e6, 2),
        "extras_static_mb": round(extra_static_bytes / 1e6, 2),
        "interm_floor_mb": round(interm_floor / 1e6, 2),
        "target_infeasible": False,
        "n_cold_starts": len(neutral.cold_starts),
        "n_prefetches": len(neutral.prefetches),
        "n_evicts": len(neutral.evicts),
        "diagnostics": {
            "pool_size": len(pool),
            "resident_count": len(resident),
            "forced_cold_count": len(forced_cold),
            "paced_pool_mb": B / 1e6,
        },
    }
    return neutral
