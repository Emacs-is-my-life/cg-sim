"""Order-axis residency MILP weight-streaming scheduler.

Residency windows live on the consumer-ORDER axis (stretch-invariant,
matches the order-driven executor by construction); time enters only the
channel-lateness model. See DESIGN.md for the measured rationale.

Pairs with the sim-side faithful executor knobs:
  DAV_PACED_PREFETCH_MB=<B>   bound claimed-but-unconsumed bytes
  DAV_PF_WAIT_ON_FULL=1       claim-miss waits for planned evicts
"""

from __future__ import annotations

import bisect
import os
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


def estimate_iterations(P_t, tids, forced_cold) -> int:
    """Median consumer count over multi-use streamable tids ≈ the number
    of decode iterations (each weight is used ~once per iteration)."""
    lens = sorted(
        len(P_t[t]) for t in tids
        if t not in forced_cold and len(P_t[t]) > 2
    )
    return max(1, lens[len(lens) // 2] if lens else 1)


def build_seed_walk(pool, tids, events, P_t, e_keys, forced_cold,
                    budget: float, policy: str = "belady"):
    """Seed plan at cold budget `budget`. Returns (g, eset, drop_n).

    policy="belady": sliding farthest-next-use eviction (order-exact).
      Measured flaw: evicts intra-iteration use-windows too, refetching
      some weights more than once per iteration (~13.9GB/iter vs the
      (total−budget)/iter ≈ 10.3 optimum on llama8b@6).
    policy="static": fixed partition — cold set stays resident the whole
      run; every streamed tid is evicted in each INTER-iteration gap
      (position-gap > θ ≈ half an iteration) and stays resident across
      intra-iteration gaps. For a cyclic access pattern this achieves the
      per-iteration volume optimum exactly (it is what SwapAdvisorRuntime
      realizes at steady state).
    """
    M = len(events)
    g = {t: 1.0 if t in forced_cold else 0.0 for t in tids}
    order = sorted(
        (t for t in tids if t not in forced_cold),
        key=lambda t: (pool[t].c_feasibility, P_t[t][0]),
    )
    rb = float(sum(pool[t].size_bytes for t in forced_cold))
    for t in order:
        sz = float(pool[t].size_bytes)
        if rb + sz <= budget:
            g[t] = 1.0
            rb += sz
    eset: set[tuple[int, int]] = set()
    drop_n = 0

    if policy == "static":
        n_iter = estimate_iterations(P_t, tids, forced_cold)
        theta = max(2, M // max(1, n_iter) // 2)
        for t in tids:
            if g[t] >= 0.5 or t in forced_cold:
                continue
            ps = P_t[t]
            for k in range(len(ps) - 1):
                if ps[k + 1] - ps[k] > theta:
                    if (t, k) in e_keys:
                        eset.add((t, k))
                    else:
                        drop_n += 1
        return g, eset, drop_n

    # belady (sliding farthest-next-use)
    resident = {t for t in tids if g[t] >= 0.5}
    rb = float(sum(pool[t].size_bytes for t in resident))
    last_k = {t: -1 for t in tids}
    for _p, (_s, _n, t_, k_) in enumerate(events):
        if t_ not in resident:
            sz = float(pool[t_].size_bytes)
            while rb + sz > budget and resident:
                cands = [u for u in resident if u not in forced_cold]
                if not cands:
                    break
                victim = max(
                    cands,
                    key=lambda u: (
                        P_t[u][last_k[u] + 1]
                        if last_k[u] + 1 < len(P_t[u]) else M + 1
                    ),
                )
                vk = last_k[victim]
                if vk == -1:
                    g[victim] = 0.0
                elif (victim, vk) in e_keys:
                    eset.add((victim, vk))
                else:
                    drop_n += 1
                resident.discard(victim)
                rb -= float(pool[victim].size_bytes)
            resident.add(t_)
            rb += sz
        last_k[t_] = k_
    return g, eset, drop_n


def seed_volume_bytes(pool, g, eset) -> int:
    """H2D bytes the seed plan implies: init loads of streamed tids +
    one refetch per chosen eviction."""
    init = sum(pool[t].size_bytes for t, v in g.items() if v < 0.5)
    ref = sum(pool[t].size_bytes for (t, _k) in eset)
    return int(init + ref)


def _solve_orderax(
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
    extra_positions: list[int] | None = None,
) -> _LPResult:
    B_pool = _paced_budget_bytes()
    bw = max(float(effective_h2d_bw(hw)), 1e-9)          # bytes / ns

    # Executor-matched release semantics: the real PyTorch executor only
    # realizes lifetime release for STREAMED tensors (tail evict + cyclic
    # reload); cold tensors stay resident forever. Crediting their release
    # makes the modeled peak unrealizable (~1.6GB of dead TE weights on
    # SDXL). cg-sim's DAV releases at retire natively, so legacy harnesses
    # keep the credit unless this is set.
    cold_no_release = os.environ.get("MILP_COLD_NO_RELEASE") == "1"

    tids = sorted(t for t, pt in pool.items() if pt.consumers)

    # ---- order axis: global consumer-event positions ----
    events, P_t = order_axis(pool, tids)
    M = len(events)
    event_start_ns = [e[0] for e in events]

    # ---- classes ----
    # big (> pool budget B): refetch claims can't be bounded by the paced
    # pool → no e vars (load once, stay). If also c-infeasible: pinned
    # cold (the embedding class — Belady's choice).
    forced_cold: set[int] = set()
    for tid in tids:
        pt = pool[tid]
        if not pt.c_feasibility and (
                pt.size_bytes > B_pool or not relax_cinfeasible):
            forced_cold.add(tid)

    # ---- variables ----
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
    total_vars = col

    bounds: list[tuple[float, float | None]] = []
    for tid in tids:
        bounds.append((1.0, 1.0) if tid in forced_cold else (0.0, 1.0))
    bounds.extend([(0.0, 1.0)] * n_e)
    bounds.append((0.0, None))                     # P
    bounds.extend([(0.0, None)] * NB)              # C_b
    bounds.append((0.0, None))                     # L
    bounds.append((0.0, None) if peak_target_bytes is not None
                  else (0.0, 0.0))                 # s_P

    integrality = np.zeros(total_vars, dtype=np.int64)
    integrality[: nv + n_e] = 1

    # ---- objective: L + Λ·s_P + (1/bw)·streamed volume ----
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

    # ---- intermediates overlay, mapped time → position ----
    interm_pos: list[tuple[int, int, float]] = []   # (p_start, p_end, MB)
    for s_, e_, sz_ in intermediates:
        ps = bisect.bisect_left(event_start_ns, int(s_))
        pe = bisect.bisect_right(event_start_ns, int(e_))
        if pe > ps:
            interm_pos.append((ps, pe, sz_ / MODEL_SCALE))
    interm_pos.sort()

    def _interm_at(p: int) -> float:
        v = 0.0
        for ps, pe, mb in interm_pos:
            if ps > p:
                break
            if p < pe:
                v += mb
        return v

    # Floor = static extras + the paced in-flight pool B + ONE inverted
    # claim: sync-fallback arrivals (consumer-before-arrival repairs)
    # enter the FIFO out of order, so a later-consumer tid can hold a
    # claim while the front's demand claim lands — bounded by the largest
    # streamed tensor (the >B class is pinned cold, so ≤ B).
    _max_streamed = max(
        (pool[t].size_bytes for t in tids
         if t not in forced_cold and pool[t].size_bytes <= B_pool),
        default=0,
    )
    constant_floor = (
        float(extra_static_bytes) + float(B_pool) + float(_max_streamed)
    ) / MODEL_SCALE

    # ---- peak rows at sampled positions ----
    sample_pos: list[int] = []
    if M:
        step = max(1, M // max(1, int(max_peak_samples)))
        sample_pos = list(range(0, M, step))
        if sample_pos[-1] != M - 1:
            sample_pos.append(M - 1)
    if extra_positions:
        have = set(sample_pos)
        for p in extra_positions:
            if 0 <= int(p) < M and int(p) not in have:
                sample_pos.append(int(p)); have.add(int(p))
        sample_pos.sort()

    for p in sample_pos:
        const = constant_floor + _interm_at(p)
        coefs: dict[int, float] = {}
        for tid in tids:
            size = pool[tid].size_bytes / MODEL_SCALE
            ps = P_t[tid]
            if p > ps[-1]:
                if cold_no_release:
                    # released only if streamed: size·c stays after last use
                    coefs[c_idx[tid]] = coefs.get(c_idx[tid], 0.0) + size
                continue                            # lifetime-released
            if p < ps[0]:
                coefs[c_idx[tid]] = coefs.get(c_idx[tid], 0.0) + size
                continue                            # cold pre-use: size·c
            j = bisect.bisect_left(ps, p)
            if j < len(ps) and ps[j] == p:
                const += size                       # consumed at p
                continue
            k = j - 1                               # in gap (k, k+1)
            ei = e_idx.get((tid, k))
            if ei is None:
                const += size                       # un-evictable gap
            else:
                const += size                       # size·(1 − e)
                coefs[ei] = coefs.get(ei, 0.0) - size
        rows.append(row); cols_.append(P_IDX); vals.append(-1.0)
        for ci, cf in coefs.items():
            if abs(cf) >= 1e-9:
                rows.append(row); cols_.append(ci); vals.append(float(cf))
        ub.append(-float(const))
        row += 1

    # ---- soft cap ----
    # MILP_LP_PEAK_BUFFER_MB: tighten ONLY the LP's cap by a small buffer
    # so the exact position-sweep (checked against the TRUE target) passes
    # despite few-MB peaks at unsampled positions. Without it, tiny
    # oscillating overshoots (4-40MB at sdxl@4) defeat the lazy rounds and
    # force the Belady seed fallback. Same discipline as seed autocal.
    target_adj_mb = 0.0
    if peak_target_bytes is not None:
        target_adj_mb = max(
            0.0, float(peak_target_bytes) * (1.0 - float(safety_margin_frac))
        ) / MODEL_SCALE
        lp_buf_mb = max(
            0.0, float(os.environ.get("MILP_LP_PEAK_BUFFER_MB", "0")))
        rows.append(row); cols_.append(P_IDX); vals.append(1.0)
        rows.append(row); cols_.append(SP_IDX); vals.append(-1.0)
        ub.append(max(0.0, target_adj_mb - lp_buf_mb))
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

    # ---- exact position-space sweep for any (c, e) plan ----
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

    def _exact_violations(c_sol, e_sol, thr_mb, top_k=None) -> list[int]:
        if top_k is None:
            top_k = int(os.environ.get("MILP_LAZY_K", "512"))
        over = [(v, p) for p, v in _exact_sweep(c_sol, e_sol)
                if v > thr_mb + 1.0]
        if not over:
            return []
        over.sort(reverse=True)
        keep = {p for _v, p in over[:64]}
        rest = [p for _v, p in over[64:]]
        n_more = max(0, top_k - len(keep))
        if rest and n_more:
            step = max(1.0, len(rest) / float(n_more))
            i = 0.0
            while int(i) < len(rest) and len(keep) < top_k:
                keep.add(rest[int(i)])
                i += step
        return sorted(keep)

    # ---- seed walk (policy-selectable) + autocal ----
    seed_c: dict[int, float] = {t: 0.0 for t in tids}
    seed_e: dict[tuple[int, int], float] = {}
    cap_b = (float(peak_target_bytes) * (1.0 - float(safety_margin_frac))
             if peak_target_bytes is not None else float("inf"))

    def _autocal(policy: str):
        """Shrink the cold budget until the policy's seed fits the exact
        order grid. Returns (g, eset, ok, volume_bytes) or None."""
        budget = cap_b
        for it in range(6):
            g, eset, drop_n = build_seed_walk(
                pool, tids, events, P_t, e_idx, forced_cold,
                budget, policy=policy)
            e_full = {key: (1.0 if key in eset else 0.0) for key in e_idx}
            pk = _exact_peak(g, e_full)
            if audit:
                print(f"[ct_milp_orderax:seed/{policy}] iter {it}: "
                      f"budget={budget/1e6:.0f}MB exact_peak={pk:.0f}MB "
                      f"target={target_adj_mb:.0f}MB "
                      f"vol={seed_volume_bytes(pool, g, eset)/1e9:.1f}GB "
                      f"dropped={drop_n}", flush=True)
            if pk <= target_adj_mb + 1.0:
                return g, eset, True, seed_volume_bytes(pool, g, eset)
            budget -= (pk - target_adj_mb) * MODEL_SCALE
        return g, eset, False, seed_volume_bytes(pool, g, eset)

    seed_ok = False
    if peak_target_bytes is not None:
        policy_env = os.environ.get("MILP_SEED_POLICY", "best")
        cands = (["belady", "static"] if policy_env == "best"
                 else [policy_env])
        best = None
        for pol in cands:
            g, eset, ok, vol = _autocal(pol)
            if best is None or (ok, -vol) > (best[3], -best[4]):
                best = (pol, g, eset, ok, vol)
        pol, seed_c, eset, seed_ok, _vol = best
        seed_e = {key: (1.0 if key in eset else 0.0) for key in e_idx}
        if audit:
            print(f"[ct_milp_orderax:seed] selected policy={pol} "
                  f"exact_feasible={seed_ok} "
                  f"volume={_vol/1e9:.1f}GB", flush=True)

    warm = np.zeros(total_vars, dtype=np.float64)
    for t in tids:
        warm[c_idx[t]] = seed_c.get(t, 0.0)
    for key, ci in e_idx.items():
        warm[ci] = seed_e.get(key, 0.0)
    seed_pk_mb = _exact_peak(seed_c, seed_e)
    warm[P_IDX] = seed_pk_mb
    if peak_target_bytes is not None:
        warm[SP_IDX] = max(0.0, seed_pk_mb - target_adj_mb)
    seed_C = t0_ms
    seed_L = 0.0
    for b in range(NB):
        Wb = b_const[b] + sum(cf * float(warm[ci]) for ci, cf in b_terms[b])
        Wb = max(0.0, Wb)
        left = t0 + b * bucket_w
        r_b = max(t0_ms, left / MODEL_SCALE - W_ns / MODEL_SCALE)
        T_b = (t0 + (b + 1) * bucket_w) / MODEL_SCALE
        seed_C = max(seed_C, r_b) + Wb
        warm[C_BASE + b] = seed_C
        seed_L = max(seed_L, seed_C - T_b)
    warm[L_IDX] = seed_L

    # ---- solve ----
    # MILP_NO_SEED=1: ablation — solve without the Belady warm-start
    # incumbent (phase-2 falls back to the rounded LP seed). Combine with
    # MILP_SEED_FALLBACK=0 to also disable the ship-time seed fallback.
    no_seed = os.environ.get("MILP_NO_SEED") == "1"
    if no_seed and audit:
        print("[ct_milp_orderax] NO-SEED ablation: Belady incumbent "
              "withheld from the solver", flush=True)
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
        feasible_fallback=None if no_seed else warm,
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
        c_sol = dict(seed_c)
        e_sol = dict(seed_e)
        lateness_ns = int(seed_L * MODEL_SCALE)
    # fill non-var gaps as e=0 for emit
    for t in tids:
        for k in range(len(pool[t].consumers) - 1):
            e_sol.setdefault((t, k), 0.0)

    peak_mb = _exact_peak(c_sol, e_sol)
    violations: list[int] = []
    if peak_target_bytes is not None:
        violations = _exact_violations(c_sol, e_sol, target_adj_mb)
    target_infeasible = bool(violations)

    diagnostics = {
        "pool_size": len(pool),
        "order_events": M,
        "forced_cold_count": len(forced_cold),
        "e_var_count": n_e,
        "n_samples": len(sample_pos),
        "solver_success": bool(success),
        "solver_status": str(message),
        "fell_back_to_lp": bool(lp_only),
        "seed_exact_feasible": bool(seed_ok),
        "paced_pool_mb": B_pool / 1e6,
    }

    res = _LPResult(
        c_solution=c_sol,
        e_solution=e_sol,
        forced_cold=set(forced_cold),
        feasible_tids=list(tids),
        peak_bytes=int(peak_mb * MODEL_SCALE),
        lateness_ns=int(lateness_ns),
        peak_overrun_bytes=int(max(0.0, peak_mb - target_adj_mb)
                               * MODEL_SCALE)
        if peak_target_bytes is not None else 0,
        target_infeasible=target_infeasible,
        solver_status=str(message),
        diagnostics=diagnostics,
        violated_sample_ts=violations,        # POSITIONS here, not times
        seed_c_solution=dict(seed_c),
        seed_e_solution={k: v for k, v in seed_e.items()},
        seed_peak_bytes=int(seed_pk_mb * MODEL_SCALE),
        seed_lateness_ns=int(seed_L * MODEL_SCALE),
    )
    return res


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
    """Order-axis solve → NeutralSchedule (same I/O shape as overlap).

    ``schedulable_tids``, when given, restricts the pool to trace tids the
    executor can actually retarget (e.g. compile-space-mappable graph
    inputs for the PyTorch wrapper). Excluded tensors stay resident; their
    bytes are carried as static floor so the cap stays honest.
    """
    # Emit semantics: lifetime release, evicts on every gap the solution
    # chose, pinned tags for the cold c-infeasible class.
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
                    f"[ct_milp_orderax] {len(dropped)} pool tensors "
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
    lazy_rounds = int(os.environ.get("MILP_LAZY_ROUNDS", "2"))
    extra_pos: list[int] = []
    result = None
    for ri in range(max(1, lazy_rounds)):
        result = _solve_orderax(
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
            extra_positions=extra_pos or None,
        )
        new_p = [p for p in result.violated_sample_ts
                 if p not in set(extra_pos)]
        if not new_p:
            break
        extra_pos.extend(new_p)
        if audit:
            print(f"[ct_milp_orderax] lazy round {ri + 1}: +{len(new_p)} "
                  f"violated positions — re-solving", flush=True)

    # Seed fallback: ship the exact-feasible Belady incumbent if the LP
    # plan is still exact-infeasible.
    if (result.violated_sample_ts and result.seed_c_solution
            and peak_target_bytes is not None
            and result.seed_peak_bytes
            <= peak_target_bytes * (1.0 - safety_margin_frac) + 2e6
            and os.environ.get("MILP_SEED_FALLBACK", "1") == "1"):
        if audit:
            print(f"[ct_milp_orderax] SEED FALLBACK: LP exact-infeasible "
                  f"(peak={result.peak_bytes/1e6:.0f}MB); shipping Belady "
                  f"incumbent (peak={result.seed_peak_bytes/1e6:.0f}MB, "
                  f"L={result.seed_lateness_ns/1e6:.0f}ms)", flush=True)
        result.c_solution = dict(result.seed_c_solution)
        result.e_solution = dict(result.seed_e_solution)
        for t in result.feasible_tids:
            for k in range(len(pool[t].consumers) - 1):
                result.e_solution.setdefault((t, k), 0.0)
        result.peak_bytes = int(result.seed_peak_bytes)
        result.lateness_ns = int(result.seed_lateness_ns)
        result.target_infeasible = False
        result.peak_overrun_bytes = 0
        result.diagnostics["seed_fallback"] = True

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
        "io_model": "ct_milp_orderax",
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
