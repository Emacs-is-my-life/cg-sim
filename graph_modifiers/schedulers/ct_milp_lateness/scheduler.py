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

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

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
# Λ in the objective: per-byte penalty on peak-target overrun, in
# lateness-ns units. 1e6 = 1 ms of lateness per 1 MB of overrun — heavy
# enough that the LP always prefers a slow plan to one over cap.
PEAK_SLACK_PENALTY = 1.0e6


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
    # (node_id, trace_start_ns, trace_end_ns) sorted by trace_start_ns.
    consumers: list[tuple[int, int, int]]
    tau_h2d_ns: int                  # δ_t — per-tid H2D transfer time
    tau_d2h_ns: int                  # per-tid D2H transfer time
    # True iff consumer_{k+1}.start − consumer_k.end ≥ τ_h2d:
    #   the gap admits an evict + refetch round trip. Otherwise the
    #   tid stays loaded across the gap regardless of c_t.
    gap_feasibility: list[bool]
    # True iff consumer_0.start ≥ τ_h2d: the initial prefetch can finish
    # before the first consumer fires. Otherwise the tid must be cold.
    c_feasibility: bool
    # Per-consumer compiled_launch_id / compiled_graph_id from the trace
    # (when present). Used only to populate NeutralTensor fields for the
    # schedule JSON's compile-side metadata; the injector's fast path
    # doesn't depend on them.
    consumer_graph_ids: list[int] = field(default_factory=list)
    consumer_launch_ids: list[int] = field(default_factory=list)


def _build_pool(trace: Trace, hw: HwParams) -> dict[int, _PoolTensor]:
    """Build the cgsim_tid pool of cuda-resident WEIGHT/LEAF/INPUT tensors.

    A tid is admitted iff:
      - tensor_type ∈ {WEIGHT, LEAF, INPUT}
      - device starts with "cuda"
      - size_bytes > 0
      - at least one gpu trace consumer

    The trace loader has already merged aliases by (device, storage_id),
    so each entry corresponds to a distinct physical storage.
    """
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

    # Per-graph earliest gpu node start (used for c_feasibility — the
    # initial prefetch can only fire at or after a gpu node dispatches
    # in the consumer's graph, since emit's issuer-picker only searches
    # within the consumer's graph). Tids whose consumer_0 is itself the
    # earliest gpu node in its graph have no valid async issuer and
    # must be cold (or accept a sync prefetch that stalls the
    # consumer).
    graph_first_gpu_ns: dict[int, int] = {}
    for nid, node in trace.node_map.items():
        rk = str((node.args or {}).get("resource_kind") or "")
        if rk not in _GPU_RESOURCE_KINDS:
            continue
        start_ns = int((node.args or {}).get("start_ns") or 0)
        if start_ns <= 0:
            continue
        gid_raw = (node.args or {}).get("compiled_graph_id")
        try:
            gid_i = int(gid_raw) if gid_raw is not None else -1
        except (TypeError, ValueError):
            gid_i = -1
        cur = graph_first_gpu_ns.get(gid_i)
        if cur is None or start_ns < cur:
            graph_first_gpu_ns[gid_i] = start_ns

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
        gap_feas: list[bool] = []
        for i in range(len(consumers) - 1):
            gap_ns = consumers[i + 1][1] - consumers[i][2]
            gap_feas.append(gap_ns >= tau_h2d)
        # c_feasibility: consumer_0.start − consumer_graph's first gpu
        # node start ≥ τ_h2d. Otherwise no async issuer in the
        # consumer's graph can fit a full transfer before consumer_0
        # fires — sim falls back to sync prefetch (which stalls
        # consumer by τ_h2d) and the injector's coverage_repair often
        # silently demotes the tid back to cold. Forcing cold here
        # makes the LP's plan match the injector's behaviour.
        consumer_0_gid = graph_ids[0] if graph_ids else -1
        origin_for_c = graph_first_gpu_ns.get(consumer_0_gid, consumers[0][1])
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
        )
    return pool


def _build_gpu_consumer_timeline(
    trace: Trace,
) -> list[tuple[int, int]]:
    """All gpu compute nodes with valid trace_start_ns, sorted by start.

    Returns (node_id, trace_start_ns) tuples. Used as the sample grid for
    peak and lateness constraints — sampling at every gpu compute event
    keeps the LP aligned with sim's moment-by-moment vram state and the
    PCIe queue's cumulative load.
    """
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


def _solve_milp(
    pool: dict[int, _PoolTensor],
    trace: Trace,
    hw: HwParams,
    *,
    peak_target_bytes: int | None,
    extra_static_bytes: int,
    safety_margin_frac: float,
    max_peak_samples: int,
    time_limit_s: float | None,
    lp_relaxation: bool,
    audit: bool,
) -> _LPResult:
    """Build and solve the lateness MILP.

    Variables:
        c_t       binary per feasible pool tid
        e_{t,k}   binary per feasible cross-iter gap
        P         continuous ≥ 0  — modeled peak (driven by per-sample rows)
        L_max     continuous ≥ 0  — max consumer lateness (ns)
        s_P       continuous ≥ 0  — peak overrun slack (bytes)
    """
    # ---- 1. Feasibility filter ----
    feasible_tids: list[int] = []
    forced_cold: set[int] = set()
    for tid, pt in pool.items():
        # Forced cold iff initial prefetch cannot fit (consumer_0 too
        # early) AND no cross-iter gap admits a refetch either: the LP
        # has no choice but to keep the tid resident from layout.
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
    L_MAX_IDX = col
    col += 1
    S_PEAK_IDX = col
    col += 1
    total_vars = col

    # ---- 3. Variable bounds + integrality ----
    bounds_list: list[tuple[float, float | None]] = []
    for tid in feasible_tids:
        pt = pool[tid]
        if not pt.c_feasibility:
            # No room for an async initial prefetch in the consumer's
            # graph — emit would fall back to a sync prefetch, the
            # injector would silently demote, and sim would carry the
            # tid as effectively cold anyway. Pin c=1 so the LP plans
            # the residency explicitly and the peak constraint sees it.
            bounds_list.append((1.0, 1.0))
        else:
            bounds_list.append((0.0, 1.0))
    bounds_list.extend([(0.0, 1.0)] * n_e)
    bounds_list.append((0.0, None))                          # P
    bounds_list.append((0.0, None))                          # L_max
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
    c_obj[L_MAX_IDX] = 1.0
    if peak_target_bytes is not None:
        c_obj[S_PEAK_IDX] = PEAK_SLACK_PENALTY
    for tid in feasible_tids:
        size = pool[tid].size_bytes
        c_obj[c_var_idx[tid]] = -float(size) * epsilon_per_byte
    for (tid, k), col in e_var_idx.items():
        size = pool[tid].size_bytes
        c_obj[col] = float(size) * epsilon_per_byte

    # ---- 5. Conditional e-c coupling ----
    #
    # For tids that CAN be JIT-prefetched (c_feasibility=True), we
    # impose ``e_{t,k} = 1 − c_t`` on every feasible gap. This locks
    # the LP into one of two natural patterns: cold = pinned the whole
    # run (e=0 everywhere), or streamed = per-iter prefetch+evict
    # cycle (e=1 everywhere). The cycle pattern matches jit_sim_prune
    # and is empirically the right peak/PCIe tradeoff for "normal"
    # tids — without it, the LP gravitates toward c=0, e=0 (streamed
    # but never evicted, alive the whole run) which collapses peak
    # to near-cold-floor and fails cap.
    #
    # For tids that CAN'T be JIT-prefetched (c_feasibility=False, c
    # pinned to 1 by bounds), we DO NOT impose the coupling. This
    # frees the LP to pick e=1 on selective gaps for forced-cold
    # tids — the "hybrid" pattern where the tensor loads at layout
    # for early consumers, gets evicted mid-run when its VRAM is
    # needed elsewhere, then refetched from RAM before later
    # consumers. Without this exception, a forced-cold tid would
    # have e=0 forced on every gap → resident the whole run, even
    # when there's room to evict and refetch later.
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    ub_list: list[float] = []
    row = 0

    for tid in feasible_tids:
        pt = pool[tid]
        if not pt.c_feasibility:
            # Forced cold — let e be free for hybrid pattern.
            continue
        for k, feas in enumerate(pt.gap_feasibility):
            if not feas:
                continue
            ci = c_var_idx[tid]
            ei = e_var_idx[(tid, k)]
            # c + e ≤ 1
            rows.extend([row, row])
            cols.extend([ci, ei])
            vals.extend([1.0, 1.0])
            ub_list.append(1.0)
            row += 1
            # c + e ≥ 1   (encoded as −c − e ≤ −1)
            rows.extend([row, row])
            cols.extend([ci, ei])
            vals.extend([-1.0, -1.0])
            ub_list.append(-1.0)
            row += 1

    # ---- 6. Sample grid (shared by peak & lateness rows) ----
    gpu_consumers = _build_gpu_consumer_timeline(trace)
    if not gpu_consumers:
        raise RuntimeError(
            "[ct_milp_lateness] no gpu consumer events in trace; "
            "cannot build LP sample grid."
        )
    samples = _select_sample_points(gpu_consumers, max_samples=max_peak_samples)
    if audit:
        print(
            f"[ct_milp_lateness:audit] gpu_consumer_events={len(gpu_consumers)} "
            f"sampled_points={len(samples)}"
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
    constant_floor = float(forced_cold_bytes) + float(extra_static_bytes)

    for nid_sample, t_l in samples:
        const_addons = constant_floor
        var_coefs: dict[int, float] = {}
        for tid in feasible_tids:
            pt = pool[tid]
            size = pt.size_bytes
            # Currently consumed at this exact node?
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
                # Sample sits on a consumer's [start, end] boundary
                # without matching node id — treat as alive.
                const_addons += size
                continue
            arc_kp1_start = pt.consumers[k_in + 1][1] - tau
            if t_l >= arc_kp1_start:
                const_addons += size
            elif pt.gap_feasibility[k_in]:
                # Dead zone of a feasible gap, alive iff NOT evicted:
                # contribution = size · (1 − e_{t,k}).
                #
                # For c_feasibility=True tids (under c+e=1 coupling),
                # (1 − e) = c, so the contribution reduces to size · c.
                # Encode as a positive coefficient on c_var — preserves
                # the legacy row shape for solver well-conditioning.
                #
                # For c_feasibility=False tids (c pinned to 1, no
                # coupling), encode as `const += size` plus `−size`
                # on the e_var. This is the encoding that lets the
                # LP actually observe the VRAM savings from picking
                # e=1 mid-run (the hybrid pattern).
                if pt.c_feasibility:
                    var_coefs[c_var_idx[tid]] = (
                        var_coefs.get(c_var_idx[tid], 0.0) + size
                    )
                else:
                    e_col = e_var_idx.get((tid, k_in))
                    if e_col is not None:
                        const_addons += size
                        var_coefs[e_col] = (
                            var_coefs.get(e_col, 0.0) - size
                        )
                    else:
                        const_addons += size
            else:
                # Infeasible gap — no evict can fit, tensor stays.
                const_addons += size

        # P ≥ const_addons + Σ var_coef · c   ⇒   Σ var_coef · c − P ≤ −const_addons
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

    # ---- 7b. Soft peak cap: P − s_P ≤ target·(1 − margin) ----
    if peak_target_bytes is not None:
        target_adj = max(
            0.0,
            float(peak_target_bytes) * (1.0 - float(safety_margin_frac)),
        )
        rows.append(row)
        cols.append(P_IDX)
        vals.append(1.0)
        rows.append(row)
        cols.append(S_PEAK_IDX)
        vals.append(-1.0)
        ub_list.append(target_adj)
        row += 1
        if audit:
            print(
                f"[ct_milp_lateness:audit] peak cap: target={peak_target_bytes/1e6:.1f}MB "
                f"margin={safety_margin_frac*100:.1f}% → P ≤ {target_adj/1e6:.1f}MB"
            )

    # ---- 8. Lateness rows (cumulative PCIe H2D ≤ T + L_max) ----
    #
    # At each sample T_i, the LP requires that the cumulative H2D work
    # for all transfers whose deadline ≤ T_i fits in [0, T_i] plus
    # slack L_max:
    #
    #   Σ_{t: any consumer ≤ T_i}    δ_t · (1 − c_t)
    # + Σ_{(t,k): consumer_{k+1} ≤ T_i, feasible} δ_t · e_{t,k}
    #   ≤ T_i + L_max
    #
    # Expanding 1 − c_t and rearranging:
    #   Σ (−δ_t · c_t) + Σ (δ_t · e_{t,k}) − L_max
    #     ≤ T_i − Σ δ_t  (the trailing sum is the "const_lhs" below)
    #
    # The constraint's slack is L_max — minimizing L_max in the
    # objective drives the schedule to keep PCIe load below the
    # cumulative wall-clock at every sample. The first sample where
    # cumulative outruns T_i sets L_max for the run.
    for _nid_sample, t_l in samples:
        const_lhs = 0.0
        rows.append(row)
        cols.append(L_MAX_IDX)
        vals.append(-1.0)
        for tid in feasible_tids:
            pt = pool[tid]
            delta = pt.tau_h2d_ns
            # Initial prefetch deadline = first_consumer's trace_start.
            if pt.consumers[0][1] <= t_l:
                const_lhs += delta
                rows.append(row)
                cols.append(c_var_idx[tid])
                vals.append(-float(delta))
            # Cross-iter refetch deadlines per feasible gap.
            for k in range(len(pt.consumers) - 1):
                if (tid, k) not in e_var_idx:
                    continue
                if pt.consumers[k + 1][1] <= t_l:
                    rows.append(row)
                    cols.append(e_var_idx[(tid, k)])
                    vals.append(float(delta))
        ub_list.append(float(t_l) - const_lhs)
        row += 1

    nb = row

    # ---- 9. Solve ----
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

    fell_back = False
    if not res.success and integrality_arr is not None and not lp_relaxation:
        if audit:
            print(
                f"[ct_milp_lateness:solver] MILP failed: "
                f"status={getattr(res, 'message', '')!r} — "
                f"falling back to LP relaxation."
            )
        kwargs.pop("integrality", None)
        res = linprog(c_obj, **kwargs)
        fell_back = True

    if audit:
        print(
            f"[ct_milp_lateness:solver] success={res.success} fell_back={fell_back} "
            f"status={getattr(res, 'message', '')!r}"
        )
        if res.success and res.x is not None and fell_back:
            cvals = [float(res.x[c_var_idx[t]]) for t in feasible_tids]
            n_zero = sum(1 for v in cvals if v < 0.01)
            n_one = sum(1 for v in cvals if v > 0.99)
            n_frac = sum(1 for v in cvals if 0.01 <= v <= 0.99)
            print(
                f"[ct_milp_lateness:audit] c-value distribution (LP relaxation): "
                f"≈0: {n_zero}, ≈1: {n_one}, fractional: {n_frac}"
            )

    # ---- 10. Decode ----
    c_solution: dict[int, float] = {}
    e_solution: dict[tuple[int, int], float] = {}
    target_infeasible = False
    peak_overrun_bytes = 0
    peak_bytes = 0
    lateness_ns = 0

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
        lateness_ns = int(float(x[L_MAX_IDX]))
        if audit:
            # Verify cumulative-load constraint at the last sample.
            _last_sample = samples[-1]
            t_last = _last_sample[1]
            total_h2d_ns = 0.0
            for tid in feasible_tids:
                pt = pool[tid]
                if pt.consumers[0][1] <= t_last:
                    total_h2d_ns += pt.tau_h2d_ns * (1.0 - c_solution[tid])
                for k in range(len(pt.consumers) - 1):
                    if (tid, k) not in e_var_idx:
                        continue
                    if pt.consumers[k + 1][1] <= t_last:
                        total_h2d_ns += pt.tau_h2d_ns * e_solution[(tid, k)]
            implied_lateness_ns = total_h2d_ns - float(t_last)
            print(
                f"[ct_milp_lateness:audit] verification @ last sample "
                f"t_last={t_last/1e6:.2f}ms, "
                f"cumulative_h2d={total_h2d_ns/1e6:.2f}ms, "
                f"implied_lateness={implied_lateness_ns/1e6:.2f}ms, "
                f"LP_L_max={lateness_ns/1e6:.2f}ms"
            )
        if peak_target_bytes is not None:
            peak_overrun_bytes = int(float(x[S_PEAK_IDX]))
            if peak_overrun_bytes > 1:
                target_infeasible = True
    else:
        # Hard-fallback: cold-start everything feasible.
        for tid in feasible_tids:
            c_solution[tid] = 1.0
            pt = pool[tid]
            for k in range(len(pt.consumers) - 1):
                e_solution[(tid, k)] = 0.0
        peak_bytes = int(
            constant_floor + sum(pool[t].size_bytes for t in feasible_tids)
        )
        lateness_ns = 0
        if peak_target_bytes is not None and peak_bytes > peak_target_bytes:
            target_infeasible = True

    diagnostics = {
        "pool_size": len(pool),
        "forced_cold_count": len(forced_cold),
        "forced_cold_bytes": forced_cold_bytes,
        "feasible_var_count": nv,
        "e_var_count": n_e,
        "n_samples": len(samples),
        "solver_success": bool(res.success),
        "solver_status": str(getattr(res, "message", "")),
        "fell_back_to_lp": bool(fell_back),
        "target_infeasible": bool(target_infeasible),
        "peak_overrun_bytes": int(peak_overrun_bytes),
        "lateness_ns": int(lateness_ns),
        "lp_relaxation": bool(lp_relaxation),
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

    # Per-graph sorted list (trace_start_ns, node_id) for issuer lookup.
    nodes_by_graph_trace: dict[int, list[tuple[int, int]]] = {}
    for nid, node in trace.node_map.items():
        rk = str((node.args or {}).get("resource_kind") or "")
        if rk not in _GPU_RESOURCE_KINDS:
            continue
        start_ns = int((node.args or {}).get("start_ns") or 0)
        if start_ns <= 0:
            continue
        gid_raw = (node.args or {}).get("compiled_graph_id")
        try:
            gid_i = int(gid_raw) if gid_raw is not None else -1
        except (TypeError, ValueError):
            gid_i = -1
        nodes_by_graph_trace.setdefault(gid_i, []).append(
            (int(start_ns), int(nid))
        )
    for g in nodes_by_graph_trace:
        nodes_by_graph_trace[g].sort(key=lambda x: x[0])

    def _pick_issuer_node(
        consumer_gid: int, consumer_start_ns: int, tau_h2d_ns: int,
    ) -> tuple[int, int]:
        """Latest gpu node in ``consumer_gid`` whose trace_start_ns ≤
        consumer_start_ns − τ_h2d. Falls back to the consumer node
        itself when no such predecessor exists (sync prefetch).
        Returns (issuer_node_id, issuer_trace_start_ns).
        """
        target = consumer_start_ns - tau_h2d_ns
        lst = nodes_by_graph_trace.get(consumer_gid, ())
        best_ns = -1
        best_nid = -1
        for ts, nid in lst:
            if ts <= target:
                if ts > best_ns:
                    best_ns = ts
                    best_nid = nid
            else:
                break
        if best_nid < 0:
            return -1, -1
        return best_nid, best_ns

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
        neutral_tensors.append(NeutralTensor(
            uid=uid,
            graph_id=int(primary_gid),
            compiled_tensor_id=int(tid),
            graph_input_name=pt.name or f"cgtid_{tid}",
            size_bytes=int(pt.size_bytes),
            dtype=pt.dtype,
            used_by_launch_ids=sorted(set(pt.consumer_launch_ids)),
            shape=[],
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
        # (Per-gap evicts emit below independently of c_t.)
        if is_cold:
            cold_starts.append(NeutralColdStart(
                tensor_uid=uid,
                anchor_launch_id=(
                    pt.consumer_launch_ids[0]
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
            c0_nid, c0_start, _c0_end = consumer_0
            c0_gid = pt.consumer_graph_ids[0]
            c0_lid = pt.consumer_launch_ids[0]
            issue_nid, _ = _pick_issuer_node(
                c0_gid, c0_start, pt.tau_h2d_ns,
            )
            if issue_nid < 0:
                # No earlier gpu node available — sync prefetch
                # (issuer == consumer). Injector treats this as a
                # blocking H2D before consumer dispatch.
                issue_nid = c0_nid
            prefetches.append(NeutralPrefetch(
                tensor_uid=uid,
                issue_launch_id=int(c0_lid),
                wait_launch_id=int(c0_lid),
                transfer_start_ns=int(max(0, c0_start - pt.tau_h2d_ns)),
                transfer_end_ns=int(c0_start),
                reason="lateness_initial",
                issue_node_id=int(issue_nid),
                wait_node_id=int(c0_nid),
                cgsim_tid=int(tid),
                trusted_async=(issue_nid != c0_nid),
                issue_graph_id=-1,
                iter_mask=[],
            ))

        # Per-gap evict + refetch — INDEPENDENT of c_t. A cold tid
        # (c=1) may still be evicted mid-run if the LP found that
        # freeing the dst pages during a gap reduces per-sample VRAM
        # below the cap, then refetched from RAM before the next
        # consumer. This is the "hybrid" pattern the old e = 1 − c
        # coupling foreclosed.
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
                "lateness_hybrid_gap_evict" if is_cold
                else "lateness_gap_evict"
            )
            refetch_reason = (
                "lateness_hybrid_gap_refetch" if is_cold
                else "lateness_gap_refetch"
            )
            evicts.append(NeutralEvict(
                tensor_uid=uid,
                issue_launch_id=int(pt.consumer_launch_ids[k]),
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
                issue_launch_id=int(pt.consumer_launch_ids[k + 1]),
                wait_launch_id=int(kp1_lid),
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
    peak_target_bytes: int | None = None,
    safety_margin_frac: float = 0.07,
    max_peak_samples: int = 256,
    time_limit_s: float | None = 240.0,
    lp_relaxation: bool = False,
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
      ``lp_relaxation``       — Skip integrality, solve continuous LP
                                (debug aid).
      ``audit``               — Print pool/LP/solver diagnostics.
      ``sidecars``            — Accepted for interface parity with
                                ct_milp_multistream's main.py; ignored.

    Returns:
      A NeutralSchedule with cgsim_tid pre-resolved on every entry.
    """
    pool = _build_pool(trace, hw)
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

    result = _solve_milp(
        pool, trace, hw,
        peak_target_bytes=peak_target_bytes,
        extra_static_bytes=extra_static_bytes,
        safety_margin_frac=safety_margin_frac,
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
        "io_model": "ct_milp_lateness",
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
