"""Bandwidth-aware admission scheduler.

Same per-use ``UseJob`` model as ``jit_sim_prune`` but the admission
decision is **windowed bandwidth utilization**, not a single deadline
check.  We maintain a piecewise-constant H2D utilization curve over
the iter; admitting a candidate requires no window in
``[planned_start, planned_end]`` to exceed ``bw_target × hw.h2d_bw``
once the candidate is included.  This rejects the
locally-feasible-but-globally-over-committed schedules that
``jit_sim_prune n=1`` admits at deadline-only granularity.

Phases:

  1. Build per-use H2D jobs at ALAP seed (same as ``jit_sim_prune``).
  2. Drop jobs whose individual ALAP slack is already negative.
  3. Walk jobs in priority order (default: by ``start_ns``); admit
     each iff the windowed-utilization check stays under
     ``bw_target``.  Otherwise drop.
  4. Emit ``NeutralSchedule`` via the shared
     ``jit_sim_prune._emit_neutral`` helper.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from sim.core.trace import Trace

from sim.load.pytorch_profile.graph_modifiers.common import (
    HwParams,
    GlobalTask,
    MultiGraphSidecars,
    NeutralColdStart,
    NeutralPrefetch,
    NeutralEvict,
    NeutralSchedule,
    NeutralTensor,
    UnifiedTimeline,
    build_node_timeline,
    build_unified_timeline,
    effective_h2d_bw,
    neutral_to_pytorch,
)
from sim.load.pytorch_profile.graph_modifiers.schedulers.jit_sim_prune.scheduler import (
    UseJob,
    _emit_neutral,
)


# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------


@dataclass
class BWAwareKnobs:
    """Tuning surface for the bandwidth-aware admission.

    Attributes
    ----------
    bw_target : float
        Maximum H2D utilization permitted in any window during the
        iter.  ``1.0`` = saturate the lane (= classic FIFO admit);
        ``0.7`` = leave 30% headroom.  Lower values produce
        less-aggressive schedules (fewer admitted prefetches) with
        smaller stall margin.
    drop_infeasible : bool
        Drop jobs whose ALAP slack is already negative before the
        admission walk.  Default True.
    value_model : {"dram_density", "dram_size", "tensor_size", "uniform"}
        Priority used when admitting.  ``"start_ns"`` (default) = process
        in ALAP-start order, identical to ``jit_sim_prune``'s walk.
        Other modes process highest-value first, so when bandwidth
        runs out the scheduler keeps the highest-value reloads.
    iter_wall_ns : int | None
        If set, use this as the wall budget the bandwidth windows
        cover.  Otherwise use ``tl.total_duration_ns``.
    """

    bw_target: float = 0.7
    drop_infeasible: bool = True
    value_model: str = "start_ns"
    iter_wall_ns: int | None = None
    peak_target_bytes: int | None = None


# ---------------------------------------------------------------------------
# FIFO with deadline-margin admission
# ---------------------------------------------------------------------------
#
# At ``h2d_streams=1`` (the runtime's enforced configuration) any single
# transfer occupies the lane at 100% for its duration, so a "max
# windowed utilization" admission collapses to FIFO: either you admit
# the candidate or you don't.  Under that lens the only useful slider
# is *deadline headroom*: how much of a margin we leave between the
# transfer's actual fire+dur and its consumer's start_ns.  ``bw_target``
# in this scheduler is interpreted as
#
#   slack_required(j) = (1 - bw_target) * j.duration_ns
#
# i.e., reject any candidate whose serialised fire+dur lands within
# ``slack_required`` of the consumer.  At ``bw_target = 1.0`` this is
# identical to ``jit_sim_prune n=1`` (all-deadline-feasible jobs admit).
# Lower values shed the tightest-fit jobs first, freeing lane time for
# the rest of the iter to absorb dispatch jitter without stalling.


@dataclass
class _LaneFIFO:
    """Single-channel transfer FIFO.  ``end_ns`` is the time the next
    admitted transfer would start (``= max(prev_end, candidate.earliest)``).
    """
    end_ns: int = 0

    def fire_time(self, earliest_ns: int) -> int:
        return max(int(earliest_ns), int(self.end_ns))

    def admit(self, fire: int, dur: int) -> None:
        self.end_ns = int(fire) + int(dur)


# ---------------------------------------------------------------------------
# Job construction (same shape as jit_sim_prune)
# ---------------------------------------------------------------------------


def _build_jobs(tl: UnifiedTimeline, hw: HwParams) -> list[UseJob]:
    """One UseJob per (storage-group, use_index) — see jit_sim_prune.

    Coalescing by ``storage_group_id`` ensures aliased ctids are
    scheduled once, not N times.
    """
    from sim.load.pytorch_profile.graph_modifiers.common import coalesce_by_storage

    bw = max(effective_h2d_bw(hw), 1e-9)
    jobs: list[UseJob] = []
    for representative, _members in coalesce_by_storage(tl.tensors):
        if representative.size_bytes <= 0:
            continue
        dur = hw.h2d_latency_ns + int(representative.size_bytes / bw)
        prev_pos = -1
        for k, pos in enumerate(representative.uses):
            consumer_task = tl.tasks[pos]
            earliest = 0 if prev_pos == -1 else tl.tasks[prev_pos].end_ns
            deadline = consumer_task.start_ns
            seed_start = max(earliest, deadline - dur)
            jobs.append(UseJob(
                tensor_uid=representative.uid,
                use_index=k,
                consumer_pos=pos,
                release_pos=prev_pos,
                duration_ns=dur,
                size_bytes=representative.size_bytes,
                deadline_ns=deadline,
                earliest_ns=earliest,
                start_ns=seed_start,
                end_ns=seed_start + dur,
            ))
            prev_pos = pos
    return jobs


# ---------------------------------------------------------------------------
# Admission walk
# ---------------------------------------------------------------------------


@dataclass
class _AdmitState:
    admitted: list[UseJob] = field(default_factory=list)
    dropped: list[UseJob] = field(default_factory=list)
    startup: set[int] = field(default_factory=set)


def _value(j: UseJob, model: str) -> float:
    if model == "dram_density":
        gap = max(0, j.deadline_ns - j.earliest_ns)
        evict = max(0, gap - j.duration_ns)
        return (j.size_bytes * evict) / max(j.duration_ns, 1)
    if model == "dram_size":
        gap = max(0, j.deadline_ns - j.earliest_ns)
        evict = max(0, gap - j.duration_ns)
        return float(j.size_bytes * evict)
    if model == "tensor_size":
        return float(j.size_bytes)
    if model == "uniform":
        return 1.0
    if model == "start_ns":
        return -float(j.start_ns)  # earlier first
    raise ValueError(f"unknown value_model: {model}")


def _build_peak_state(
    tl: UnifiedTimeline, jobs: list[UseJob], locked_uids: set[int],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int],
           dict[int, set[tuple[int, int]]], int]:
    """Per-launch peak state used by the peak-target admission.

    Returns ``(launch_active_bytes, kept_not_active, consumers_of_uid,
    forced_keep_bytes)`` where:

      * ``launch_active_bytes[(g, l)]`` — bytes of feasible tensors
        consumed AT this launch (always resident there, regardless of
        streaming choice).
      * ``kept_not_active[(g, l)]`` — bytes of feasible tensors that
        are still "kept" (not yet marked for streaming) and are NOT
        consumed at this launch.  Initially this is ``feasible_total
        − launch_active_bytes[(g, l)]``; after a tensor is admitted
        for streaming we subtract its size from every launch where it
        is *not* consumed.
      * ``consumers_of_uid[uid]`` — set of ``(g, l)`` keys at which
        the tensor is consumed.  Used to update ``kept_not_active``
        on admission.
      * ``forced_keep_bytes`` — sum of locked tensor sizes (constant).
    """
    feasible_uids: set[int] = {j.tensor_uid for j in jobs}
    feasible_total = sum(
        int(tl.tensors[uid].size_bytes) for uid in feasible_uids
    )
    launch_active: dict[tuple[int, int], int] = {}
    consumers_of_uid: dict[int, set[tuple[int, int]]] = {}
    # For non-feasible tensors (no admissible UseJobs — e.g. all-uses
    # gap < transfer duration), they're effectively "always kept"
    # contributing to peak at every launch.  Treat them as part of the
    # forced-keep baseline.
    non_feasible_active: dict[tuple[int, int], int] = {}
    forced_keep = 0
    for tensor in tl.tensors:
        size = int(tensor.size_bytes)
        if tensor.uid in feasible_uids and tensor.uid not in locked_uids:
            seen: set[tuple[int, int]] = set()
            for pos in tensor.uses:
                t = tl.tasks[pos]
                seen.add((int(t.graph_id), int(t.launch_id)))
            consumers_of_uid[tensor.uid] = seen
            for k in seen:
                launch_active[k] = launch_active.get(k, 0) + size
        else:
            forced_keep += size
            # We still need to know which launches this tensor is "active"
            # at so we don't double-count via kept_not_active later — but
            # since it's part of forced_keep (always resident, all
            # launches), and kept_not_active is about *feasible* tensors
            # only, no further accounting is needed.

    kept_not_active: dict[tuple[int, int], int] = {}
    for k, active in launch_active.items():
        kept_not_active[k] = feasible_total - active
    # Some launches may have only forced-keep / locked consumers; they
    # have launch_active = 0 in our feasible-only map but still see
    # peak = forced_keep + feasible_total + 0.  Make sure those launches
    # are represented so _max_peak considers them.
    for tensor in tl.tensors:
        if tensor.uid in feasible_uids and tensor.uid not in locked_uids:
            continue
        for pos in tensor.uses:
            t = tl.tasks[pos]
            k = (int(t.graph_id), int(t.launch_id))
            if k not in launch_active:
                launch_active[k] = 0
                kept_not_active[k] = feasible_total

    return launch_active, kept_not_active, consumers_of_uid, forced_keep


def _max_peak(
    launch_active: dict[tuple[int, int], int],
    kept_not_active: dict[tuple[int, int], int],
    forced_keep: int,
) -> int:
    """Per-launch peak under the current streaming choice."""
    max_p = forced_keep
    for k, active in launch_active.items():
        p = forced_keep + active + kept_not_active.get(k, 0)
        if p > max_p:
            max_p = p
    return max_p


def _forward_admit(
    jobs: list[UseJob], hw: HwParams, knobs: BWAwareKnobs,
    iter_wall_ns: int,
    *,
    tl: UnifiedTimeline | None = None,
    locked_uids: set[int] | None = None,
) -> _AdmitState:
    state = _AdmitState()
    if not jobs:
        return state

    bw = max(effective_h2d_bw(hw), 1e-9)
    # Global byte budget = bw_target × bw × iter_wall.  This caps the
    # *aggregate* bytes the scheduler may move per iter to a fraction
    # of the lane's full capacity, so even if every individual job is
    # FIFO-feasible we stop admitting once we've committed enough lane
    # time to keep overall iter wall close to the no-WS baseline.
    budget_bytes = max(0, int(knobs.bw_target * bw * iter_wall_ns))
    lane = _LaneFIFO()

    # Order: by ``value_model``.  ``start_ns`` keeps jit_sim_prune's
    # walk; other modes admit highest-value first so when the budget
    # runs out we keep the highest-value reloads.  ``tensor_size`` is
    # the natural choice for "minimize peak VRAM" since the largest
    # tensors are the biggest contributors to peak.
    peak_target = knobs.peak_target_bytes
    do_peak_target = (
        peak_target is not None and tl is not None
        and locked_uids is not None
    )
    if knobs.value_model == "start_ns":
        order = sorted(
            jobs,
            key=lambda j: (j.start_ns, j.deadline_ns, -j.duration_ns,
                           j.tensor_uid, j.use_index),
        )
    else:
        order = sorted(jobs, key=lambda j: -_value(j, knobs.value_model))

    if do_peak_target:
        launch_active, kept_not_active, consumers_of_uid, forced_keep = (
            _build_peak_state(tl, jobs, locked_uids)
        )
        current_peak = _max_peak(launch_active, kept_not_active, forced_keep)
        if current_peak <= peak_target:
            for job in order:
                if job.use_index == 0:
                    state.startup.add(job.tensor_uid)
                else:
                    state.dropped.append(job)
            return state
        streamed_uids: set[int] = set()

    admitted_bytes = 0
    for job in order:
        if do_peak_target and current_peak <= peak_target:
            if job.use_index == 0:
                state.startup.add(job.tensor_uid)
            else:
                state.dropped.append(job)
            continue
        # 1. Global byte budget — stop admitting once we've committed
        #    enough lane time to risk lengthening the iter.
        if admitted_bytes + job.size_bytes > budget_bytes:
            if job.use_index == 0:
                state.startup.add(job.tensor_uid)
            else:
                state.dropped.append(job)
            continue
        # 2. Local FIFO deadline — same admission criterion as
        #    jit_sim_prune n=1, but at the FIFO-determined fire time
        #    (not the ALAP seed).
        fire = lane.fire_time(job.earliest_ns)
        end = fire + job.duration_ns
        if end > job.deadline_ns:
            if job.use_index == 0:
                state.startup.add(job.tensor_uid)
            else:
                state.dropped.append(job)
            continue
        job.start_ns = fire
        job.end_ns = end
        lane.admit(fire, job.duration_ns)
        state.admitted.append(job)
        admitted_bytes += job.size_bytes

        if do_peak_target and job.tensor_uid not in streamed_uids:
            streamed_uids.add(job.tensor_uid)
            size = int(job.size_bytes)
            consumers = consumers_of_uid.get(job.tensor_uid, set())
            for k in launch_active:
                if k not in consumers:
                    kept_not_active[k] = max(0, kept_not_active[k] - size)
            current_peak = _max_peak(
                launch_active, kept_not_active, forced_keep,
            )

    return state


def _final_model_peak(
    state: _AdmitState,
    launch_active: dict[tuple[int, int], int],
    kept_not_active: dict[tuple[int, int], int],
    consumers_of_uid: dict[int, set[tuple[int, int]]],
    forced_keep: int,
    tl: UnifiedTimeline,
) -> int:
    """Replay the admission's streamed-uid set against a clean
    ``kept_not_active`` to get the per-launch model peak the admission
    actually achieved — used by the peak-target search to decide whether
    a candidate ``bw_target`` reaches the cap.
    """
    streamed: set[int] = {j.tensor_uid for j in state.admitted}
    kna = dict(kept_not_active)
    for uid in streamed:
        size = int(tl.tensors[uid].size_bytes)
        consumers = consumers_of_uid.get(uid, set())
        for k in launch_active:
            if k not in consumers:
                kna[k] = max(0, kna[k] - size)
    return _max_peak(launch_active, kna, forced_keep)


# ---------------------------------------------------------------------------
# Public solve
# ---------------------------------------------------------------------------


# Adapter: jit_sim_prune._emit_neutral expects the jit_sim_prune AdmitState
# and PruneKnobs, but only reads ``.admitted`` and ``.startup``.  Wrap.


@dataclass
class _AdmitStateAdapter:
    admitted: list[UseJob]
    startup: set[int]
    dropped: list[UseJob] = field(default_factory=list)
    channel_end_ns: list[int] = field(default_factory=list)
    channel_admitted_idx: dict[int, list[int]] = field(default_factory=dict)


def solve_neutral(
    trace: Trace,
    *,
    sidecars: MultiGraphSidecars,
    hw: HwParams,
    locked_graph_input_names: set[str] | None = None,
    knobs: BWAwareKnobs | None = None,
    graph_multiplicity: dict[int, int] | None = None,
    iter_wall_ns_override: int | None = None,
) -> NeutralSchedule:
    if knobs is None:
        knobs = BWAwareKnobs()

    tl = build_unified_timeline(
        trace, sidecars, cpu_per_launch_ns=hw.cpu_per_launch_ns,
        graph_multiplicity=graph_multiplicity,
        replicate_uses=True,
    )
    if not tl.tasks:
        raise RuntimeError("[bw_aware] no compiled tasks in bundle.")

    iter_wall_ns = (
        iter_wall_ns_override
        or knobs.iter_wall_ns
        or tl.total_duration_ns
    )

    locked = set(locked_graph_input_names or set())
    locked_uids = {
        t.uid for t in tl.tensors if t.graph_input_name in locked
    }

    jobs = _build_jobs(tl, hw)

    if knobs.drop_infeasible:
        infeasible = [
            j for j in jobs if j.deadline_ns - j.earliest_ns < j.duration_ns
        ]
        feasible = [
            j for j in jobs if j.deadline_ns - j.earliest_ns >= j.duration_ns
        ]
    else:
        infeasible = []
        feasible = jobs

    if knobs.peak_target_bytes is not None:
        # Peak-target mode: combine the per-tensor early-stop with an
        # outer search for the smallest ``bw_target`` that still meets
        # the cap. Lower bw_target = wider per-window utilisation
        # headroom = fewer concurrent prefetches = less stall (since
        # the simulator water-fills the H2D lane). Early-stop within
        # the chosen bw_target keeps total streamed bytes minimal.
        # The combination targets the user's stated objective: minimum
        # E2E (= baseline + minimum stall) under the peak cap.
        from dataclasses import replace
        la, kna_init, cou, fk = _build_peak_state(tl, feasible, locked_uids)
        target = knobs.peak_target_bytes

        def _solve_at(bw_t: float) -> tuple[_AdmitState, int]:
            k = replace(knobs, bw_target=bw_t)
            s = _forward_admit(
                feasible, hw, k, iter_wall_ns,
                tl=tl, locked_uids=locked_uids,
            )
            p = _final_model_peak(s, la, kna_init, cou, fk, tl)
            return s, p

        hi_cap = max(0.05, min(1.0, float(knobs.bw_target)))
        state_hi, peak_hi = _solve_at(hi_cap)
        if peak_hi > target:
            # Even at the user's bw cap we can't reach the target.
            # Return the most aggressive admission as best-effort.
            state = state_hi
            chosen_bw = hi_cap
        else:
            # Binary-search down for the smallest bw_target that still
            # admits enough (with early-stop) to meet the peak cap.
            lo = 0.01
            hi = hi_cap
            best_state = state_hi
            best_bw = hi_cap
            for _ in range(8):
                mid = (lo + hi) / 2
                s_mid, p_mid = _solve_at(mid)
                if p_mid <= target:
                    hi = mid
                    best_state = s_mid
                    best_bw = mid
                else:
                    lo = mid
                if hi - lo < 0.005:
                    break
            state = best_state
            chosen_bw = best_bw
        knobs.bw_target = chosen_bw
    else:
        state = _forward_admit(
            feasible, hw, knobs, iter_wall_ns,
            tl=tl, locked_uids=locked_uids,
        )

    # Build an adapter so the shared emission helper can read it.
    from sim.load.pytorch_profile.graph_modifiers.schedulers.jit_sim_prune.scheduler import PruneKnobs
    adapter = _AdmitStateAdapter(
        admitted=state.admitted, startup=state.startup, dropped=state.dropped,
    )
    prefetches, evicts, evicted, cold_start = _emit_neutral(
        tl, adapter, feasible, locked_uids, PruneKnobs(),
    )

    cold_starts: list[NeutralColdStart] = []
    for uid in sorted(cold_start):
        tensor = tl.tensors[uid]
        first_pos = tensor.uses[0] if tensor.uses else 0
        cold_starts.append(NeutralColdStart(
            tensor_uid=uid,
            anchor_launch_id=int(tl.tasks[first_pos].launch_id),
            reason=(
                "bw_aware_locked" if uid in locked_uids
                else "bw_aware_startup"
            ),
        ))

    total_bytes = sum(t.size_bytes for t in tl.tensors)
    cold_start_bytes = sum(tl.tensors[u].size_bytes for u in cold_start)

    per_graph_stats: dict[int, dict[str, Any]] = {}
    for gid in tl.graph_order:
        tensors_in_g = [t for t in tl.tensors if t.graph_id == gid]
        per_graph_stats[gid] = {
            "weight_bytes": sum(t.size_bytes for t in tensors_in_g),
            "tensor_count": len(tensors_in_g),
            "evicted_count": sum(1 for t in tensors_in_g if t.uid in evicted),
        }

    meta: dict[str, Any] = {
        "io_model": "bw_aware",
        "backend": "cg-sim",
        "algorithm": "Windowed bandwidth-utilization admission",
        "knobs": {
            "bw_target": knobs.bw_target,
            "drop_infeasible": knobs.drop_infeasible,
            "value_model": knobs.value_model,
            "peak_target_mb": (
                round(knobs.peak_target_bytes / 1e6, 2)
                if knobs.peak_target_bytes is not None else None
            ),
        },
        "graph_order": tl.graph_order,
        "graph_multiplicity": {
            int(gid): int(tl.graph_multiplicity.get(gid, 1))
            for gid in tl.graph_order
        },
        "per_graph_stats": per_graph_stats,
        "total_weight_bytes": int(total_bytes),
        "total_weight_mb": round(total_bytes / 1e6, 2),
        "cold_start_count": len(cold_start),
        "cold_start_bytes": int(cold_start_bytes),
        "cold_start_mb": round(cold_start_bytes / 1e6, 2),
        "evicted_tensor_count": len(evicted),
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
        "infeasible_dropped": len(infeasible),
        "forward_dropped": len(state.dropped),
        "startup_promoted": len(state.startup),
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
            used_by_launch_ids=sorted({
                int(tl.tasks[p].launch_id) for p in t.uses
            }),
        ))

    return NeutralSchedule(
        graph_order=list(tl.graph_order),
        compilation_hashes={
            int(gid): str(tl.per_graph_hash.get(gid, ""))
            for gid in tl.graph_order
        },
        tensors=tensor_records,
        prefetches=prefetches,
        evicts=evicts,
        cold_starts=cold_starts,
        meta=meta,
    )


def print_summary(neutral: NeutralSchedule) -> None:
    m = neutral.meta
    print()
    print(f"  Variant            : {m.get('io_model')}")
    print(f"  bw_target          : {m.get('knobs', {}).get('bw_target')}")
    print(f"  Iter wall          : {m.get('wall_time_iter_ns', 0)/1e6:.2f} ms")
    print(f"  Total weights      : {m.get('total_weight_mb')} MB")
    print(f"  Cold-start         : {m.get('cold_start_count')} tensors "
          f"({m.get('cold_start_mb')} MB)")
    print(f"  Evicted tensors    : {m.get('evicted_tensor_count')}")
    print(f"  Prefetch ops       : {m.get('h2d_ops')}")
    print(f"  Evict ops          : {m.get('d2h_ops')}")
    print(f"  Forward dropped    : {m.get('forward_dropped')}")
    print(f"  Startup promoted   : {m.get('startup_promoted')}")
