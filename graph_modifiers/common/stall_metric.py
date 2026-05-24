"""Analytic two-lane PCIe stall metric for NeutralSchedule evaluation.

Forward-simulates a ``NeutralSchedule`` on a ``UnifiedTimeline`` under
independent H2D and D2H FIFOs, mirroring the dual-lane PCIe model the
cg-sim engine's ``ScheduleReplay`` uses. Returns:

* ``total_stall_ns`` — sum over kernels of ``max(0, prefetch_h2d_end −
  kernel_start_ns)``, the per-kernel unhidden H2D latency.
* ``peak_vram_bytes`` — residency including the destination region held
  from ``h2d_start`` and the source region held until ``d2h_end`` (same
  semantics as ``schedule_replay._vram_evict_pending_release``).
* ``pending_evict_bytes_peak`` — peak of source regions waiting on D2H
  retirement, i.e. the "hidden overhead" the analytic model usually
  drops.
* Lane utilisations, total bytes, and H2D/D2H overcommit flags
  (``bytes_per_iter > bw * iter_length``). Overcommit predicts the
  simulator will back up.
* ``missing_tensor_count`` — kernels whose input isn't resident at start
  (cold-start evicted without reload, or a prefetch scheduled past the
  iteration boundary).

Use to compare schedulers without paying the cost of a full cg-sim run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hw import HwParams
from .multigraph_timeline import UnifiedTimeline
from .neutral import NeutralSchedule


@dataclass
class StallMetrics:
    """Result of :func:`compute_stall_metrics`. Times ns, sizes bytes."""

    total_stall_ns: int
    max_stall_ns: int
    stalled_kernel_count: int
    pipeline_stretch_ns: int  # how much iter wall-time inflated end-to-end

    peak_vram_bytes: int
    peak_vram_mb: float
    pending_evict_bytes_peak: int

    h2d_lane_busy_ns: int
    d2h_lane_busy_ns: int
    h2d_lane_util_pct: float
    d2h_lane_util_pct: float

    h2d_bytes_total: int
    d2h_bytes_total: int
    h2d_overcommit: bool
    d2h_overcommit: bool

    missing_tensor_count: int
    total_duration_ns: int

    def summary_dict(self) -> dict[str, Any]:
        return {
            "total_stall_ns": self.total_stall_ns,
            "total_stall_ms": round(self.total_stall_ns / 1e6, 3),
            "max_stall_ns": self.max_stall_ns,
            "stalled_kernel_count": self.stalled_kernel_count,
            "pipeline_stretch_ns": self.pipeline_stretch_ns,
            "pipeline_stretch_ms": round(self.pipeline_stretch_ns / 1e6, 3),
            "peak_vram_bytes": self.peak_vram_bytes,
            "peak_vram_mb": self.peak_vram_mb,
            "pending_evict_bytes_peak": self.pending_evict_bytes_peak,
            "pending_evict_peak_mb": round(
                self.pending_evict_bytes_peak / 1e6, 2
            ),
            "h2d_lane_util_pct": self.h2d_lane_util_pct,
            "d2h_lane_util_pct": self.d2h_lane_util_pct,
            "h2d_bytes_total": self.h2d_bytes_total,
            "d2h_bytes_total": self.d2h_bytes_total,
            "h2d_overcommit": self.h2d_overcommit,
            "d2h_overcommit": self.d2h_overcommit,
            "missing_tensor_count": self.missing_tensor_count,
            "total_duration_ns": self.total_duration_ns,
        }


@dataclass
class _PrefetchRun:
    tl_uid: int
    issue_pos: int
    wait_pos: int
    size: int
    dur: int
    h2d_start: int = 0
    h2d_end: int = 0


@dataclass
class _EvictRun:
    tl_uid: int
    issue_pos: int
    size: int
    dur: int
    d2h_start: int = 0
    d2h_end: int = 0


def _map_sched_to_tl(
    tl: UnifiedTimeline, schedule: NeutralSchedule
) -> dict[int, int]:
    """Map schedule tensor uid → UnifiedTimeline tensor uid via (graph, tid)."""
    tl_key = {(t.graph_id, t.compiled_tensor_id): t.uid for t in tl.tensors}
    out: dict[int, int] = {}
    for st in schedule.tensors:
        key = (st.graph_id, st.compiled_tensor_id)
        if key in tl_key:
            out[st.uid] = tl_key[key]
    return out


def _pos_of(tl: UnifiedTimeline, graph_id: int, launch_id: int) -> int:
    """Return the FIRST task position for (graph_id, launch_id).

    Under replicate_uses=True the per_graph_launch_to_task entries are
    list[int] (one position per iter). Take the first (= iter 0) so the
    analytic FIFO model lines up with the LP's "iter 0 deadline" view.
    Single-iter entries are stored as int directly (legacy).
    """
    hit = tl.per_graph_launch_to_task.get(int(graph_id), {}).get(
        int(launch_id), -1
    )
    if isinstance(hit, list):
        return hit[0] if hit else -1
    return hit


def _run_h2d_fifo(
    tl: UnifiedTimeline,
    schedule: NeutralSchedule,
    hw: HwParams,
    sched_to_tl: dict[int, int],
) -> list[_PrefetchRun]:
    runs: list[_PrefetchRun] = []
    for pf in schedule.prefetches:
        tl_uid = sched_to_tl.get(pf.tensor_uid)
        if tl_uid is None:
            continue
        tensor = tl.tensors[tl_uid]
        # issue_launch_id < 0 → treat as sync at wait_launch_id.
        issue_lid = (
            pf.issue_launch_id if pf.issue_launch_id >= 0 else pf.wait_launch_id
        )
        issue_pos = _pos_of(tl, tensor.graph_id, issue_lid)
        wait_pos = _pos_of(tl, tensor.graph_id, pf.wait_launch_id)
        if issue_pos < 0 or wait_pos < 0:
            continue
        size = tensor.size_bytes
        dur = hw.h2d_latency_ns + int(size / max(hw.h2d_bw, 1e-9))
        runs.append(_PrefetchRun(
            tl_uid=tl_uid, issue_pos=issue_pos, wait_pos=wait_pos,
            size=size, dur=dur,
        ))
    runs.sort(key=lambda r: (tl.tasks[r.issue_pos].start_ns, r.issue_pos))
    lane_free = 0
    for r in runs:
        earliest = tl.tasks[r.issue_pos].start_ns
        r.h2d_start = max(lane_free, earliest)
        r.h2d_end = r.h2d_start + r.dur
        lane_free = r.h2d_end
    return runs


def _run_d2h_fifo(
    tl: UnifiedTimeline,
    schedule: NeutralSchedule,
    hw: HwParams,
    sched_to_tl: dict[int, int],
) -> list[_EvictRun]:
    runs: list[_EvictRun] = []
    for ev in schedule.evicts:
        tl_uid = sched_to_tl.get(ev.tensor_uid)
        if tl_uid is None:
            continue
        tensor = tl.tensors[tl_uid]
        issue_pos = _pos_of(tl, tensor.graph_id, ev.issue_launch_id)
        if issue_pos < 0:
            continue
        size = tensor.size_bytes
        dur = hw.d2h_latency_ns + int(size / max(hw.d2h_bw, 1e-9))
        runs.append(_EvictRun(
            tl_uid=tl_uid, issue_pos=issue_pos, size=size, dur=dur,
        ))
    runs.sort(key=lambda r: (tl.tasks[r.issue_pos].end_ns, r.issue_pos))
    lane_free = 0
    for r in runs:
        earliest = tl.tasks[r.issue_pos].end_ns
        r.d2h_start = max(lane_free, earliest)
        r.d2h_end = r.d2h_start + r.dur
        lane_free = r.d2h_end
    return runs


def _sweep_peak(events: list[tuple[int, int]]) -> int:
    if not events:
        return 0
    events.sort()
    cur = 0
    peak = 0
    for _t, delta in events:
        cur += delta
        if cur > peak:
            peak = cur
    return peak


def compute_stall_metrics(
    tl: UnifiedTimeline,
    schedule: NeutralSchedule,
    hw: HwParams,
    *,
    iter_ns_override: int | None = None,
) -> StallMetrics:
    """Simulate the schedule on independent H2D/D2H FIFOs; report metrics.

    ``iter_ns_override`` — use for PCIe budget / lane util instead of
    ``tl.total_duration_ns``. The unified timeline sums GPU kernel compute
    only; real wall-time also includes CPU dispatch gaps. Pass the real
    value (e.g. from ``ct_belady_pcie``'s ``wall_time_iter_ns``) for
    accurate lane-util and overcommit reporting.
    """
    iter_ns = max(
        iter_ns_override if iter_ns_override else int(tl.total_duration_ns),
        1,
    )
    if not tl.tasks:
        return StallMetrics(
            total_stall_ns=0, max_stall_ns=0, stalled_kernel_count=0,
            pipeline_stretch_ns=0,
            peak_vram_bytes=0, peak_vram_mb=0.0, pending_evict_bytes_peak=0,
            h2d_lane_busy_ns=0, d2h_lane_busy_ns=0,
            h2d_lane_util_pct=0.0, d2h_lane_util_pct=0.0,
            h2d_bytes_total=0, d2h_bytes_total=0,
            h2d_overcommit=False, d2h_overcommit=False,
            missing_tensor_count=0, total_duration_ns=iter_ns,
        )

    sched_to_tl = _map_sched_to_tl(tl, schedule)
    cold_tl_uids: set[int] = set()
    for cs in schedule.cold_starts:
        tl_uid = sched_to_tl.get(cs.tensor_uid)
        if tl_uid is not None:
            cold_tl_uids.add(tl_uid)

    pf_runs = _run_h2d_fifo(tl, schedule, hw, sched_to_tl)
    ev_runs = _run_d2h_fifo(tl, schedule, hw, sched_to_tl)

    # --- Stall per kernel (uncascaded; prefetches anchored by wait_pos) ---
    pf_by_wait: dict[int, list[_PrefetchRun]] = {}
    for r in pf_runs:
        pf_by_wait.setdefault(r.wait_pos, []).append(r)

    total_stall = 0
    max_stall = 0
    stalled_count = 0
    # pipeline_stretch cascades each kernel's delay forward so the caller
    # sees the bottom-line iter inflation, not the sum of per-kernel tail
    # lateness (which double-counts cascaded delays).
    last_actual_end = 0
    for task in tl.tasks:
        runs = pf_by_wait.get(task.global_pos)
        required_ready = max(r.h2d_end for r in runs) if runs else 0
        actual_start = max(task.start_ns, required_ready, last_actual_end)
        stall = actual_start - task.start_ns
        if stall > 0:
            total_stall += stall
            stalled_count += 1
            if stall > max_stall:
                max_stall = stall
        last_actual_end = actual_start + task.duration_ns
    pipeline_stretch = max(0, last_actual_end - tl.tasks[-1].end_ns)

    # --- VRAM residency sweep ---
    # Pair each prefetch/cold-start (supply) with the next evict (consume) in
    # time order. Supply adds size at h2d_start (or 0 for cold-start);
    # consume removes size at d2h_end of the matched evict (or never).
    pf_by_uid: dict[int, list[_PrefetchRun]] = {}
    for r in pf_runs:
        pf_by_uid.setdefault(r.tl_uid, []).append(r)
    ev_by_uid: dict[int, list[_EvictRun]] = {}
    for r in ev_runs:
        ev_by_uid.setdefault(r.tl_uid, []).append(r)

    vram_events: list[tuple[int, int]] = []
    pending_events: list[tuple[int, int]] = []
    for uid in range(len(tl.tensors)):
        size = tl.tensors[uid].size_bytes
        pfs = sorted(pf_by_uid.get(uid, []), key=lambda r: r.h2d_start)
        evs = sorted(ev_by_uid.get(uid, []), key=lambda r: r.d2h_start)
        ev_idx = 0
        if uid in cold_tl_uids:
            end = evs[ev_idx].d2h_end if ev_idx < len(evs) else iter_ns
            vram_events.append((0, +size))
            vram_events.append((end, -size))
            if ev_idx < len(evs):
                ev = evs[ev_idx]
                pending_events.append((ev.d2h_start, +size))
                pending_events.append((ev.d2h_end, -size))
                ev_idx += 1
        for r in pfs:
            end = evs[ev_idx].d2h_end if ev_idx < len(evs) else iter_ns
            vram_events.append((r.h2d_start, +size))
            vram_events.append((end, -size))
            if ev_idx < len(evs):
                ev = evs[ev_idx]
                pending_events.append((ev.d2h_start, +size))
                pending_events.append((ev.d2h_end, -size))
                ev_idx += 1

    peak_vram = _sweep_peak(vram_events)
    pending_peak = _sweep_peak(pending_events)

    # --- Missing tensor check (position-based, not time-based) ---
    #
    # A prefetch with wait_pos ≤ K supplies the tensor by K (late arrival is
    # counted as stall above, not as "missing"). An evict with issue_pos < K
    # has already consumed the resident region by K. Cold-start adds 1 to the
    # initial residency. If supply − consume ≤ 0 at K, the schedule is broken.
    missing = 0
    for task in tl.tasks:
        for uid in task.used_tensors:
            supply = 1 if uid in cold_tl_uids else 0
            supply += sum(
                1 for r in pf_by_uid.get(uid, [])
                if r.wait_pos <= task.global_pos
            )
            consume = sum(
                1 for r in ev_by_uid.get(uid, [])
                if r.issue_pos < task.global_pos
            )
            if supply - consume <= 0:
                missing += 1

    # --- Lane utilisation + budget overcommit ---
    h2d_busy = sum(r.dur for r in pf_runs)
    d2h_busy = sum(r.dur for r in ev_runs)
    h2d_bytes = sum(r.size for r in pf_runs)
    d2h_bytes = sum(r.size for r in ev_runs)
    h2d_budget_bytes = (
        int(hw.h2d_bw * iter_ns) if hw.h2d_bw > 0 else (1 << 62)
    )
    d2h_budget_bytes = (
        int(hw.d2h_bw * iter_ns) if hw.d2h_bw > 0 else (1 << 62)
    )

    return StallMetrics(
        total_stall_ns=int(total_stall),
        max_stall_ns=int(max_stall),
        stalled_kernel_count=int(stalled_count),
        pipeline_stretch_ns=int(pipeline_stretch),
        peak_vram_bytes=int(peak_vram),
        peak_vram_mb=round(peak_vram / 1e6, 2),
        pending_evict_bytes_peak=int(pending_peak),
        h2d_lane_busy_ns=int(h2d_busy),
        d2h_lane_busy_ns=int(d2h_busy),
        h2d_lane_util_pct=round(100.0 * h2d_busy / iter_ns, 2),
        d2h_lane_util_pct=round(100.0 * d2h_busy / iter_ns, 2),
        h2d_bytes_total=int(h2d_bytes),
        d2h_bytes_total=int(d2h_bytes),
        h2d_overcommit=h2d_bytes > h2d_budget_bytes,
        d2h_overcommit=d2h_bytes > d2h_budget_bytes,
        missing_tensor_count=int(missing),
        total_duration_ns=iter_ns,
    )
