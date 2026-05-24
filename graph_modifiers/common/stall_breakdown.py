"""Stall breakdown diagnostic for ``StreamedProfile``.

Simulates the baseline iteration with *explicit* lanes (compute / cpu /
h2d), and reports per-launch waits. The point is to prove (or disprove)
that a lane-aware scheduler can recover stall that the 1D model invents.

Per launch L, four components are computed:

* ``cpu_dispatch_wait[L]`` — CPU lane was busy dispatching prior launches,
  so this launch's ``cudaLaunchKernel`` could not start at its gpu_start
  arrival time. Only nonzero when the CPU lane is the binding constraint.

* ``compute_wait_for_cpu[L]`` — the compute stream was idle waiting for
  ``cpu_dispatch_end`` of this launch (kernel queued late).

* ``compute_wait_for_h2d[L]`` — the compute stream was idle waiting for a
  weight H2D that this launch depends on. In the baseline profile there is
  no weight streaming, so this is always 0; the diagnostic reports it so
  the same metric can be computed after applying a schedule.

* ``h2d_idle_during_compute[L]`` — time the h2d_stream was idle while the
  compute stream was busy with L's kernel. This is the *recoverable*
  bandwidth a scheduler can spend on async prefetches. Summed across all
  launches, this is the upper bound on VRAM reduction via async H2D under
  the current profile.

The tool deliberately does NOT require a schedule — it characterizes the
baseline profile so we can decide whether a Level 2 solver is worth
building for this workload.
"""

from __future__ import annotations

from dataclasses import dataclass

from .streamed_profile import StreamedProfile


@dataclass
class StallSummary:
    iter_wall_ns: int                    # raw Kineto wall
    iter_wall_calibrated_ns: int         # after profiler-overhead subtraction
    gpu_busy_ns: int
    gpu_idle_ns: int
    cpu_dispatch_ns: int
    profiler_overhead_per_event_ns: int

    cpu_dispatch_wait_total_ns: int
    compute_wait_for_cpu_total_ns: int
    compute_wait_for_h2d_total_ns: int
    h2d_idle_during_compute_ns: int

    max_bandwidth_recoverable_bytes: int
    max_bandwidth_full_iter_bytes: int   # at calibrated iter wall
    launches_with_valid_cpu_pair: int
    total_launches: int

    def pretty(self, h2d_bw_Bps: float) -> str:
        wall = max(self.iter_wall_ns, 1)
        cal = max(self.iter_wall_calibrated_ns, 1)
        util = 100.0 * self.gpu_busy_ns / wall
        idle = 100.0 * self.gpu_idle_ns / wall
        cpu_pct = 100.0 * self.cpu_dispatch_ns / wall
        cpu_wait = 100.0 * self.cpu_dispatch_wait_total_ns / wall
        cw_cpu = 100.0 * self.compute_wait_for_cpu_total_ns / wall
        cw_h2d = 100.0 * self.compute_wait_for_h2d_total_ns / wall
        h2d_slack_pct = 100.0 * self.h2d_idle_during_compute_ns / wall
        h2d_slack_mb = self.max_bandwidth_recoverable_bytes / 1e6
        full_iter_mb = self.max_bandwidth_full_iter_bytes / 1e6
        lines = [
            "=== Stall Breakdown (StreamedProfile) ===",
            f"  Iter wall (Kineto, raw):     {wall/1e6:.3f} ms",
            f"  Iter wall (calibrated):      {cal/1e6:.3f} ms "
            f"(overhead={self.profiler_overhead_per_event_ns} ns/event)",
            f"  GPU busy:                    {self.gpu_busy_ns/1e6:.3f} ms ({util:.1f}% of raw)",
            f"  GPU idle:                    {self.gpu_idle_ns/1e6:.3f} ms ({idle:.1f}% of raw)",
            f"  CPU dispatch busy:           {self.cpu_dispatch_ns/1e6:.3f} ms ({cpu_pct:.1f}% of raw)",
            "",
            "  --- stall components (raw profile) ---",
            f"  CPU dispatch queued-wait:    {self.cpu_dispatch_wait_total_ns/1e6:.3f} ms ({cpu_wait:.1f}%)",
            f"  Compute waited for CPU:      {self.compute_wait_for_cpu_total_ns/1e6:.3f} ms ({cw_cpu:.1f}%)",
            f"  Compute waited for H2D:      {self.compute_wait_for_h2d_total_ns/1e6:.3f} ms ({cw_h2d:.1f}%)",
            "",
            "  --- recoverable bandwidth ---",
            f"  H2D idle during compute:     {self.h2d_idle_during_compute_ns/1e6:.3f} ms ({h2d_slack_pct:.1f}%)",
            f"  @ {h2d_bw_Bps/1e9:.2f} GB/s during compute-busy only:  {h2d_slack_mb:.2f} MB/iter",
            f"  @ {h2d_bw_Bps/1e9:.2f} GB/s across calibrated iter:    {full_iter_mb:.2f} MB/iter",
            "",
            f"  Launches with valid CPU pair: {self.launches_with_valid_cpu_pair}/{self.total_launches}",
        ]
        return "\n".join(lines)


def compute_stall_breakdown(
    sp: StreamedProfile, *, h2d_bw_Bps: float = 25e9,
) -> StallSummary:
    """Walk launches in order; track per-lane occupancy.

    The compute lane's "free time" starts at each launch's observed
    ``gpu_exec_end_ns``; the CPU lane's at each launch's ``cpu_dispatch_end_ns``.
    Idle intervals on the h2d lane during compute-busy windows are tallied
    as recoverable bandwidth.
    """
    if not sp.launches:
        return StallSummary(
            iter_wall_ns=0, iter_wall_calibrated_ns=0,
            gpu_busy_ns=0, gpu_idle_ns=0, cpu_dispatch_ns=0,
            profiler_overhead_per_event_ns=0,
            cpu_dispatch_wait_total_ns=0, compute_wait_for_cpu_total_ns=0,
            compute_wait_for_h2d_total_ns=0, h2d_idle_during_compute_ns=0,
            max_bandwidth_recoverable_bytes=0, max_bandwidth_full_iter_bytes=0,
            launches_with_valid_cpu_pair=0, total_launches=0,
        )

    iter_wall = sp.iter_wall_ns
    gpu_busy = sp.gpu_busy_ns
    gpu_idle = max(0, iter_wall - gpu_busy)

    # Sweep in GPU execution order.
    cpu_lane_free = 0
    compute_lane_free = 0
    cpu_dispatch_wait_total = 0
    cw_cpu_total = 0
    cw_h2d_total = 0  # always 0 for baseline (no h2d deps in profile)
    h2d_idle_during_compute_total = 0
    valid_pairs = 0

    # iter_base is the earliest GPU start — subtract to normalize.
    iter_base = sp.launches[0].gpu_exec_start_ns
    prev_compute_end = 0  # relative time; compute lane idle until first launch

    for L in sp.launches:
        gs = L.gpu_exec_start_ns - iter_base
        ge = L.gpu_exec_end_ns - iter_base

        if L.cpu_dispatch_valid:
            valid_pairs += 1
            cs = L.cpu_dispatch_start_ns - iter_base
            ce = L.cpu_dispatch_end_ns - iter_base
            # CPU lane queued wait: if the CPU wanted to start at cs but the
            # lane was still busy from a prior dispatch.
            cpu_queue = max(0, cpu_lane_free - cs)
            cpu_dispatch_wait_total += cpu_queue
            # Advance the lane.
            effective_cpu_end = max(ce, cpu_lane_free + (ce - cs))
            cpu_lane_free = max(cpu_lane_free, ce)
            # Compute waited for CPU dispatch to finish.
            if ce > gs:
                # Kernel queued only after ce, so compute could not start at
                # gs (this is a symptom — usually a Python-side stall).
                cw_cpu_total += min(ce - gs, ge - gs)
            else:
                # Dispatch landed before GPU start. Compute idle interval
                # (gs - max(prev_compute_end, ce)) is general GPU idle, not
                # a CPU stall. We do not count it here to avoid double-
                # attribution with gpu_idle_ns.
                pass

        # h2d lane idle during this launch's compute busy window = full
        # (ge - gs). Sum conservatively as "time compute was busy AND h2d
        # was available." (h2d lane has no ops in baseline.)
        if ge > gs:
            h2d_idle_during_compute_total += (ge - gs)

        # compute_wait_for_h2d is always 0 in the baseline.
        # Advance compute lane.
        compute_lane_free = max(compute_lane_free, ge)
        prev_compute_end = compute_lane_free

    max_bw_bytes = int(h2d_idle_during_compute_total * h2d_bw_Bps / 1e9)
    calibrated_wall = sp.iter_wall_calibrated_ns or iter_wall
    max_bw_full_iter = int(calibrated_wall * h2d_bw_Bps / 1e9)

    return StallSummary(
        iter_wall_ns=iter_wall,
        iter_wall_calibrated_ns=sp.iter_wall_calibrated_ns,
        gpu_busy_ns=gpu_busy,
        gpu_idle_ns=gpu_idle,
        cpu_dispatch_ns=sp.cpu_dispatch_ns,
        profiler_overhead_per_event_ns=sp.profiler_overhead_per_event_ns,
        cpu_dispatch_wait_total_ns=cpu_dispatch_wait_total,
        compute_wait_for_cpu_total_ns=cw_cpu_total,
        compute_wait_for_h2d_total_ns=cw_h2d_total,
        h2d_idle_during_compute_ns=h2d_idle_during_compute_total,
        max_bandwidth_recoverable_bytes=max_bw_bytes,
        max_bandwidth_full_iter_bytes=max_bw_full_iter,
        launches_with_valid_cpu_pair=valid_pairs,
        total_launches=len(sp.launches),
    )
