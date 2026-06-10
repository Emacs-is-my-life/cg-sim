"""SwapAdvisor realized as an ONLINE RUNTIME policy.

SwapAdvisor (Huang et al., ASPLOS'20) plans swapping OFFLINE to maximize
compute/communication overlap: it proactively prefetches swapped-out tensors
*as early as possible* and, under memory pressure, swaps out the resident
tensor whose *next use is furthest in the future* (Belady). This is NOT a
demand-driven policy — on-demand swapping is precisely the ODSwap baseline the
paper beats by up to 80x; proactive overlap is the central mechanism. The
planner reasons over the MXNet allocator's physical memory objects; its
residency unit is the **physical storage object**, not a logical tensor id.

Porting that as an offline per-tensor schedule into cg-sim's graph_modifiers
mis-modeled the residency unit: the unified timeline represents each physical
weight as several logical tids (compile-side sidecar + synthesized trace
duplicates), so the planner saw ~2x the true weight mass (llama8b: 30 vs 15
GiB), manufacturing spurious budget pressure -> chronic over-streaming. And the
per-tid schedule had to be gated per-tid by the injector, which deadlocked once
storage was coalesced for correct accounting.

This runtime sidesteps both problems by reasoning over the simulator's **real
physical storage** (the page allocator already coalesces aliases by storage),
exactly as SwapAdvisor reasons over physical memory objects. It subclasses
``DeviceAwareVanillaAsync`` and adds the two things DAV lacks — proactive
prefetch and online eviction:

  * weights are RAM-homed (streamed), so they start ABSENT in VRAM;
  * proactive prefetch (``_proactive_prefetch``, ON by default) pulls the
    weights of upcoming nodes into spare VRAM ahead of the execution frontier,
    overlapping the RAM->VRAM copy with compute — SwapAdvisor's central
    mechanism. Demand swap-in (DAV's inherited ``_ensure_inputs_resident``) is
    the safety net for anything prefetch didn't cover in time. NB: the timing
    is approximate vs the paper, which fires each prefetch at the binding
    victim's last-use (victim-paired ASAP); this online form fires
    opportunistically into currently-free space over a bounded horizon;
  * when a VRAM claim can't fit (real memory pressure), instead of aborting we
    evict the resident streamable storage with the *furthest next use*
    (Belady), freeing VRAM-only and keeping the RAM mirror so the next swap-in
    is a cheap RAM->VRAM copy;
  * full-trace lookahead (the sim has the whole trace) gives the exact next-use
    ordering Belady needs — faithful to "Belady is given the schedule".

What is NOT ported (and why): SwapAdvisor's defining contribution — a genetic
algorithm jointly searching operator schedule x memory-pool allocation (§5) —
is structurally out of scope here: Inductor fixes the launch order at compile
and there is no MXNet-style size-class pool to configure. So this is faithful
to SwapAdvisor's swap-PLANNER intent (Belady + overlap on physical storage),
not the SwapAdvisor SYSTEM. The same-size-class victim restriction (§4.1) is an
opt-in ablation (SWAPRT_SIZE_CLASS=1), OFF by default: it is a limitation forced
by MXNet's fixed-object pool, not an optimization (global Belady is a strict
victim superset, never slower on makespan), and it triggers heavy re-streaming
churn that makes the sim >10x slower here even after eviction was optimized.

Budget = the VRAM hardware size in the input.yaml (``vram0.memory_size_KB``);
no separate knob. The input.yaml must carry NO injected schedule, so DAV's
injector hooks stay inert and only this online policy acts.

Env knobs (debug / ablation):
  SWAPRT_NO_EVICT=1   — disable Belady eviction (skeleton test: should abort at
                        the VRAM claim under pressure, locating the hook).
  SWAPRT_HEADROOM_FRAC=F — static activation headroom the cold seed reserves,
                        as a fraction of global-peak activation (default 0.0 =
                        seed every weight that fits the budget; pass-2 repair is
                        runtime Belady). Raise toward 1.0 for a more
                        conservative seed that reserves room up front.
  SWAPRT_PREFETCH=0   — disable proactive async prefetch (overlap). Prefetch is
                        ON by default, faithful to SwapAdvisor's "prefetch as
                        early as possible" overlap (§4.1/§4.2); set to 0 to fall
                        back to pure demand swap-in for ablation.
  SWAPRT_SIZE_CLASS=1 — enable the same-size-class victim restriction (§4.1
                        fixed-size-object pool) with a global-Belady fallback;
                        the ``size_class_tol`` scheduler arg coarsens classes
                        (adjacent sizes within the ratio merge; 0.0 -> exact).
                        OFF by default — paper-literal but never faster on
                        makespan and >10x slower to simulate (re-streaming
                        churn). Opt-in ablation only.
"""

from __future__ import annotations

import os

from sim.core.job import BaseJob
from sim.core.trace import NodeStatus, Tensor, Trace
from sim.hw.common import DataRegionAccess
from sim.hw.memory.common import BaseMemory
from sim.sched.device_aware_vanilla_async import DeviceAwareVanillaAsync
from sim.sched.device_aware_vanilla_async.device_aware_vanilla_async import (
    _ABSENT, _LOADED, _LOADING, _RESIDENT,
)


class SwapAdvisorRuntime(DeviceAwareVanillaAsync):
    """Online SwapAdvisor: proactive prefetch + Belady eviction on physical storage."""

    _STREAMABLE_TYPES = frozenset({"WEIGHT", "LEAF"})

    def __init__(self, obj_id, name, log, sys, args=None):
        super().__init__(obj_id, name, log, sys, args)
        self._belady_evict = os.environ.get("SWAPRT_NO_EVICT") != "1"
        self._seed_initial = os.environ.get("SWAPRT_NO_INITIAL") != "1"
        # Proactive async prefetch (SwapAdvisor's overlap, §4.1 3-stream model)
        # is ON by default — this is SwapAdvisor's central mechanism: "prefetch
        # previously-swapped-out tensors as early as possible" to overlap the
        # RAM->VRAM transfer with current compute instead of stalling consumers
        # on demand (§4.2). Disable with SWAPRT_PREFETCH=0 to ablate down to
        # pure demand swap-in. Caveat for bandwidth-bound regimes: where the
        # H2D stream is already saturated (tight budgets) overlap cannot cut
        # total bytes, and budget-bounded cold-residency can leave little free
        # space to prefetch into; in those regimes prefetch is roughly neutral
        # (the win is from residency — streaming less volume — not overlap).
        self._prefetch_on = os.environ.get("SWAPRT_PREFETCH") != "0"
        self._pf_per_tick = int(self.args.get("prefetch_per_tick", 32))
        # how many upcoming nodes to prefetch weights for (the overlap horizon)
        self._lookahead = int(self.args.get("prefetch_lookahead_nodes", 128))
        self._exec_ptr = 0  # frontier index into _sched_nodes (monotonic)
        # tid -> sorted [(start_ns, node_id)] of all consumers (built in compile)
        self._consumer_starts: dict[int, list[tuple[int, int]]] = {}
        # all streamable weight tids (prefetch candidate pool, built in compile)
        self._streamable_tids: list[int] = []
        # amortized next-use cursor: index into _consumer_starts[tid] of the
        # first possibly-not-DONE consumer (only ever advances forward).
        self._nu_cursor: dict[int, int] = {}
        # weights cold-resident from layout (VRAM-homed, up to budget); the rest
        # are RAM-homed and demand-streamed.
        self._cold_tids: set[int] = set()
        # online stats
        self._sa_evictions = 0
        self._sa_swap_ins = 0  # reserved for proactive prefetch (v2)
        # SwapAdvisor size-class-restricted victim pool (§4.1/§4.2), opt-in via
        # SWAPRT_SIZE_CLASS=1. Mirrors MXNet's fixed-size-object pool: a size-s
        # claim may only reuse a freed object from s's size-class, with a
        # global-Belady fallback when the class is exhausted. OFF by default:
        # it is an MXNet-pool limitation, not an optimization (global Belady is a
        # strict victim superset, never slower on makespan), AND it triggers
        # heavy re-streaming churn — even with the batched-eviction optimization
        # it ran >10x slower (>570s vs 30s on sdxl@1.6G; >1hr/cell in the sweep)
        # because the restriction evicts soon-needed weights. Kept as an opt-in
        # ablation for paper-literal runs; not viable as a default here.
        self._size_class_on = os.environ.get("SWAPRT_SIZE_CLASS") == "1"
        # adjacent observed sizes within this ratio merge into one class
        # (0.0 -> one class per distinct page-count).
        self._size_class_tol = float(self.args.get("size_class_tol", 0.0))
        self._size_class: dict[int, int] = {}  # tid -> class id
        self._sa_sizeclass_fallbacks = 0  # picks that fell to global Belady
        # Initial-resident headroom scale — SwapAdvisor's two-pass (§4.2)
        # realized for the online setting. The cold seed reserves
        # (global-peak activation x frac) pages for activations and fills the
        # rest with the soonest-first-use weights. Default 0.0 = the two-pass
        # intent: PASS 1 seeds every weight that fits the budget (the cyclic
        # Belady-optimal t=0 set is exactly soonest-first-use, since the bulk
        # weight mass is re-used K>=3x per trace); PASS-2 repair — dropping a
        # seeded weight when a later activation spike causes pressure — is done
        # DYNAMICALLY by runtime Belady (cold weights keep a RAM mirror, so the
        # eviction is free), not as a redundant compile-time fixpoint. Measured
        # -3..-5% e2e on tight diffusion caps, neutral on llama (activations
        # ~0). frac>0 reserves static headroom instead (more conservative).
        self._headroom_frac = float(os.environ.get("SWAPRT_HEADROOM_FRAC", "0.0"))

    # ----------------------------------------------------------- compile
    def compile(self, trace: Trace) -> None:
        super().compile(trace)
        # Next-use lookahead index: per-tid consumer start_ns (profiler wall
        # clock), sorted ascending. Belady picks the resident tid whose first
        # not-yet-DONE consumer is furthest out.
        starts: dict[int, list[tuple[int, int]]] = {}
        for nid, node in trace.node_map.items():
            s = int(node.args.get("start_ns", nid))
            for tid in (node.input_tensors or []):
                starts.setdefault(int(tid), []).append((s, int(nid)))
        for tid in starts:
            starts[tid].sort()
        self._consumer_starts = starts
        # Make every streamable weight evictable, so the inherited
        # permanent-type guard does not block online eviction/streaming.
        self._streamable_tids = [
            int(tid) for tid, t in trace.tensor_map.items()
            if t.args.get("tensor_type") in self._STREAMABLE_TYPES
        ]
        self._streamable_set = set(self._streamable_tids)
        for tid in self._streamable_tids:
            self._evictable_tensor_ids.add(tid)
        if self._size_class_on:
            self._size_class = self._build_size_classes(trace)
        # GPU/compute nodes in execution (start_ns) order — the prefetch
        # horizon walks this list ahead of the execution frontier.
        self._sched_nodes = sorted(
            (int(n.args.get("start_ns", nid)), int(nid))
            for nid, n in trace.node_map.items()
        )

        # --- SwapAdvisor two-pass initial residency (§4.2), realized online.
        # PASS 1: cold-resident the soonest-first-use weights up to the budget
        # (minus an optional static headroom; see _headroom_frac). PASS-2 repair
        # is dynamic — runtime Belady evicts a cold weight if/when an activation
        # spike actually needs the room (cold weights keep a DRAM mirror, so the
        # eviction is free). So over-seeding is self-correcting and no
        # compile-time fixpoint is needed: the online Belady IS pass 2.
        if self._seed_initial:
            self._cold_tids = self._select_cold_residents(trace, starts)

    def _select_cold_residents(self, trace: Trace, starts) -> set[int]:
        """Pass 1 of SwapAdvisor's two-pass initial residency: the soonest-
        first-use weights that fit (budget - static headroom). Pass-2 repair is
        runtime Belady (see compile()). Returns the cold-resident weight tids."""
        budget_pages = self._vram.space.num_total_pages
        # First-use (start_ns) per tensor, and producer start per tensor.
        first_use: dict[int, int] = {}
        producer_start: dict[int, int] = {}
        for nid, node in trace.node_map.items():
            s = int(node.args.get("start_ns", nid))
            for tid in (node.input_tensors or []):
                tid = int(tid)
                if tid not in first_use or s < first_use[tid]:
                    first_use[tid] = s
            for tid in (node.output_tensors or []):
                tid = int(tid)
                if tid not in producer_start or s < producer_start[tid]:
                    producer_start[tid] = s
        # Activation headroom = peak concurrent INTERMEDIATE footprint (pages),
        # estimated from producer..last-consumer lifetimes in profiler time.
        events: list[tuple[int, int]] = []
        for tid, t in trace.tensor_map.items():
            if t.args.get("tensor_type") != "INTERMEDIATE":
                continue
            ps = producer_start.get(int(tid))
            cs = starts.get(int(tid))
            if ps is None or not cs:
                continue
            events.append((ps, t.num_pages))
            events.append((cs[-1][0] + 1, -t.num_pages))
        events.sort()
        cur = peak = 0
        for _, d in events:
            cur += d
            peak = max(peak, cur)
        headroom_pages = int(peak * self._headroom_frac)
        cold_target = max(0, budget_pages - headroom_pages)
        # Cold-resident streamable weights in first-use order until the target
        # is reached (earliest-needed weights resident first -> no startup stall).
        weights = sorted(
            (first_use.get(int(tid), 1 << 62), int(tid), t.num_pages)
            for tid, t in trace.tensor_map.items()
            if t.args.get("tensor_type") in self._STREAMABLE_TYPES
        )
        cold: set[int] = set()
        used = 0
        for _, tid, pg in weights:
            if used + pg <= cold_target:
                cold.add(tid)
                used += pg
        print(
            f"[swapadvisor_runtime] budget={budget_pages * 4 / 1024:.0f}MiB "
            f"act-headroom={headroom_pages * 4 / 1024:.0f}MiB cold-resident "
            f"{len(cold)}/{len(weights)} weights ({used * 4 / 1024:.0f}MiB), "
            f"{len(weights) - len(cold)} streamed",
            flush=True,
        )
        return cold

    def _build_size_classes(self, trace: Trace) -> dict[int, int]:
        """Map each streamable tid -> a size-class id, mirroring SwapAdvisor's
        fixed-size-object memory pool (§4.1): a size-s allocation may only reuse
        a freed object of the same size-class. Classes are the sorted set of
        observed streamable page-counts; adjacent sizes within
        ``size_class_tol`` merge into one class (tol=0 -> exact-size classes)."""
        tmap = trace.tensor_map
        sizes = sorted({tmap[tid].num_pages for tid in self._streamable_tids})
        size_to_class: dict[int, int] = {}
        cls = -1
        base = None
        for s in sizes:
            if base is None or s > base * (1.0 + self._size_class_tol):
                cls += 1
                base = s
            size_to_class[s] = cls
        return {
            tid: size_to_class[tmap[tid].num_pages]
            for tid in self._streamable_tids
        }

    # ----------------------------------------------------------- homing
    def _memory_for_tensor(self, tensor: Tensor) -> BaseMemory:
        # Stream weights: RAM-home them so they start ABSENT in VRAM and are
        # demand-streamed + Belady-evicted at runtime, instead of being
        # cold-resident from (free) layout. (Budget-bounded initial residency
        # is layered on top in a later pass.)
        if tensor.args.get("tensor_type") in self._STREAMABLE_TYPES:
            if int(tensor.id) in self._cold_tids:
                # cold-resident: VRAM home + DRAM staging (inherited CUDA path),
                # so it's loaded free at layout and still evictable at runtime.
                return self.memory_by_device[self.cuda_device]
            return self.memory_by_device["cpu"]
        return super()._memory_for_tensor(tensor)

    # ----------------------------------------------------------- eviction
    def _next_use(self, tid: int) -> float:
        """Start_ns of this tid's first not-yet-DONE consumer; +inf if none
        remain (the tid is dead -> ideal Belady victim). Amortized O(1) via a
        forward-only cursor over the (ascending) consumer list."""
        cs = self._consumer_starts.get(tid)
        if not cs:
            return float("inf")
        node_map = self.sys.trace.node_map
        i = self._nu_cursor.get(tid, 0)
        while i < len(cs):
            node = node_map.get(cs[i][1])
            if node is None or node.status == NodeStatus.DONE:
                i += 1
            else:
                break
        self._nu_cursor[tid] = i
        return float(cs[i][0]) if i < len(cs) else float("inf")

    def _evict_until_fits(self, memory: BaseMemory, incoming: Tensor) -> None:
        """Evict furthest-next-use victims (immediate VRAM-only free, RAM mirror
        preserved) until ``incoming`` fits or nothing more is evictable.

        Belady (§4.2): evict the resident storage whose next use is furthest in
        the future. Only IDLE regions are eligible (a BEING_READ/WRITTEN region
        can't be freed this tick). With size-class ON (§4.1 fixed-size pool),
        same-class victims are tried first, then a global-Belady fallback (a
        page allocator isn't a fixed-object pool, so a class can drain before
        ``incoming`` fits; the fallback keeps the budget-safe / abort-free
        invariant).

        The candidate set is gathered and ordered in ONE scan per claim (and
        ``_next_use`` is stable within a tick), instead of rescanning every
        resident region once per eviction — the latter is quadratic when a
        single claim needs many evictions (size-class + fragmentation), which
        dominated wall-clock at packed budgets."""
        need = incoming.num_pages
        if self._find_free_page(memory, need) is not None:
            return
        sc = self._size_class.get(incoming.id) if self._size_class_on else None
        # ONE scan: collect IDLE evictable victims with next-use, split by class.
        same: list[tuple[float, int]] = []
        other: list[tuple[float, int]] = []
        for region in memory.space._regions_by_page_idx_start.values():
            tid = getattr(region, "tensor_id", None)
            if tid is None or tid == incoming.id:
                continue
            if tid not in self._evictable_tensor_ids:
                continue
            if region.access_status != DataRegionAccess.IDLE:
                continue
            nu = self._next_use(tid)
            if sc is not None and self._size_class.get(tid) == sc:
                same.append((nu, tid))
            else:
                other.append((nu, tid))
        # Furthest next use first; same-class tier ahead of the global fallback.
        same.sort(reverse=True)
        other.sort(reverse=True)
        order = same + other  # (same is empty when size-class is OFF)
        n_same = len(same)
        for i, (_nu, tid) in enumerate(order):
            if self._find_free_page(memory, need) is not None:
                break  # fits now -> stop evicting
            if i >= n_same and sc is not None:
                self._sa_sizeclass_fallbacks += 1
            self._release_vram_only(tid)
            self._sa_evictions += 1
        # If still no contiguous hole, the caller's claim aborts (visible
        # failure) — every IDLE evictable victim has been freed.

    def _claim_region(self, memory: BaseMemory, tensor: Tensor):
        # Belady eviction only on the GPU under real pressure; other memories
        # (RAM staging) keep the inherited behavior.
        if self._belady_evict and memory is self._vram:
            if self._find_free_page(memory, tensor.num_pages) is None:
                self._evict_until_fits(memory, tensor)
        return super()._claim_region(memory, tensor)

    # ----------------------------------------------------------- prefetch
    def runtime(self, retired_jobs: list[BaseJob]) -> None:
        super().runtime(retired_jobs)
        if self._prefetch_on:
            self._proactive_prefetch()

    def _proactive_prefetch(self) -> None:
        """SwapAdvisor's "prefetch as early as possible" overlap, realized
        online with a bounded horizon: prefetch the absent weights consumed by
        the next ``_lookahead`` upcoming nodes into spare VRAM, so their
        RAM->VRAM transfer overlaps current compute instead of stalling the
        consumer on demand. Bounded by the horizon (no far-future thrash) and
        by free contiguous space (never evicts to prefetch — displacing a
        resident weight is Belady's job on the demand path)."""
        vram = self._vram
        if vram.space.num_used_pages >= vram.space.num_total_pages:
            return  # full -> nothing to prefetch into (cheap early-out)
        nm = self.sys.trace.node_map
        sn = self._sched_nodes
        # Advance the execution frontier past completed nodes (monotonic).
        p = self._exec_ptr
        while p < len(sn) and nm[sn[p][1]].status == NodeStatus.DONE:
            p += 1
        self._exec_ptr = p
        # Walk the next _lookahead nodes; prefetch their absent weight inputs in
        # execution order (soonest-needed first).
        issued = 0
        seen: set[int] = set()
        for k in range(p, min(p + self._lookahead, len(sn))):
            node = nm.get(sn[k][1])
            if node is None:
                continue
            for tid in (node.input_tensors or []):
                tid = int(tid)
                if tid in seen or tid not in self._streamable_set:
                    continue
                seen.add(tid)
                if self._xfer_state.get(tid, _ABSENT) in (_LOADING, _LOADED, _RESIDENT):
                    continue
                if vram.space.get_by_tensor_id(tid):
                    continue
                t = self.sys.trace.tensor_map.get(tid)
                if t is None:
                    continue
                if self._find_free_page(vram, t.num_pages) is None:
                    return  # no free contiguous room -> stop (don't evict to prefetch)
                if self._issue_prefetch([tid]):
                    issued += 1
                    self._sa_swap_ins += 1
                    if issued >= self._pf_per_tick:
                        return

    # ----------------------------------------------------------- logging
    def log_counters(self):
        base = super().log_counters() or {}
        base.update({
            "swapadvisor_runtime_evictions": self._sa_evictions,
            "swapadvisor_runtime_prefetches": self._sa_swap_ins,
            "swapadvisor_runtime_sizeclass_fallbacks": self._sa_sizeclass_fallbacks,
        })
        return base
