"""AccelerateCPUOffload — synchronous cpu_offload scheduler.

A THIN, FAITHFUL executor of the plan that `PytorchOffloadLoader` (with
`offload_reconstruct`) emits. It does not reverse-engineer anything at runtime:
the loader already collapsed the reincarnation soup into 255 RAM masters,
redirected every gpu weight read to its master tid, and tagged each recorded
`Memcpy HtoD` node as an explicit transfer trigger. See `docs/known_problems.md`.

FAITHFUL-TRANSLATION PRINCIPLE (mirror the real-run trace as-is, for accelerate
AND, later, diffusers):
  - compute / CPU nodes -> ComputeJobs replaying their (probe-compensated)
    durations.
  - each recorded tensor transfer -> ONE explicit TransferJob at that node's
    position in the graph. A node tagged `args["offload_transfer"]={master,dir}`
    is a TRANSFER node: when it becomes ready we claim the master's VRAM region
    and fire a same-tid RAM->VRAM TransferJob; the node is marked DONE when that
    job retires. Its gpu consumer (a stream_order child) then passes the engine's
    built-in input-residency check (master now in VRAM) = load-before-use as a
    real dependency. No residency/stream-on-demand heuristic.

Eviction:
  - masters: per the loader's `evict_after_node` schedule (accelerate has no
    recorded free event; the schedule frees a weight's VRAM after the last gpu
    use in each forward, so the next forward re-streams it => recorded 108 GB).
  - intermediates: released when their last consumer retires (refcount liveness),
    so VRAM peak tracks the real working set (~the 788 MB embedding).

Engine constraints honored (see docs/known_problems.md §3): `job_waiting` is FIFO
with head-of-line blocking (we only submit runnable jobs, capped per compute
device per tick); `claim` needs an explicit free page; `TransferJob` needs
`src.tid == dest.tid` (the master is one tid in both RAM and VRAM).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, TYPE_CHECKING

from sim.core.job import BaseJob, ComputeJob, TransferJob
from sim.core.log import Log
from sim.core.trace import Node, NodeStatus, TerminalNode, Trace
from sim.hw.common import DataRegion, DataRegionAccess
from sim.hw.compute.common import BaseCompute
from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage
from sim.sched.common import BaseScheduler

if TYPE_CHECKING:
    from sim.core.system import System


class AccelerateCPUOffload(BaseScheduler):
    _INITIAL_TYPES = frozenset({"WEIGHT", "INPUT", "LEAF"})

    def __init__(
        self,
        obj_id: int,
        name: str,
        log: Log,
        sys: System,
        args: dict[str, Any] | None = None,
    ):
        super().__init__(obj_id, name, log, sys, args)

        # ---- hardware discovery ----
        self.compute_by_name: dict[str, BaseCompute] = {}
        self.memory_by_name: dict[str, BaseMemory] = {}
        self.storage: BaseStorage | None = None
        for hw in sys.hw.values():
            if isinstance(hw, BaseCompute):
                self.compute_by_name[hw.name] = hw
            elif isinstance(hw, BaseMemory):
                self.memory_by_name[hw.name] = hw
            elif isinstance(hw, BaseStorage) and self.storage is None:
                self.storage = hw

        cpu_name = self.args.get("cpu_compute", "cpu")
        cuda_name = self.args.get("cuda_compute", "gpu0")
        self.cuda_device = str(self.args.get("cuda_device", "cuda:0")).lower()
        if cpu_name not in self.compute_by_name:
            raise Exception(f"[{type(self).__name__}] CPU compute '{cpu_name}' does not exist.")
        if cuda_name not in self.compute_by_name:
            raise Exception(f"[{type(self).__name__}] CUDA compute '{cuda_name}' does not exist.")
        self.cpu_compute = self.compute_by_name[cpu_name]
        self.cuda_compute = self.compute_by_name[cuda_name]
        self.ram: BaseMemory = self.cpu_compute.memory
        self.vram: BaseMemory = self.cuda_compute.memory
        self.memory_by_device: dict[str, BaseMemory] = {
            "cpu": self.ram,
            self.cuda_device: self.vram,
        }

        nm = sys.trace.node_map

        # ---- offload contract from the loader ----
        offload = sys.trace.args.get("offload") or {}
        self.master_tids: set[int] = set(offload.get("master_tids", []))
        ev = offload.get("evict_after_node") or {}
        self.evict_after_node: dict[int, list[int]] = {int(k): list(v) for k, v in ev.items()}

        # ---- control-graph readiness (parents DONE + start-gated parents STARTED) ----
        self._pending_parent: dict[int, int] = {nid: len(n.parent_nodes) for nid, n in nm.items()}
        self._started_children: dict[int, list[int]] = defaultdict(list)
        self._pending_started: dict[int, int] = defaultdict(int)
        seen: set[tuple[int, int]] = set()
        for p, c in (sys.trace.args.get("start_gated_edges") or []):
            p, c = int(p), int(c)
            if (p, c) in seen:
                continue
            seen.add((p, c))
            self._started_children[p].append(c)
            self._pending_started[c] += 1

        self.ready: deque[int] = deque(
            nid for nid in nm
            if self._pending_parent[nid] == 0 and self._pending_started.get(nid, 0) == 0
        )

        # ---- intermediate liveness (refcount): release when last consumer retires ----
        self._remaining_consumers: dict[int, int] = defaultdict(int)
        for n in nm.values():
            for tid in n.input_tensors:
                self._remaining_consumers[tid] += 1

        # ---- runtime state ----
        self._submitted: set[int] = set()
        self._transfer_job_node: dict[Any, int] = {}   # transfer job id -> trigger node id
        self._pending_release: set[int] = set()        # tids whose release deferred (region busy)
        self._layout_phase: int = 0

        # ---- stale-duplicate reclamation (double-claim leak) ----
        # A tid produced twice (a dispatcher pre-claim AND a normal producer)
        # can end up with two regions when the first is still busy (BEING_READ)
        # at the second claim: the second write invalidate()s the first, which
        # then leaks as an IDLE + is_latest=False region nobody releases. A
        # stale-IDLE region is DEAD (no ComputeJob consumer can read it — the
        # input check requires is_latest — and the only transfer source-select
        # here requires is_latest too), so reclaiming it is correct, not just
        # space-saving. We register a tid as a suspect at the moment a second
        # region is claimed for it, then sweep only that small set per tick.
        self._release_dups: bool = bool(self.args.get("release_stale_duplicates", True))
        self._suspect_dups: set[int] = set()

        # diagnostics (visible via log_states / MCP)
        self._stat_transfers_fired = 0
        self._stat_masters_evicted = 0
        self._stat_intermediates_freed = 0
        self._stat_stale_dups_freed = 0

        # ---- dead-intermediate GC ----
        # The refcount free (_release_intermediate) only fires when a CONSUMER
        # retires and decrements rem to 0. That misses two classes that the SD3
        # MMDiT trace produces in bulk (~200 MiB by the peak; SDXL/accelerate
        # produce ~none): (a) ORPHANS — intermediates produced with no consumer
        # at all (rem==0 from the start, so no retire ever triggers a release);
        # (b) tids whose region is finalized through a cross-device / view-alias
        # path AFTER rem already hit 0, so the one-shot release found nothing.
        # A periodic sweep reclaims both: any INTERMEDIATE region that is IDLE
        # with rem==0 is dead (no normal consumer needs it — custom_deps
        # consumers bypass residency — and the real allocator frees such buffers
        # at once). Periodic (not per-tick) keeps the full-region scan cheap;
        # between sweeps only ~1 MiB of dead accrues, so the peak stays accurate.
        # Default OFF for accelerate: its dead-intermediate count is tiny and the
        # real allocator HOLDS those buffers at its peak (the manifest peak
        # includes them), so sweeping them makes accelerate LESS accurate vs the
        # target (768.55→763.3, i.e. −0.55%→−1.23% vs the real 772.8). The
        # phenomenon is diffusers/MMDiT-specific (thousands of orphan activations
        # the real run frees promptly), so `DiffusersGroupOffload` turns it on.
        self._gc_tick: int = 0
        self._gc_period: int = int(self.args.get("dead_gc_period", 0))
        self._stat_dead_freed: int = 0
        return

    # ============================================================ compile
    def compile(self, trace: Trace) -> None:
        # The loader (PytorchOffloadLoader + offload_reconstruct) owns the rewrite.
        return

    # ============================================================ helpers
    def _home_memory(self, tensor) -> BaseMemory:
        dev = str(tensor.args.get("device", "cpu")).lower()
        if dev in self.memory_by_device:
            return self.memory_by_device[dev]
        if dev.startswith("cuda"):
            return self.vram
        return self.ram

    def _compute_for_node(self, node: Node) -> BaseCompute:
        if isinstance(node, TerminalNode):
            return self.cpu_compute
        if str(node.args.get("device_type", "CPU")).upper() in ("CUDA", "GPU"):
            return self.cuda_compute
        return self.cpu_compute

    def _find_free_page(self, memory: BaseMemory, num_pages: int) -> int | None:
        cursor = 0
        for region in memory.space._regions_by_page_idx_start.values():
            if region.page_idx_start - cursor >= num_pages:
                return cursor
            cursor = max(cursor, region.page_idx_end)
        if memory.space.num_total_pages - cursor >= num_pages:
            return cursor
        return None

    def _claim(self, memory: BaseMemory, tensor) -> DataRegion | None:
        page = self._find_free_page(memory, tensor.num_pages)
        if page is None:
            self.sys.abort({
                "from": self.name,
                "error": "OOM",
                "msg": f"No room on {memory.name} for tensor {tensor.id} "
                       f"({tensor.num_pages} pages); used={memory.space.num_used_pages}/"
                       f"{memory.space.num_total_pages}.",
            })
            return None
        return self.sys.claim(memory, tensor, page)

    def _mark_started(self, nid: int) -> None:
        for c in self._started_children.get(nid, []):
            if self._pending_started.get(c, 0) > 0:
                self._pending_started[c] -= 1
            self._maybe_ready(c)

    def _mark_done(self, nid: int) -> None:
        for c in self.sys.trace.node_map[nid].children_nodes:
            if self._pending_parent.get(c, 0) > 0:
                self._pending_parent[c] -= 1
            self._maybe_ready(c)

    def _maybe_ready(self, c: int) -> None:
        if (c not in self._submitted
                and self._pending_parent.get(c, 0) == 0
                and self._pending_started.get(c, 0) == 0):
            self.ready.append(c)

    # ============================================================ layout
    def layout(self, init_storage: BaseStorage) -> bool:
        """Phase 0: claim home regions for initial tensors (cpu->RAM, cuda->VRAM).
        Phase 1: one batched SSD->RAM transfer (cpu-home initials, incl. masters).
        Phase 2: one batched SSD->VRAM transfer (cuda-home initials). Done.
        Each pass emits a single non-contending TransferJob (SimpleSSD is
        1-concurrent), and layout jobs finish instantly (no sim time)."""
        tm = self.sys.trace.tensor_map
        initials = [t for t in tm.values()
                    if t.args.get("tensor_type") in self._INITIAL_TYPES]

        if self._layout_phase == 0:
            for t in initials:
                home = self._home_memory(t)
                if home.space.get_by_tensor_id(t.id):
                    continue
                if self._claim(home, t) is None:
                    return True  # abort signalled
            self._layout_phase = 1
            return False

        if self._layout_phase in (1, 2):
            dest_mem = self.ram if self._layout_phase == 1 else self.vram
            batch: list[tuple[DataRegion, DataRegion]] = []
            for t in initials:
                if self._home_memory(t) is not dest_mem:
                    continue
                src = init_storage.space.get_by_tensor_id(t.id)
                dst = dest_mem.space.get_by_tensor_id(t.id)
                if not src or not dst:
                    self.sys.abort({
                        "from": self.name, "error": "LAYOUT_FAILURE",
                        "msg": f"tensor {t.id} missing storage/home region in phase {self._layout_phase}.",
                    })
                    return True
                batch.append((src[0], dst[0]))
            if batch:
                self.sys.transfer(batch)
            done = self._layout_phase == 2
            self._layout_phase += 1
            return done

        return True

    # ============================================================ runtime
    def runtime(self, retired_jobs: list[BaseJob]) -> None:
        for job in retired_jobs:
            if isinstance(job, TransferJob):
                nid = self._transfer_job_node.pop(job.id, None)
                if nid is not None:                       # a weight-load trigger finished
                    self.sys.trace.node_map[nid].status = NodeStatus.DONE
                    self._mark_done(nid)
            elif isinstance(job, ComputeJob):
                node = job.node                           # engine already set status DONE
                self._mark_done(node.id)
                self._evict_masters(self.evict_after_node.get(node.id, []))
                self._consume_inputs(node)

        self._retry_pending_release()
        self._release_stale_duplicates()
        self._gc_tick += 1
        if self._gc_period > 0 and self._gc_tick % self._gc_period == 0:
            self._gc_dead_intermediates()
        self._submit_ready()
        return

    def _submit_ready(self) -> None:
        committed: dict[BaseCompute, int] = {}
        requeue: list[int] = []
        nm = self.sys.trace.node_map
        for _ in range(len(self.ready)):
            nid = self.ready.popleft()
            if nid in self._submitted:
                continue
            if self._pending_parent.get(nid, 0) != 0 or self._pending_started.get(nid, 0) != 0:
                continue  # stale queue entry; will be re-added when it truly clears
            node = nm[nid]

            if "offload_transfer" in node.args:
                if not self._dispatch_transfer(node):
                    requeue.append(nid)
                    continue
                self._submitted.add(nid)
                self._mark_started(nid)
                continue

            compute = self._compute_for_node(node)
            cap = getattr(compute, "max_concurrent_jobs", 1)
            if len(compute.job_running) + committed.get(compute, 0) >= cap:
                requeue.append(nid)
                continue
            # Don't submit a gpu compute whose cuda inputs aren't VRAM-resident yet:
            # job_waiting is FIFO with head-of-line blocking, so parking a not-yet-
            # runnable node at the head would block the very TransferJob that makes it
            # runnable -> engine "Deadlock detected". Accelerate never hits this (its
            # gpu nodes are stream_order children of their own weight Memcpy, so they
            # become ready only AFTER that transfer retires); diffusers streams on a
            # SIDE stream, so a compute node can become ready before its master's
            # cross-stream transfer completes. Defer it so the transfer runs first.
            # Skip custom_deps nodes (alias/dispatcher) — they bypass residency by
            # design. No-op for accelerate (its inputs are resident once ready).
            if (compute is self.cuda_compute and not node.custom_deps
                    and not self._inputs_resident_vram(node)):
                requeue.append(nid)
                continue
            if not self._ensure_outputs_claimed(node, compute):
                requeue.append(nid)
                continue
            if node.args.get("dispatcher_outputs"):
                self._preclaim_dispatcher_outputs(node)
            self.sys.compute(compute, node)
            self._submitted.add(nid)
            committed[compute] = committed.get(compute, 0) + 1
            self._mark_started(nid)

        for nid in requeue:
            self.ready.append(nid)

    def _dispatch_transfer(self, node: Node) -> bool:
        """Fire one same-tid RAM->VRAM TransferJob for this trigger's master.
        Returns False (requeue) if the master isn't resident in RAM yet (e.g. an
        INTERMEDIATE master still being produced) or VRAM can't be claimed."""
        info = node.args["offload_transfer"]
        master_tid = info["master"]
        tm = self.sys.trace.tensor_map
        master = tm[master_tid]

        src = next((r for r in self.ram.space.get_by_tensor_id(master_tid)
                    if r.is_ready and r.is_latest
                    and r.access_status in (DataRegionAccess.IDLE, DataRegionAccess.BEING_READ)),
                   None)
        if src is None:
            return False  # master not in RAM yet -> retry next tick

        dst = next((r for r in self.vram.space.get_by_tensor_id(master_tid)
                    if r.access_status == DataRegionAccess.IDLE), None)
        if dst is None:
            dst = self._claim(self.vram, master)
            if dst is None:
                return False  # VRAM full -> retry (or abort already signalled)

        jid = self.sys.transfer([(src, dst)])
        self._transfer_job_node[jid] = node.id
        node.status = NodeStatus.RUNNING
        self._stat_transfers_fired += 1
        return True

    def _inputs_resident_vram(self, node: Node) -> bool:
        """True iff every input tensor of this gpu node has a ready+latest VRAM
        region (a master after its RAM->VRAM transfer retired, or an activation
        already produced on the gpu). Used pre-submission to avoid parking a
        not-yet-runnable node at the FIFO head (see _submit_ready)."""
        for tid in node.input_tensors:
            regions = self.vram.space.get_by_tensor_id(tid)
            if not any(r.is_ready and r.is_latest
                       and r.access_status in (DataRegionAccess.IDLE, DataRegionAccess.BEING_READ)
                       for r in regions):
                return False
        return True

    def _ensure_outputs_claimed(self, node: Node, compute: BaseCompute) -> bool:
        for tid in node.output_tensors:
            if tid in node.input_tensors:
                continue  # in-place: input region is the output region
            regions = compute.memory.space.get_by_tensor_id(tid)
            if any(r.access_status == DataRegionAccess.IDLE for r in regions):
                continue
            # No IDLE region but one already exists (busy) -> claiming a second
            # region; the older copy will go stale on this write and leak. Watch it.
            if regions and self._release_dups:
                self._suspect_dups.add(tid)
            if self._claim(compute.memory, self.sys.trace.tensor_map[tid]) is None:
                return False
        return True

    def _preclaim_dispatcher_outputs(self, node: Node) -> None:
        """Cross-device outputs (loader stripped them from output_tensors into
        args['dispatcher_outputs']) are pre-claimed on their home memory and
        marked ready/latest so downstream consumers find them resident."""
        tm = self.sys.trace.tensor_map
        for tid in node.args.get("dispatcher_outputs") or []:
            t = tm.get(tid)
            if t is None:
                continue
            home = self._home_memory(t)
            existing = home.space.get_by_tensor_id(tid)
            region = next((r for r in existing
                           if r.access_status == DataRegionAccess.IDLE), None)
            if region is None:
                if existing and self._release_dups:
                    self._suspect_dups.add(tid)
                region = self._claim(home, t)
                if region is None:
                    continue
            region.is_ready = True
            region.is_latest = True

    def _release_stale_duplicates(self) -> None:
        """Reclaim double-claim leaks: for each suspect tid, on each memory that
        holds a fresh (is_latest=True) region for it, release the dead
        (is_latest=False) IDLE region(s). Same-memory gating keeps the fresh
        copy and never drops a tid whose only copy here is stale (conservative:
        if no fresh copy is present yet, leave it and revisit next tick). Skip
        masters/initials — they are never invalidated, so never stale. A tid is
        dropped from the watch set once it holds at most one region everywhere.
        No-op (early return) whenever no suspect exists — so accelerate, whose
        serial single-stream execution never claims a second region, pays only
        one set-emptiness check per tick."""
        if not self._suspect_dups:
            return
        for tid in list(self._suspect_dups):
            if tid in self.master_tids:
                self._suspect_dups.discard(tid)
                continue
            t = self.sys.trace.tensor_map.get(tid)
            if t is None or t.args.get("tensor_type") in self._INITIAL_TYPES:
                self._suspect_dups.discard(tid)
                continue
            total_regions = 0
            for hw in (self.vram, self.ram):
                regions = hw.space.get_by_tensor_id(tid)
                total_regions += len(regions)
                if len(regions) < 2:
                    continue
                if not any(r.is_latest for r in regions):
                    continue  # fresh copy not here yet -> revisit next tick
                for r in regions:
                    if (not r.is_latest) and r.access_status == DataRegionAccess.IDLE:
                        self.sys.release(r)
                        self._stat_stale_dups_freed += 1
                        total_regions -= 1
            if total_regions <= 1:
                self._suspect_dups.discard(tid)

    # ------------------------------------------------------------ eviction
    def _evict_masters(self, master_tids: list[int]) -> None:
        for mtid in master_tids:
            deferred = False
            for r in list(self.vram.space.get_by_tensor_id(mtid)):
                if r.access_status == DataRegionAccess.IDLE:
                    self.sys.release(r)
                    self._stat_masters_evicted += 1
                else:
                    deferred = True
            if deferred:
                self._pending_release.add(mtid)

    def _consume_inputs(self, node: Node) -> None:
        for tid in node.input_tensors:
            c = self._remaining_consumers.get(tid, 0)
            if c <= 0:
                continue
            c -= 1
            self._remaining_consumers[tid] = c
            if c == 0:
                self._release_intermediate(tid)

    def _release_intermediate(self, tid: int) -> None:
        if tid in self.master_tids:
            return  # masters are schedule-evicted, never refcount-freed
        t = self.sys.trace.tensor_map.get(tid)
        if t is None or t.args.get("tensor_type") in self._INITIAL_TYPES:
            return  # initial tensors stay resident
        deferred = False
        for hw in (self.vram, self.ram):
            for r in list(hw.space.get_by_tensor_id(tid)):
                if r.access_status == DataRegionAccess.IDLE:
                    self.sys.release(r)
                    self._stat_intermediates_freed += 1
                else:
                    deferred = True
        if deferred:
            self._pending_release.add(tid)

    def _gc_dead_intermediates(self) -> None:
        """Periodic reclaim of dead intermediates the consumer-triggered
        refcount free misses (orphans with no consumer, and tids whose region
        is finalized after rem already hit 0 — see __init__). Frees every IDLE
        INTERMEDIATE region whose tid has remaining_consumers==0 (skipping
        masters/initials). Safe by construction: rem==0 means no normal consumer
        needs the buffer (custom_deps consumers bypass residency), so e2e is
        unaffected — nothing ever reads a freed region."""
        rem = self._remaining_consumers
        tm = self.sys.trace.tensor_map
        for hw in (self.vram, self.ram):
            for r in list(hw.space._regions_by_page_idx_start.values()):
                if r.access_status != DataRegionAccess.IDLE:
                    continue
                tid = r.tensor_id
                if rem.get(tid, 0) != 0 or tid in self.master_tids:
                    continue
                t = tm.get(tid)
                if t is None or t.args.get("tensor_type") in self._INITIAL_TYPES:
                    continue
                self.sys.release(r)
                self._stat_dead_freed += 1

    def _retry_pending_release(self) -> None:
        if not self._pending_release:
            return
        for tid in list(self._pending_release):
            still_busy = False
            for hw in (self.vram, self.ram):
                for r in list(hw.space.get_by_tensor_id(tid)):
                    if r.access_status == DataRegionAccess.IDLE:
                        self.sys.release(r)
                    else:
                        still_busy = True
            if not still_busy:
                self._pending_release.discard(tid)

    # ============================================================ logging
    def log_states(self) -> dict[str, Any] | None:
        return {
            "transfers_fired": self._stat_transfers_fired,
            "masters_evicted": self._stat_masters_evicted,
            "intermediates_freed": self._stat_intermediates_freed,
            "stale_dups_freed": self._stat_stale_dups_freed,
            "dead_freed": self._stat_dead_freed,
            "vram_used_KB": 4 * self.vram.space.num_used_pages,
            "vram_peak_KB": 4 * self.vram.space.peak_num_used_pages,
            "ready_queue": len(self.ready),
            "pending_release": len(self._pending_release),
            "suspect_dups": len(self._suspect_dups),
        }
