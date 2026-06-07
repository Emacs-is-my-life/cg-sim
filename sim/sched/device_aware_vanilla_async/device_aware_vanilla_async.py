"""Device-aware async-prefetch scheduler.

Keeps each tensor in a home memory derived from ``tensor.args["device"]``,
submits CPU nodes to the CPU compute and CUDA nodes to the GPU compute,
and lazily creates cross-device copies when a node consumes a tensor whose
latest copy is resident elsewhere.

When the trace carries ``xfer_arrivals`` populated by
``graph_modifiers.inject_schedule``, this scheduler additionally fires
RAM->VRAM transfers in the background when each issuer node retires,
gating the consumer node's compute submission on those transfers'
completion.

Mechanics mirror ``sim.sched.flexinfer``:

  - Each prefetch tensor carries a per-tensor ``xfer_state`` flag
    (RESIDENT / ABSENT / LOADING / LOADED) on the scheduler.
  - At issuer-retire, ``sys.transfer`` is invoked for the planned
    RAM->VRAM moves; the dst VRAM region is claimed at that moment, so
    peak VRAM accounting tracks "VRAM occupied during transfer" — the
    real PyTorch behavior. Tensor flips ABSENT -> LOADING.
  - ``sys.transfer`` runs the byte movement in parallel with GPU
    compute on the existing memory subsystem (PCIe bandwidth enforced
    by SimpleRAM/SimpleVRAM bandwidths). No new resource needed.
  - On TransferJob retire, tensor flips LOADING -> LOADED.
  - The submit loop is gated: a node that has any required tensor
    with state != LOADED/RESIDENT is re-queued, retried next tick.
  - Eviction is driven by ``trace.args["evict_after_node"]`` populated
    by the injector.
  - ``trace.args["d2h_xfer_arrivals"]`` (also written by the injector)
    fires VRAM->RAM TransferJobs on issuer-retire that *transfer* the
    streamed bytes back to RAM (consuming PCIe bandwidth) and release
    the VRAM region on transfer retire. Used by hf_accelerate's
    ``model`` / ``module-hook`` modes whose real HF API does
    ``module.to("cpu")`` on offload.

This keeps the simulator's wall-clock asymmetry between async and
sync prefetch realistic without inventing a phantom copy resource.
"""

from __future__ import annotations

import os
from collections import defaultdict, deque
from typing import Any, TYPE_CHECKING

from sim.core.job import BaseJob, ComputeJob, TransferJob
from sim.core.log import Log
from sim.core.trace import Node, NodeStatus, Tensor, TerminalNode, Trace
from sim.core.trace.custom_dep import NodeDoneDep
from sim.hw.common import DataRegion, DataRegionAccess
from sim.hw.compute.common import BaseCompute
from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage
from sim.sched.common import BaseScheduler

if TYPE_CHECKING:
    from sim.core.system import System


# Tensor xfer states.
_RESIDENT = "RESIDENT"  # placed in VRAM at layout, never moved
_ABSENT = "ABSENT"      # lives in RAM only (or already evicted)
_LOADING = "LOADING"    # transfer in flight to VRAM
_LOADED = "LOADED"      # transfer finished, region resident


class DeviceAwareVanillaAsync(BaseScheduler):
    """Device-aware scheduler with optional async prefetch (driven by injector)."""

    # Tensor types that stay resident for the whole run. Everything else
    # is released as soon as its last consumer retires.
    _PERMANENT_TYPES = frozenset({"WEIGHT", "INPUT", "LEAF"})

    def __init__(
        self,
        obj_id: int,
        name: str,
        log: Log,
        sys: System,
        args: dict[str, Any] | None = None,
    ):
        super().__init__(obj_id, name, log, sys, args)

        # ---------------- device-aware state ----------------
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

        cpu_compute_name = self.args.get("cpu_compute", "cpu")
        cuda_compute_name = self.args.get("cuda_compute", "gpu0")
        self.cuda_device = str(self.args.get("cuda_device", "cuda:0")).lower()

        if cpu_compute_name not in self.compute_by_name:
            raise Exception(f"[{type(self).__name__}] CPU compute '{cpu_compute_name}' does not exist.")
        if cuda_compute_name not in self.compute_by_name:
            raise Exception(f"[{type(self).__name__}] CUDA compute '{cuda_compute_name}' does not exist.")

        self.cpu_compute = self.compute_by_name[cpu_compute_name]
        self.cuda_compute = self.compute_by_name[cuda_compute_name]

        self.memory_by_device: dict[str, BaseMemory] = {
            "cpu": self.cpu_compute.memory,
            self.cuda_device: self.cuda_compute.memory,
        }
        self.initial_tensor_types = set(self.args.get("initial_tensor_types", ["WEIGHT", "INPUT", "LEAF"]))
        self.node_ids: list[int] = list(self.sys.trace.node_map.keys())

        # Start-gated edges: (parent_id, child_id) pairs the loader
        # carved out of the trace's control graph because they model
        # CUDA's async-launch semantics — a `submit` node on the CPU
        # thread is `cudaLaunchKernel`; once it begins on the engine,
        # the kernel is enqueued on the stream and the gpu_runtime
        # child can dispatch independently of the CPU side completing
        # its RecordFunction-wrapped duration.
        #
        # The loader stores these on the trace's args side channel
        # (`trace.args["start_gated_edges"]`) rather than as control
        # edges on the Node graph, so the engine's compute_assertion
        # — which would otherwise force the gpu_runtime to wait for
        # the submit's DONE — never sees them. This scheduler is the
        # only place that enforces their "parent must have at least
        # started" gate.
        start_gated_raw = self.sys.trace.args.get("start_gated_edges") or []
        self._started_edges: set[tuple[int, int]] = set()
        self._started_children_by_parent: dict[int, list[int]] = defaultdict(list)
        self._pending_started_count: dict[int, int] = {}
        for parent_id, child_id in start_gated_raw:
            pid, cid = int(parent_id), int(child_id)
            if (pid, cid) in self._started_edges:
                continue
            self._started_edges.add((pid, cid))
            self._started_children_by_parent[pid].append(cid)
            self._pending_started_count[cid] = self._pending_started_count.get(cid, 0) + 1

        self.pending_parent_count: dict[int, int] = {
            node.id: len(node.parent_nodes) for node in self.sys.trace.node_map.values()
        }
        self.ready_node_ids: deque[int] = deque(
            node.id for node in self.sys.trace.node_map.values()
            if self.pending_parent_count[node.id] == 0
            and self._pending_started_count.get(node.id, 0) == 0
        )

        # Multi-phase layout state. See `layout` for the phase meanings.
        self._layout_phase: int = 0
        self._ram: BaseMemory = self.cpu_compute.memory
        self._vram: BaseMemory = self.cuda_compute.memory
        # Tensors that live on CUDA but also keep a DRAM staging copy. Maps
        # tensor_id -> (ram_region, vram_region).
        self._cuda_staging: dict[int, tuple[DataRegion, DataRegion]] = {}

        # Lifetime tracking for intermediate release. Remaining-consumer
        # count per tensor, derived from node.input_tensors after the
        # loader's storage aliasing has collapsed views / allocators onto
        # their real identities.
        self._remaining_consumers: dict[int, int] = {}
        # CPU-only consumer count: cpu_leaf nodes that have the tid in
        # their input_tensors. This models the Python refcount — the
        # bytes become available to the caching allocator the moment the
        # last CPU op that referenced the tensor retires, even if
        # downstream GPU consumers are still pending in the stream
        # chain. Real PyTorch's caching allocator marks the block "free
        # pending event" at this point and lets new aten::empty calls
        # reuse it; subsequent kernels serialize on the GPU stream
        # behind the still-pending old consumer. (`submit` and `wait`
        # nodes are excluded because the loader strips their
        # input_tensors via `_is_pointer_only`.)
        self._remaining_cpu_consumers: dict[int, int] = {}
        for node in self.sys.trace.node_map.values():
            for tid in node.input_tensors:
                self._remaining_consumers[tid] = self._remaining_consumers.get(tid, 0) + 1
            if (node.args.get("runtime_role") or "") == "cpu_leaf":
                for tid in node.input_tensors:
                    self._remaining_cpu_consumers[tid] = self._remaining_cpu_consumers.get(tid, 0) + 1
        # Tensor ids that have been Python-released (last CPU consumer
        # retired but some GPU consumer still pending). Their VRAM
        # regions have been freed; pending consumers' input residency
        # check is bypassed via custom_deps. Custom_deps are attached
        # at python-release time (so a node already in engine.job_waiting
        # gets the bypass on the engine's next is_runnable check) and
        # also at submit time (covers the case where the consumer
        # entered ready_node_ids after the python-release).
        self._python_released_tids: set[int] = set()

        # Reverse index of real (non-alias) consumers per tensor:
        # tid → list of node_ids that take tid as input but not output.
        # Used by `_maybe_python_release` to patch pending consumers'
        # `custom_deps` so they bypass the engine's input residency
        # check after their input's region is freed.
        self._consumers_by_tid: dict[int, list[int]] = {}
        for nid, n in self.sys.trace.node_map.items():
            for tid in n.input_tensors:
                if tid in n.output_tensors:
                    continue
                self._consumers_by_tid.setdefault(tid, []).append(nid)

        # Tensors whose last consumer has retired but whose regions were
        # not IDLE when we tried to release (e.g. being read by an
        # in-flight transfer). Retried every engine tick until all their
        # regions are freed — see _retry_pending_releases.
        self._pending_releases: set[int] = set()

        # Diagnostic: outstanding kernel count (gpu_runtime submitted
        # but not retired). Track max and a histogram bucketed log.
        self._diag_outstanding_gpu: int = 0
        self._diag_max_outstanding_gpu: int = 0
        self._diag_outstanding_log: list = []  # (sim_t_ms, outstanding)
        # Hook the terminal node's post-run callback so we dump the
        # diagnostic at simulation end (the engine returns immediately
        # at terminal retire, before sched.runtime() runs).
        for n in self.sys.trace.node_map.values():
            if isinstance(n, TerminalNode):
                n.hook_post_run = lambda sys: self._diag_dump_outstanding()
                break
        # Opt-in per-tensor release set. Tensors in this set are released
        # after their last consumer retires, regardless of tensor_type.
        # The schedule-injection trace transformer populates this with
        # tensors that the WS schedule plans to evict (so the simulator
        # mirrors real PyTorch's resize_(0) eviction). Use the trace's
        # ``args["evictable_tensor_ids"]`` if present.
        evictable = self.sys.trace.args.get("evictable_tensor_ids")
        self._evictable_tensor_ids: set[int] = (
            set(int(x) for x in evictable) if evictable else set()
        )

        # ---------------- async-prefetch state ----------------

        # Per cgsim-tensor-id residency state.
        self._xfer_state: dict[int, str] = {}

        # TransferJob.id -> [cgsim_tid, ...] for retire callback.
        self._inflight_jobs: dict[Any, list[int]] = {}
        self._active_prefetch_jobs = 0

        # Optional H2D FIFO width. 0 means legacy/unlimited behavior.
        stream_arg = (args or {}).get(
            "h2d_streams",
            self.sys.trace.args.get("xfer_h2d_streams", 0),
        )
        try:
            self._h2d_streams = max(0, int(stream_arg))
        except (TypeError, ValueError):
            self._h2d_streams = 0
        # Queued transfers waiting for an h2d_stream slot. Each entry is
        # ``(kind, cgsim_tids)`` where ``kind`` is ``"h2d"`` (RAM→VRAM
        # prefetch) or ``"d2h"`` (VRAM→RAM eviction). Both directions
        # share the same slot counter so they serialize when
        # ``h2d_streams=1`` — matches real PyTorch's default-stream
        # cudaMemcpyAsync ordering at HF accelerate hook callbacks.
        self._prefetch_queue: deque[tuple[str, list[int]]] = deque()

        # issuer_node_id -> list of arrival dicts (fire on retire).
        self._arrivals_by_issuer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        # consumer_node_id -> set of cgsim_tids that must be LOADED before dispatch.
        self._gate_by_consumer: dict[int, set[int]] = defaultdict(set)
        # tid -> set of consumer node_ids still **unsubmitted** for it.
        # Initialised from `_gate_by_consumer`, decremented at
        # consumer-submit time (`_drop_pending_consumer`). Eviction
        # via ``_release_vram_only`` defers while non-empty so we
        # don't strand an unsubmitted consumer on an absent tid.
        self._pending_consumers_by_tid: dict[int, set[int]] = defaultdict(set)
        # consumer_id -> set of tids already credited as gate-clear,
        # independent of current xfer_state. Populated when an arrival
        # fires while the tid is already LOADED (the load has been
        # observed) and when an in-flight load this arrival was
        # piggybacking on retires. A later eviction dropping state to
        # ABSENT won't strand a pre-satisfied consumer.
        self._pre_satisfied_gate: dict[int, set[int]] = defaultdict(set)
        # tid -> list of (consumer_id, tid) pairs whose arrival fired
        # while a load was already in flight (state LOADING) and so
        # registered to inherit the load. When the corresponding
        # TransferJob retires we move them into ``_pre_satisfied_gate``.
        self._loading_awaiters: dict[int, list[int]] = defaultdict(list)

        # D2H (VRAM -> RAM) arrivals: issuer_node_id -> list of tid lists.
        # Populated by injectors that emit `d2h_xfer_arrivals` (e.g.
        # hf_accelerate `model` / `module-hook` modes whose real HF API
        # does `module.to("cpu")` on offload). After the D2H TransferJob
        # retires, the VRAM region is released — this is the *only*
        # eviction path for tids on this list (the solver excludes them
        # from `evict_after_node` / `evictable_tensor_ids`).
        self._d2h_arrivals_by_issuer: dict[int, list[list[int]]] = defaultdict(list)
        # D2H TransferJob.id -> [cgsim_tid, ...] for the retire callback
        # that releases the VRAM region after the transfer completes.
        self._inflight_d2h_jobs: dict[Any, list[int]] = {}
        # D2H-released tids whose VRAM region was BUSY at release time
        # (retried on subsequent ticks). Distinct from `_pending_releases`
        # because the retry must call `_release_vram_only` rather than
        # the full RAM+VRAM `_release_tensor_regions` (which would drop
        # the RAM mirror that the next H2D needs to source).
        self._d2h_pending_vram: set[int] = set()

        self._build_arrival_index()
        self._init_xfer_states()

        return

    # ============================================================ compile
    def compile(self, trace: Trace) -> None:
        # DAV_REACTIVE_EVICT=1: reactive farthest-next-use (Belady) eviction
        # fallback in _claim_region. Makes an injected OFFLINE schedule robust:
        # the schedule's planned evictions handle the steady state, and when a
        # scheduled load can't fit (planned-eviction timing vs actual need
        # mismatch) we evict the resident evictable weight whose next use is
        # furthest out instead of aborting (what SwapAdvisorRuntime does
        # online). Needs per-tid consumer start times for _next_use.
        self._reactive_evict = os.environ.get("DAV_REACTIVE_EVICT") == "1"
        # Plan-fidelity instrumentation: count UNPLANNED reactive evictions (i.e.
        # how much cache-replacement the runtime does that the MILP plan did NOT
        # schedule). High values ⇒ sim is overriding the offline residency plan.
        self._reactive_evict_count = 0
        self._reactive_evict_bytes = 0
        if os.environ.get("DAV_REACTIVE_REPORT") == "1":
            import atexit
            atexit.register(
                lambda: print(
                    f"[DAV:reactive_report] reactive_evict_count="
                    f"{self._reactive_evict_count} reactive_evict_bytes_MB="
                    f"{self._reactive_evict_bytes/1e6:.0f}",
                    flush=True,
                )
            )
        self._consumer_starts: dict[int, list[tuple[int, int]]] = {}
        self._nu_cursor: dict[int, int] = {}
        if self._reactive_evict:
            starts: dict[int, list[tuple[int, int]]] = {}
            for nid, node in trace.node_map.items():
                s = int(node.args.get("start_ns", nid))
                for tid in (node.input_tensors or []):
                    starts.setdefault(int(tid), []).append((s, int(nid)))
            for tid in starts:
                starts[tid].sort()
            self._consumer_starts = starts
        return

    def _next_use(self, tid: int) -> float:
        """Start_ns of tid's first not-yet-DONE consumer; +inf if none remain
        (ideal Belady victim). Amortized O(1) via a forward-only cursor."""
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

    def _reactive_evict_until_fits(self, memory: BaseMemory, tensor: Tensor) -> None:
        """Evict IDLE evictable resident weights, furthest-next-use first, until
        `tensor` fits or no victim remains. Mirrors SwapAdvisorRuntime."""
        need = tensor.num_pages
        cands: list[tuple[float, int]] = []
        for region in memory.space._regions_by_page_idx_start.values():
            tid = getattr(region, "tensor_id", None)
            if tid is None or tid == tensor.id:
                continue
            if tid not in self._evictable_tensor_ids:
                continue
            if region.access_status != DataRegionAccess.IDLE:
                continue
            cands.append((self._next_use(tid), int(tid)))
        cands.sort(reverse=True)   # furthest next use first
        for _nu, tid in cands:
            if self._find_free_page(memory, need) is not None:
                break
            self._reactive_evict_count += 1
            self._reactive_evict_bytes += int(
                getattr(self.sys.trace.tensor_map.get(int(tid)), "size_bytes", 0)
                or 0
            )
            self._release_vram_only(tid)

    # ============================================================ helpers
    def _memory_for_tensor(self, tensor: Tensor) -> BaseMemory:
        device = str(tensor.args.get("device", "cpu")).lower()
        if device in self.memory_by_device:
            return self.memory_by_device[device]
        if device.startswith("cuda"):
            return self.memory_by_device[self.cuda_device]
        return self.memory_by_device["cpu"]

    def _compute_for_node(self, node: Node) -> BaseCompute:
        if isinstance(node, TerminalNode):
            return self.cpu_compute

        device_type = str(node.args.get("device_type", "CPU")).upper()
        if device_type in ("CUDA", "GPU"):
            return self.cuda_compute
        return self.cpu_compute

    @staticmethod
    def _region_readable(region: DataRegion) -> bool:
        return (
            region.is_ready
            and region.is_latest
            and region.access_status in (DataRegionAccess.IDLE, DataRegionAccess.BEING_READ)
        )

    def _find_free_page(self, memory: BaseMemory, num_pages: int) -> int | None:
        # Placement policy via scheduler arg `alloc_policy`:
        #   "first_fit" (default) — first gap from page 0.
        #   "best_fit"            — smallest gap that fits.
        # First-fit drifts allocations low and shreds free space under the
        # heavy evict/refetch churn of a tight weight-streaming schedule — a
        # tid can fail to find a CONTIGUOUS slot even when total free ≫ its
        # size (e.g. llama3b@4gib: 366 MiB free, largest gap 43.9 MiB, a
        # 48 MiB weight aborts). Best-fit reuses an evicted same-size
        # weight's slot for the next same-size load, which is how the real
        # CUDA caching allocator (size-segregated free lists) avoids exactly
        # this fragmentation. Placement is invariant for peak (byte-sum) and
        # makespan (address-independent timing), so this only changes which
        # runs can place — never the resulting peaks/makespans.
        bestfit = str(self.args.get("alloc_policy", "first_fit")) == "best_fit"
        cursor = 0
        best_start: int | None = None
        best_gap: int | None = None
        for region in memory.space._regions_by_page_idx_start.values():
            gap = region.page_idx_start - cursor
            if gap >= num_pages:
                if not bestfit:
                    return cursor
                if best_gap is None or gap < best_gap:
                    best_start, best_gap = cursor, gap
            cursor = max(cursor, region.page_idx_end)

        tail = memory.space.num_total_pages - cursor
        if tail >= num_pages:
            if not bestfit:
                return cursor
            if best_gap is None or tail < best_gap:
                best_start, best_gap = cursor, tail
        return best_start

    def _claim_region(self, memory: BaseMemory, tensor: Tensor):
        page_idx = self._find_free_page(memory, tensor.num_pages)
        if (page_idx is None and getattr(self, "_reactive_evict", False)
                and memory is self._vram):
            # Reactive Belady fallback: free room by evicting furthest-next-use
            # evictable weights, so an offline schedule's planned-eviction
            # timing mismatch doesn't abort (see compile()).
            self._reactive_evict_until_fits(memory, tensor)
            page_idx = self._find_free_page(memory, tensor.num_pages)
        if page_idx is None:
            self._dump_abort_diag(memory, tensor)
            self.sys.abort({
                "from": self.name,
                "error": "LAYOUT_FAILURE",
                "msg": f"Not enough space on {memory.name} for tensor {tensor.id}.",
                "tensor": {
                    "id": tensor.id,
                    "name": tensor.name,
                    "num_pages": tensor.num_pages,
                    "device": tensor.args.get("device"),
                },
                "memory": {
                    "name": memory.name,
                    "used_pages": memory.space.num_used_pages,
                    "total_pages": memory.space.num_total_pages,
                },
            })
            return None

        return self.sys.claim(memory, tensor, page_idx)

    def _dump_abort_diag_consumers(self, memory: BaseMemory) -> None:
        """For each resident intermediate, classify pending consumers
        by role to test the Python-ref-drop hypothesis."""
        from collections import Counter, defaultdict
        resident = set()
        for r in memory.space._regions_by_page_idx_start.values():
            tid = getattr(r, "tensor_id", None)
            if tid is None: continue
            t = self.sys.trace.tensor_map.get(tid)
            if t and t.args.get("tensor_type") == "INTERMEDIATE":
                resident.add(tid)
        # Build per-tid consumer list
        all_consumers = defaultdict(list)
        for nid, n in self.sys.trace.node_map.items():
            for tid in n.input_tensors:
                if tid in resident and tid not in n.output_tensors:
                    all_consumers[tid].append((nid, n.args.get("runtime_role") or "", n.status))
        # Classify each resident: how many CPU vs GPU pending?
        c = Counter()
        cpu_done_gpu_pending = 0
        any_pending_cpu = 0
        for tid in resident:
            cons = all_consumers.get(tid, [])
            pending_cpu = sum(1 for _, role, st in cons
                              if role in ("cpu_leaf", "submit") and st != NodeStatus.DONE)
            pending_gpu = sum(1 for _, role, st in cons
                              if role == "gpu_runtime" and st != NodeStatus.DONE)
            done_cpu = sum(1 for _, role, st in cons
                           if role in ("cpu_leaf", "submit") and st == NodeStatus.DONE)
            done_gpu = sum(1 for _, role, st in cons
                           if role == "gpu_runtime" and st == NodeStatus.DONE)
            if pending_cpu == 0 and pending_gpu > 0:
                cpu_done_gpu_pending += 1
            if pending_cpu > 0:
                any_pending_cpu += 1
            c[(pending_cpu, pending_gpu)] += 1
        print(f"[DAV diag2] resident intermediates: {len(resident)}", flush=True)
        print(f"  all CPU consumers DONE, GPU pending: {cpu_done_gpu_pending}", flush=True)
        print(f"  some CPU consumer pending:           {any_pending_cpu}", flush=True)
        print(f"  (pending_cpu, pending_gpu) histogram (top 10):", flush=True)
        for k, v in c.most_common(10):
            print(f"    {k}: {v}", flush=True)

    def _dump_abort_diag(self, memory: BaseMemory, failed_tensor: Tensor) -> None:
        self._dump_abort_diag_consumers(memory)
        """Diagnose VRAM-cap abort: dump sim_t, what's resident,
        the producer/consumer state for resident intermediates, and
        how trace timestamps compare to sim wall time."""
        from collections import Counter, defaultdict
        sim_t = self.sys.engine.timestamp_now
        print(f"\n[DAV diag] === abort at sim_t={sim_t/1000:.2f} ms; need tensor {failed_tensor.id} ({failed_tensor.name}) {failed_tensor.num_pages*4/1024:.2f}MB ===", flush=True)

        # Resident summary
        per_tensor_pages = defaultdict(int)
        for r in memory.space._regions_by_page_idx_start.values():
            tid = getattr(r, "tensor_id", None)
            if tid is not None:
                per_tensor_pages[tid] += r.page_idx_end - r.page_idx_start
        by_type = defaultdict(lambda: [0, 0])
        for tid, p in per_tensor_pages.items():
            t = self.sys.trace.tensor_map.get(tid)
            ttype = t.args.get("tensor_type") if t else "?"
            by_type[ttype][0] += 1
            by_type[ttype][1] += p
        for ttype, (n, p) in sorted(by_type.items(), key=lambda kv: -kv[1][1]):
            print(f"  resident {ttype!s:<14} n={n:6} pages={p:>10} ({p*4/1024:8.1f} MB)", flush=True)

        # Status counts by role
        status_by_role = Counter()
        for n in self.sys.trace.node_map.values():
            status_by_role[(n.args.get("runtime_role") or "", str(n.status).split(".")[-1])] += 1
        print(f"  status by role:", flush=True)
        for (role, status), c in sorted(status_by_role.items()):
            print(f"    ({role!s:<14}, {status:<8}): {c}", flush=True)

        # Producer info for top resident intermediates
        producer_of = {}
        for nid, n in self.sys.trace.node_map.items():
            for tid in n.output_tensors:
                if tid not in n.input_tensors:
                    producer_of.setdefault(tid, nid)
            for tid in (n.args.get("dispatcher_outputs") or []):
                producer_of.setdefault(tid, nid)
        kept = []
        for tid, p in per_tensor_pages.items():
            t = self.sys.trace.tensor_map.get(tid)
            if t is None: continue
            if t.args.get("tensor_type") != "INTERMEDIATE": continue
            kept.append((p, tid, t.name))
        kept.sort(reverse=True)
        print(f"  top 5 resident INTERMEDIATES, producer info + trace times:", flush=True)
        for p, tid, name in kept[:5]:
            pnid = producer_of.get(tid)
            pn = self.sys.trace.node_map.get(pnid) if pnid is not None else None
            prole = (pn.args.get("runtime_role") or "") if pn else "?"
            pstatus = str(pn.status).split(".")[-1] if pn else "?"
            p_start = pn.args.get("start_ns") if pn else None
            rc = self._remaining_consumers.get(tid, 0)
            # Find earliest unfired consumer
            earliest_unfired = None
            for nid, n in self.sys.trace.node_map.items():
                if tid in n.input_tensors and tid not in n.output_tensors:
                    if n.status != NodeStatus.DONE:
                        s = n.args.get("start_ns")
                        if s is not None and (earliest_unfired is None or s < earliest_unfired[0]):
                            earliest_unfired = (s, nid, n.args.get("runtime_role") or "")
            ratio = ""
            if p_start is not None and sim_t > 0:
                # profile_time_at_producer (ns) → ms
                ratio = f" trace_t={p_start/1e6:.1f}ms  sim_advance={p_start/1e6 / (sim_t/1000):.1f}x"
            ef = f"  next_consumer=({earliest_unfired[1]}, {earliest_unfired[2]}, trace_t={earliest_unfired[0]/1e6:.1f}ms)" if earliest_unfired else ""
            print(f"    tid={tid} {p*4/1024:.1f}MB rc={rc} prod=({pnid},{prole},{pstatus}){ratio}{ef}", flush=True)

    def _ensure_home_region(self, tensor: Tensor):
        memory = self._memory_for_tensor(tensor)
        regions = memory.space.get_by_tensor_id(tensor.id)
        if regions:
            return regions[0]
        return self._claim_region(memory, tensor)

    # ============================================================ layout
    def layout(self, init_storage: BaseStorage) -> bool:
        """Multi-phase layout that stages weights SSD -> DRAM -> VRAM.

        The engine's layout drain loop requires every job submitted in one pass
        to be immediately runnable — it aborts as "Deadlock detected" if the
        first queued job isn't. That means: a single layout pass may only emit
        TransferJobs that share src/dst hardware or that do not contend.
        SimpleSSD admits one concurrent job, so we cannot mix ssd->ram and
        ssd->vram in the same pass.

        Phases:
          0 -> claim home regions for all tensors and DRAM staging regions for
               every CUDA-homed initial tensor. No transfers. Returns False.
          1 -> one batched ssd->ram TransferJob covering (a) every CPU-homed
               initial tensor going to its RAM home and (b) every CUDA-homed
               initial tensor going to its DRAM staging region. Returns False.
          2 -> one batched ram->vram TransferJob moving each CUDA-homed
               initial tensor from its DRAM staging region to its VRAM home.
               Returns True (done).

        The DRAM staging copy is intentionally kept resident after layout; it
        matches real PyTorch semantics (weights live in DRAM and are copied to
        VRAM on use) and lets runtime CPU ops read the tensor without a new
        vram->ram transfer.
        """
        if self._layout_phase == 0:
            # Claim home regions and staging ONLY for initial tensors
            # (WEIGHT / INPUT / LEAF). Intermediates — which dominate tensor
            # count — are claimed lazily in runtime by _ensure_outputs_claimed
            # at the moment their producer runs, so layout peak stays bounded
            # by the actual weight / input footprint rather than the sum of
            # every transient tensor's size.
            for tensor in self.sys.trace.tensor_map.values():
                tensor_type = tensor.args.get("tensor_type")
                if tensor_type not in self.initial_tensor_types:
                    continue

                home_region = self._ensure_home_region(tensor)
                if home_region is None:
                    return True  # abort already signalled

                # For CUDA-homed initial tensors, also reserve a DRAM staging
                # region so phase 1 can batch ssd->ram for every weight in
                # one TransferJob, and phase 2 can then ram->vram them.
                if home_region.hw is self._vram:
                    stage = self._claim_region(self._ram, tensor)
                    if stage is None:
                        return True
                    self._cuda_staging[tensor.id] = (stage, home_region)

            self._layout_phase = 1
            return False

        if self._layout_phase == 1:
            ssd_to_ram: list[tuple[DataRegion, DataRegion]] = []
            for tensor in self.sys.trace.tensor_map.values():
                tensor_type = tensor.args.get("tensor_type")
                if tensor_type not in self.initial_tensor_types:
                    continue

                stor_regions = init_storage.space.get_by_tensor_id(tensor.id)
                if not stor_regions:
                    self.sys.abort({
                        "from": self.name,
                        "error": "LAYOUT_FAILURE",
                        "msg": f"Initial tensor {tensor.id} has no storage placement.",
                        "tensor": {
                            "id": tensor.id,
                            "name": tensor.name,
                            "tensor_type": tensor_type,
                        },
                    })
                    return True
                src = stor_regions[0]

                # DAV_STREAM_FROM_SSD=1: model weights streaming SSD->GPU with
                # NO DRAM staging. Skip the SSD->RAM load for streamed (RAM-homed)
                # WEIGHT tensors — their RAM home stays empty, so at runtime
                # _find_latest_region (prefer-memory, fall through to storage)
                # finds only the SSD copy and the prefetch runs SSD->VRAM at the
                # SSD read_io_curve bandwidth. Cold (VRAM-homed) weights still
                # stage through DRAM at layout, but layout is ~free (≈20us), so
                # this models "runtime streaming is SSD->GPU".
                if (os.environ.get("DAV_STREAM_FROM_SSD") == "1"
                        and tensor.id not in self._cuda_staging
                        and tensor_type == "WEIGHT"):
                    continue

                if tensor.id in self._cuda_staging:
                    dest = self._cuda_staging[tensor.id][0]  # DRAM staging
                else:
                    # CPU-homed initial tensor: dest is its RAM home.
                    home_regions = self._ram.space.get_by_tensor_id(tensor.id)
                    if not home_regions:
                        self.sys.abort({
                            "from": self.name,
                            "error": "LAYOUT_FAILURE",
                            "msg": f"Tensor {tensor.id} has no RAM home region.",
                        })
                        return True
                    dest = home_regions[0]

                ssd_to_ram.append((src, dest))

            if ssd_to_ram:
                self.sys.transfer(ssd_to_ram)

            self._layout_phase = 2
            return False

        if self._layout_phase == 2:
            ram_to_vram: list[tuple[DataRegion, DataRegion]] = []
            for tensor_id, (stage, home) in self._cuda_staging.items():
                ram_to_vram.append((stage, home))

            if ram_to_vram:
                self.sys.transfer(ram_to_vram)

            self._layout_phase = 3
            return True

        return True

    # ============================================================ residency
    def _find_latest_region(self, tensor_id: int, exclude_hw: BaseMemory | BaseStorage | None = None) -> DataRegion | None:
        # Prefer memory copies over storage. Runtime transfers from storage are
        # still supported for initial tensors that were not already laid out.
        for hw in self.sys.hw.values():
            if hw is exclude_hw or not isinstance(hw, BaseMemory):
                continue
            for region in hw.space.get_by_tensor_id(tensor_id):
                if self._region_readable(region):
                    return region

        for hw in self.sys.hw.values():
            if hw is exclude_hw or not isinstance(hw, BaseStorage):
                continue
            for region in hw.space.get_by_tensor_id(tensor_id):
                if self._region_readable(region):
                    return region

        return None

    def _find_or_claim_dest_region(self, memory: BaseMemory, tensor: Tensor) -> DataRegion | None:
        for region in memory.space.get_by_tensor_id(tensor.id):
            if region.access_status == DataRegionAccess.IDLE:
                return region

        return self._claim_region(memory, tensor)

    def _ensure_inputs_resident(self, node: Node, memory: BaseMemory) -> list[tuple[DataRegion, DataRegion]] | None:
        transfers: list[tuple[DataRegion, DataRegion]] = []

        pending_dest_ids: set[int] = set()
        for job in self.sys.engine.job_waiting:
            if isinstance(job, TransferJob):
                for _src, dst in job.batch:
                    pending_dest_ids.add(id(dst))
        for job in self.sys.engine.job_running:
            if isinstance(job, TransferJob):
                for _src, dst in job.batch:
                    pending_dest_ids.add(id(dst))
        pending_dest_ids.update(self._tick_pending_dest_ids)

        for tensor_id in node.input_tensors:
            # Python-released tids have had their VRAM region freed at
            # last-CPU-consumer retire; we don't try to find or stage
            # them — the consumer's compute_assertion is bypassed via
            # custom_deps added at submit time.
            if tensor_id in self._python_released_tids:
                continue

            tensor = self.sys.trace.tensor_map[tensor_id]
            target_regions = memory.space.get_by_tensor_id(tensor_id)

            if any(self._region_readable(region) for region in target_regions):
                continue

            if any(
                (r.access_status == DataRegionAccess.BEING_WRITTEN) or (id(r) in pending_dest_ids)
                for r in target_regions
            ):
                return None

            src_region = self._find_latest_region(tensor_id, exclude_hw=memory)
            if src_region is None:
                return None

            dest_region = self._find_or_claim_dest_region(memory, tensor)
            if dest_region is None or dest_region.access_status != DataRegionAccess.IDLE:
                return None

            transfers.append((src_region, dest_region))

        return transfers

    def _preclaim_dispatcher_outputs(self, node: Node) -> None:
        """Reserve cross-device output regions for a dispatcher node on
        the tensor's home memory before submission.

        The loader stashes the cross-device tensor_ids in
        `node.args["dispatcher_outputs"]` and removes them from
        `node.output_tensors`, so the engine's begin_mutation never sees
        them and never invalidates the pre-claimed region. Downstream
        consumers find the tensor sitting ready/latest where it belongs.

        Dead outputs (no consumer in the DAG) are skipped to avoid
        leaking regions nobody will read.
        """
        cross_outs = node.args.get("dispatcher_outputs") or []
        for tensor_id in cross_outs:
            tensor = self.sys.trace.tensor_map.get(tensor_id)
            if tensor is None:
                continue
            home = self._memory_for_tensor(tensor)
            if self._remaining_consumers.get(tensor_id, 0) <= 0:
                if tensor.args.get("tensor_type") not in self._PERMANENT_TYPES:
                    continue
            regions = home.space.get_by_tensor_id(tensor_id)
            target = None
            for r in regions:
                if r.access_status == DataRegionAccess.IDLE:
                    target = r
                    break
            if target is None:
                target = self._claim_region(home, tensor)
                if target is None:
                    continue
            target.is_ready = True
            target.is_latest = True

    def _consume_inputs(self, node: Node) -> None:
        """Decrement per-tensor remaining-consumer counts for this node's
        input tensors. Two release events are possible per tensor:

        * Python-release (when the last CPU consumer retires while some
          GPU consumer is still pending): genuinely free the VRAM region
          and mark the tid as python-released. Models the CUDA caching
          allocator returning the block to its cache when Python's
          refcount drops at end-of-scope. Pending GPU consumers' input
          residency check is bypassed via custom_deps at submit time
          (the data they need is conceptually still consistent because
          stream_order serializes any subsequent kernel that reuses the
          bytes).

        * Full release (when all consumers — CPU and GPU — have
          retired): no-op for already python-released tids; releases
          regions for tids that never had CPU consumers.

        Skipped for permanent tensors (WEIGHT / INPUT / LEAF) unless
        the tid is in the explicit ``_evictable_tensor_ids`` set.
        """
        is_cpu_role = (node.args.get("runtime_role") or "") == "cpu_leaf"
        for tid in node.input_tensors:
            # CPU-side decrement (Python refcount analogue)
            if is_cpu_role:
                cpu_left = self._remaining_cpu_consumers.get(tid, 0)
                if cpu_left > 0:
                    cpu_left -= 1
                    if cpu_left == 0:
                        self._remaining_cpu_consumers.pop(tid, None)
                        self._maybe_python_release(tid)
                    else:
                        self._remaining_cpu_consumers[tid] = cpu_left

            # Full-release decrement (all consumers)
            remaining = self._remaining_consumers.get(tid)
            if remaining is None:
                continue
            remaining -= 1
            if remaining > 0:
                self._remaining_consumers[tid] = remaining
                continue
            self._remaining_consumers.pop(tid, None)
            tensor = self.sys.trace.tensor_map.get(tid)
            if tensor is None:
                continue
            ttype = tensor.args.get("tensor_type")
            if ttype in self._PERMANENT_TYPES and tid not in self._evictable_tensor_ids:
                continue
            # If already python-released, the region was freed earlier
            # (or deferred to `_pending_releases` if it was BUSY then).
            # Still call _release_tensor_regions: it's a no-op when the
            # region is already gone, and it retries the deferred case.
            if tid in self._python_released_tids:
                self._python_released_tids.discard(tid)
            # Streamed weights / leaves (permanent type in evictable
            # set) need their RAM mirror preserved across the entire
            # run — every later run's prefetch sources from it. Only
            # release VRAM here; ``_release_tensor_regions`` would
            # drop RAM too, breaking subsequent reloads.
            if ttype in self._PERMANENT_TYPES and tid in self._evictable_tensor_ids:
                self._release_vram_only(tid)
            else:
                self._release_tensor_regions(tid)

    def _maybe_python_release(self, tid: int) -> None:
        """Last CPU consumer of `tid` just retired. If some GPU consumer
        is still pending, release the VRAM region now (Python refcount
        drop) and mark the tid python-released. Pending GPU consumers
        of the tid will have their input residency check bypassed at
        submit time via custom_deps.
        """
        if tid in self._python_released_tids:
            return
        if self._remaining_consumers.get(tid, 0) <= 0:
            # All consumers (incl. GPU) have already retired; normal
            # full-release path will handle it.
            return
        tensor = self.sys.trace.tensor_map.get(tid)
        if tensor is None:
            return
        ttype = tensor.args.get("tensor_type")
        # All permanents (WEIGHT / INPUT / LEAF) are skipped, whether
        # always-resident or LP-managed evictable. For LP-managed
        # streamed weights in `_evictable_tensor_ids`, the schedule
        # injector owns the lifecycle: prefetch arrivals load the
        # weight into VRAM, `evict_after_node` releases it. That
        # cycle is timed against the LP's planned epochs. Python-
        # releasing inside an epoch (at last-CPU-consumer-retire,
        # which is earlier than `evict_after_node`) marks
        # `xfer_state = ABSENT` for the still-active epoch, parking
        # subsequent gated consumers on a prefetch that won't fire
        # until the next epoch — silent coverage_demoted, peak
        # blow-up. Always-resident permanents (not in the evictable
        # set) are pinned for the run and never python-released.
        if ttype in self._PERMANENT_TYPES:
            return
        self._python_released_tids.add(tid)
        self._release_tensor_regions(tid)
        # Patch pending consumers' custom_deps so the engine's
        # compute_assertion bypasses the input-residency check on the
        # tid we just freed. Necessary for consumers that were
        # already submitted to engine.job_waiting before this
        # python-release fired — they get the bypass on the engine's
        # next is_runnable retry.
        for cid in self._consumers_by_tid.get(tid, []):
            consumer = self.sys.trace.node_map.get(cid)
            if consumer is None:
                continue
            if consumer.status == NodeStatus.DONE:
                continue
            if consumer.custom_deps:
                continue
            for parent_id in consumer.parent_nodes:
                consumer.custom_deps.append(NodeDoneDep(parent_id))

    def _python_release_producer_outputs(self, node: Node) -> None:
        """At a producer's retirement, python-release any output tid
        that has zero remaining CPU consumers (the producer was the
        last Python touch). Covers both `output_tensors` and
        `dispatcher_outputs` (which the loader stripped from
        output_tensors). Pending GPU consumers of a python-released
        tid get their `custom_deps` patched inside
        `_maybe_python_release` so the engine's subsequent
        compute_assertion takes the bypass branch."""
        candidates = []
        for tid in node.output_tensors:
            if tid in node.input_tensors:
                continue
            candidates.append(tid)
        for tid in (node.args.get("dispatcher_outputs") or []):
            candidates.append(tid)
        for tid in candidates:
            if tid in self._python_released_tids:
                continue
            if self._remaining_cpu_consumers.get(tid, 0) > 0:
                continue
            self._maybe_python_release(tid)

    def _release_tensor_regions(self, tensor_id: int) -> None:
        """Free every region holding this tensor. If any region is busy
        (BEING_READ by a transfer src, BEING_WRITTEN, etc.) at the moment
        of call, core's release_assertion rejects it — we record the
        tensor as pending and retry on the next engine tick.

        Also flips the per-tensor xfer_state back to ABSENT once VRAM
        is freed, so the next prefetch can re-LOAD it.
        """
        deferred = False
        for hw in self.sys.hw.values():
            if not isinstance(hw, BaseMemory):
                continue
            for region in list(hw.space.get_by_tensor_id(tensor_id)):
                if region.access_status == DataRegionAccess.IDLE:
                    self.sys.release(region)
                else:
                    deferred = True

        if deferred:
            self._pending_releases.add(tensor_id)
        else:
            self._pending_releases.discard(tensor_id)

        # Mirror residency state: if VRAM no longer holds this tensor and
        # we previously marked it RESIDENT/LOADED, flip back to ABSENT.
        if not self._vram.space.get_by_tensor_id(tensor_id):
            if self._xfer_state.get(tensor_id) in (_LOADED, _RESIDENT):
                self._xfer_state[tensor_id] = _ABSENT

    def _drop_pending_consumer(self, node: Node) -> None:
        """Mark this node as submitted in the per-tid pending-consumer
        index. Once a consumer is submitted, the gate has been
        satisfied and a later state-change to ABSENT won't strand it
        (compute_assertion is checked at submit, not throughout the
        node's runtime). So the eviction path can release VRAM as
        soon as no remaining consumer is *unsubmitted*.
        """
        for tid in node.input_tensors:
            pending = self._pending_consumers_by_tid.get(tid)
            if pending and node.id in pending:
                pending.discard(node.id)
                if not pending:
                    del self._pending_consumers_by_tid[tid]

    def _release_vram_only(self, tensor_id: int) -> None:
        """Release only the VRAM region(s) for the tensor, preserving
        any RAM mirror. Used by the D2H eviction path so the next H2D
        prefetch can still pull from RAM. Out-of-order arrival firing
        is handled by ``_pre_satisfied_gate`` — consumers that saw the
        tid LOADED at their arrival's fire moment are credited and
        won't strand even when state later flips to ABSENT here.
        """
        deferred = False
        for region in list(self._vram.space.get_by_tensor_id(tensor_id)):
            if region.access_status == DataRegionAccess.IDLE:
                self.sys.release(region)
            else:
                deferred = True
        if deferred:
            self._d2h_pending_vram.add(tensor_id)
        else:
            self._d2h_pending_vram.discard(tensor_id)
        if not self._vram.space.get_by_tensor_id(tensor_id):
            if self._xfer_state.get(tensor_id) in (_LOADED, _RESIDENT):
                self._xfer_state[tensor_id] = _ABSENT

    def _retry_pending_releases(self) -> None:
        """Re-attempt any deferred releases. Called once per engine tick
        from runtime(); a region transitions out of BEING_READ/WRITTEN
        only through a job's end_mutation, which always runs inside
        _runtime_forward immediately before the next sched.runtime call,
        so one retry per tick catches every freed region."""
        if self._pending_releases:
            for tensor_id in list(self._pending_releases):
                self._release_tensor_regions(tensor_id)
        if self._d2h_pending_vram:
            for tensor_id in list(self._d2h_pending_vram):
                self._release_vram_only(tensor_id)

    def _outputs_free(self, node: Node, memory: BaseMemory) -> bool:
        """Returns True iff every non-aliased output tensor has an IDLE
        region on `memory` with no pending transfer targeting it."""
        pending_dest_ids: set[int] = set(self._tick_pending_dest_ids)
        for job in self.sys.engine.job_waiting:
            if isinstance(job, TransferJob):
                for _src, dst in job.batch:
                    pending_dest_ids.add(id(dst))
        for job in self.sys.engine.job_running:
            if isinstance(job, TransferJob):
                for _src, dst in job.batch:
                    pending_dest_ids.add(id(dst))

        for tensor_id in node.output_tensors:
            if tensor_id in node.input_tensors:
                continue
            regions = memory.space.get_by_tensor_id(tensor_id)
            has_idle = False
            for r in regions:
                if r.access_status == DataRegionAccess.IDLE and id(r) not in pending_dest_ids:
                    has_idle = True
                    break
            if not has_idle:
                return False
        return True

    def _ensure_outputs_claimed(self, node: Node, memory: BaseMemory) -> bool:
        for tensor_id in node.output_tensors:
            # Output aliased to an input (pure view/in-place): input region
            # IS the output region, nothing to claim.
            if tensor_id in node.input_tensors:
                continue

            tensor = self.sys.trace.tensor_map[tensor_id]
            regions = memory.space.get_by_tensor_id(tensor_id)
            if any(region.access_status == DataRegionAccess.IDLE for region in regions):
                continue

            region = self._claim_region(memory, tensor)
            if region is None:
                return False

        return True

    def _release_started_children(self, parent: Node) -> None:
        """When a parent with start-gated outgoing edges is submitted
        to the engine, its start-gated children's started-parent
        counter is decremented immediately (rather than waiting for
        the parent to reach DONE). Any child whose remaining gate
        (regular pending_parent_count AND started count) has cleared
        is enqueued as ready."""
        children = self._started_children_by_parent.get(parent.id)
        if not children:
            return
        for child_id in children:
            remaining = self._pending_started_count.get(child_id, 0)
            if remaining > 0:
                self._pending_started_count[child_id] = remaining - 1
                remaining -= 1
            if (remaining == 0
                    and self.pending_parent_count.get(child_id, 0) == 0):
                child = self.sys.trace.node_map[child_id]
                if child.status == NodeStatus.TODO:
                    self.ready_node_ids.append(child_id)

    def _parents_satisfied(self, node: Node) -> bool:
        """All control-graph parents must be DONE. Start-gated edges
        are deliberately *not* in `node.parent_nodes` (the loader
        excludes them from the control graph and routes them through
        the started-children index); their gate is already enforced
        by `_pending_started_count`, checked at the readiness site."""
        node_map = self.sys.trace.node_map
        for parent_id in node.parent_nodes:
            if node_map[parent_id].status != NodeStatus.DONE:
                return False
        return True

    def _submit_transfer_batches(self, transfers: list[tuple[DataRegion, DataRegion]]) -> None:
        grouped: dict[tuple[int, int], list[tuple[DataRegion, DataRegion]]] = defaultdict(list)
        for src_region, dest_region in transfers:
            grouped[(src_region.hw.id, dest_region.hw.id)].append((src_region, dest_region))

        for batch in grouped.values():
            self.sys.transfer(batch)

        return

    # ============================================================ submit loop
    def _submit_ready_nodes_core(self) -> bool:
        """Submit ready nodes each tick, packing independent compute devices.

        The engine's runtime drain loop head-of-line-blocks: if the first job
        in job_waiting isn't immediately runnable, it aborts. So this method
        never queues a ComputeJob behind a TransferJob in the same tick.

        For each ready node:
          - if its inputs need transferring, submit the transfers only and
            re-queue the node. The compute goes next tick after the transfer
            completes and the inputs become readable.
          - if its inputs are already resident and outputs claimed, submit the
            compute directly.

        Multiple nodes on different devices (e.g. one CPU op and one GPU
        kernel) can be queued in the same tick. Within a single device stop
        at its concurrency cap.

        IMPORTANT: the `committed_per_compute` cap is load-bearing — it
        prevents back-to-back same-compute commits within one invocation.
        The engine's drain loop dispatches FIFO from job_waiting and HOL-
        blocks the first non-runnable entry; if we queue [CPU_A, CPU_B,
        GPU_X] in this order, CPU_B fails `can_run` (CPU already at cap=1)
        and HOL breaks before reaching GPU_X — losing parallelism (and on
        traces where the deadlock check fires, aborting). The cap forces
        commits to interleave (CPU, GPU, CPU, GPU, …) so the drain reaches
        both computes.
        """
        submitted_any = False
        committed_per_compute: dict = {}
        # Tracks dst region ids that this tick's newly-issued transfers are
        # targeting, so a second ready node doesn't emit a duplicate before
        # those transfers are visible in engine.job_waiting.
        self._tick_pending_dest_ids: set[int] = set()
        num_ready = len(self.ready_node_ids)
        for _ in range(num_ready):
            node_id = self.ready_node_ids.popleft()
            node = self.sys.trace.node_map[node_id]
            if node.status != NodeStatus.TODO:
                continue
            if (self.pending_parent_count[node_id] != 0
                    or self._pending_started_count.get(node_id, 0) != 0
                    or not self._parents_satisfied(node)):
                self.ready_node_ids.append(node_id)
                continue

            compute = self._compute_for_node(node)

            # Alias / dispatcher nodes carry custom_deps set by the
            # loader. The engine's compute_assertion takes the bypass
            # branch on those, skipping the input-residency and
            # output-IDLE checks that don't apply to pointer-only ops.
            # We just need to pre-claim cross-device outputs (so the
            # tensor lands on its home memory before any downstream
            # consumer wakes up) and submit on the node's natural compute.
            #
            # Edge case: a dispatcher node with no parents (e.g., the
            # very first `aten::empty(device="cuda")` on a cpu_thread)
            # has dispatcher_outputs set but empty custom_deps (the
            # loader only adds NodeDoneDeps from parent_nodes, which
            # are []). Without pre-claim its cross-device output never
            # lands on the home memory, and downstream consumers
            # deadlock waiting for a region that never appears. Route
            # such nodes through the same dispatcher branch.
            is_dispatcher_path = bool(node.custom_deps) or bool(node.args.get("dispatcher_outputs"))
            if is_dispatcher_path:
                cap_x = getattr(compute, "max_concurrent_jobs", 1)
                running_x = len(compute.job_running)
                committed_x = committed_per_compute.get(compute, 0)
                if running_x + committed_x >= cap_x:
                    self.ready_node_ids.append(node_id)
                    continue
                # If the node has real outputs (e.g. a gpu_runtime kernel
                # that landed on this branch because python-release
                # patched custom_deps onto it to bypass input residency
                # for a freed tid), claim their regions here. The
                # original dispatcher / alias cases have their real
                # output_tensors stripped, so this is a no-op for them.
                if node.output_tensors:
                    if not self._ensure_outputs_claimed(node, compute.memory):
                        self.ready_node_ids.appendleft(node_id)
                        continue
                    if not self._outputs_free(node, compute.memory):
                        self.ready_node_ids.append(node_id)
                        continue
                if node.args.get("dispatcher_outputs"):
                    self._preclaim_dispatcher_outputs(node)
                self.sys.compute(compute, node)
                self._drop_pending_consumer(node)
                self._diag_on_submit(node)
                self._release_started_children(node)
                committed_per_compute[compute] = committed_x + 1
                submitted_any = True
                continue

            cap = getattr(compute, "max_concurrent_jobs", 1)
            already_running = len(compute.job_running)
            committed = committed_per_compute.get(compute, 0)
            if already_running + committed >= cap:
                self.ready_node_ids.append(node_id)
                continue

            memory = compute.memory
            if not self._ensure_outputs_claimed(node, memory):
                self.ready_node_ids.appendleft(node_id)
                continue

            transfers = self._ensure_inputs_resident(node, memory)
            if transfers is None:
                self.ready_node_ids.append(node_id)
                continue

            if transfers:
                self._submit_transfer_batches(transfers)
                for _src, dst in transfers:
                    self._tick_pending_dest_ids.add(id(dst))
                self.ready_node_ids.append(node_id)
                continue

            # Guard: if any output region has a transfer queued or running
            # targeting it, the compute would begin() and fail the
            # output-IDLE check. Wait.
            if not self._outputs_free(node, memory):
                self.ready_node_ids.append(node_id)
                continue

            # If any input is python-released (last CPU consumer has
            # retired, VRAM region freed), the engine's compute_assertion
            # would fail its input-residency check. Add NodeDoneDep
            # custom_deps so the engine takes the bypass branch on this
            # node: it skips the input/output residency checks and
            # gates only on the parents' DONE status. Output regions
            # claimed above by `_ensure_outputs_claimed` are still
            # picked up by begin_mutation.
            if any(tid in self._python_released_tids for tid in node.input_tensors):
                if not node.custom_deps:
                    for parent_id in node.parent_nodes:
                        node.custom_deps.append(NodeDoneDep(parent_id))

            self.sys.compute(compute, node)
            self._drop_pending_consumer(node)
            self._diag_on_submit(node)
            self._release_started_children(node)
            committed_per_compute[compute] = committed + 1
            submitted_any = True

        return submitted_any

    def _diag_on_submit(self, node: Node) -> None:
        if (node.args.get("runtime_role") or "") != "gpu_runtime":
            return
        self._diag_outstanding_gpu += 1
        if self._diag_outstanding_gpu > self._diag_max_outstanding_gpu:
            self._diag_max_outstanding_gpu = self._diag_outstanding_gpu
        # Sparse log: only when count crosses each 50-step or each 100ms.
        sim_t = self.sys.engine.timestamp_now
        if (not self._diag_outstanding_log
                or sim_t - self._diag_outstanding_log[-1][0] > 1000.0
                or self._diag_outstanding_gpu - self._diag_outstanding_log[-1][1] >= 50
                or self._diag_outstanding_log[-1][1] - self._diag_outstanding_gpu >= 50):
            self._diag_outstanding_log.append((sim_t, self._diag_outstanding_gpu))

    def _diag_on_retire(self, node: Node) -> None:
        if (node.args.get("runtime_role") or "") != "gpu_runtime":
            return
        if self._diag_outstanding_gpu > 0:
            self._diag_outstanding_gpu -= 1
        # Final dump at terminal node retire (= sim end).
        if isinstance(node, TerminalNode):
            self._diag_dump_outstanding()

    def _diag_dump_outstanding(self) -> None:
        from collections import Counter
        print(f"[DAV diag] max outstanding gpu_runtime: {self._diag_max_outstanding_gpu}", flush=True)
        if not self._diag_outstanding_log:
            return
        buckets = Counter()
        for _, n in self._diag_outstanding_log:
            if n < 16:    buckets['0-15'] += 1
            elif n < 64:  buckets['16-63'] += 1
            elif n < 256: buckets['64-255'] += 1
            elif n < 1024: buckets['256-1023'] += 1
            else:          buckets['>=1024'] += 1
        print(f"[DAV diag] outstanding histogram (sampled):", flush=True)
        for k, v in buckets.most_common():
            print(f"    {k}: {v}", flush=True)
        # Show evolution: every Nth sample
        n = len(self._diag_outstanding_log)
        step = max(1, n // 20)
        print(f"[DAV diag] outstanding timeline (sampled @ ~{step}-step intervals):", flush=True)
        for i in range(0, n, step):
            sim_t, count = self._diag_outstanding_log[i]
            print(f"    sim_t={sim_t/1000:.1f}ms outstanding={count}", flush=True)

    def _submit_ready_nodes(self) -> bool:
        """Wrap the core submit loop with the prefetch gate.

        We pre-filter ``ready_node_ids``: any node whose gate is
        unsatisfied is moved to a parked queue and re-injected at the
        end of the tick. The core loop runs over the remaining
        gate-satisfied nodes.
        """
        if not self._gate_by_consumer:
            return self._submit_ready_nodes_core()

        parked: deque[int] = deque()
        passable: deque[int] = deque()
        while self.ready_node_ids:
            nid = self.ready_node_ids.popleft()
            if self._node_xfer_gate_satisfied(nid):
                passable.append(nid)
            else:
                parked.append(nid)

        self.ready_node_ids = passable
        any_submitted = self._submit_ready_nodes_core()

        # Re-merge: parked nodes go back; whatever the parent left in
        # ready_node_ids stays ahead so it's tried first next tick.
        parked.extend(self.ready_node_ids)
        self.ready_node_ids = parked
        return any_submitted

    # ============================================================ retire
    def _retire_completed_nodes(self, retired_jobs: list[BaseJob]) -> None:
        for job in retired_jobs:
            if not isinstance(job, ComputeJob):
                continue
            self._diag_on_retire(job.node)

            self._consume_inputs(job.node)
            self._python_release_producer_outputs(job.node)
            self._release_dead_outputs(job.node)
            # Per-node explicit release (set by schedule injection): if
            # the trace's evict_after_node[node_id] lists tensors, they
            # are released NOW even if their type is permanent. This is
            # how cg-sim mirrors real PyTorch's resize_(0) eviction at
            # the schedule's evict_lid (last_use boundary), without
            # waiting for the global last-consumer-retire.
            evict_after = self.sys.trace.args.get("evict_after_node", {})
            if evict_after:
                tids = evict_after.get(job.node.id) or evict_after.get(str(job.node.id))
                if tids:
                    for tid in tids:
                        # Free VRAM mirror only; keep RAM home so the
                        # next reload is RAM→VRAM (mirroring real
                        # PyTorch's pinned backup pattern). Releasing
                        # RAM too would force SSD→RAM→VRAM and inflate
                        # sim time. Deferred releases route through
                        # ``_d2h_pending_vram`` (not ``_pending_releases``)
                        # so the retry path stays VRAM-only — the latter
                        # uses the full RAM+VRAM ``_release_tensor_regions``
                        # which would drop the RAM mirror needed by the
                        # next reload arrival.
                        deferred = False
                        for region in list(
                            self._vram.space.get_by_tensor_id(int(tid))
                        ):
                            if region.access_status == DataRegionAccess.IDLE:
                                self.sys.release(region)
                            else:
                                deferred = True
                        if deferred:
                            self._d2h_pending_vram.add(int(tid))
                        # Also flip xfer_state back so the next prefetch
                        # arrival sees ABSENT and re-issues the H2D.
                        if not self._vram.space.get_by_tensor_id(int(tid)):
                            if self._xfer_state.get(int(tid)) in (_LOADED, _RESIDENT):
                                self._xfer_state[int(tid)] = _ABSENT

            # Start-gated edges live on the trace's args side channel,
            # not in `children_nodes`, so this iteration only sees
            # regular control children — no special-case needed.
            for child_id in job.node.children_nodes:
                if child_id not in self.pending_parent_count:
                    continue
                if self.pending_parent_count[child_id] > 0:
                    self.pending_parent_count[child_id] -= 1
                if (self.pending_parent_count[child_id] == 0
                        and self._pending_started_count.get(child_id, 0) == 0):
                    child = self.sys.trace.node_map[child_id]
                    if child.status == NodeStatus.TODO:
                        self.ready_node_ids.append(child_id)

        return

    def _release_dead_outputs(self, node: Node) -> None:
        """Free outputs that no downstream node reads. Real PyTorch drops
        such tensors via Python ref-counting as soon as the producing op
        returns; without this, we'd hold their region for the rest of the
        run."""
        for tid in node.output_tensors:
            if tid in node.input_tensors:
                continue
            if self._remaining_consumers.get(tid, 0) > 0:
                continue
            tensor = self.sys.trace.tensor_map.get(tid)
            if tensor is None:
                continue
            if tensor.args.get("tensor_type") in self._PERMANENT_TYPES:
                continue
            self._release_tensor_regions(tid)

    # ============================================================ async hooks
    def _build_arrival_index(self) -> None:
        arrivals = self.sys.trace.args.get("xfer_arrivals") or []
        for a in arrivals:
            issuer = int(a["issuer_node_id"])
            consumer = int(a["consumer_node_id"])
            tids = [int(t) for t in a["cgsim_tids"]]
            self._arrivals_by_issuer[issuer].append({
                "consumer_node_id": consumer,
                "cgsim_tids": tids,
            })
            self._gate_by_consumer[consumer].update(tids)
            for t in tids:
                self._pending_consumers_by_tid[t].add(consumer)

        # D2H (VRAM -> RAM) arrivals: fire a VRAM->RAM TransferJob when
        # the issuer node retires. The issuer is the streamed tid's last
        # GPU consumer (so the VRAM region is IDLE by then), and on
        # TransferJob retire the VRAM region is freed.
        d2h_arrivals = self.sys.trace.args.get("d2h_xfer_arrivals") or []
        for a in d2h_arrivals:
            issuer = int(a["issuer_node_id"])
            tids = [int(t) for t in a["cgsim_tids"]]
            if tids:
                self._d2h_arrivals_by_issuer[issuer].append(tids)

    def _init_xfer_states(self) -> None:
        for tid, tensor in self.sys.trace.tensor_map.items():
            ttype = tensor.args.get("tensor_type")
            if ttype not in ("WEIGHT", "LEAF"):
                continue
            device = str(tensor.args.get("device", "")).lower()
            if device.startswith("cuda"):
                self._xfer_state[tid] = _RESIDENT
            else:
                self._xfer_state[tid] = _ABSENT

    def _handle_transfer_retires(self, retired_jobs: list[BaseJob]) -> None:
        for job in retired_jobs:
            if not isinstance(job, TransferJob):
                continue
            tids = self._inflight_jobs.pop(job.id, None)
            if tids:
                self._active_prefetch_jobs = max(0, self._active_prefetch_jobs - 1)
                for tid in tids:
                    if self._xfer_state.get(tid) == _LOADING:
                        self._xfer_state[tid] = _LOADED
                    # Drain awaiters registered for this tid: they
                    # piggybacked on this in-flight load and are now
                    # credited as gate-clear regardless of later
                    # evictions.
                    awaiters = self._loading_awaiters.pop(tid, None)
                    if awaiters:
                        for cid in awaiters:
                            self._pre_satisfied_gate[cid].add(tid)
                continue
            # D2H retire: release ONLY the VRAM region for each tid so
            # peak accounting drops the moment the bytes finish moving
            # back to RAM (mirrors real HF's `module.to("cpu")` storage
            # release after the cudaMemcpyAsync DtoH completes). The
            # RAM mirror MUST be preserved — the next H2D prefetch
            # reads from it (real PyTorch keeps the pinned CPU copy
            # around exactly so reloads don't have to go to SSD).
            d2h_tids = self._inflight_d2h_jobs.pop(job.id, None)
            if d2h_tids:
                self._active_prefetch_jobs = max(0, self._active_prefetch_jobs - 1)
                for tid in d2h_tids:
                    self._release_vram_only(int(tid))

    def _fire_prefetches_for_retired(self, retired_jobs: list[BaseJob]) -> None:
        for job in retired_jobs:
            if not isinstance(job, ComputeJob):
                continue
            arrivals = self._arrivals_by_issuer.pop(job.node.id, None)
            if not arrivals:
                continue
            for a in arrivals:
                consumer_id = int(a["consumer_node_id"])
                tids = a["cgsim_tids"]
                # Per-tid pre-satisfaction bookkeeping. The arrival's
                # contract is "ensure this consumer has its tids
                # loaded when it's ready to dispatch." Three cases:
                #   - LOADED/RESIDENT: data already there — credit
                #     consumer immediately so a later eviction won't
                #     strand it.
                #   - LOADING: a peer arrival is loading it. Register
                #     this consumer to inherit the load when the
                #     in-flight TransferJob retires.
                #   - ABSENT: a fresh prefetch will be issued below.
                #     Register so when that prefetch's TransferJob
                #     retires this consumer is credited.
                for tid in tids:
                    st = self._xfer_state.get(tid, _ABSENT)
                    if st in (_LOADED, _RESIDENT):
                        self._pre_satisfied_gate[consumer_id].add(tid)
                    else:
                        self._loading_awaiters[tid].append(consumer_id)
                if self._h2d_streams > 0:
                    self._prefetch_queue.append(("h2d", tids))
                else:
                    self._issue_prefetch(tids)

    def _fire_d2h_for_retired(self, retired_jobs: list[BaseJob]) -> None:
        """Schedule VRAM->RAM transfers for any d2h_xfer_arrivals
        attached to the retired GPU compute nodes. When ``h2d_streams``
        is configured (e.g. =1 for default-stream HF accelerate), D2H
        shares the same stream slot as H2D so the two cannot run
        concurrently — modelling real PyTorch's default-stream
        cudaMemcpyAsync serialization at ``cpu_offload_with_hook``'s
        ``pre_forward`` (which queues D2H of the previous component
        and H2D of the next back-to-back on the default stream).
        """
        for job in retired_jobs:
            if not isinstance(job, ComputeJob):
                continue
            batches = self._d2h_arrivals_by_issuer.pop(job.node.id, None)
            if not batches:
                continue
            for tids in batches:
                if self._h2d_streams > 0:
                    self._prefetch_queue.append(("d2h", tids))
                else:
                    self._issue_d2h(tids)

    def _issue_d2h(self, cgsim_tids: list[int]) -> bool:
        """Fire a VRAM->RAM transfer for the given tensors."""
        batch: list[tuple[DataRegion, DataRegion]] = []
        moved_tids: list[int] = []
        for tid in cgsim_tids:
            vram_regions = self._vram.space.get_by_tensor_id(int(tid))
            if not vram_regions:
                # Already evicted (e.g. via a different path) — skip.
                continue
            src = vram_regions[0]
            ram_regions = self._ram.space.get_by_tensor_id(int(tid))
            if not ram_regions:
                # No RAM mirror (streamed tids should always have one).
                continue
            dst = ram_regions[0]
            if src.access_status == DataRegionAccess.BEING_WRITTEN:
                # Still inbound; cannot start D2H. Skip — eviction will
                # be redundant once the H2D + last consumer have run.
                continue
            batch.append((src, dst))
            moved_tids.append(int(tid))
        if not batch:
            return False
        job_id = self.sys.transfer(batch)
        self._inflight_d2h_jobs[job_id] = moved_tids
        return True

    def _drain_prefetch_queue(self) -> None:
        if self._h2d_streams <= 0:
            return
        while (
            self._prefetch_queue
            and self._active_prefetch_jobs < self._h2d_streams
        ):
            kind, cgsim_tids = self._prefetch_queue.popleft()
            issued = (
                self._issue_d2h(cgsim_tids) if kind == "d2h"
                else self._issue_prefetch(cgsim_tids)
            )
            if issued:
                self._active_prefetch_jobs += 1

    def _issue_prefetch(self, cgsim_tids: list[int]) -> bool:
        """Fire async RAM->VRAM transfer for the given tensors."""
        batch: list[tuple[DataRegion, DataRegion]] = []
        loaded_tids: list[int] = []
        for tid in cgsim_tids:
            st = self._xfer_state.get(tid, _ABSENT)
            if st in (_RESIDENT, _LOADED, _LOADING):
                continue
            tensor = self.sys.trace.tensor_map.get(tid)
            if tensor is None:
                continue
            # Normally source from the RAM home. Under SSD-streaming
            # (DAV_STREAM_FROM_SSD) the RAM home was never loaded at
            # layout, so it exists but is empty/not-readable — fall back
            # to the latest readable region (the SSD copy). The transfer
            # then runs SSD->VRAM at the SSD read_io_curve bandwidth.
            src_regions = self._ram.space.get_by_tensor_id(tid)
            src = src_regions[0] if src_regions else None
            if src is None or not self._region_readable(src):
                src = self._find_latest_region(tid, exclude_hw=self._vram)
            if src is None:
                continue
            dst = self._claim_region(self._vram, tensor)
            if dst is None:
                continue
            batch.append((src, dst))
            self._xfer_state[tid] = _LOADING
            loaded_tids.append(tid)
        if not batch:
            return False
        job_id = self.sys.transfer(batch)
        self._inflight_jobs[job_id] = loaded_tids
        return True

    def _node_xfer_gate_satisfied(self, node_id: int) -> bool:
        needed = self._gate_by_consumer.get(node_id)
        if not needed:
            return True
        pre = self._pre_satisfied_gate.get(node_id) or ()
        for tid in needed:
            if tid in pre:
                continue
            st = self._xfer_state.get(tid, _ABSENT)
            if st not in (_RESIDENT, _LOADED):
                return False
        return True

    # ============================================================ runtime
    def runtime(self, retired_jobs: list[BaseJob]) -> None:
        # Phase 1: walk retired jobs.
        # - TransferJob retire flips LOADING -> LOADED.
        # - ComputeJob retire fires any prefetches scheduled on this node
        #   AFTER the standard _retire_completed_nodes runs (which honors
        #   evict_after_node, releasing VRAM regions before we try to
        #   claim them for prefetch destinations).
        self._handle_transfer_retires(retired_jobs)
        self._retire_completed_nodes(retired_jobs)
        # Retry deferred releases BEFORE firing new prefetches. A
        # pending VRAM-only eviction (from the prior run's
        # evict_after_node, deferred because the region was being read
        # at the moment of the trigger node's retire) needs to flip
        # `xfer_state` back to ABSENT before the next run's prefetch
        # arrival fires. Otherwise `_issue_prefetch` sees state LOADED,
        # silently skips, returns False, and the popped arrival is
        # lost — leaving the next consumer permanently parked on a
        # gate that nothing will satisfy.
        self._retry_pending_releases()
        self._fire_prefetches_for_retired(retired_jobs)
        self._fire_d2h_for_retired(retired_jobs)
        self._drain_prefetch_queue()

        # Iterate: retiring view-like ops in place unblocks children that
        # may themselves be ready to schedule (or be view-like). Keep going
        # until a pass makes no further progress.
        while self._submit_ready_nodes():
            pass

        if not self.sys.engine.job_running and not self.sys.engine.job_waiting:
            todo = [
                n for n in self.sys.trace.node_map.values()
                if n.status == NodeStatus.TODO
            ]
            if todo:
                blocked = todo[0]
                gate = self._gate_by_consumer.get(blocked.id, set())
                gate_status = [
                    {"tid": t, "state": self._xfer_state.get(t, "?")}
                    for t in gate
                ]
                # Identify the prefetch arrival keyed to this consumer:
                # which issuer was supposed to fire it, has that issuer
                # already retired, and is the arrival still pending in
                # the per-issuer index? Distinguishes "issuer never ran"
                # (upstream blocker) from "issuer ran but prefetch was
                # silently dropped" (the bug class to chase).
                arrival_diag = []
                for tid in gate:
                    if self._xfer_state.get(tid) != _ABSENT:
                        continue
                    # Find any issuer whose pending arrivals include this tid.
                    for issuer_id, arrs in self._arrivals_by_issuer.items():
                        for a in arrs:
                            if tid in a["cgsim_tids"]:
                                issuer_node = self.sys.trace.node_map.get(issuer_id)
                                issuer_status = (
                                    str(issuer_node.status) if issuer_node else "?"
                                )
                                arrival_diag.append({
                                    "tid": tid,
                                    "issuer_id": issuer_id,
                                    "issuer_status": issuer_status,
                                    "consumer_id": a["consumer_node_id"],
                                })
                                break
                self.sys.abort({
                    "from": self.name,
                    "error": "SCHEDULER_DEADLOCK",
                    "msg": "No runnable node is available.",
                    "node": {
                        "id": blocked.id,
                        "name": blocked.name,
                        "parent_nodes": blocked.parent_nodes,
                        "input_tensors": blocked.input_tensors,
                        "output_tensors": blocked.output_tensors,
                        "args": blocked.args,
                    },
                    "xfer_gate": gate_status,
                    "pending_arrivals": arrival_diag,
                    "active_prefetch_jobs": self._active_prefetch_jobs,
                    "prefetch_queue_len": len(self._prefetch_queue),
                })

        return
