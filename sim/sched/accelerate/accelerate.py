"""Simulator implementation of ``accelerate.cpu_offload`` (HuggingFace).

Algorithm (from ``accelerate_cpu-offload-spec.md`` in the repo root):

  Phase 1 — Attach (modeled at layout time):
    Every WEIGHT tensor is parked in the pageable CPU master copy (RAM).
    Loaded once via SSD->RAM, then never modified. No WEIGHTs are
    pre-loaded into VRAM. Non-WEIGHT INPUT/LEAF tensors are placed on
    their declared device (matches accelerate's view of activations
    and the "buffers stay resident" rule under ``offload_buffers=False``).

  Phase 2 — Per-forward (modeled at runtime, per cgsim compute node):
    For every ready GPU compute node:
      - For each WEIGHT input that lacks a ready VRAM copy: claim a
        VRAM region and queue a RAM->VRAM transfer. Mark the tensor
        ``LOADING``; the compute is gated until the transfer retires
        (``_LOADED``). One synchronous logical stream.
      - For each non-WEIGHT input: standard "ensure resident on
        compute.memory" check (queues cross-device transfers as
        needed for activations).
      - On compute retire: release the VRAM region of every WEIGHT
        input whose last consumer has now retired. The RAM master
        copy is *never* touched — accelerate post_forward replaces
        the on-module tensor with a zero-byte meta placeholder, the
        CPU copy is untouched.

Knobs (spec section 9):
  - ``offload_buffers`` (default False): True is not supported because
    the pytorch profile bundle CSV does not distinguish parameter and
    buffer tensors. Raises NotImplementedError if set.
  - ``preload_module_classes`` (default None / []): list of nn.Module
    class names. WEIGHTs whose owning-module path falls under an
    ancestor in this set are coalesced into one offload group and
    loaded/evicted atomically (real accelerate's ``place_submodules``
    behavior). Default config → 1 group per cuda-WEIGHT leaf.

Notes vs the spec:
  - cg-sim transfers run on the memory subsystem (PCIe bandwidth)
    parallel to GPU compute on the compute unit, so H2D can overlap
    with unrelated GPU work — peak weight residency exceeds spec's
    strict "1 leaf module at a time" by a small constant.
  - Speculative H2D prefetch: a CPU dispatcher or CPU-compute that
    references a cuda-WEIGHT input kicks the RAM->VRAM load alongside
    its own work so downstream GPU kernels find the weight in flight.
    Single logical stream, no pinning, no write-back (RAM master is
    never modified — matches accelerate's CPU state_dict semantics).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, TYPE_CHECKING

from sim.core.job import BaseJob, ComputeJob, TransferJob
from sim.core.log import Log
from sim.core.trace import Node, NodeHW, NodeStatus, Tensor, TerminalNode, Trace
from sim.hw.common import DataRegion, DataRegionAccess
from sim.hw.compute.common import BaseCompute
from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage
from sim.sched.common import BaseScheduler

if TYPE_CHECKING:
    from sim.core.system import System


# Per-WEIGHT residency states. CONTEXT / INPUT / LEAF tensors track
# residency via the cg-sim DataRegion's own (is_ready, is_latest,
# access_status) flags, so they don't need a parallel state machine.
_ABSENT = "ABSENT"    # WEIGHT lives in RAM only
_LOADING = "LOADING"  # RAM->VRAM transfer in flight
_LOADED = "LOADED"    # VRAM region ready


class Accelerate(BaseScheduler):
    """Implements ``accelerate.cpu_offload``."""

    # Non-WEIGHT tensors whose region must never be released.
    # WEIGHTs are intentionally excluded — they are evicted from VRAM
    # after each leaf module's last consumer retires, while the RAM
    # master copy stays for the whole run.
    _PERMANENT_TYPES = frozenset({"INPUT", "LEAF"})

    # Page size used by every memory in cg-sim (see base_memory.py).
    _PAGE_SIZE_KB: int = 4

    def __init__(
        self,
        obj_id: int,
        name: str,
        log: Log,
        sys: System,
        args: dict[str, Any] | None = None,
    ):
        super().__init__(obj_id, name, log, sys, args)

        # ---------------- Topology discovery ----------------
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
            raise Exception(f"[Accelerate] CPU compute '{cpu_compute_name}' does not exist.")
        if cuda_compute_name not in self.compute_by_name:
            raise Exception(f"[Accelerate] CUDA compute '{cuda_compute_name}' does not exist.")

        self.cpu_compute: BaseCompute = self.compute_by_name[cpu_compute_name]
        self.cuda_compute: BaseCompute = self.compute_by_name[cuda_compute_name]
        self._ram: BaseMemory = self.cpu_compute.memory
        self._vram: BaseMemory = self.cuda_compute.memory

        # ---------------- Algorithmic knobs (spec section 9) ----------------
        # offload_buffers: real accelerate's flag for treating buffers as
        # offload participants. The bundle CSV only emits WEIGHT vs
        # CONTEXT — there is no BUFFER tag — so the True branch cannot
        # be faithfully simulated without changes to the bundle producer.
        self.offload_buffers: bool = bool(self.args.get("offload_buffers", False))
        if self.offload_buffers:
            raise NotImplementedError(
                "[Accelerate] offload_buffers=True is not yet supported: "
                "the pytorch profile bundle CSV does not distinguish "
                "parameters from buffers (all are tagged tensor_kind=WEIGHT). "
                "Emit a BUFFER tag from the bundle producer, then teach "
                "the loader/scheduler to treat them separately."
            )

        # preload_module_classes: list of nn.Module class names whose
        # whole subtree should load/evict as a single unit (place_submodules
        # in real accelerate). Empty/None → per-leaf hook granularity
        # (every direct-tensor-owning leaf is its own group). The bundle
        # CSV's `module_class` column carries ancestor-class annotations
        # for nodes whose immediate owning module is the ancestor (e.g.
        # the residual-add op inside LlamaDecoderLayer is tagged
        # module_class=LlamaDecoderLayer), so we can resolve group roots
        # from the trace directly.
        raw_preload = self.args.get("preload_module_classes")
        if raw_preload is None:
            raw_preload = []
        if not isinstance(raw_preload, (list, tuple)):
            raise Exception(
                f"[Accelerate] preload_module_classes must be a list of strings; "
                f"got {type(raw_preload).__name__}."
            )
        self.preload_module_classes: frozenset[str] = frozenset(str(s) for s in raw_preload)

        # ---------------- Trace bookkeeping ----------------
        self.pending_parent_count: dict[int, int] = {
            n.id: len(n.parent_nodes) for n in self.sys.trace.node_map.values()
        }
        self.ready_node_ids: deque[int] = deque(
            n.id for n in self.sys.trace.node_map.values()
            if self.pending_parent_count[n.id] == 0
        )

        # Per-tensor remaining-consumer counter (over the whole DAG).
        # WEIGHT VRAM regions are evicted when their *group's* total
        # counter hits zero; CONTEXT tensor regions are released across
        # all memories when this hits zero (unless permanent).
        self._remaining_consumers: dict[int, int] = {}
        for node in self.sys.trace.node_map.values():
            for tid in node.input_tensors:
                self._remaining_consumers[tid] = self._remaining_consumers.get(tid, 0) + 1

        # ---------------- Offload-group construction ----------------
        # tid -> group_id (string). Default config: every cuda-WEIGHT is
        # its own singleton group, preserving per-leaf hook granularity.
        # With preload_module_classes, cuda-WEIGHTs whose owning module
        # path falls under a matching ancestor are merged into one group.
        self._tensor_group: dict[int, str] = {}
        self._group_tids: dict[str, list[int]] = {}
        self._build_offload_groups()

        # Per-group residency state and consumer accounting (cuda-WEIGHT
        # only; cpu-device WEIGHTs stay in RAM forever).
        self._group_state: dict[str, str] = {gid: _ABSENT for gid in self._group_tids}
        self._group_remaining: dict[str, int] = {}
        for gid, members in self._group_tids.items():
            self._group_remaining[gid] = sum(
                self._remaining_consumers.get(tid, 0) for tid in members
            )

        # TransferJob.id -> list of WEIGHT tensor ids it is loading.
        self._inflight_load_tids: dict[Any, list[int]] = {}

        # Releases deferred because regions were BEING_READ at the time
        # we tried to release them; retried each tick. Keyed by group_id.
        self._pending_vram_group_releases: set[str] = set()
        self._pending_releases: set[int] = set()

        # Multi-phase layout: 0=claim, 1=ssd->ram, 2=ssd->vram, 3=done.
        self._layout_phase: int = 0

        # Outputs of dispatcher / alias nodes that the loader hid in
        # ``node.args["dispatcher_outputs"]``. Pre-claim on the home
        # memory before submission so downstream consumers find them.
        # Same as DAV — the loader-driven contract.
        self._tick_pending_dest_ids: set[int] = set()

        # ---------------- Explicit spec-state accumulators (spec §8) ----------------
        # Cached after layout completes; gpu_resident_bytes is computed
        # on demand because it changes throughout the run. peak_gpu_resident_bytes
        # tracks the high-water mark of WEIGHT VRAM residency.
        self.cpu_state_dict_bytes: int = 0
        self.gpu_persistent_buffer_bytes: int = 0
        self.peak_gpu_resident_bytes: int = 0
        return

    # ============================================================ groups
    def _build_offload_groups(self) -> None:
        """Assign every cuda-device WEIGHT to an offload group.

        Default (no ``preload_module_classes``): each cuda-WEIGHT is its
        own singleton, matching per-leaf hook granularity in real
        accelerate.

        With ``preload_module_classes``: scan the bundle's per-node
        ``module`` annotation to build ``module_path -> module_class``
        (PyTorch profiler tags each op with the innermost owning
        nn.Module, which may be an ancestor of the leaf weight's
        Linear/etc., so ancestor paths show up naturally for ops
        executed directly in the ancestor's forward). Paths whose class
        is in the preload set become group roots; cuda-WEIGHTs whose
        owning-leaf path falls under a root share that root's group_id.
        Deeper roots win on ties.
        """
        trace = self.sys.trace
        # All cuda-device WEIGHTs participate in load/evict; cpu-device
        # WEIGHTs stay in RAM forever and are not group members.
        weight_tids = [
            tid for tid, t in trace.tensor_map.items()
            if t.args.get("tensor_type") == "WEIGHT"
            and str(t.args.get("device", "")).lower().startswith("cuda")
        ]

        if not self.preload_module_classes:
            for tid in weight_tids:
                gid = f"w:{tid}"
                self._tensor_group[tid] = gid
                self._group_tids[gid] = [tid]
            return

        # path -> module_class (first seen wins; all node rows on the
        # same path carry the same class in practice).
        path_to_class: dict[str, str] = {}
        for node in trace.node_map.values():
            mod = node.args.get("module")
            if not mod:
                continue
            path = mod.get("module_path")
            cls = mod.get("module_class")
            if path and cls:
                path_to_class.setdefault(path, cls)

        # tid -> set of all consumer-module paths. Tracking every path
        # (not just the first-seen) closes a coverage gap: a WEIGHT
        # consumed by both a leaf Linear (path under the layer) and an
        # un-annotated submit/wait node would have missed the layer
        # group if first-seen happened to be the un-annotated one.
        tid_to_paths: dict[int, set[str]] = {}
        for node in trace.node_map.values():
            mod = node.args.get("module")
            if not mod:
                continue
            path = mod.get("module_path")
            if not path:
                continue
            for tid in node.input_tensors:
                t = trace.tensor_map.get(tid)
                if t is None or t.args.get("tensor_type") != "WEIGHT":
                    continue
                tid_to_paths.setdefault(tid, set()).add(path)

        def _matching_root(path: str) -> str | None:
            """Walk the dot-hierarchy from leaf to root; return the
            deepest ancestor whose class is in the preload set."""
            parts = path.split(".")
            for depth in range(len(parts), 0, -1):
                ancestor = ".".join(parts[:depth])
                if path_to_class.get(ancestor) in self.preload_module_classes:
                    return ancestor
            return None

        for tid in weight_tids:
            best_root: str | None = None
            best_depth = -1
            for path in tid_to_paths.get(tid, ()):
                root = _matching_root(path)
                if root is None:
                    continue
                d = root.count(".")
                if d > best_depth:
                    best_depth = d
                    best_root = root
            gid = f"g:{best_root}" if best_root is not None else f"w:{tid}"
            self._tensor_group[tid] = gid
            self._group_tids.setdefault(gid, []).append(tid)
        return

    # ============================================================ spec-state accumulators
    @property
    def gpu_resident_bytes(self) -> int:
        """Current WEIGHT bytes resident on VRAM (spec §8 state field).
        Sums over *all* WEIGHTs — grouped cuda-WEIGHTs plus any cpu-device
        WEIGHT that was promoted via the singleton-fallback path."""
        total_pages = 0
        for tid, t in self.sys.trace.tensor_map.items():
            if t.args.get("tensor_type") != "WEIGHT":
                continue
            for r in self._vram.space.get_by_tensor_id(tid):
                total_pages += r.page_idx_end - r.page_idx_start
        return total_pages * self._PAGE_SIZE_KB * 1024

    def _refresh_peak_gpu_resident(self) -> None:
        cur = self.gpu_resident_bytes
        if cur > self.peak_gpu_resident_bytes:
            self.peak_gpu_resident_bytes = cur
        return

    # ============================================================ compile
    def compile(self, trace: Trace) -> None:
        return

    # ============================================================ topology helpers
    def _compute_for_node(self, node: Node) -> BaseCompute:
        # Terminal nodes are synthetic — keep them on CPU.
        if isinstance(node, TerminalNode):
            return self.cpu_compute
        if node.hw == NodeHW.GPU:
            return self.cuda_compute
        return self.cpu_compute

    def _home_memory_for_non_weight(self, tensor: Tensor) -> BaseMemory:
        """Where a non-WEIGHT tensor's home region lives. Driven by the
        tensor's declared device. WEIGHTs always live in RAM and are
        handled separately."""
        device = str(tensor.args.get("device", "cpu")).lower()
        if device.startswith("cuda"):
            return self._vram
        return self._ram

    @staticmethod
    def _region_readable(region: DataRegion) -> bool:
        return (
            region.is_ready
            and region.is_latest
            and region.access_status in (DataRegionAccess.IDLE, DataRegionAccess.BEING_READ)
        )

    def _find_free_page(self, memory: BaseMemory, num_pages: int) -> int | None:
        cursor = 0
        for region in memory.space._regions_by_page_idx_start.values():
            if region.page_idx_start - cursor >= num_pages:
                return cursor
            cursor = max(cursor, region.page_idx_end)
        if memory.space.num_total_pages - cursor >= num_pages:
            return cursor
        return None

    def _claim_region(self, memory: BaseMemory, tensor: Tensor):
        page_idx = self._find_free_page(memory, tensor.num_pages)
        if page_idx is None:
            self.sys.abort({
                "from": self.name,
                "error": "OOM",
                "msg": f"No free pages on {memory.name} for tensor {tensor.id} ({tensor.name}, {tensor.num_pages} pages).",
                "memory": {
                    "name": memory.name,
                    "used_pages": memory.space.num_used_pages,
                    "total_pages": memory.space.num_total_pages,
                },
            })
            return None
        return self.sys.claim(memory, tensor, page_idx)

    # ============================================================ layout
    def layout(self, init_storage: BaseStorage) -> bool:
        """Three-phase layout.

        Phase 0: Claim home regions.
          - Every WEIGHT tensor: RAM region (the CPU master copy).
            No VRAM stamp — that's the whole point of cpu_offload.
          - Every non-WEIGHT INPUT/LEAF tensor: region on its declared
            home memory (RAM or VRAM).

        Phase 1: One batched SSD->RAM transfer covering every WEIGHT
        and every cpu-homed initial tensor.

        Phase 2: One batched SSD->VRAM transfer for every cuda-homed
        non-WEIGHT initial tensor.

        SSD can only run one job at a time, so phases 1 and 2 must be
        separate layout-loop iterations (one transfer drains before
        the next is queued).
        """
        if self._layout_phase == 0:
            for tensor in self.sys.trace.tensor_map.values():
                ttype = tensor.args.get("tensor_type")
                if ttype == "WEIGHT":
                    if self._claim_region(self._ram, tensor) is None:
                        return True  # abort already signalled
                elif ttype in ("INPUT", "LEAF"):
                    home = self._home_memory_for_non_weight(tensor)
                    if self._claim_region(home, tensor) is None:
                        return True
            self._layout_phase = 1
            return False

        if self._layout_phase == 1:
            ssd_to_ram: list[tuple[DataRegion, DataRegion]] = []
            for tensor in self.sys.trace.tensor_map.values():
                ttype = tensor.args.get("tensor_type")
                if ttype not in ("WEIGHT", "INPUT", "LEAF"):
                    continue
                stor_regions = init_storage.space.get_by_tensor_id(tensor.id)
                if not stor_regions:
                    continue
                src = stor_regions[0]
                if ttype == "WEIGHT":
                    dest_regions = self._ram.space.get_by_tensor_id(tensor.id)
                else:
                    home = self._home_memory_for_non_weight(tensor)
                    if home is not self._ram:
                        continue  # SSD->VRAM goes in phase 2
                    dest_regions = home.space.get_by_tensor_id(tensor.id)
                if not dest_regions:
                    continue
                ssd_to_ram.append((src, dest_regions[0]))

            if ssd_to_ram:
                self.sys.transfer(ssd_to_ram)
            self._layout_phase = 2
            return False

        if self._layout_phase == 2:
            ssd_to_vram: list[tuple[DataRegion, DataRegion]] = []
            for tensor in self.sys.trace.tensor_map.values():
                ttype = tensor.args.get("tensor_type")
                if ttype not in ("INPUT", "LEAF"):
                    continue
                home = self._home_memory_for_non_weight(tensor)
                if home is not self._vram:
                    continue
                stor_regions = init_storage.space.get_by_tensor_id(tensor.id)
                if not stor_regions:
                    continue
                dest_regions = self._vram.space.get_by_tensor_id(tensor.id)
                if not dest_regions:
                    continue
                ssd_to_vram.append((stor_regions[0], dest_regions[0]))

            if ssd_to_vram:
                self.sys.transfer(ssd_to_vram)
            self._layout_phase = 3

            # Cache spec-section-8 set-once accumulators now that the
            # attach phase is complete.
            self.cpu_state_dict_bytes = sum(
                (r.page_idx_end - r.page_idx_start)
                for tid, t in self.sys.trace.tensor_map.items()
                if t.args.get("tensor_type") == "WEIGHT"
                for r in self._ram.space.get_by_tensor_id(tid)
            ) * self._PAGE_SIZE_KB * 1024
            self.gpu_persistent_buffer_bytes = sum(
                (r.page_idx_end - r.page_idx_start)
                for tid, t in self.sys.trace.tensor_map.items()
                if t.args.get("tensor_type") in self._PERMANENT_TYPES
                for r in self._vram.space.get_by_tensor_id(tid)
            ) * self._PAGE_SIZE_KB * 1024
            return True

        return True

    # ============================================================ residency
    def _find_latest_region(
        self, tensor_id: int, exclude_hw: BaseMemory | BaseStorage | None = None
    ) -> DataRegion | None:
        # Memory copies preferred over storage.
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

    def _idle_region_on(self, memory: BaseMemory, tensor_id: int) -> DataRegion | None:
        for region in memory.space.get_by_tensor_id(tensor_id):
            if region.access_status == DataRegionAccess.IDLE:
                return region
        return None

    def _pending_dest_ids(self) -> set[int]:
        ids: set[int] = set(self._tick_pending_dest_ids)
        for job in self.sys.engine.job_waiting:
            if isinstance(job, TransferJob):
                for _src, dst in job.batch:
                    ids.add(id(dst))
        for job in self.sys.engine.job_running:
            if isinstance(job, TransferJob):
                for _src, dst in job.batch:
                    ids.add(id(dst))
        return ids

    def _ensure_weight_loaded(
        self, tensor: Tensor, transfers: list[tuple[DataRegion, DataRegion]], loading_tids: list[int]
    ) -> bool | None:
        """For a WEIGHT input of a GPU compute node, make sure a ready
        VRAM region exists. If a transfer is already in flight, return
        True (still wait). If we need to queue a new one, append to
        ``transfers`` / ``loading_tids`` and return True. Returns False
        only if we can't make progress (no idle dest etc.) so the node
        is re-queued for next tick.

        Group-aware: staging operates on the tensor's whole offload
        group at once (under ``preload_module_classes``), so a single
        consumer can trigger an atomic load of every cuda-WEIGHT in
        the matching ancestor's subtree.
        """
        # If we already have a readable VRAM copy, we're done.
        for region in self._vram.space.get_by_tensor_id(tensor.id):
            if self._region_readable(region):
                return True

        gid = self._tensor_group.get(tensor.id)
        if gid is None:
            # cpu-device WEIGHT being consumed on a GPU compute: stage
            # it as a singleton, matching the pre-refactor behavior.
            # (Per spec these should be permanent-on-GPU buffers, but
            # the trace's reality may differ.)
            ram_regions = self._ram.space.get_by_tensor_id(tensor.id)
            if not ram_regions or not ram_regions[0].is_ready:
                return False
            dst = self._claim_region(self._vram, tensor)
            if dst is None:
                return False
            transfers.append((ram_regions[0], dst))
            loading_tids.append(tensor.id)
            return True

        state = self._group_state.get(gid, _ABSENT)
        if state == _LOADING:
            return True
        if state == _LOADED:
            # Anomaly: group says LOADED but this member isn't readable
            # (early-return check above already missed). With the
            # all-or-nothing release in _release_group_vram this branch
            # is normally unreachable; resync state to ABSENT and let
            # _stage_group repair the missing members. Other members
            # whose regions are still readable are skipped inside
            # _stage_group.
            self._group_state[gid] = _ABSENT
        return self._stage_group(gid, transfers, loading_tids)

    def _stage_group(
        self,
        gid: str,
        transfers: list[tuple[DataRegion, DataRegion]],
        loading_tids: list[int],
    ) -> bool:
        """Atomic: either every needs-staging member gets claimed+queued,
        or we touch nothing and return False. Two passes — first verify
        every member's RAM master is ready, then claim VRAM."""
        tensor_map = self.sys.trace.tensor_map
        to_stage: list[tuple[Tensor, DataRegion]] = []
        for member_tid in self._group_tids.get(gid, ()):
            member = tensor_map.get(member_tid)
            if member is None:
                continue
            if any(self._region_readable(r)
                   for r in self._vram.space.get_by_tensor_id(member_tid)):
                continue
            ram_regions = self._ram.space.get_by_tensor_id(member_tid)
            if not ram_regions or not ram_regions[0].is_ready:
                return False  # bail before claiming anything
            to_stage.append((member, ram_regions[0]))

        for member, src in to_stage:
            dst = self._claim_region(self._vram, member)
            if dst is None:
                # _claim_region already signalled OOM → abort path;
                # partial claims here are moot since the sim ends.
                return False
            transfers.append((src, dst))
            loading_tids.append(member.id)

        if to_stage:
            self._group_state[gid] = _LOADING
        return True

    def _ensure_non_weight_resident(
        self, tensor: Tensor, memory: BaseMemory, transfers: list[tuple[DataRegion, DataRegion]]
    ) -> bool | None:
        """Standard "input must be readable on compute.memory" pattern.
        For non-WEIGHT tensors (CONTEXT / INPUT / LEAF) the home device
        is set by the trace; if the compute is on a different device we
        stage a transfer."""
        target_regions = memory.space.get_by_tensor_id(tensor.id)
        if any(self._region_readable(r) for r in target_regions):
            return True

        pending = self._pending_dest_ids()
        if any(
            (r.access_status == DataRegionAccess.BEING_WRITTEN) or (id(r) in pending)
            for r in target_regions
        ):
            # Already being filled; just wait.
            return True

        src = self._find_latest_region(tensor.id, exclude_hw=memory)
        if src is None:
            return False  # not produced yet

        # An IDLE existing region we can reuse, else claim new.
        dst = self._idle_region_on(memory, tensor.id)
        if dst is None:
            dst = self._claim_region(memory, tensor)
            if dst is None:
                return False
        transfers.append((src, dst))
        return True

    def _ensure_output_claimed(self, node: Node, memory: BaseMemory) -> bool:
        for tid in node.output_tensors:
            if tid in node.input_tensors:
                continue  # in-place / view: input region IS output region
            tensor = self.sys.trace.tensor_map[tid]
            regions = memory.space.get_by_tensor_id(tid)
            if any(r.access_status == DataRegionAccess.IDLE for r in regions):
                continue
            region = self._claim_region(memory, tensor)
            if region is None:
                return False
        return True

    def _outputs_free(self, node: Node, memory: BaseMemory) -> bool:
        pending = self._pending_dest_ids()
        for tid in node.output_tensors:
            if tid in node.input_tensors:
                continue
            regions = memory.space.get_by_tensor_id(tid)
            has_idle = False
            for r in regions:
                if r.access_status == DataRegionAccess.IDLE and id(r) not in pending:
                    has_idle = True
                    break
            if not has_idle:
                return False
        return True

    def _preclaim_dispatcher_outputs(self, node: Node) -> None:
        """Loader-driven contract: dispatcher nodes (e.g. cpu thread
        aten::empty producing a CUDA tensor) have their cross-device
        outputs stashed in ``node.args["dispatcher_outputs"]`` and
        removed from ``node.output_tensors``. We pre-claim those
        regions on the tensor's home memory so consumers see them
        ready/latest."""
        cross = node.args.get("dispatcher_outputs") or []
        for tid in cross:
            tensor = self.sys.trace.tensor_map.get(tid)
            if tensor is None:
                continue
            ttype = tensor.args.get("tensor_type")
            if ttype == "WEIGHT":
                # WEIGHTs already have their home RAM region from layout.
                home = self._ram
            else:
                home = self._home_memory_for_non_weight(tensor)
            target = self._idle_region_on(home, tid)
            if target is None:
                if self._remaining_consumers.get(tid, 0) <= 0 and ttype not in self._PERMANENT_TYPES:
                    continue
                target = self._claim_region(home, tensor)
                if target is None:
                    continue
            target.is_ready = True
            target.is_latest = True

    # ============================================================ submit loop
    def _parents_done(self, node: Node) -> bool:
        node_map = self.sys.trace.node_map
        return all(node_map[pid].status == NodeStatus.DONE for pid in node.parent_nodes)

    def _submit_transfer_batches(self, transfers: list[tuple[DataRegion, DataRegion]]) -> None:
        grouped: dict[tuple[int, int], list[tuple[DataRegion, DataRegion]]] = defaultdict(list)
        for src, dst in transfers:
            grouped[(src.hw.id, dst.hw.id)].append((src, dst))
        for batch in grouped.values():
            self.sys.transfer(batch)

    def _kick_cuda_weight_prefetch(self, node: Node) -> None:
        """For every offload group touched by ``node``'s WEIGHT inputs,
        kick a RAM->VRAM transfer if the group isn't already in flight
        or resident. Mirrors accelerate's "leaf module's forward starts
        -> H2D the weights" — even when the consumer node is the
        CPU-thread dispatcher whose own compute doesn't need them
        resident on VRAM. Doesn't gate the consumer; the transfer
        fires alongside.

        Group-aware: under ``preload_module_classes``, kicking on any
        member of a group stages the whole group atomically.
        """
        tensor_map = self.sys.trace.tensor_map
        seen_groups: set[str] = set()
        for tid in node.input_tensors:
            tensor = tensor_map.get(tid)
            if tensor is None:
                continue
            gid = self._tensor_group.get(tid)
            if gid is None or gid in seen_groups:
                continue
            seen_groups.add(gid)
            state = self._group_state.get(gid, _ABSENT)
            if state in (_LOADING, _LOADED):
                continue
            # If the group's members are already on VRAM and readable,
            # sync state to LOADED without staging.
            members = self._group_tids.get(gid, [])
            if members and all(
                any(self._region_readable(r)
                    for r in self._vram.space.get_by_tensor_id(mt))
                for mt in members
            ):
                self._group_state[gid] = _LOADED
                continue

            batch: list[tuple[DataRegion, DataRegion]] = []
            staged: list[int] = []
            if not self._stage_group(gid, batch, staged):
                # Master copy not ready or OOM; try again next tick.
                continue
            if batch:
                job_id = self.sys.transfer(batch)
                self._inflight_load_tids[job_id] = staged
                for _src, dst in batch:
                    self._tick_pending_dest_ids.add(id(dst))

    def _submit_ready_nodes(self) -> bool:
        """One pass over ``ready_node_ids``; returns True iff anything
        was queued so the caller can re-drive (view-only retires cascade)."""
        submitted_any = False
        committed_per_compute: dict[BaseCompute, int] = {}
        self._tick_pending_dest_ids = set()

        num_ready = len(self.ready_node_ids)
        for _ in range(num_ready):
            node_id = self.ready_node_ids.popleft()
            node = self.sys.trace.node_map[node_id]
            if node.status != NodeStatus.TODO:
                continue
            if self.pending_parent_count[node_id] != 0 or not self._parents_done(node):
                self.ready_node_ids.append(node_id)
                continue

            compute = self._compute_for_node(node)

            # Alias / dispatcher nodes: pure pointer ops. The loader
            # gave them ``custom_deps``; the engine bypasses input/
            # output residency checks. The dispatcher itself does not
            # need its WEIGHT inputs resident on VRAM to run, but the
            # downstream GPU kernels do — fire the RAM->VRAM transfer
            # for any cuda-homed WEIGHTs alongside the dispatcher.
            if node.custom_deps:
                cap = getattr(compute, "max_concurrent_jobs", 1)
                if len(compute.job_running) + committed_per_compute.get(compute, 0) >= cap:
                    self.ready_node_ids.append(node_id)
                    continue
                self._kick_cuda_weight_prefetch(node)
                if node.args.get("dispatcher_outputs"):
                    self._preclaim_dispatcher_outputs(node)
                self.sys.compute(compute, node)
                committed_per_compute[compute] = committed_per_compute.get(compute, 0) + 1
                submitted_any = True
                continue

            cap = getattr(compute, "max_concurrent_jobs", 1)
            if len(compute.job_running) + committed_per_compute.get(compute, 0) >= cap:
                self.ready_node_ids.append(node_id)
                continue

            memory = compute.memory

            # 1. Output claiming first — needed before submit. If we
            # can't even claim outputs (OOM or busy), retry later.
            if not self._ensure_output_claimed(node, memory):
                self.ready_node_ids.appendleft(node_id)
                continue

            # 2. Per-input residency. WEIGHT inputs landing on a GPU
            # compute trigger RAM->VRAM staging that the compute will
            # block on; everything else uses the standard cross-device
            # residency machinery. CPU computes that happen to consume
            # a WEIGHT find it in RAM and don't trigger any transfer
            # via this path — the prefetch hook above handles their
            # GPU-side staging.
            transfers: list[tuple[DataRegion, DataRegion]] = []
            loading_tids: list[int] = []
            blocked = False
            tensor_map = self.sys.trace.tensor_map
            for tid in node.input_tensors:
                t = tensor_map[tid]
                ttype = t.args.get("tensor_type")
                if ttype == "WEIGHT" and memory is self._vram:
                    ok = self._ensure_weight_loaded(t, transfers, loading_tids)
                else:
                    ok = self._ensure_non_weight_resident(t, memory, transfers)
                if not ok:
                    blocked = True
                    break

            if blocked:
                self.ready_node_ids.append(node_id)
                continue

            # CPU compute nodes consume WEIGHT tensors via RAM (already
            # present from layout). But the downstream GPU kernels need
            # those weights on VRAM — kick the H2D prefetch alongside
            # this node's submission, same as the dispatcher path.
            if memory is self._ram:
                self._kick_cuda_weight_prefetch(node)

            # 3. Submit any new transfers; node compute waits another
            # tick until the transfer retires and the dest region is
            # readable.
            if transfers:
                weight_transfers: list[tuple[DataRegion, DataRegion]] = []
                other_transfers: list[tuple[DataRegion, DataRegion]] = []
                for src, dst in transfers:
                    if dst.hw is self._vram and src.hw is self._ram and dst.tensor_id in loading_tids:
                        weight_transfers.append((src, dst))
                    else:
                        other_transfers.append((src, dst))
                if weight_transfers:
                    job_id = self.sys.transfer(weight_transfers)
                    self._inflight_load_tids[job_id] = [d.tensor_id for _s, d in weight_transfers]
                if other_transfers:
                    self._submit_transfer_batches(other_transfers)
                for _src, dst in transfers:
                    self._tick_pending_dest_ids.add(id(dst))
                self.ready_node_ids.append(node_id)
                continue

            # 4. Outputs must also have no pending transfer landing on
            # their region; otherwise the compute's begin() would race.
            if not self._outputs_free(node, memory):
                self.ready_node_ids.append(node_id)
                continue

            self.sys.compute(compute, node)
            committed_per_compute[compute] = committed_per_compute.get(compute, 0) + 1
            submitted_any = True

        return submitted_any

    # ============================================================ retire
    def _handle_transfer_retires(self, retired_jobs: list[BaseJob]) -> None:
        for job in retired_jobs:
            if not isinstance(job, TransferJob):
                continue
            tids = self._inflight_load_tids.pop(job.id, None)
            if not tids:
                continue
            # Identify which groups this transfer satisfied. A group
            # only flips to LOADED when every member is readable on
            # VRAM — for singleton groups that's the trivial case; for
            # coalesced groups it guards against partial loads.
            touched_groups = {self._tensor_group.get(tid) for tid in tids}
            touched_groups.discard(None)
            for gid in touched_groups:
                if self._group_state.get(gid) != _LOADING:
                    continue
                members = self._group_tids.get(gid, [])
                if all(
                    any(self._region_readable(r)
                        for r in self._vram.space.get_by_tensor_id(mt))
                    for mt in members
                ):
                    self._group_state[gid] = _LOADED
        self._refresh_peak_gpu_resident()

    def _release_group_vram(self, gid: str) -> None:
        """All-or-nothing: if any member's VRAM region is still non-IDLE
        (BEING_READ / BEING_WRITTEN), defer the whole group; otherwise
        release every member and flip state to ABSENT. This prevents
        partial release from leaving group state stuck at _LOADED with
        a subset of members missing."""
        members = self._group_tids.get(gid, ())
        for member_tid in members:
            for region in self._vram.space.get_by_tensor_id(member_tid):
                if region.access_status != DataRegionAccess.IDLE:
                    self._pending_vram_group_releases.add(gid)
                    return

        for member_tid in members:
            for region in list(self._vram.space.get_by_tensor_id(member_tid)):
                self.sys.release(region)

        self._pending_vram_group_releases.discard(gid)
        self._group_state[gid] = _ABSENT

    def _release_non_weight_regions(self, tensor_id: int) -> None:
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

    def _consume_inputs(self, node: Node) -> None:
        tensor_map = self.sys.trace.tensor_map
        for tid in node.input_tensors:
            remaining = self._remaining_consumers.get(tid)
            if remaining is None:
                continue
            remaining -= 1
            tensor = tensor_map.get(tid)
            ttype = tensor.args.get("tensor_type") if tensor is not None else None

            # WEIGHT consumers also decrement the group counter, which
            # gates VRAM eviction so a coalesced group stays resident
            # until every member's last consumer has retired.
            gid = self._tensor_group.get(tid) if ttype == "WEIGHT" else None
            if gid is not None:
                self._group_remaining[gid] = self._group_remaining.get(gid, 0) - 1

            if remaining > 0:
                self._remaining_consumers[tid] = remaining
                continue
            self._remaining_consumers.pop(tid, None)

            if tensor is None:
                continue

            if ttype == "WEIGHT":
                # Free the VRAM mirror once *all* group members are
                # consumed (under preload_module_classes the group may
                # cover many leaves; with the default it's just this
                # tensor). RAM master stays for the whole run. A WEIGHT
                # outside any group (cpu-device WEIGHT promoted to VRAM
                # by a GPU consumer) is released as a singleton via the
                # generic path, but skips RAM (RAM is its permanent
                # home).
                if gid is not None and self._group_remaining.get(gid, 0) <= 0:
                    self._release_group_vram(gid)
                elif gid is None:
                    for region in list(self._vram.space.get_by_tensor_id(tid)):
                        if region.access_status == DataRegionAccess.IDLE:
                            self.sys.release(region)
                continue

            if ttype in self._PERMANENT_TYPES:
                continue  # INPUT / LEAF — held forever

            self._release_non_weight_regions(tid)

    def _release_dead_outputs(self, node: Node) -> None:
        """Producer outputs with no remaining consumer get released as
        soon as the producing op retires — mirrors PyTorch ref-counting.
        WEIGHTs are never node outputs in practice; skip."""
        for tid in node.output_tensors:
            if tid in node.input_tensors:
                continue
            if self._remaining_consumers.get(tid, 0) > 0:
                continue
            tensor = self.sys.trace.tensor_map.get(tid)
            if tensor is None:
                continue
            ttype = tensor.args.get("tensor_type")
            if ttype == "WEIGHT" or ttype in self._PERMANENT_TYPES:
                continue
            self._release_non_weight_regions(tid)

    def _retry_pending_releases(self) -> None:
        for gid in list(self._pending_vram_group_releases):
            self._release_group_vram(gid)
        for tid in list(self._pending_releases):
            self._release_non_weight_regions(tid)

    def _retire_computes(self, retired_jobs: list[BaseJob]) -> None:
        for job in retired_jobs:
            if not isinstance(job, ComputeJob):
                continue
            self._consume_inputs(job.node)
            self._release_dead_outputs(job.node)
            for child_id in job.node.children_nodes:
                if child_id not in self.pending_parent_count:
                    continue
                if self.pending_parent_count[child_id] > 0:
                    self.pending_parent_count[child_id] -= 1
                child = self.sys.trace.node_map[child_id]
                if self.pending_parent_count[child_id] == 0 and child.status == NodeStatus.TODO:
                    self.ready_node_ids.append(child_id)

    # ============================================================ runtime
    def runtime(self, retired_jobs: list[BaseJob]) -> None:
        self._handle_transfer_retires(retired_jobs)
        self._retire_computes(retired_jobs)
        self._retry_pending_releases()

        while self._submit_ready_nodes():
            pass

        self._refresh_peak_gpu_resident()

        # Deadlock guard: nothing in flight, but nodes still TODO.
        if not self.sys.engine.job_running and not self.sys.engine.job_waiting:
            todo = [n for n in self.sys.trace.node_map.values() if n.status == NodeStatus.TODO]
            if todo:
                blocked = todo[0]
                self.sys.abort({
                    "from": self.name,
                    "error": "SCHEDULER_DEADLOCK",
                    "msg": "No runnable node and queues empty.",
                    "node": {
                        "id": blocked.id,
                        "name": blocked.name,
                        "parent_nodes": blocked.parent_nodes,
                        "input_tensors": blocked.input_tensors,
                        "output_tensors": blocked.output_tensors,
                    },
                })
        return
