from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, TYPE_CHECKING

from sim.core.job import BaseJob, ComputeJob, TransferJob
from sim.core.log import Log
from sim.core.trace import Node, NodeStatus, Tensor, TerminalNode, Trace
from sim.hw.common import DataRegion, DataRegionAccess
from sim.hw.compute.common import BaseCompute
from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage
from sim.sched.common import BaseScheduler

if TYPE_CHECKING:
    from sim.core.system import System


class DeviceAwareVanilla(BaseScheduler):
    """
    Vanilla scheduler for traces with explicit CPU/CUDA device metadata.

    It keeps each tensor in a home memory derived from tensor.args["device"],
    submits CPU nodes to the CPU compute and CUDA nodes to the GPU compute, and
    lazily creates cross-device copies when a node consumes a tensor whose
    latest copy is resident elsewhere.
    """

    def __init__(self, obj_id: int, name: str, log: Log, sys: System, args: dict[str, Any] | None = None):
        super().__init__(obj_id, name, log, sys, args)

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
            raise Exception(f"[DeviceAwareVanilla] CPU compute '{cpu_compute_name}' does not exist.")
        if cuda_compute_name not in self.compute_by_name:
            raise Exception(f"[DeviceAwareVanilla] CUDA compute '{cuda_compute_name}' does not exist.")

        self.cpu_compute = self.compute_by_name[cpu_compute_name]
        self.cuda_compute = self.compute_by_name[cuda_compute_name]

        self.memory_by_device: dict[str, BaseMemory] = {
            "cpu": self.cpu_compute.memory,
            self.cuda_device: self.cuda_compute.memory,
        }
        self.initial_tensor_types = set(self.args.get("initial_tensor_types", ["WEIGHT", "INPUT", "LEAF"]))
        self.node_ids: list[int] = list(self.sys.trace.node_map.keys())
        self.pending_parent_count: dict[int, int] = {
            node.id: len(node.parent_nodes) for node in self.sys.trace.node_map.values()
        }
        self.ready_node_ids: deque[int] = deque(
            node.id for node in self.sys.trace.node_map.values()
            if self.pending_parent_count[node.id] == 0
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
        for node in self.sys.trace.node_map.values():
            for tid in node.input_tensors:
                self._remaining_consumers[tid] = self._remaining_consumers.get(tid, 0) + 1

        # Tensors whose last consumer has retired but whose regions were
        # not IDLE when we tried to release (e.g. being read by an
        # in-flight transfer). Retried every engine tick until all their
        # regions are freed — see _retry_pending_releases.
        self._pending_releases: set[int] = set()

        # Timing shims: alias / dispatcher nodes need their CPU dispatch
        # time charged but can't safely run as real ComputeJobs (the
        # engine's compute_assertion would force phantom transfers for
        # CUDA-tensor inputs on a CPU-thread node, or fail the output-
        # IDLE check on aliased outputs). Solution: submit a synthetic
        # Node with empty input/output tensors and the original's
        # compute_time_micros, queued on cpu_compute. The engine runs it
        # serially through the CPU like any other op (correctly charging
        # the dispatch time and respecting CPU concurrency = 1) without
        # ever asking about tensor residency. When the shim retires, the
        # scheduler finalizes the original node's data semantics
        # (dispatcher output promotion, child unblock, lifetime release).
        self._shim_id_counter: int = 0
        self._shim_to_original: dict[int, Node] = {}
        return

    # Tensor types that stay resident for the whole run. Everything else
    # is released as soon as its last consumer retires.
    _PERMANENT_TYPES = frozenset({"WEIGHT", "INPUT", "LEAF"})

    def compile(self, trace: Trace) -> None:
        return

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

    def _ensure_home_region(self, tensor: Tensor):
        memory = self._memory_for_tensor(tensor)
        regions = memory.space.get_by_tensor_id(tensor.id)
        if regions:
            return regions[0]
        return self._claim_region(memory, tensor)

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

    def _retire_kind(self, node: Node) -> str | None:
        """Decide whether a node can be retired in place instead of running
        as a ComputeJob. Returns the reason string or None.

        Fully data-driven from tensor devices:

          - "alias"      : every output tensor_id is already an input
                           tensor_id (view/in-place ops merged by the
                           loader's storage aliasing).
          - "dispatcher" : at least one output tensor's home memory
                           differs from the compute's local memory. The
                           node runs on one device but produces a tensor
                           on another — e.g. aten::empty(device=cuda) on
                           the CPU thread. In real PyTorch this is just a
                           pointer/allocator call; running it as a real
                           ComputeJob would place the output on the wrong
                           memory and force every downstream consumer to
                           transfer it.

        A node with neither property runs as a normal ComputeJob."""
        if node.output_tensors and all(t in node.input_tensors for t in node.output_tensors):
            return "alias"

        compute = self._compute_for_node(node)
        for tid in node.output_tensors:
            tensor = self.sys.trace.tensor_map[tid]
            if self._memory_for_tensor(tensor) is not compute.memory:
                return "dispatcher"

        return None

    def _submit_timing_shim(self, node: Node, kind: str) -> None:
        """Submit a synthetic CPU ComputeJob carrying only the original
        node's compute_time_micros — no input or output tensors.

        Why: the original node is a pure-alias or dispatcher op whose
        real execution on the CPU thread is dispatch-only (pointer
        arithmetic, cudaMalloc, etc.). It still costs CPU time. We can't
        run the original directly because core's compute_assertion would
        demand input residency on cpu.memory — wrong for CUDA tensors —
        and refuse the output-IDLE check on aliased outputs. Submitting
        a stripped synthetic ComputeJob charges the engine for the time
        on the CPU thread (serially, respecting CPU max_concurrent_jobs)
        without triggering any data-residency machinery.

        On the shim's retirement, _retire_completed_nodes detects the
        marker in args and runs _finalize_in_place_after_shim on the
        original node — that handles dispatcher output promotion,
        consumer counts, dead-output release, and child unblocking.
        """
        self._shim_id_counter += 1
        shim_id = -1_000_000_000 - self._shim_id_counter
        shim = Node(
            shim_id,
            f"shim<{node.name}>",
            float(node.compute_time_micros),
            args={"timing_shim_for": node.id, "shim_kind": kind},
        )
        self._shim_to_original[shim_id] = node
        node.status = NodeStatus.WAITING
        self.sys.compute(self.cpu_compute, shim)

    def _finalize_in_place_after_shim(self, node: Node, kind: str) -> None:
        """Run the data-side bookkeeping for a node whose timing shim
        just retired. Same effects the old eager retire-in-place had,
        but now happens after the engine has charged the CPU for the
        node's compute_time."""
        node.status = NodeStatus.DONE

        if kind == "dispatcher":
            for tensor_id in node.output_tensors:
                if tensor_id in node.input_tensors:
                    continue
                tensor = self.sys.trace.tensor_map[tensor_id]
                if self._remaining_consumers.get(tensor_id, 0) <= 0:
                    if tensor.args.get("tensor_type") not in self._PERMANENT_TYPES:
                        continue
                home = self._memory_for_tensor(tensor)
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

        self._consume_inputs(node)
        self._release_dead_outputs(node)

        for child_id in node.children_nodes:
            if child_id not in self.pending_parent_count:
                continue
            if self.pending_parent_count[child_id] > 0:
                self.pending_parent_count[child_id] -= 1
            child = self.sys.trace.node_map[child_id]
            if self.pending_parent_count[child_id] == 0 and child.status == NodeStatus.TODO:
                self.ready_node_ids.append(child_id)

    def _consume_inputs(self, node: Node) -> None:
        """Decrement per-tensor remaining-consumer counts for this node's
        input tensors, releasing regions of tensors whose last consumer
        has now retired (unless the tensor is permanent: WEIGHT/INPUT/LEAF)."""
        for tid in node.input_tensors:
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
            if tensor.args.get("tensor_type") in self._PERMANENT_TYPES:
                continue
            self._release_tensor_regions(tid)

    def _release_tensor_regions(self, tensor_id: int) -> None:
        """Free every region holding this tensor. If any region is busy
        (BEING_READ by a transfer src, BEING_WRITTEN, etc.) at the moment
        of call, core's release_assertion rejects it — we record the
        tensor as pending and retry on the next engine tick."""
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

    def _retry_pending_releases(self) -> None:
        """Re-attempt any deferred releases. Called once per engine tick
        from runtime(); a region transitions out of BEING_READ/WRITTEN
        only through a job's end_mutation, which always runs inside
        _runtime_forward immediately before the next sched.runtime call,
        so one retry per tick catches every freed region."""
        if not self._pending_releases:
            return
        for tensor_id in list(self._pending_releases):
            self._release_tensor_regions(tensor_id)

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

    def _parents_done(self, node: Node) -> bool:
        node_map = self.sys.trace.node_map
        return all(node_map[parent_id].status == NodeStatus.DONE for parent_id in node.parent_nodes)

    def _submit_transfer_batches(self, transfers: list[tuple[DataRegion, DataRegion]]) -> None:
        grouped: dict[tuple[int, int], list[tuple[DataRegion, DataRegion]]] = defaultdict(list)
        for src_region, dest_region in transfers:
            grouped[(src_region.hw.id, dest_region.hw.id)].append((src_region, dest_region))

        for batch in grouped.values():
            self.sys.transfer(batch)

        return

    def _submit_ready_nodes(self) -> bool:
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
            if self.pending_parent_count[node_id] != 0 or not self._parents_done(node):
                self.ready_node_ids.append(node_id)
                continue

            compute = self._compute_for_node(node)

            # Pure-alias / dispatcher nodes: route through a CPU timing
            # shim so their dispatch time is charged on the CPU thread
            # without triggering data-residency machinery (which would
            # demand inputs on cpu.memory and fail for CUDA tensors).
            # The shim consumes a CPU concurrency slot like any other
            # CPU op, so respect cpu_compute's cap.
            kind = self._retire_kind(node)
            if kind is not None:
                shim_compute = self.cpu_compute
                cap_shim = getattr(shim_compute, "max_concurrent_jobs", 1)
                running_shim = len(shim_compute.job_running)
                committed_shim = committed_per_compute.get(shim_compute, 0)
                if running_shim + committed_shim >= cap_shim:
                    self.ready_node_ids.append(node_id)
                    continue
                self._submit_timing_shim(node, kind)
                committed_per_compute[shim_compute] = committed_shim + 1
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

            self.sys.compute(compute, node)
            committed_per_compute[compute] = committed + 1
            submitted_any = True

        return submitted_any

    def _retire_completed_nodes(self, retired_jobs: list[BaseJob]) -> None:
        for job in retired_jobs:
            if not isinstance(job, ComputeJob):
                continue

            shim_for = job.node.args.get("timing_shim_for")
            if shim_for is not None:
                # Timing shim retiring: fold its effect onto the original.
                original = self._shim_to_original.pop(job.node.id, None)
                if original is None:
                    original = self.sys.trace.node_map.get(shim_for)
                if original is not None:
                    self._finalize_in_place_after_shim(original, job.node.args.get("shim_kind"))
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

    def runtime(self, retired_jobs: list[BaseJob]) -> None:
        self._retire_completed_nodes(retired_jobs)
        self._retry_pending_releases()

        # Iterate: retiring view-like ops in place unblocks children that
        # may themselves be ready to schedule (or be view-like). Keep going
        # until a pass makes no further progress.
        while self._submit_ready_nodes():
            pass

        if not self.sys.engine.job_running and not self.sys.engine.job_waiting:
            todo_nodes = [
                node for node in self.sys.trace.node_map.values()
                if node.status == NodeStatus.TODO
            ]
            if todo_nodes:
                blocked = todo_nodes[0]
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
                })

        return
