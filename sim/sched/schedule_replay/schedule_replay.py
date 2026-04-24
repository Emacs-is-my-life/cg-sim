"""Replay a ``jit_sim_prune_schedule.json`` through cg-sim's simulator.

Use case: you have a schedule produced by ``graph_modifiers/ct_belady_pcie``
(or similar) and want cg-sim to tell you its actual peak VRAM, peak DRAM,
and makespan under realistic IO curves / concurrency / page fragmentation.

Design
------
* Treat UNet weights as **RAM-resident by default**. At layout, each WEIGHT
  (or LEAF) tensor is claimed on the CPU memory and streamed in from
  storage — just like non-GPU tensors. Tensors listed in the schedule's
  ``cold_start_prefetches`` are additionally copied to VRAM.
* At runtime, walk GPU nodes in topological order. For each GPU node's
  ``compiled_launch_id``:
    - Pre-ops: ``sync_h2d`` and ``vram_prefetch_h2d`` with matching
      ``before_launch_id`` → RAM-to-VRAM transfer, must complete before
      submitting the compute job.
    - Submit the ComputeJob to the GPU.
    - Post-ops: ``vram_evict_d2h`` with matching ``after_launch_id`` →
      VRAM-to-RAM transfer.
* Maps ``compiled_tensor_id`` to cg-sim ``tensor_id`` by intersecting the
  sidecar's ``used_by_launch_ids`` with the trace's per-node input tensors.
  A single ``compiled_tensor_id`` can resolve to multiple cg-sim tensors
  (PyTorch's storage dedup); each is streamed together.

This is an **approximation**: we don't model the schedule's async H2D /
h2d_wait nuances exactly — cross-iter reloads land at the next consumer's
pre-op so they still complete before the kernel that needs them. Peak-VRAM
and makespan are realistic within that approximation.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
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


class ScheduleReplay(BaseScheduler):
    """Replay a weight-streaming schedule JSON through the full simulator."""

    def __init__(
        self, obj_id: int, name: str, log: Log, sys: System,
        args: dict[str, Any] | None = None,
    ):
        super().__init__(obj_id, name, log, sys, args)

        schedule_path = self.args.get("schedule_path")
        if not schedule_path:
            raise Exception(
                "[ScheduleReplay] args.schedule_path is required "
                "(path to jit_sim_prune_schedule.json)."
            )
        with open(schedule_path) as f:
            self._schedule_doc: dict[str, Any] = json.load(f)

        cpu_compute_name = self.args.get("cpu_compute", "cpu")
        cuda_compute_name = self.args.get("cuda_compute", "gpu0")
        self.cuda_device = str(self.args.get("cuda_device", "cuda:0")).lower()

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
        if cpu_compute_name not in self.compute_by_name:
            raise Exception(f"[ScheduleReplay] CPU compute '{cpu_compute_name}' not found.")
        if cuda_compute_name not in self.compute_by_name:
            raise Exception(f"[ScheduleReplay] GPU compute '{cuda_compute_name}' not found.")
        self.cpu_compute = self.compute_by_name[cpu_compute_name]
        self.gpu_compute = self.compute_by_name[cuda_compute_name]
        self.cpu_memory = self.cpu_compute.memory
        self.gpu_memory = self.gpu_compute.memory

        # Map compiled_tensor_id → cg-sim tensor_ids (can be multiple aliases).
        self._tid_to_cgsim_tensors: dict[int, list[int]] = {}
        # Schedule ops keyed by launch_id.
        self._pre_by_launch: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._post_by_launch: dict[int, list[dict[str, Any]]] = defaultdict(list)
        # Cold-start compiled_tensor_ids that start resident in VRAM.
        self._cold_start_tids: set[int] = set()

        # Per-node bookkeeping
        self.pending_parent_count: dict[int, int] = {}
        self.ready_node_ids: deque[int] = deque()
        self._nodes_in_flight: set[int] = set()
        self._tensor_regions_in_transit: set[int] = set()  # cgsim tensor_ids currently being written (h2d)
        # cg-sim tensor ids that have been evicted from VRAM and whose source
        # VRAM region should be released once the D2H transfer retires
        # (source becomes IDLE again).
        self._vram_evict_pending_release: set[int] = set()
        return

    # -----------------------------------------------------------------------
    # Compile: load schedule, build tid mapping
    # -----------------------------------------------------------------------

    def compile(self, trace: Trace) -> None:
        # Partition ops by launch id.
        for op in self._schedule_doc.get("io_operations", []):
            t = op.get("type", "")
            if t == "vram_prefetch_h2d":
                lid = int(op.get("before_launch_id", -1))
                if lid >= 0:
                    self._pre_by_launch[lid].append(op)
            elif t == "vram_evict_d2h":
                lid = int(op.get("after_launch_id", -1))
                if lid >= 0:
                    self._post_by_launch[lid].append(op)
            # "prefetch" (SSD→DRAM) we ignore for the simulator; DRAM is
            # preloaded at layout anyway.

        for cs in self._schedule_doc.get("cold_start_prefetches", []):
            tid = int(cs.get("compiled_tensor_id", -1))
            if tid >= 0:
                self._cold_start_tids.add(tid)

        # Build compiled_tensor_id → cg-sim tensor_id mapping via usage
        # intersection.
        # 1. Build launch_id → list of cg-sim node ids.
        lid_to_node_ids: dict[int, list[int]] = defaultdict(list)
        for nid, node in trace.node_map.items():
            lid = node.args.get("compiled_launch_id")
            if lid is None:
                continue
            lid_int = int(lid)
            if lid_int < 0:
                continue
            lid_to_node_ids[lid_int].append(nid)

        # 2. For each ops-referenced compiled_tensor_id, find cg-sim tensors
        #    input to ALL the consumer launches. Use tensor_map if available.
        # We get used_by_launch_ids from the schedule's own op anchors as a
        # secondary signal. For the primary signal we need the sidecar's
        # tensor_map. Embed a compact "used_by" from schedule ops:
        tid_launches: dict[int, set[int]] = defaultdict(set)
        for op in self._schedule_doc.get("io_operations", []):
            tid = op.get("compiled_tensor_id", -1)
            if tid < 0:
                continue
            for key in ("before_launch_id", "after_launch_id"):
                lid = op.get(key, -1)
                if lid is not None and int(lid) >= 0:
                    tid_launches[int(tid)].add(int(lid))
        # Cold-start ops too
        for cs in self._schedule_doc.get("cold_start_prefetches", []):
            tid = cs.get("compiled_tensor_id", -1)
            lid = cs.get("before_launch_id", -1)
            if tid is not None and int(tid) >= 0 and lid is not None and int(lid) >= 0:
                tid_launches[int(tid)].add(int(lid))

        # For each tid, intersect input tensors across all consumer nodes.
        for tid, launches in tid_launches.items():
            common: set[int] | None = None
            for lid in launches:
                nodes_for_lid = lid_to_node_ids.get(lid, [])
                per_launch: set[int] = set()
                for nid in nodes_for_lid:
                    node = trace.node_map[nid]
                    for t_id in node.input_tensors:
                        tensor = trace.tensor_map.get(t_id)
                        if tensor is None:
                            continue
                        if tensor.args.get("tensor_type") != "WEIGHT":
                            continue
                        per_launch.add(t_id)
                if common is None:
                    common = per_launch
                else:
                    common &= per_launch
                if not common:
                    break
            if common:
                self._tid_to_cgsim_tensors[tid] = sorted(common)

        # Initialize dependency book-keeping.
        self.pending_parent_count = {
            nid: len(node.parent_nodes) for nid, node in trace.node_map.items()
        }
        self.ready_node_ids = deque(
            nid for nid, n in trace.node_map.items()
            if self.pending_parent_count[nid] == 0
        )
        return

    # -----------------------------------------------------------------------
    # Layout: place tensors
    # -----------------------------------------------------------------------

    def _find_free_page(self, memory: BaseMemory, num_pages: int) -> int | None:
        cursor = 0
        for region in memory.space._regions_by_page_idx_start.values():
            if region.page_idx_start - cursor >= num_pages:
                return cursor
            cursor = max(cursor, region.page_idx_end)
        if memory.space.num_total_pages - cursor >= num_pages:
            return cursor
        return None

    def _claim_region(self, memory: BaseMemory, tensor: Tensor) -> DataRegion | None:
        page_idx = self._find_free_page(memory, tensor.num_pages)
        if page_idx is None:
            self.sys.abort({
                "from": self.name,
                "error": "LAYOUT_FAILURE",
                "msg": f"Not enough space on {memory.name} for tensor {tensor.id}.",
                "tensor": {"id": tensor.id, "num_pages": tensor.num_pages},
                "memory": {
                    "name": memory.name,
                    "used": memory.space.num_used_pages,
                    "total": memory.space.num_total_pages,
                },
            })
            return None
        return self.sys.claim(memory, tensor, page_idx)

    def _cold_start_cgsim_tids(self) -> set[int]:
        out: set[int] = set()
        for tid in self._cold_start_tids:
            for cg_tid in self._tid_to_cgsim_tensors.get(tid, []):
                out.add(cg_tid)
        return out

    def layout(self, init_storage: BaseStorage) -> None:
        """Place tensors at the start.

        * WEIGHT/LEAF tensors: home = RAM. Copy from storage → RAM.
        * Cold-start tensors (from the schedule): additionally copied RAM → VRAM.
        * All other tensors (INPUT, INTERMEDIATE, KVCACHE): home = their
          profile device (RAM for CPU, VRAM for CUDA).
        """
        cold_cgsim_tids = self._cold_start_cgsim_tids()
        initial_transfers: list[tuple[DataRegion, DataRegion]] = []

        for tensor in self.sys.trace.tensor_map.values():
            ttype = tensor.args.get("tensor_type", "")
            device = str(tensor.args.get("device") or "").lower()

            # Pick home memory for this tensor.
            if ttype in ("WEIGHT", "LEAF"):
                home = self.cpu_memory
            elif device.startswith("cuda"):
                home = self.gpu_memory
            else:
                home = self.cpu_memory

            region = self._claim_region(home, tensor)
            if region is None:
                return

            # If tensor is pre-placed in storage, stream it in.
            stor_regions = init_storage.space.get_by_tensor_id(tensor.id)
            if stor_regions and ttype in ("WEIGHT", "LEAF", "INPUT"):
                initial_transfers.append((stor_regions[0], region))

            # Cold-start tensors also get a VRAM mirror.
            if tensor.id in cold_cgsim_tids and home is not self.gpu_memory:
                gpu_region = self._claim_region(self.gpu_memory, tensor)
                if gpu_region is None:
                    return
                # First hydrate RAM, then mirror to VRAM. We schedule the
                # storage→RAM transfer above; and the RAM→VRAM transfer here.
                # The simulator will serialize them via job dependencies.
                initial_transfers.append((region, gpu_region))

        if initial_transfers:
            grouped: dict[tuple[int, int], list[tuple[DataRegion, DataRegion]]] = defaultdict(list)
            for s, d in initial_transfers:
                grouped[(s.hw.id, d.hw.id)].append((s, d))
            for batch in grouped.values():
                self.sys.transfer(batch)
        return

    # -----------------------------------------------------------------------
    # Runtime: schedule-driven compute + H2D/D2H
    # -----------------------------------------------------------------------

    @staticmethod
    def _is_gpu_node(node: Node) -> bool:
        if isinstance(node, TerminalNode):
            return False
        device_type = str(node.args.get("device_type") or "").upper()
        if device_type in ("CUDA", "GPU"):
            return True
        rk = str(node.args.get("resource_kind") or "")
        return rk in ("gpu_stream", "gpu")

    def _compute_for_node(self, node: Node) -> BaseCompute:
        if self._is_gpu_node(node):
            return self.gpu_compute
        return self.cpu_compute

    @staticmethod
    def _region_readable(region: DataRegion) -> bool:
        return (
            region.is_ready and region.is_latest
            and region.access_status in (DataRegionAccess.IDLE, DataRegionAccess.BEING_READ)
        )

    def _ensure_vram(self, cgsim_tid: int) -> list[tuple[DataRegion, DataRegion]]:
        """Return transfers needed to get ``cgsim_tid`` into VRAM."""
        tensor = self.sys.trace.tensor_map.get(cgsim_tid)
        if tensor is None:
            return []
        gpu_regions = self.gpu_memory.space.get_by_tensor_id(cgsim_tid)
        if any(self._region_readable(r) for r in gpu_regions):
            return []
        # Find RAM region, or claim one; transfer.
        cpu_regions = self.cpu_memory.space.get_by_tensor_id(cgsim_tid)
        src = next((r for r in cpu_regions if self._region_readable(r)), None)
        if src is None:
            return []
        # Claim fresh VRAM region if needed.
        dest: DataRegion | None = None
        for r in gpu_regions:
            if r.access_status == DataRegionAccess.IDLE:
                dest = r
                break
        if dest is None:
            dest = self._claim_region(self.gpu_memory, tensor)
        if dest is None or dest.access_status != DataRegionAccess.IDLE:
            return []
        return [(src, dest)]

    def _schedule_evict_to_ram(self, cgsim_tid: int) -> list[tuple[DataRegion, DataRegion]]:
        """Move ``cgsim_tid`` from VRAM back to RAM (if in VRAM).

        Marks the tid for a VRAM-region release once the transfer retires
        (so the VRAM page budget is actually reclaimed).
        """
        tensor = self.sys.trace.tensor_map.get(cgsim_tid)
        if tensor is None:
            return []
        gpu_regions = [
            r for r in self.gpu_memory.space.get_by_tensor_id(cgsim_tid)
            if self._region_readable(r)
        ]
        if not gpu_regions:
            return []
        src = gpu_regions[0]
        # If RAM already has a readable copy, just release the VRAM region —
        # the RAM "backup" is still valid so we don't need another transfer.
        cpu_readable = any(
            self._region_readable(r)
            for r in self.cpu_memory.space.get_by_tensor_id(cgsim_tid)
        )
        if cpu_readable:
            # Release synchronously if possible. Otherwise defer.
            if src.access_status == DataRegionAccess.IDLE:
                self.sys.release(src)
            else:
                self._vram_evict_pending_release.add(cgsim_tid)
            return []
        # Find/claim RAM dest.
        cpu_regions = self.cpu_memory.space.get_by_tensor_id(cgsim_tid)
        dest: DataRegion | None = None
        for r in cpu_regions:
            if r.access_status == DataRegionAccess.IDLE:
                dest = r
                break
        if dest is None:
            dest = self._claim_region(self.cpu_memory, tensor)
        if dest is None or dest.access_status != DataRegionAccess.IDLE:
            return []
        self._vram_evict_pending_release.add(cgsim_tid)
        return [(src, dest)]

    def _reclaim_from_retired(self, retired_jobs: list[BaseJob]) -> None:
        """Release VRAM source regions for retired VRAM→RAM transfer jobs.

        Only scans ``retired_jobs`` (cheap, O(retired)) — no per-tick full
        sweep of pending tids. Only fires when a D2H transfer we submitted
        earlier actually lands.
        """
        for job in retired_jobs:
            if not isinstance(job, TransferJob):
                continue
            batch = getattr(job, "batch", None)
            if not batch:
                continue
            src0 = batch[0][0]
            dest0 = batch[0][1]
            # Was this a VRAM → RAM eviction transfer?
            if src0.hw is not self.gpu_memory:
                continue
            if dest0.hw is not self.cpu_memory:
                continue
            for src_region, _dest_region in batch:
                # If this tid is waiting for release and its source is now
                # idle + stale, release it to reclaim the page budget.
                tid = src_region.tensor_id
                if tid not in self._vram_evict_pending_release:
                    continue
                if (
                    src_region.hw is self.gpu_memory
                    and not src_region.is_latest
                    and src_region.access_status == DataRegionAccess.IDLE
                ):
                    self.sys.release(src_region)
                    self._vram_evict_pending_release.discard(tid)

    def _apply_schedule_transfers(self, transfers: list[tuple[DataRegion, DataRegion]]) -> None:
        if not transfers:
            return
        grouped: dict[tuple[int, int], list[tuple[DataRegion, DataRegion]]] = defaultdict(list)
        for s, d in transfers:
            grouped[(s.hw.id, d.hw.id)].append((s, d))
        for batch in grouped.values():
            self.sys.transfer(batch)

    def _parents_done(self, node: Node) -> bool:
        return all(
            self.sys.trace.node_map[pid].status == NodeStatus.DONE
            for pid in node.parent_nodes
        )

    def _ensure_outputs_claimed(self, node: Node, memory: BaseMemory) -> bool:
        """Make sure the node has IDLE regions on ``memory`` for all output tensors."""
        for t_id in node.output_tensors:
            tensor = self.sys.trace.tensor_map[t_id]
            regions = memory.space.get_by_tensor_id(t_id)
            if any(r.access_status == DataRegionAccess.IDLE for r in regions):
                continue
            region = self._claim_region(memory, tensor)
            if region is None:
                return False
        return True

    def _ensure_inputs_for_node(
        self, node: Node, memory: BaseMemory
    ) -> list[tuple[DataRegion, DataRegion]] | None:
        """Return transfers to materialise all node inputs on ``memory``.

        ``None`` means some tensor has no source anywhere — treat as transient
        (parent hasn't written yet) and retry.
        """
        transfers: list[tuple[DataRegion, DataRegion]] = []
        for t_id in node.input_tensors:
            tensor = self.sys.trace.tensor_map.get(t_id)
            if tensor is None:
                continue
            target_regions = memory.space.get_by_tensor_id(t_id)
            if any(self._region_readable(r) for r in target_regions):
                continue
            # Find a source on a different memory/storage.
            src = self._find_readable_elsewhere(t_id, exclude=memory)
            if src is None:
                return None
            dest = next(
                (r for r in target_regions if r.access_status == DataRegionAccess.IDLE),
                None,
            )
            if dest is None:
                dest = self._claim_region(memory, tensor)
            if dest is None or dest.access_status != DataRegionAccess.IDLE:
                return None
            transfers.append((src, dest))
        return transfers

    def _find_readable_elsewhere(
        self, tensor_id: int, exclude: BaseMemory
    ) -> DataRegion | None:
        # Prefer other memories, then storage.
        for hw in self.sys.hw.values():
            if hw is exclude or not isinstance(hw, BaseMemory):
                continue
            for r in hw.space.get_by_tensor_id(tensor_id):
                if self._region_readable(r):
                    return r
        for hw in self.sys.hw.values():
            if hw is exclude or not isinstance(hw, BaseStorage):
                continue
            for r in hw.space.get_by_tensor_id(tensor_id):
                if self._region_readable(r):
                    return r
        return None

    def _submit_one_ready(self) -> bool:
        attempts = len(self.ready_node_ids)
        for _ in range(attempts):
            if not self.ready_node_ids:
                return False
            nid = self.ready_node_ids.popleft()
            node = self.sys.trace.node_map[nid]
            if node.status != NodeStatus.TODO:
                continue
            if self.pending_parent_count[nid] != 0 or not self._parents_done(node):
                self.ready_node_ids.append(nid)
                continue

            compute = self._compute_for_node(node)
            if not compute.can_run(None):
                self.ready_node_ids.append(nid)
                continue
            memory = compute.memory

            # Claim output regions first so eviction isn't racing us.
            if not self._ensure_outputs_claimed(node, memory):
                self.ready_node_ids.appendleft(nid)
                return False

            # Apply schedule-driven H2D pre-ops (WEIGHT tensors).
            transfers: list[tuple[DataRegion, DataRegion]] = []
            lid = node.args.get("compiled_launch_id")
            if self._is_gpu_node(node) and lid is not None and int(lid) >= 0:
                for op in self._pre_by_launch.get(int(lid), []):
                    tid = op.get("compiled_tensor_id", -1)
                    for cg_tid in self._tid_to_cgsim_tensors.get(int(tid), []):
                        transfers.extend(self._ensure_vram(cg_tid))

            # Ensure ALL node inputs are resident on this compute's memory
            # (regardless of compiled_launch_id). This covers CUDA memcpys
            # and other profile nodes that don't appear in the compile map.
            extra = self._ensure_inputs_for_node(node, memory)
            if extra is None:
                # Some input has no source yet: a parent is still running.
                self.ready_node_ids.append(nid)
                continue
            transfers.extend(extra)

            if transfers:
                self._apply_schedule_transfers(transfers)
                # Re-queue node; transfers will retire before it runs.
                self.ready_node_ids.appendleft(nid)
                return False

            self.sys.compute(compute, node)
            self._nodes_in_flight.add(nid)
            return True
        return False

    def _retire(self, retired_jobs: list[BaseJob]) -> None:
        for job in retired_jobs:
            if not isinstance(job, ComputeJob):
                continue
            node = job.node
            self._nodes_in_flight.discard(node.id)
            for child_id in node.children_nodes:
                if child_id in self.pending_parent_count:
                    self.pending_parent_count[child_id] = max(
                        0, self.pending_parent_count[child_id] - 1
                    )
                    if self.pending_parent_count[child_id] == 0:
                        child = self.sys.trace.node_map[child_id]
                        if child.status == NodeStatus.TODO:
                            self.ready_node_ids.append(child_id)

            # Apply schedule's post-ops for this launch.
            lid = node.args.get("compiled_launch_id")
            if self._is_gpu_node(node) and lid is not None and int(lid) >= 0:
                transfers: list[tuple[DataRegion, DataRegion]] = []
                for op in self._post_by_launch.get(int(lid), []):
                    tid = op.get("compiled_tensor_id", -1)
                    for cg_tid in self._tid_to_cgsim_tensors.get(int(tid), []):
                        transfers.extend(self._schedule_evict_to_ram(cg_tid))
                self._apply_schedule_transfers(transfers)

    def runtime(self, retired_jobs: list[BaseJob]) -> None:
        self._retire(retired_jobs)
        self._reclaim_from_retired(retired_jobs)
        if self.sys.engine.job_waiting:
            return
        if self._submit_one_ready():
            return
        # Deadlock watchdog.
        if (
            not self.sys.engine.job_running
            and not self.sys.engine.job_waiting
            and not self.ready_node_ids
        ):
            # If all compute nodes done, end stage.
            todo_left = [
                n for n in self.sys.trace.node_map.values()
                if n.status == NodeStatus.TODO
            ]
            if not todo_left:
                self.sys.end_stage()
        return
