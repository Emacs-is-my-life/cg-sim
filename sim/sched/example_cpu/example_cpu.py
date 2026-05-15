from __future__ import annotations

from typing import Any, TYPE_CHECKING

from sim.sched.common import BaseScheduler
from sim.core.log import Log
from sim.core.trace import Trace
from sim.core.job import BaseJob
from sim.hw.compute.common import BaseCPU
from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage

if TYPE_CHECKING:
    from sim.core.system import System


class ExampleCPU(BaseScheduler):
    """
    ExampleCPU scheduler (barebone).
    """

    def __init__(self, obj_id: int, name: str, log: Log, sys: System, args: dict[str, Any] | None = None):
        super().__init__(obj_id, name, log, sys, args)

        # Read a tunable parameter from the simulation configuration.
        self.lookahead: int = int(self.args["lookahead"])

        # Check hardware configuration: expect exactly one CPU, one RAM
        # (connected to that CPU), and one Storage.
        cpus = [hw for hw in sys.hw.values() if isinstance(hw, BaseCPU)]
        rams = [hw for hw in sys.hw.values() if isinstance(hw, BaseMemory)]
        storages = [hw for hw in sys.hw.values() if isinstance(hw, BaseStorage)]

        if len(cpus) != 1 or len(rams) != 1 or len(storages) != 1:
            self.sys.abort({
                "from": self.name,
                "msg": (
                    f"ExampleCPU expects exactly 1 CPU / 1 RAM / 1 Storage; "
                    f"got {len(cpus)} / {len(rams)} / {len(storages)}."
                ),
            })
            return

        self.cpu: BaseCPU = cpus[0]
        self.ram: BaseMemory = rams[0]
        self.storage: BaseStorage = storages[0]

        if self.cpu.memory is not self.ram:
            self.sys.abort({
                "from": self.name,
                "msg": (
                    f"ExampleCPU expects RAM {self.ram.name!r} to be connected to "
                    f"CPU {self.cpu.name!r}, but CPU's memory is {self.cpu.memory.name!r}."
                ),
            })
            return

        # Runtime state.
        self.node_ids: list[int] = list(self.sys.trace.node_map.keys())
        self.next_node_idx: int = 0
        self.pending_compute_ids: set = set()
        self.regions_to_release: list = []

        return

    def compile(self, trace: Trace) -> None:
        # ExampleCPU doesn't do compute graph transformation.
        return

    def layout(self, init_storage: BaseStorage) -> bool:
        # Find the peak memory (in pages) required to compute any single
        # node: sum of input + output tensor pages, taken over all nodes.
        tensor_map = self.sys.trace.tensor_map
        peak_node_pages = 0
        for node in self.sys.trace.node_map.values():
            node_pages = sum(tensor_map[tid].num_pages for tid in node.input_tensors)
            node_pages += sum(tensor_map[tid].num_pages for tid in node.output_tensors)
            if node_pages > peak_node_pages:
                peak_node_pages = node_pages
        self.peak_node_pages: int = peak_node_pages

        # Reserve the first (peak_node_pages * lookahead) of RAM as the
        # compute buffer, so tensors for up to `lookahead` nodes can be
        # loaded concurrently. Pinned tensors live in pages
        # [page_idx_start, num_total_pages).
        self.page_idx_start: int = peak_node_pages * self.lookahead

        # Pin tensors into RAM from storage in tensor_map order, packing
        # contiguously starting at page_idx_start, until RAM is full.
        batch = []
        next_page = self.page_idx_start
        for tensor in tensor_map.values():
            stor_regions = self.sys.find(self.storage, tensor)
            if not stor_regions:
                continue
            if not self.ram.space.check_avail(next_page, tensor.num_pages):
                break
            mem_region = self.sys.claim(self.ram, tensor, next_page)
            if mem_region is None:
                break
            batch.append((stor_regions[0], mem_region))
            tensor.args["pin"] = True
            next_page += tensor.num_pages

        if batch:
            self.sys.transfer(batch)

        return True

    def _find_buffer_slot(self, num_pages: int) -> int | None:
        """First-fit search for `num_pages` of contiguous free space in
        the buffer area [0, page_idx_start). Walks live regions in
        page order and returns the first gap big enough, or None."""
        regions = self.ram.space._regions_by_page_idx_start
        cursor = 0
        for region in regions.values():
            if region.page_idx_start >= self.page_idx_start:
                break
            if region.page_idx_start - cursor >= num_pages:
                return cursor
            cursor = max(cursor, region.page_idx_end)
        if self.page_idx_start - cursor >= num_pages:
            return cursor
        return None

    def runtime(self, retired_jobs: list[BaseJob]) -> None:
        # Block until the previous batch's compute jobs have all retired.
        if self.pending_compute_ids:
            retired_ids = {j.id for j in retired_jobs}
            self.pending_compute_ids -= retired_ids
            if self.pending_compute_ids:
                return

        # Offload non-pinned tensors from the just-finished batch.
        for region in self.regions_to_release:
            self.sys.release(region)
        self.regions_to_release = []

        # All nodes processed → end runtime stage.
        if self.next_node_idx >= len(self.node_ids):
            self.sys.end_stage()
            return

        # Take the next `lookahead` nodes.
        batch_end = min(self.next_node_idx + self.lookahead, len(self.node_ids))
        batch_node_ids = self.node_ids[self.next_node_idx:batch_end]
        self.next_node_idx = batch_end

        tensor_map = self.sys.trace.tensor_map

        # For every unique input/output tensor in the batch: if it is not
        # already resident in RAM (pinned), claim a region in the compute
        # buffer; if it has a storage source, queue a storage->RAM load.
        # Only the loaded regions are eligible for release after the batch
        # finishes — intermediates we just claimed for outputs stay so
        # downstream nodes can consume them (their data only exists in
        # RAM; there is no storage source to reload from).
        transfer_batch = []
        loaded_regions = []
        seen: set[int] = set()
        for nid in batch_node_ids:
            node = self.sys.trace.node_map[nid]
            for tid in node.input_tensors + node.output_tensors:
                if tid in seen:
                    continue
                seen.add(tid)
                tensor = tensor_map[tid]

                if self.sys.find(self.ram, tensor):
                    continue

                slot = self._find_buffer_slot(tensor.num_pages)
                if slot is None:
                    self.sys.abort({
                        "from": self.name,
                        "msg": f"Buffer area exhausted for tensor {tid} ({tensor.name}, {tensor.num_pages} pages).",
                    })
                    return
                mem_region = self.sys.claim(self.ram, tensor, slot)
                if mem_region is None:
                    return

                stor_regions = self.sys.find(self.storage, tensor)
                if stor_regions:
                    transfer_batch.append((stor_regions[0], mem_region))
                    loaded_regions.append(mem_region)

        self.regions_to_release = loaded_regions

        if transfer_batch:
            self.sys.transfer(transfer_batch)

        # Dispatch the batch's compute jobs.
        for nid in batch_node_ids:
            node = self.sys.trace.node_map[nid]
            job_id = self.sys.compute(self.cpu, node)
            self.pending_compute_ids.add(job_id)

        return
