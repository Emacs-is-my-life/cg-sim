from typing import Any

from sim.core.log import Log
from sim.hw.common import BaseHardware

from .memory_region import MemorySpace
from .utils import KB_to_num_pages


class BaseMemory(BaseHardware):
    """Base class for memory hardwares"""

    def __init__(self, obj_id: int, name: str, log: Log, memory_size_KB: int):
        if memory_size_KB < 4:
            raise ValueError(f"[Memory] Memory size cannot be: {memory_size_KB}")

        super().__init__(obj_id, name, log)
        self.space: MemorySpace = MemorySpace(self, KB_to_num_pages(memory_size_KB))
        return

    def log_counters(self) -> dict[str, Any]:
        total_transfers = 0
        for job in self.job_running:
            total_transfers += job.work_rate

        counters = {
            "memory_used_KB": 4 * self.space.num_used_pages,
            "memory_transfer_KBps": 1_000_000 * total_transfers
        }

        return counters

    def log_states(self) -> dict[str, Any]:
        states = {
            "regions": []
        }

        for mem_region in self.space._regions_by_page_idx_start.values():
            states["regions"].append({
                "page_idx_start": mem_region.page_idx_start,
                "page_idx_end": mem_region.page_idx_end,
                "tensor_id": mem_region.tensor_id,
                "access_status": mem_region.access_status.name,
                "is_latest": mem_region.is_latest,
                "is_ready": mem_region.is_ready
            })

        return states
