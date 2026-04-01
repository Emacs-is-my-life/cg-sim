from typing import Any, TYPE_CHECKING

from sim.core.log import Log
from sim.hw.common.base_hardware import BaseHardware

from .utils import KB_to_num_pages

if TYPE_CHECKING:
    from .memory_region import MemorySpace


class BaseMemory(BaseHardware):
    """Base class for memory hardwares"""

    def __init__(self, obj_id: int, name: str, log: Log, memory_size_KB: int):
        if memory_size_KB < 4:
            raise ValueError(f"[Memory] Memory size cannot be: {memory_size_KB}")

        super().__init__(obj_id, name, log)
        from .memory_region import MemorySpace
        self.space: MemorySpace = MemorySpace(self, KB_to_num_pages(memory_size_KB))
        return

    def log_counters(self) -> dict[str, Any]:
        total_transfers = 0
        for job in self.job_running:
            total_transfers += (job.work_rate or 0)

        counters = {
            "memory_used_KB": 4 * self.space.num_used_pages,
            "transfer_KBps": 1_000_000 * total_transfers
        }

        return counters

    def log_states(self) -> dict[str, Any]:
        states = {
            "tensors": []
        }

        for mem_region in self.space._regions_by_page_idx_start.values():
            states["tensors"].append(mem_region.tensor_id)

        return states
