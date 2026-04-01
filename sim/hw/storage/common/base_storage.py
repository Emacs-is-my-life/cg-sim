from typing import Any

from sim.core.log import Log
from sim.hw.common import BaseHardware

from .storage_region import StorageSpace


class BaseStorage(BaseHardware):
    """Base class for storage hardware models"""

    def __init__(self, obj_id: int, name: str, log: Log):
        super().__init__(obj_id, name, log)
        self.space: StorageSpace = StorageSpace(self)
        self.fixed_latency_micros: float = 0.0
        return

    def log_counters(self) -> dict[str, Any]:
        total_transfers = 0
        for job in self.job_running:
            total_transfers += job.work_rate

        counters = {
            "transfer_KBps": 1_000_000 * total_transfers
        }

        return counters

    def log_states(self) -> dict[str, Any]:
        states = {
            "tensors": []
        }

        for stor_region in self.space._regions:
            states["tensors"].append(stor_region.tensor_id)

        return states
