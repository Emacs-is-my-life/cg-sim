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

    # TODO: After defining storage job
    def log_counters(self) -> dict[str, Any]:

        counters = {
            "read_bandwidth_KBps": -1,
            "write_bandwidth_KBps": -1
        }
        return counters

    def log_states(self) -> dict[str, Any]:
        states = {
            "regions": []
        }

        for stor_region in self.space._regions:
            states["regions"].append({
                "tensor_id": stor_region.tensor_id,
                "access_status": stor_region.access_status.name,
                "is_latest": stor_region.is_latest,
                "is_ready": stor_region.is_ready
            })

        return states
