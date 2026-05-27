from typing import Any, TYPE_CHECKING

from sim.core.log import Log
from sim.hw.common.base_hardware import BaseHardware

if TYPE_CHECKING:
    from .storage_region import StorageSpace


class BaseStorage(BaseHardware):
    """Base class for storage hardware models"""

    def __init__(self, obj_id: int, name: str, log: Log, args: dict[str, Any] | None = None):
        super().__init__(obj_id, name, log, args)
        from .storage_region import StorageSpace
        self.space: StorageSpace = StorageSpace(self)
        self.fixed_latency_micros: float = 0.0

        self.initial_placement: bool = False
        return

    def log_counters(self) -> dict[str, Any]:
        total_transfers = 0
        for job in self.job_running:
            total_transfers += (job.work_rate or 0)

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
