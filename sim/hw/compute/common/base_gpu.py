from typing import Any

from sim.core.log import Log
from sim.core.job import BaseJob
from sim.hw.memory.common import BaseMemory

from .base_compute import BaseCompute


class BaseGPU(BaseCompute):
    def __init__(self, obj_id: int, name: str, log: Log, memory: BaseMemory, args: dict[str, Any] | None = None):
        super().__init__(obj_id, name, log, memory, args)
        return

    def can_run(self, job: BaseJob) -> bool:
        pass

    def max_work_rate(self) -> float:
        pass
