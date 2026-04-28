from typing import Any

from sim.core.log import Log
from sim.core.job import BaseJob
from sim.hw.memory.common import BaseMemory

from .base_compute import BaseCompute


class BaseCPU(BaseCompute):
    def __init__(self, obj_id: int, name: str, log: Log, memory: BaseMemory, args: dict[str, Any]):
        pass

    def can_run(self, job: BaseJob) -> bool:
        pass

    def max_work_rate(self) -> float:
        pass
