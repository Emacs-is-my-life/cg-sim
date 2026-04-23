from typing import Any

from sim.core.log import Log
from sim.core.job import BaseJob
from sim.hw.compute.common import BaseCompute
from sim.hw.memory.common import BaseMemory


class SimpleCPU(BaseCompute):
    """
    Simplistic model of CPU.
    It has only one tunable parameter: modifier
    Calculates latency based on Node profiling info and this modifier

    Latency = node.compute_time_micros / modifier
    """

    def __init__(self, obj_id: int, name: str, log: Log, memory: BaseMemory, args: dict[str, Any]):
        super().__init__(obj_id, name, log, memory)

        modifier: float = args["modifier"]
        if modifier <= 0:
            raise ValueError("[Compute]: modifier must be > 0.")

        self.modifier = modifier
        self.args["HW_type"] = "CPU"
        return

    def can_run(self, job: BaseJob) -> bool:
        return len(self.job_running) == 0

    def max_work_rate(self) -> float:
        if self.job_running:
            return self.modifier
        else:
            return 0
