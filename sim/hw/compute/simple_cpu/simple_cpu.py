from typing import Any

from sim.core.log import Log
from sim.hw.compute.common import BaseCompute


class SimpleCPU(BaseCompute):
    """
    Simplistic model of CPU.
    It has only one tunable parameter: modifier
    Calculates latency based on Node profiling info and this modifier

    Latency = node.compute_time_micros / modifier
    """

    def init(self, obj_id: int, name: str, log: Log, args: dict[str, Any]):
        super().__init__(obj_id, name, log)

        modifier: float = args["modifier"]
        if modifier <= 0:
            raise ValueError("[Compute]: modifier must be > 0.")

        self.modifier = modifier
        return

    def is_avail(self) -> bool:
        return len(self.job_running) == 0

    def update_work_rate(self) -> None:
        self.max_rate.compute = self.modifier
        return
