from typing import Any

from sim.core.log import Log
from sim.core.job import BaseJob
from sim.hw.compute.common import BaseCompute
from sim.hw.memory.common import BaseMemory


class SimpleGPU(BaseCompute):
    """
    Simplistic GPU model.

    Like SimpleCPU, this uses profiled node latency as the amount of work and
    scales it by a configurable modifier. The optional concurrency limit is
    conservative by default because the engine currently exposes one aggregate
    compute work rate per hardware.
    """

    def __init__(self, obj_id: int, name: str, log: Log, memory: BaseMemory, args: dict[str, Any]):
        super().__init__(obj_id, name, log, memory)

        modifier: float = float(args["modifier"])
        if modifier <= 0:
            raise ValueError("[Compute]: modifier must be > 0.")

        self.modifier = modifier
        self.max_concurrent_jobs = int(args.get("max_concurrent_jobs", 1))
        if self.max_concurrent_jobs <= 0:
            raise ValueError("[Compute]: max_concurrent_jobs must be > 0.")

        return

    def can_run(self, job: BaseJob) -> bool:
        return len(self.job_running) < self.max_concurrent_jobs

    def max_work_rate(self) -> float:
        if self.job_running:
            return self.modifier
        else:
            return 0
