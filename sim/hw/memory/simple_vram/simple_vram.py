from typing import Any

from sim.core.log import Log
from sim.core.job import BaseJob
from sim.hw.memory.common import BaseMemory


class SimpleVRAM(BaseMemory):
    """
    Simplistic VRAM model.

    This is intentionally equivalent to SimpleRAM for now. Keeping a distinct
    class lets configs and logs distinguish host RAM from GPU-local memory.
    """

    def __init__(self, obj_id: int, name: str, log: Log, args: dict[str, Any]):
        memory_size_KB: int = int(args["memory_size_KB"])
        super().__init__(obj_id, name, log, memory_size_KB, args=args)
        self.memory_bandwidth_KBps = float(args["memory_bandwidth_KBps"])
        self.max_concurrent_jobs = int(args.get("max_concurrent_jobs", 4))
        if self.max_concurrent_jobs <= 0:
            raise ValueError("[Memory] max_concurrent_jobs must be > 0.")
        return

    def can_run(self, job: BaseJob) -> bool:
        return len(self.job_running) < self.max_concurrent_jobs

    def max_work_rate(self) -> float:
        if self.job_running:
            return self.memory_bandwidth_KBps / 1_000_000
        else:
            return 0
