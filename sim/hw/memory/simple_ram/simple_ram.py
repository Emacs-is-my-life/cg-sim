from typing import Any

from sim.core.log import Log
from sim.core.job import BaseJob
from sim.hw.memory.common import BaseMemory


class SimpleRAM(BaseMemory):
    """
    Simplistic model of RAM

    - Only memory size is configurable
    - Instant memory move(memory-to-memory transfer)
    """

    def __init__(self, obj_id: int, name: str, log: Log, args: dict[str, Any]):
        memory_size_KB: int = int(args["memory_size_KB"])
        super().__init__(obj_id, name, log, memory_size_KB)
        self.memory_bandwidth_KBps = float(args["memory_bandwidth_KBps"])
        return

    def can_run(self, job: BaseJob) -> bool:
        return len(self.job_running) < 4

    def max_work_rate(self) -> float:
        if self.job_running: 
            return (self.memory_bandwidth_KBps / 1_000_000)   # KB per microsecond
        else:
            return 0
