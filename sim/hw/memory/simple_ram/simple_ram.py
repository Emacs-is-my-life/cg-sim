from typing import Any

from sim.core.log import Log
from sim.hw.memory.common import BaseMemory


class SimpleRAM(BaseMemory):
    """
    Simplistic model of RAM

    - Only memory size is configurable
    - Instant memory move(memory-to-memory transfer)
    """

    def init(self, obj_id: int, name: str, log: Log, args: dict[str, Any]):
        memory_size_KB: int = int(args["memory_size_KB"])
        super().__init__(obj_id, name, log, memory_size_KB)
        self.memory_bandwidth_KBps = float(args["memory_bandwidth_KBps"])
        return

    def is_avail(self) -> bool:
        return len(self.job_running) < 4

    def update_work_rate(self) -> None:
        self.max_rate.rw_total = (self.memory_bandwidth_KBps / 1_000_000)   # KB per microsecond
        return
