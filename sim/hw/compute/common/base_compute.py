from typing import Any

from sim.core.log import Log
from sim.core.trace import Node
from sim.hw.common import BaseHardware
from sim.hw.memory.common import BaseMemory


class BaseCompute(BaseHardware):
    """Base class for compute hardwares"""

    def __init__(self, obj_id: int, name: str, log: Log, memory: BaseMemory):
        super().__init__(obj_id, name, log)
        self.memory: BaseMemory = memory
        return

    def log_counters(self) -> dict[str, Any]:
        """
        Log processing speed of currently running job
        """

        counters = {
            "compute_speed_AUps": 0
        }

        if self.job_running:
            job = self.job_running[0]
            counters["compute_speed_AUps"] = 1_000_000 * job.work_rate  # AU / second

        return counters

    def log_states(self) -> dict[str, Any]:
        """
        Show what job is running now
        """

        states = {
            "computing_now": {}
        }

        if self.job_running:
            job = self.job_running[0]
            node: Node = job.node
            states["computing_now"] = {
                "id": node.id,
                "name": node.name,
                "progress": f"{100 * (job.work_done / job.work_total)} %",
                "args": node.args
            }

        return states
