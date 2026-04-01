from typing import Any

from sim.sched.common import BaseScheduler
from sim.core.log import Log
from sim.core.trace import Trace
from sim.core import System
from sim.core.job import BaseJob
from sim.hw.compute.common import BaseCompute
from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage


class Vanilla(BaseScheduler):
    """
    Vanilla scheduler
    Only runs when memory is adequate
    """

    def __init__(self, obj_id: int, name: str, log: Log, sys: System, args: dict[str, Any] | None = None):
        super().__init__(obj_id, name, log, sys, args)

        """
        Assumption: Vanila scheduler expects one of each:

        - One compute unit
        - One memory unit
        - One storage unit
        """
        for hw in sys.hw.values():
            if isinstance(hw, BaseCompute):
                self.compute = hw
            elif isinstance(hw, BaseMemory):
                self.memory = hw
            elif isinstance(hw, BaseStorage):
                self.storage = hw

        self.avail_page_idx: int = 0

        self.node_ids: list[int] = list(self.sys.trace.node_map.keys())
        self.last_job_id = None

        return

    def compile(self, trace: Trace) -> None:
        """Vanilla Scheduler won't do compilation."""
        return

    def layout(self, retired_jobs: list[BaseJob]) -> None:
        tensor_map = self.sys.trace.tensor_map

        # Check if memory size adequate to hold all tensors
        required_memory_pages = 0
        for tensor in tensor_map.values():
            required_memory_pages += tensor.num_pages

        if self.memory.space.num_total_pages < required_memory_pages:
            args = {
                "from": self.name,
                "error": "LAYOUT_FAILURE",
                "msg": f"Not enough memory size({self.memory.space.num_total_pages}). {required_memory_pages} required."
            }
            self.sys.abort(args)
            return

        # Claim Memory Regions for tensors
        batch = []
        for tensor in tensor_map.values():
            mem_region = self.sys.claim(self.memory, tensor, self.avail_page_idx)
            self.avail_page_idx += tensor.num_pages
            if tensor.args["tensor_type"] in ("INPUT", "WEIGHT"):
                stor_region = self.sys.find(self.storage, tensor)[0]
                batch.append((stor_region, mem_region))

        # Submit transfer job
        if batch:
            self.sys.transfer(batch)

        # End Layout stage
        self.sys.end_stage()
        return

    def runtime(self, retired_jobs: list[BaseJob]) -> None:
        if self.last_job_id is not None:
            retired_job_ids = {job.id for job in retired_jobs}
            if self.last_job_id not in retired_job_ids:
                return

        if not self.node_ids:
            self.sys.end_stage()
            return

        num_to_submit = min(8, len(self.node_ids))
        for _ in range(num_to_submit):
            node_id = self.node_ids.pop(0)
            self.last_job_id = self.sys.compute(self.compute, self.sys.trace.node_map[node_id])

        return
