from typing import Any

from sim.sched.common import BaseScheduler
from sim.core.log import Log
from sim.core.trace import Trace
from sim.core import System
from sim.core.job import BaseJob, ComputeJob, ClaimJob, ReleaseJob, TransferJob
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
        for hw in sys.hw.values:
            if isinstance(hw, BaseCompute):
                self.compute = hw
            elif isinstance(hw, BaseMemory):
                self.memory = hw
            elif isinstance(hw, BaseStorage):
                self.storage = hw

        return

    def compile(self, trace: Trace) -> None:
        """Vanilla Scheduler won't do compilation."""
        return

    def layout(self, retired_jobs: list[BaseJob]) -> None:
        tensor_map = self.sys.trace.tensor_map

        # Check the memory size adequate
        # - sum(tensor.num_pages in tensor_map) <= self.memory.space.num_total_pages
        # - if not, call sys.abort() with args: {"from": self.name, "msg": "Total Tensor Size: {sum_tensor_pages} while Memory Size: {self.memory.space.num_total_pages}"}

        # Load tensors one by one into Memory
        # - Keep a avail_page_idx = 0, which tracks the first available page_idx in memory now
        # Iterate over tensor in tensor_map
        # if tensor_type = tensor.args["tensor_type"] is "INPUT" or "WEIGHT"
        #   - find StorageRegion in self.storage by tensor.id
        #   - Allocate MemoryRegion in self.memory with tensor.id and tensor.num_pages, then increment avail_page_idx
        #   - 

        return

    def runtime(self, retired_jobs: list[BaseJob]) -> None:
        return
