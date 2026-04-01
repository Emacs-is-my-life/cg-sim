from sim.core.log import Log
from sim.core import System
from sim.core.trace import Tensor
from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage

from .job import BaseJob

from .assertion.claim_assertion import assertion
from .mutation.claim_mutation import begin_mutation, end_mutation
from .logging.claim_logging import begin_log, end_log


class ClaimJob(BaseJob):
    """
    Data Region claim job
    """
    def __init__(self, hw: BaseMemory | BaseStorage, tensor: Tensor, page_idx_start: int = -1):
        work_total = 0                # Instant event
        super().__init__(work_total)
        self.work_rate = 1

        self.running_on.append(hw)
        self.tensor_id = tensor.id
        self.num_pages = tensor.num_pages
        self.page_idx_start = page_idx_start
        return

    def is_runnable(self, sys: System) -> bool:
        return assertion(self, sys)

    def begin_mut(self, sys: System) -> None:
        begin_mutation(self, sys)
        return

    def begin_log(self, log: Log) -> None:
        begin_log(self, log)
        return

    def end_mut(self, sys: System) -> None:
        end_mutation(self, sys)
        return

    def end_log(self, log: Log) -> None:
        end_log(self, log)
        return
