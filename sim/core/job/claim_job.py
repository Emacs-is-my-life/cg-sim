from sim.core.log import Log
from sim.core import System
from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage

from .job import BaseJob


class ClaimJob(BaseJob):
    """
    Data Region claim job
    """
    def __init__(self, hw: BaseMemory | BaseStorage, tensor_id: int, page_idx_start: int = -1, num_pages: int = -1):
        work_total = 0                # Instant event
        super().__init__(work_total)
        self.work_rate = 1

        self.running_on.append(hw)
        self.tensor_id = tensor_id
        self.page_idx_start = page_idx_start
        self.num_pages = num_pages
        return

    # TODO
    def is_runnable(self, sys: System) -> bool:
        pass

    def begin_mut(self, sys: System) -> None:
        pass

    def begin_log(self, log: Log, timestamp: float) -> None:
        pass

    def end_mut(self, sys: System) -> None:
        pass

    def end_log(self, log: Log, timestamp: float) -> None:
        pass
