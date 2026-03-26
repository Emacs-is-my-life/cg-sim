from sim.core.log import Log
from sim.core import System
from sim.hw.common import DataRegion

from .job import BaseJob


class TransferJob(BaseJob):
    """
    Data Region -> Data Region, data transfer job
    """
    def __init__(self, batch: list[(DataRegion, DataRegion)]):
        work_total = 0
        for src_region, _ in batch:
            work_total += 4 * src_region.num_pages    # KB

        super().__init__(work_total)
        self.work_rate = 1

        self.batch = batch
        src0, dest0 = batch[0]

        self.running_on.append(src0.hw)
        self.running_on.append(dest0.hw)
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
