from sim.core.log import Log
from sim.core import System
from sim.hw.common import DataRegion

from .job import BaseJob

from .assertion.transfer_assertion import assertion
from .mutation.transfer_mutation import begin_mutation, end_mutation
from .logging.transfer_logging import begin_log, end_log


class TransferJob(BaseJob):
    """
    Data Region -> Data Region, data transfer job
    """
    def __init__(self, batch: list[tuple[DataRegion, DataRegion]]):
        if len(batch) == 0:
            raise Exception("[Engine] batch size of TransferJob cannot be 0.")

        work_total = 0
        for src_region, _ in batch:
            work_total += 4 * src_region.num_pages    # KB

        super().__init__(work_total)
        self.work_rate = 1

        self.batch = batch
        src0, dest0 = batch[0]
        self.hw_from = src0.hw
        self.hw_to = dest0.hw

        self.running_on.append(src0.hw)
        self.running_on.append(dest0.hw)
        return

    def is_runnable(self, sys: System) -> bool:
        return assertion(self, sys)

    def begin_mut(self, sys: System) -> None:
        begin_mutation(self, sys)
        return

    def begin_log(self, log: Log, timestamp: float) -> None:
        begin_log(self, log, timestamp)
        return

    def end_mut(self, sys: System) -> None:
        end_mutation(self, sys)
        return

    def end_log(self, log: Log, timestamp: float) -> None:
        end_log(self, log, timestamp)
        return
