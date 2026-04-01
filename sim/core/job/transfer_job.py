from sim.core.log import Log
from sim.core import System
from sim.hw.common import DataRegion
from sim.hw.storage.common import BaseStorage

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

        # Account for fixed latency
        hw_fixed_latency_micros: float = 0.0
        for hw in [self.hw_from, self.hw_to]:
            if isinstance(hw, BaseStorage):
                hw_fixed_latency_micros = max(hw_fixed_latency_micros, hw.fixed_latency_micros)

        self.fixed_latency_micros = hw_fixed_latency_micros
        self.fixed_latency_hit: bool = False
        self.bw_work_complete: bool = False
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

    # Override update_ETA logic to anticipate fixed latency
    def update_ETA(self, timestamp_now: float, new_work_rate: float | None = None) -> None:
        if new_work_rate is not None:
            self.work_rate = new_work_rate

        if self.work_rate is None or self.work_rate == 0:
            raise Exception(f"[Job] Job ID: {self.id}, work_rate is: {self.work_rate}")

        work_left = max(self.work_total - self.work_done, 0.0)

        # Account fixed latency when updating ETA
        if self.bw_work_complete:
            if not self.fixed_latency_hit:
                self.timestamp_ETA = timestamp_now + self.fixed_latency_micros
                self.fixed_latency_hit = True
        else:
            self.timestamp_ETA = timestamp_now + (work_left / self.work_rate)

        return
