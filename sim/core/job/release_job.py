from __future__ import annotations

from typing import TYPE_CHECKING

from sim.core.log import Log
from sim.hw.common.data_region import DataRegion

from .job import BaseJob

from .assertion.release_assertion import assertion
from .mutation.release_mutation import begin_mutation, end_mutation
from .logging.release_logging import begin_log, end_log

if TYPE_CHECKING:
    from sim.core.system import System


class ReleaseJob(BaseJob):
    """
    Data Region release job
    """
    def __init__(self, region: DataRegion):
        work_total = 0                  # Instant event
        super().__init__(work_total)
        self.work_rate = 1

        hw = region.hw
        self.running_on.append(hw)
        self.region = region
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
