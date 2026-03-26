from sim.core.log import Log
from sim.core import System
from sim.hw.common import DataRegion

from .job import BaseJob


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
