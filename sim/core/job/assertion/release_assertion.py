from sim.core import System
from sim.core.job import ReleaseJob

from sim.hw.common import DataRegionAccess, DataRegion
from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage


def assertion(job: ReleaseJob, sys: System) -> bool:
    # 0. Hardware Availability
    hw: BaseMemory | BaseStorage = job.running_on[0]
    if not hw.can_run(job):
        return False

    data_region: DataRegion = job.region
    # 1. Check if access is ongoing
    if data_region.access_status != DataRegionAccess.IDLE:
        return False

    return True
