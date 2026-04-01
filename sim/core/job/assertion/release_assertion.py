from __future__ import annotations

from typing import TYPE_CHECKING

from sim.hw.common.data_region import DataRegionAccess, DataRegion
from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage

if TYPE_CHECKING:
    from sim.core.system import System
    from ..release_job import ReleaseJob


def assertion(job: ReleaseJob, sys: System) -> bool:
    data_region: DataRegion = job.region
    # 1. Check if access is ongoing
    if data_region.access_status != DataRegionAccess.IDLE:
        return False

    return True
