from sim.core import System
from sim.core.job import ClaimJob

from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage


def assertion(job: ClaimJob, sys: System) -> bool:
    # 0. Hardware Availability
    hw: BaseMemory | BaseStorage = job.running_on[0]
    if not hw.can_run(job):
        return False

    if isinstance(hw, BaseMemory):
        page_idx_start = job.page_idx_start
        num_pages = job.num_pages
        return hw.space.check_avail(page_idx_start, num_pages)
    elif isinstance(hw, BaseStorage):
        return hw.space.check_avail()
    else:
        raise Exception(f"[Engine] Cannot request claim job to {hw.name}.")
