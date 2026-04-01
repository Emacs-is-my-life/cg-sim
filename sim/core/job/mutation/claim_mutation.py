from sim.core import System
from sim.core.job import ClaimJob

from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage


def begin_mutation(job: ClaimJob, sys: System) -> None:
    # 0. Add job to hardware running slot
    for hw in job.running_on:
        hw.run(job)

    hw: BaseMemory | BaseStorage = job.running_on[0]
    hw.space.claim(job.tensor_id, job.page_idx_start, job.num_pages)
    return


def end_mutation(job: ClaimJob, sys: System) -> None:
    # 0. Retire job from hardware running slot
    for hw in job.running_on:
        hw.retire(job)

    return
