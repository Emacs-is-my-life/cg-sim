from sim.core import System
from sim.core.job import ReleaseJob


def begin_mutation(job: ReleaseJob, sys: System) -> None:
    region = job.region
    hw = region.hw
    hw.space.release(region)

    return


def end_mutation(job: ReleaseJob, sys: System) -> None:
    return
