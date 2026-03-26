from sim.core import System
from sim.core.job import ReleaseJob


def begin_mutation(job: ReleaseJob, sys: System) -> None:
    # 0. Add job to hardware running slot
    for hw in job.running_on:
        hw.run(job)

    region = job.region
    hw = region.hw
    hw.space.release(region)

    return


def end_mutation(job: ReleaseJob, sys: System) -> None:
    # 0. Retire job from hardware running slot
    for hw in job.running_on:
        hw.retire(job)

    return
