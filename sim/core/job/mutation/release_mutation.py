from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.core.system import System
    from ..release_job import ReleaseJob


def begin_mutation(job: ReleaseJob, sys: System) -> None:
    region = job.region
    hw = region.hw
    hw.space.release(region)

    return


def end_mutation(job: ReleaseJob, sys: System) -> None:
    return
