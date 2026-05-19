from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.core.system import System
    from ..release_job import ReleaseJob


def begin_mutation(job: ReleaseJob, sys: System) -> None:
    region = job.region

    # Clear sparse flag
    tensor = sys.trace.tensor_map[region.tensor_id]
    if tensor.flag_sparse:
        tensor.flag_sparse = False
        tensor.num_pages_sparse = None

    hw = region.hw
    hw.space.release(region)
    return


def end_mutation(job: ReleaseJob, sys: System) -> None:
    return
