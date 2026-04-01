from __future__ import annotations

from typing import TYPE_CHECKING

from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage

if TYPE_CHECKING:
    from sim.core.system import System
    from ..claim_job import ClaimJob


def begin_mutation(job: ClaimJob, sys: System) -> None:
    hw: BaseMemory | BaseStorage = job.running_on[0]
    region = hw.space.claim(job.tensor_id, job.page_idx_start, job.num_pages)
    if region is None:
        args = {
            "from": sys.engine.name,
            "error": "Job Execution Failure",
            "job_type": "ClaimJob",
            "msg": f"Allocation: [{job.page_idx_start}, {job.page_idx_start+job.num_pages}) failed on {hw.name}."
        }
        sys.abort(args)
        return

    job.region = region
    return


def end_mutation(job: ClaimJob, sys: System) -> None:
    return
