from sim.core import System
from sim.core.job import TransferJob

from sim.hw.common import DataRegionAccess, DataRegion
from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage


def assertion(job: TransferJob, sys: System) -> bool:
    # 0. Hardware Availability
    for hw in job.running_on:
        if not hw.can_run(job):
            return False

    batch: list[tuple(DataRegion, DataRegion)] = job.batch
    src0, dest0 = batch[0]
    src_hw = src0.hw
    dest_hw = dest0.hw
    for src_region, dest_region in batch:
        # 1. Check Src Regions
        if src_region.hw != src_hw:
            args = {
                "from": sys.engine.name,
                "error": "Job Pre-Execution Assertion Failure",
                "job_type": "TransferJob",
                "msg": "Batch in a TransferJob must be of 'One Hardware' -> 'Another Hardware'"
            }
            sys.abort(args)
            return False

        if (not src_region.is_ready) or src_region.access_status == DataRegionAccess.BEING_WRITTEN:
            return False

        # 2. Check Dest Regions
        if dest_region.hw != dest_hw:
            args = {
                "from": sys.engine.name,
                "error": "Job Pre-Execution Assertion Failure",
                "job_type": "TransferJob",
                "msg": "Batch in a TransferJob must be of 'One Hardware' -> 'Another Hardware'"
            }
            sys.abort(args)
            return False

        if dest_region.access_status != DataRegionAccess.IDLE:
            return False

        # 3. Check if both are intended for the same tensor
        if src_region.tensor_id != dest_region.tensor_id:
            return False

    return True
