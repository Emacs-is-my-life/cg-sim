from __future__ import annotations

from typing import TYPE_CHECKING

from sim.hw.common.data_region import DataRegionAccess, DataRegion
from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage

if TYPE_CHECKING:
    from sim.core.system import System
    from ..transfer_job import TransferJob


def assertion(job: TransferJob, sys: System) -> bool:
    # 0. Hardware Availability
    for hw in job.running_on:
        if not hw.can_run(job):
            return False

    batch: list[tuple[DataRegion, DataRegion]] = job.batch
    src0, dest0 = batch[0]
    src_hw = src0.hw
    dest_hw = dest0.hw
    for src_region, dest_region in batch:
        # 1. Check Src Regions
        if src_region.hw != src_hw:
            print("Error 1")

            args = {
                "from": sys.engine.name,
                "error": "Job Pre-Execution Assertion Failure",
                "job_type": "TransferJob",
                "msg": "Batch in a TransferJob must be of 'One Hardware' -> 'Another Hardware'"
            }
            sys.abort(args)
            return False

        if (not src_region.is_ready) or src_region.access_status == DataRegionAccess.BEING_WRITTEN:
            print("Error 2")
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
            print("Error 3")
            return False

        if dest_region.access_status != DataRegionAccess.IDLE:
            print("Error 4")
            return False

        # 3. Check if both are intended for the same tensor
        if src_region.tensor_id != dest_region.tensor_id:
            args = {
                "from": sys.engine.name,
                "error": "Job Pre-Execution Assertion Failure",
                "job_type": "TransferJob",
                "msg": f"Src Tensor: {src_region.tensor_id}, Dest Tensor: {dest_region.tensor_id}"
            }
            sys.abort(args)
            print("Error 5")
            return False

        # 4. Check src_size <= dest_size
        if src_region.num_pages > dest_region.num_pages:
            args = {
                "from": sys.engine.name,
                "error": "Job Pre-Execution Assertion Failure",
                "job_type": "TransferJob",
                "msg": f"Src size: {src_region.num_pages} pages, Dest Tensor: {dest_region.num_pages} pages."
            }
            sys.abort(args)
            print("Error 6")
            return False

    return True
