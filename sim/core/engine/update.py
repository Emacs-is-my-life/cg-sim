import heapq

from sim.core import System
from sim.core.job import BaseJob, ComputeJob, TransferJob
from sim.hw.common import DataRegion, BaseHardware

from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage


# TODO: Implement complex & accurate flow modeling w/ water filling algorithm
def update_running_jobs(sys: System, jobs_running: list[BaseJob], timestamp_now: float) -> None:
    """Update ETA of all jobs, based on new system state"""

    for job_r in jobs_running:
        if isinstance(job_r, ComputeJob):
            hw = job_r.running_on[0]
            new_work_rate = hw.max_rate.compute
            job_r.update_ETA(timestamp_now, new_work_rate)
        elif isinstance(job_r, TransferJob):
            batch: list[(DataRegion, DataRegion)] = job_r.batch
            src0, dest0 = batch[0]
            src_hw: BaseHardware = src0.hw
            dest_hw: BaseHardware = dest0.hw

            src_max_rate = 0
            dest_max_rate = 0

            if isinstance(src_hw, BaseMemory):
                src_max_rate = src_hw.max_rate.rw_total
            elif isinstance(src_hw, BaseStorage):
                src_max_rate = src_hw.max_rate.read_from

            if isinstance(dest_hw, BaseMemory):
                dest_max_rate = dest_hw.max_rate.rw_total
            elif isinstance(dest_hw, BaseStorage):
                dest_max_rate = dest_hw.max_rate.write_to

            new_work_rate = min(src_max_rate, dest_max_rate)
            job_r.update_ETA(timestamp_now, new_work_rate)

    # Gotta heapify jobs_running queue
    heapq.heapify(jobs_running)
    return
