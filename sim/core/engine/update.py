import heapq

from sim.core.job import BaseJob, ComputeJob, TransferJob

from .update_transfer import update_transfer_jobs


def update_running_jobs(sys, jobs_running: list[BaseJob], timestamp_now: float) -> None:
    """Update ETA of all jobs, based on new system state"""

    transfer_jobs = []
    for job in jobs_running:
        if isinstance(job, ComputeJob):
            # Process Compute jobs
            hw = job.running_on[0]
            work_rate = hw.max_work_rate()
            job.update_ETA(timestamp_now, work_rate)
        elif isinstance(job, TransferJob):
            # Transfer Jobs must be updated in wholistic manner
            transfer_jobs.append(job)
        else:
            job.update_ETA(timestamp_now, 1)  # Claim/Release jobs are instant

    # Process transfer jobs using water filling algorithm
    update_transfer_jobs(transfer_jobs, timestamp_now)

    # Gotta heapify jobs_running queue by sorthing them with ETA
    heapq.heapify(jobs_running)
    return
