from typing import Any

from sim.core.job import BaseJob
from sim.core.job import ComputeJob, TransferJob


def init_job_stats(job_stats: dict[str, Any]) -> None:
    job_stats["compute_job_counts"]: int = 0
    job_stats["compute_total_time_micros"]: float = 0
    job_stats["transfer_job_counts"]: int = 0
    job_stats["transfer_total_time_micros"]: float = 0
    job_stats["transfer_total_size_KB"]: int = 0
    return


def record_job_stats(retired_jobs: list[BaseJob], job_stats: dict[str, Any]) -> None:
    # Record job related statistics, based on their type and lifecycle timestamps

    for job in retired_jobs:
        if isinstance(job, ComputeJob):
            job_stats["compute_job_counts"] += 1

            runtime = job.timestamp_end - job.timestamp_begin
            job_stats["compute_total_time_micros"] += runtime
        elif isinstance(job, TransferJob):
            job_stats["transfer_job_counts"] += 1

            runtime = job.timestamp_end - job.timestamp_begin
            job_stats["transfer_total_time_micros"] += runtime
            batch = job.batch
            size = 0
            for src_region, _ in batch:
                size += src_region.num_pages

            job_stats["transfer_total_size_KB"] += 4 * size
    return
