from __future__ import annotations

from typing import TYPE_CHECKING

from sim.core.log import Log

if TYPE_CHECKING:
    from ..claim_job import ClaimJob


def begin_log(job: ClaimJob, log: Log) -> None:
    args = {
        "page_idx_start": job.page_idx_start,
        "num_pages": job.num_pages,
        "tensor_id": job.tensor_id
    }

    for hw in job.running_on:
        log.record(Log.event_instant(
            hw.id,
            "CLAIM_JOB",
            job.timestamp_begin,
            args
        ))

    return


def end_log(job: ClaimJob, log: Log) -> None:
    return
