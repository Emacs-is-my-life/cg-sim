from sim.core.log import Log
from sim.core.job import ClaimJob


def begin_log(job: ClaimJob, log: Log, timestamp: float) -> None:
    args = {
        "page_idx_start": job.page_idx_start,
        "num_pages": job.num_pages,
        "tensor_id": job.tensor_id
    }

    for hw in job.running_on:
        log.record(Log.event_instant(
            hw.id,
            "DATA_REGION_CLAIMED",
            timestamp,
            args
        ))

    return


def end_log(job: ClaimJob, log: Log, timestamp: float) -> None:
    return
