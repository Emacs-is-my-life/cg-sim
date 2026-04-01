from sim.core.log import Log
from sim.core.job import ReleaseJob


def begin_log(job: ReleaseJob, log: Log) -> None:
    args = {
        "num_pages": job.region.num_pages,
        "tensor_id": job.region.tensor_id
    }

    for hw in job.running_on:
        log.record(Log.event_instant(
            hw.id,
            "RELEASED_JOB",
            job.timestamp_begin,
            args
        ))

    return


def end_log(job: ReleaseJob, log: Log) -> None:
    return
