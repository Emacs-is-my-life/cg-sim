from sim.core.log import Log
from sim.core.job import TransferJob
from sim.hw.common import DataRegion


def begin_log(job: TransferJob, log: Log, timestamp: float) -> None:
    batch: list[(DataRegion, DataRegion)] = job.batch

    src0, dest0 = batch[0]
    total_transfer = 0
    for src_region, dest_region in batch:
        total_transfer += 4 * src_region.num_pages    # KB

    args = {
        "from": src0.hw.name,
        "to": dest0.hw.name,
        "size_KB": total_transfer,
    }

    for hw in job.running_on:
        log.record(Log.event_begin(
            hw.id,
            "TRANSFER_BEGIN",
            timestamp,
            args
        ))

    return


def end_log(job: TransferJob, log: Log, timestamp: float) -> None:
    batch: list[(DataRegion, DataRegion)] = job.batch

    src0, dest0 = batch[0]
    total_transfer = 0
    for src_region, dest_region in batch:
        total_transfer += 4 * src_region.num_pages    # KB

    args = {
        "from": src0.hw.name,
        "to": dest0.hw.name,
        "size_KB": total_transfer,
    }

    for hw in job.running_on:
        log.record(Log.event_end(
            hw.id,
            "TRANSFER_END",
            timestamp,
            args
        ))

    return
