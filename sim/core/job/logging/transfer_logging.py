from __future__ import annotations

from typing import TYPE_CHECKING

from sim.core.log import Log
from sim.hw.common import DataRegion

if TYPE_CHECKING:
    from ..transfer_job import TransferJob


def begin_log(job: TransferJob, log: Log) -> None:
    return


def end_log(job: TransferJob, log: Log) -> None:
    batch: list[tuple[DataRegion, DataRegion]] = job.batch

    src0, dest0 = batch[0]
    total_transfer = 0
    for src_region, dest_region in batch:
        total_transfer += 4 * src_region.num_pages    # KB

    avg_rate = float("inf")
    duration = job.timestamp_end - job.timestamp_begin
    if duration != float(0.0):
        avg_rate = (1_000_000 * total_transfer) / duration

    args = {
        "Hardware": {
            "src": {
                "id": src0.hw.id,
                "name": src0.hw.name
            },
            "dest": {
                "id": dest0.hw.id,
                "name": dest0.hw.name
            }
        },
        "Payload": {
            "size_KB": total_transfer,
            "transfer_KBps": avg_rate,
            "batch": []
        }
    }

    for src_region, _ in batch:
        tensor_id = src_region.tensor_id
        args["Payload"]["batch"].append({
            "tensor_id": tensor_id,
        })

    timestamp = job.timestamp_begin
    duration = job.timestamp_end - job.timestamp_begin

    for hw in job.running_on:
        log.record(Log.event_complete(
            hw.id,
            f"TRANSFER_JOB[{src0.hw.name} -> {dest0.hw.name}]",
            timestamp,
            duration,
            args
        ))

    return
