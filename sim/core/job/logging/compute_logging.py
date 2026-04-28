from __future__ import annotations

from typing import TYPE_CHECKING

from sim.core.log import Log
from sim.core.trace import Node
from sim.hw.compute.common import BaseCompute

if TYPE_CHECKING:
    from ..compute_job import ComputeJob


def begin_log(job: ComputeJob, log: Log) -> None:
    # node: Node = job.node
    # args = {
    #     "node_id": node.id,
    #     "node_name": node.name,
    #     "work_total": node.compute_time_micros
    # }

    # for hw in job.running_on:
    #     log.record(Log.event_begin(
    #         hw.id,
    #         "COMPUTE_BEGIN",
    #         job.timestamp_begin,
    #         args
    #     ))

    return


def end_log(job: ComputeJob, log: Log) -> None:
    node: Node = job.node
    hw: BaseCompute = job.running_on[0]
    args = {
        "Hardware": {
            "id": hw.id,
            "name": hw.name,
        },
        "Payload": {
            "id": node.id,
            "name": node.name,
            "work_total": job.work_total
        },
        "Lifecycle": {
            "timestamp_queued": job.timestamp_queued,
            "timestamp_at_head": job.timestamp_at_head,
            "timestamp_begin": job.timestamp_begin,
            "timestamp_end": job.timestamp_end
        }
    }

    timestamp = job.timestamp_begin
    duration = job.timestamp_end - job.timestamp_begin

    for hw in job.running_on:
        log.record(Log.event_complete(
            hw.id,
            "COMPUTE_JOB",
            timestamp,
            duration,
            args
        ))

    return
