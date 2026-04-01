from __future__ import annotations

from typing import TYPE_CHECKING

from sim.core.log import Log
from sim.core.trace import Node

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
    args = {
        "node_id": node.id,
        "node_name": node.name,
        "work_total": job.work_total
    }

    timestamp = job.timestamp_begin
    duration = job.timestamp_end - job.timestamp_begin

    for hw in job.running_on:
        log.record(Log.event_complete(
            hw.id,
            f"COMPUTE_JOB[Node {node.id}]",
            timestamp,
            duration,
            args
        ))

    return
