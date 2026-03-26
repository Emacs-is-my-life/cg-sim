from sim.core.log import Log
from sim.core.trace import Node
from sim.core.job import ComputeJob


def begin_log(job: ComputeJob, log: Log, timestamp: float) -> None:
    node: Node = job.node
    args = {
        "node_id": node.id,
        "node_name": node.name,
        "work_total": node.compute_time_micros
    }

    for hw in job.running_on:
        log.record(Log.event_begin(
            hw.id,
            "COMPUTE_BEGIN",
            timestamp,
            args
        ))

    return


def end_log(job: ComputeJob, log: Log, timestamp: float) -> None:
    node: Node = job.node
    args = {
        "node_id": node.id,
        "node_name": node.name,
        "work_total": node.compute_time_micros
    }

    for hw in job.running_on:
        log.record(Log.event_end(
            hw.id,
            "COMPUTE_END",
            timestamp,
            args
        ))

    return
