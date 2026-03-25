from abc import ABC
from typing import Any
import uuid
import fastuuid


class BaseJob(ABC):
    def __init__(self, work_total: float):
        # Basics
        self.id: uuid.UUID = fastuuid.uuid4()
        self.running_on: [Any] = []    # TODO: implement BaseHardware, then do type annotation
        self.args: dict[str, Any] = {}

        # Work
        self.work_total = work_total          # Total amount of work done
        self.work_done: float = 0             # Amount of work processed so far
        self.work_rate: float | None = None   # Work being processed per microsecond

        # Timestamp
        self.timestamp_begin: float | None = None  # Simulation time, that job started execution
        self.timestamp_end: float | None = None    # Simulation time, that job ended execution
        self.timestamp_ETA: float | None = None    # Simulation time, when job is expected to end

        """
        How engine keeps track of jobs in job_running queue:

        0. Sorts jobs and pops a job from the engine.job_running queue (one with closest timestamp_ETA)
           then advances simulation time to its timestamp_ETA.
             - If it's a compute job, and its node.args has "LAST_NODE" field, simulation ends.
             - Otherwise, continue running.
        1. Scheduler submits jobs to the engine.job_queue (zero or more jobs)
        2. for job in engine.job_running:
             - Update job.work_done = job.work_total - (job.work_rate * engine.time_elapsed)
             - Update job.work_rate based on current running jobs & hardwares
             - Compute work_left = job.work_total - job.work_done
             - Update job.timestamp_ETA = engine.timestamp_now + (work_left / job.work_rate)
        3. GOTO 0.

        """

    def __lt__(self, other: "BaseJob") -> bool:
        return (self.timestamp_ETA) < (other.timestamp_ETA)
