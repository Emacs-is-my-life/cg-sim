from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
import uuid
import fastuuid

from sim.core.log import Log

if TYPE_CHECKING:
    from sim.core.system import System
    from sim.hw.common.base_hardware import BaseHardware


class BaseJob(ABC):
    def __init__(self, work_total: float):
        # Basics
        self.id: uuid.UUID = fastuuid.uuid4()
        self.running_on: list["BaseHardware"] = []
        self.args: dict[str, Any] = {}

        # Work
        self.work_total = work_total          # Total amount of work done
        self.work_done: float = 0             # Amount of work processed so far
        self.work_rate: float = 0   # Work being processed per microsecond

        # Timestamp
        self.timestamp_begin: float | None = None  # Simulation time, that job started execution
        self.timestamp_end: float | None = None    # Simulation time, that job ended execution
        self.timestamp_ETA: float | None = None    # Simulation time, when job is expected to end

    def __lt__(self, other: "BaseJob") -> bool:
        my_ETA = self.timestamp_ETA if self.timestamp_ETA is not None else float("inf")
        other_ETA = other.timestamp_ETA if other.timestamp_ETA is not None else float("inf")

        return (my_ETA, self.id) < (other_ETA, other.id)

    """
    Job Lifecycle:

    - Scheduler invokes hardware.pleaseDoSomething()
    - Hardware creates a job, then invoke engine.submit(job)
    - job is dumped into engine.job_waiting queue (works in FIFO manner)

    - When the job get to the head of engine.job_waiting queue, engine checks job.is_runnable(sys)
    - if this job is runnable, then job is moved to engine.job_running queue,
      - job.begin()

    - While simulation is running:
      - job.update_progress()
      | Engine retires finished jobs ...
      | Scheduler submits new jobs ...
      | Engine recomputes hardware performance ...
      - job.update_ETA()

    - When job is finished
      - job.end()
    """

    @abstractmethod
    def is_runnable(self, sys: System) -> bool:
        """Based system state, check if this job is runnable"""
        pass

    def begin(self, log: Log, sys: System, timestamp_now: float):
        self.timestamp_begin = timestamp_now

        # Job start hook
        self.begin_mut(sys)
        self.begin_log(log)
        return

    def update_progress(self, time_elapsed: float) -> None:
        if self.work_rate is None or self.work_rate == 0:
            raise Exception(f"[Job] Job ID: {self.id}, work_rate is: {self.work_rate}")

        self.work_done = min(self.work_done + self.work_rate * time_elapsed, self.work_total)
        return

    def update_ETA(self, timestamp_now: float, new_work_rate: float | None = None) -> None:
        if new_work_rate is not None:
            self.work_rate = new_work_rate

        if self.work_rate is None or self.work_rate == 0:
            raise Exception(f"[Job] Job ID: {self.id}, work_rate is: {self.work_rate}")

        work_left = max(self.work_total - self.work_done, 0.0)
        self.timestamp_ETA = timestamp_now + (work_left / self.work_rate)
        return

    def end(self, log: Log, sys: System, timestamp_now: float):
        self.timestamp_end = timestamp_now

        # Job finish hook
        self.end_mut(sys)
        self.end_log(log)
        return

    @abstractmethod
    def begin_mut(self, sys: System) -> None:
        """job execution begins -> system state mutation"""
        pass

    @abstractmethod
    def begin_log(self, log: Log) -> None:
        """job execution begins -> log this event"""
        pass

    @abstractmethod
    def end_mut(self, sys: System) -> None:
        """job execution ends -> system state mutation"""
        pass

    @abstractmethod
    def end_log(self, log: Log) -> None:
        """job execution ends -> log this event"""
        pass
