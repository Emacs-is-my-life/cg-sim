from typing import Any
from enum import Enum, auto
from collections import deque
import heapq

from sim.core import SimObject, System
from sim.core.log import Log, TrackID
from sim.core.trace import TerminalNode
from sim.core.job import BaseJob, ComputeJob
from sim.hw.memory.common import BaseMemory

from .update import update_running_jobs


class SimStage(Enum):
    """Simulation stage"""
    COMPILE = auto()    # Scheduler modifies the compute graph
    LAYOUT = auto()     # Scheduler places tensors in storage/memory
    RUNTIME = auto()    # Scheduler makes runtime decisions
    FINISHED = auto()


class Engine(SimObject):
    """
    Engine is central point of simulation.
    """

    # TODO: polish type annotation(sys: System, sched: BaseScheduler)
    def __init__(self, obj_id: int, name: str, log: Log, sys: System, sched: Any):
        super().__init__(obj_id, name, log)

        # Key Objects
        self.log = log
        self.sys = sys
        self.sys.engine = self

        self.sched = sched

        # Simulation states
        self.stage = SimStage.COMPILE   # Simulation Stage
        self.timestamp_now: float = 0   # Simulation time, in microseconds
        self.time_elapsed: float = 0    # Time elapsed between the last job finish and the current

        # Job Queues
        self.job_waiting: deque[BaseJob] = deque()  # job_waiting.popleft()
        self.job_running: list[BaseJob] = []

        # Create subtracks for logging
        self.log.record(Log.subtrack(TrackID.Engine, self.id, self.name))
        return

    # Public Interfaces
    def submit(self, job: BaseJob) -> None:
        self.job_waiting.append(job)
        return

    def run(self) -> None:
        self._compile()
        self._layout()
        self._runtime()
        self._cleanup()
        return

    def _compile(self) -> None:
        self.sched.compile()
        return

    def _layout_forward(self) -> list[BaseJob]:
        retired_jobs: list[BaseJob] = []

        # Turn off logging
        self.log.on = False

        # Pop everything w/o advancing time
        while self.job_running:
            job = heapq.heappop(self.job_running)
            job.end(self.log, self.sys, self.timestamp_now)
            retired_jobs.append(job)

        # Turn on logging
        self.log.on = True

        return retired_jobs

    # TODO: Implement properly asserted _layout routine.
    def _layout(self) -> None:
        self.sched.layout()
        return

    def _runtime_forward(self) -> list[BaseJob]:
        retired_jobs: list[BaseJob] = []

        if self.job_running:
            job = heapq.heappop(self.job_running)

            self.time_elapsed = job.timestamp_ETA - self.timestamp_now
            self.timestamp_now = job.timestamp_ETA
            job.end(self.log, self.sys, self.timestamp_now)
            retired_jobs.append(job)

            while self.job_running and self.job_running[0].timestamp_ETA == self.timestamp_now:
                job = heapq.heappop(self.job_running)
                job.end(self.log, self.sys, self.timestamp_now)
                retired_jobs.append(job)

        return retired_jobs

    def _runtime(self) -> None:
        while True:
            # Inspect retired jobs
            retired_jobs = self._runtime_forward()
            for job in retired_jobs:
                # Check if simulation is finished
                if isinstance(job, ComputeJob) and isinstance(job.node, TerminalNode):
                    return

            # Update progress
            for job in self.job_running:
                job.update_progress(self.time_elapsed)

            # Scheduler Decisions
            self.sched.runtime(retired_jobs)

            # Drain all runnable jobs from job_waiting to job_running, in FIFO manner
            while self.job_waiting:
                job_w = self.job_waiting.popleft()
                # Start runnable jobs
                if job_w.is_runnable(self.sys):
                    job_w.begin(self.log, self.sys, self.timestamp_now)
                    self.job_running.append(job_w)  # Use append here, as ETA are None for new jobs
                else:
                    # Head-of-the-line blocking w/ no running job now
                    if len(self.job_running) == 0:
                        raise Exception("[Engine] Deadlock detected.")
                    else:
                        # Respect strict FIFO order
                        break

            # Update hardware performances, as running jobs are changed
            for hw in self.sys.hw.values():
                hw.update_work_rate()

            # Update ETA of running jobs
            update_running_jobs(self.sys, self.job_running, self.timestamp_now)

        return

    def _cleanup(self) -> None:
        """Write a report"""
        args = {
            "simulation_time": self.timestamp_now,
            "memory": []
        }

        for hw in self.sys.hw.values():
            if isinstance(hw, BaseMemory):
                args["memory"].append({
                    "id": hw.id,
                    "name": hw.name,
                    "peak_memory_usage_KB": 4 * hw.space.peak_num_used_pages
                })

        self.log.record(Log.engine(self.id, "SIMULATION_REPORT", self.timestamp_now, args))
        self.log.stop()
        return
