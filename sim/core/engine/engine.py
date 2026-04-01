from __future__ import annotations

from typing import Any, TYPE_CHECKING
from enum import Enum, auto
from collections import deque
import heapq

from sim.core.sim_object import SimObject
from sim.core.log import Log, TrackID, Level
from sim.core.trace import TerminalNode
from sim.core.job import BaseJob, ComputeJob, TransferJob
from sim.hw.memory.common import BaseMemory

from .update import update_running_jobs

if TYPE_CHECKING:
    from sim.core.system import System


class EngineSignal(Enum):
    END_STAGE = auto()
    ABORT = auto()


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

        # Signals
        self.signal_end_stage: bool = False
        self.signal_abort: bool = False

        # Simulation states
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

    def signal(self, signal: EngineSignal) -> None:
        match signal:
            case EngineSignal.END_STAGE:
                self.signal_end_stage = True
            case EngineSignal.ABORT:
                self.signal_abort = True
        return

    def run(self) -> None:
        self.log.record(Log.engine(self.id, "COMPILE_STAGE_START", self.timestamp_now))
        self.signal_end_stage: bool = False
        self._compile()

        self.log.record(Log.engine(self.id, "LAYOUT_STAGE_START", self.timestamp_now))
        self.signal_end_stage: bool = False
        self._layout()

        ## DEBUG
        # self.log.record(Log.state(self.sys.trace.id, "TRACE_MAP", self.timestamp_now, self.sys.trace.log_states()))

        self.log.record(Log.engine(self.id, "RUNTIME_STAGE_START", self.timestamp_now))
        self.signal_end_stage: bool = False
        self._runtime()

        self._cleanup()
        return

    def _compile(self) -> None:
        self.sched.compile(self.sys.trace)

        # Advance time a little bit
        self.timestamp_now += 10
        return

    def _layout_forward(self) -> list[BaseJob]:
        retired_jobs: list[BaseJob] = []

        # Pop everything w/o advancing time
        while self.job_running:
            job = self.job_running.pop(0)
            if isinstance(job, ComputeJob):
                args = {
                    "from": self.name,
                    "msg": "Cannot execute compute job in layout phase."
                }
                self._log_abort(args)
                self.signal_abort = True
                break

            job.end(self.log, self.sys, self.timestamp_now)
            retired_jobs.append(job)

        return retired_jobs

    def _layout(self) -> None:
        # Turn off logging in placement step
        self.log.on = False

        while not (
                self.signal_abort or
                (self.signal_end_stage and len(self.job_running) == 0 and len(self.job_waiting) == 0)
        ):
            retired_jobs = self._layout_forward()

            # Scheduler Placement
            self.sched.layout(retired_jobs)
            while self.job_waiting:
                job_w = self.job_waiting[0]
                if job_w.is_runnable(self.sys):
                    self.job_waiting.popleft()
                    job_w.begin(self.log, self.sys, self.timestamp_now)
                    self.job_running.append(job_w)

                    # Set their ETA to all NOW (immediately finish)
                    for job in self.job_running:
                        job.timestamp_ETA = self.timestamp_now
                else:
                    if len(self.job_running) == 0:
                        args = {
                            "from": self.name,
                            "msg": "Deadlock detected."
                        }
                        self._log_abort(args)
                        self.signal_abort = True

                    break

        # Turn on logging
        self.log.on = True

        # Advance time a little bit
        self.timestamp_now += 10
        return

    def _runtime_forward(self) -> list[BaseJob]:
        retired_jobs: list[BaseJob] = []

        if self.job_running:
            job = heapq.heappop(self.job_running)
            if job.timestamp_ETA == float("inf"):
                # None of jobs in the queue are initialized properly, yet.
                heapq.heappush(self.job_running, job)
                return retired_jobs

            self.time_elapsed = job.timestamp_ETA - self.timestamp_now
            self.timestamp_now = job.timestamp_ETA

            # Handle fixed_latency of TransferJob
            if isinstance(job, TransferJob) and (job.fixed_latency_micros > 0.0):
                if not job.bw_work_complete:
                    job.bw_work_complete = True
                    heapq.heappush(self.job_running, job)
                else:
                    # Fixed latency is handled, so retire this TransferJob
                    job.end(self.log, self.sys, self.timestamp_now)
                    retired_jobs.append(job)
            else:
                job.end(self.log, self.sys, self.timestamp_now)
                retired_jobs.append(job)

            # Don't retire multiple jobs, as this might cause overlap issue in perfetto UI viewer.
            # while self.job_running and (self.job_running[0].timestamp_ETA == self.timestamp_now):
            #     job = heapq.heappop(self.job_running)
            #     job.end(self.log, self.sys, self.timestamp_now)
            #     retired_jobs.append(job)

        # Add a bit of time, to make events not overlap
        self.timestamp_now += 0.001  # 1/1000 microsecond

        # Counter and States logging
        hw_affected = set(hw for job in retired_jobs for hw in job.running_on)
        for hw in hw_affected:
            if self.log.level.value >= Level.COUNTER.value:
                self.log.record(Log.counter(hw.id, "HW Counter", self.timestamp_now, hw.log_counters()))

            if self.log.level.value >= Level.STATE.value:
                self.log.record(Log.state(hw.id, "HW State", self.timestamp_now, hw.log_states()))

        return retired_jobs

    def _runtime(self) -> None:
        while not self.signal_abort:
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
                job_w = self.job_waiting[0]
                # Start runnable jobs
                if job_w.is_runnable(self.sys):
                    self.job_waiting.popleft()
                    job_w.begin(self.log, self.sys, self.timestamp_now)
                    self.job_running.append(job_w)  # Use append here, as ETA are None for new jobs
                else:
                    # Head-of-the-line blocking w/ no running job now
                    if len(self.job_running) == 0:
                        args = {
                            "from": self.name,
                            "msg": "Deadlock detected.",
                            "job": None
                        }

                        if self.job_waiting:
                            job = self.job_waiting[0]
                            if isinstance(job, ComputeJob):
                                node = job.node
                                args["job"] = {
                                    "JOB_TYPE": "COMPUTE",
                                    "node": {
                                        "id": node.id,
                                        "name": node.name,
                                        "parent_nodes": node.parent_nodes,
                                        "input_tensors": node.input_tensors,
                                        "output_tensors": node.output_tensors
                                    }
                                }
                            elif isinstance(job, TransferJob):
                                args["job"] = {
                                    "JOB_TYPE": "TRANSFER"
                                }

                        self._log_abort(args)
                        self.signal_abort = True

                    break

            # Update work_rate and ETA of running jobs
            update_running_jobs(self.sys, self.job_running, self.timestamp_now)
        return

    def _cleanup(self) -> None:
        """Write a report"""
        args = {
            "simulation_success": str(not self.signal_abort),
            "simulation_time": self.timestamp_now,
            "hardware": {
                "book": [],
                "memory": []
            }
        }

        for hw in self.sys.hw.values():
            args["hardware"]["book"] = {"id": hw.id, "name": hw.name}

        for hw in self.sys.hw.values():
            if isinstance(hw, BaseMemory):
                args["hardware"]["memory"].append({
                    "id": hw.id,
                    "name": hw.name,
                    "peak_memory_usage_KB": 4 * hw.space.peak_num_used_pages
                })

        self.log.record(Log.engine(self.id, "SIMULATION_REPORT", self.timestamp_now, args))
        self.log.stop()
        return

    def _log_abort(self, args: dict[str, Any] | None = None) -> None:
        args = args if args is not None else {}
        self.log.record(Log.engine(self.id, "SIMULATION_ABORT", self.timestamp_now, args))
        return

    def log_counters(self) -> dict[str, Any] | None:
        """No counters to log"""
        return None

    def log_states(self) -> dict[str, Any] | None:
        """No states to log"""
        return None
