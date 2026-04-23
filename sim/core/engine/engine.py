from __future__ import annotations

from typing import Any, TYPE_CHECKING
from enum import Enum, auto
from collections import deque
import heapq
import orjson

from sim.core.sim_object import SimObject
from sim.core.log import Log, TrackID, Level
from sim.core.trace import TerminalNode
from sim.core.job import BaseJob, ComputeJob, TransferJob
from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage
from sim.sched.common import BaseScheduler

from .update import update_running_jobs
from .job_stats import init_job_stats, record_job_stats

if TYPE_CHECKING:
    from sim.core.system import System


class EngineSignal(Enum):
    END_STAGE = auto()
    ABORT = auto()


class Engine(SimObject):
    """
    Engine is central point of simulation.
    """

    def __init__(self, obj_id: int, name: str, log: Log, sys: System, sched: BaseScheduler):
        super().__init__(obj_id, name, log)

        # Key Objects
        self.log = log
        self.sys = sys
        self.sys.engine = self
        self.sched = sched

        # Signals
        self.signal_abort: bool = False

        # Simulation states
        self.timestamp_now: float = 0   # Simulation time, in microseconds
        self.time_elapsed: float = 0    # Time elapsed between the last job finish and the current

        # Job Queues
        self.job_waiting: deque[BaseJob] = deque()  # job_waiting.popleft()
        self.job_running: list[BaseJob] = []

        # Job Statistics
        self.job_stats: dict[str, Any] = {}
        init_job_stats(self.job_stats)

        # Create subtracks for logging
        self.log.record(Log.subtrack(TrackID.Engine, self.id, "Engine"))
        return

    # Public Interfaces
    def submit(self, job: BaseJob) -> None:
        job.timestamp_queued = self.timestamp_now
        self.job_waiting.append(job)
        return

    def signal(self, signal: EngineSignal) -> None:
        match signal:
            case EngineSignal.ABORT:
                self.signal_abort = True
        return

    def run(self) -> None:
        print("[Engine] Compile stage start")
        self.log.record(Log.engine(self.id, "COMPILE_STAGE_START", self.timestamp_now))
        self._compile()

        # node_map and tensor_map dump
        arg_nodes, arg_tensors = self.log.get_trace_log(self.sys.trace)
        self.log.record(Log.engine(self.sys.trace.id, "NODES", self.timestamp_now, arg_nodes))
        self.log.record(Log.engine(self.sys.trace.id, "TENSORS", self.timestamp_now, arg_tensors))

        print("[Engine] Layout stage start")
        self.log.record(Log.engine(self.id, "LAYOUT_STAGE_START", self.timestamp_now))
        self._layout()

        print("[Engine] Runtime stage start")
        self.log.record(Log.engine(self.id, "RUNTIME_STAGE_START", self.timestamp_now))
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
            if not isinstance(job, TransferJob):
                args = {
                    "from": self.name,
                    "msg": "Scheduler can only submit TransferJob in layout phase."
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

        # Find the storage hardware with initial placement
        init_storage = None
        for hw in self.sys.hw.values():
            if isinstance(hw, BaseStorage) and hw.initial_placement:
                init_storage = hw

        # Scheduler Placement
        self.sched.layout(init_storage)

        # Run all jobs
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
                args = {
                    "from": self.name,
                    "msg": "Deadlock detected."
                }
                self._log_abort(args)
                self.signal_abort = True
                break

        # Retire all jobs
        self._layout_forward()

        # Turn logging back on
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

            # Retire jobs with the same ETA timestamp
            while self.job_running and (self.job_running[0].timestamp_ETA == self.timestamp_now):
                job = heapq.heappop(self.job_running)
                job.end(self.log, self.sys, self.timestamp_now)
                retired_jobs.append(job)

        # Handle Job Logging (Compute Stall Time, ...)
        record_job_stats(retired_jobs, self.job_stats)

        return retired_jobs

    def _runtime(self) -> None:
        while not self.signal_abort:
            # Log Hardware Counters & States
            running_hw = set(hw for job in self.job_running for hw in job.running_on)
            for hw in running_hw:
                if self.log.level.value >= Level.COUNTER.value:
                    self.log.record(Log.counter(hw.id, "HW Counter", self.timestamp_now, hw.log_counters()))

                if self.log.level.value >= Level.STATE.value:
                    self.log.record(Log.state(hw.id, "HW State", self.timestamp_now, hw.log_states()))

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

                # Mark its head arrivale time if not set
                if job_w.timestamp_at_head is None:
                    job_w.timestamp_at_head = self.timestamp_now

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

        args = {
            "simulation": {},
            "memory": [],
            "job": {}
        }

        args["simulation"]["success"] = str(not self.signal_abort)
        args["simulation"]["time"] = self.timestamp_now

        for hw in self.sys.hw.values():
            if isinstance(hw, BaseMemory):
                args["memory"].append({
                    "name": hw.name,
                    "peak_memory_usage_KB": 4 * hw.space.peak_num_used_pages
                })

        args["job"] = self.job_stats
        self.log.record(Log.engine(self.id, "SIMULATION_RESULT", self.timestamp_now, args))
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
