from typing import Any
from enum import Enum, auto
import heapq

from sim.core import SimObject
from sim.core.log import Log, TrackID
from sim.core.job import BaseJob


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
    def __init__(self, obj_id: int, name: str, log: Log, sys: Any, sched: Any):
        super().__init__(obj_id, name)

        # Key Objects
        self.log = log
        self.sys = sys
        # TODO: iterate over sys.hw[<name>], inject engine(self) to them

        self.sched = sched

        # Simulation states
        self.stage = SimStage.COMPILE   # Simulation Stage
        self.timestamp_now: float = 0   # Simulation time, in microseconds
        self.time_elapsed: float = 0    # Time elapsed between the last job finish and the current

        # Job Queues
        self.job_queue: list[BaseJob] = []
        self.job_running: list[BaseJob] = []

        # Create subtracks for logging
        self.log.record(Log.subtrack(TrackID.Engine, self.id, self.name))

    # TODO: Public Interfaces
    """
    - submit(job): Called by hardwares for enqueueing jobs to the job_queue
        - if self.stage is SimStage.LAYOUT:
            - self._instant_exec(job)
        - else:
            - if job is instanceof(claim) or job is instance of(release):
                - self._instant_exec(job)
            - else:
                - Add the job to self.job_queue

    - run(): Called by Simulator, runs the simulation
        - _compile()
        - _layout()
        - _runtime()
        - _cleanup()
    - debug(<caller>, <msg>): Leaves debugging message
    """

    # TODO: Private Interfaces
    """

    - _complile(): Calls scheduler's compiler function
    - _layout(): Calles scheduler's layout function
    - _runtime(): Runs the simulation main loop
        - while True:
            # Finish a job
            - job = self._forward()
            - if job is ComputeJob and "LAST_NODE" in job.node.args:
                - break

            # Update job progresses
            - for job in self.job_running:
                - job.work_done += self.time_elapsed * job.work_rate

            # Handle system mutations from job end
            - self._end_exec(job)

            # Let scheduler make decisions (adding jobs to the job_queue)
            - self.sched.runtime(job)  # Provide just retired job for scheduling decisions

            - for job_w in self.job_queue:
                # Check if this pending job is runnable
                - if not _assert_exec(job_w):
                    - if len(self.job_running) == 0:
                        - raise DEADLOCK_ERROR
                    - else:
                        - break

                # Handle system mutations for the very start of the job
                - self._begin_exec(job_w)

            # Update & Gather roofline performance of each hardware
            # Assign bandwidth to ongoing transfers using water fill algorithm
            - for job in self.job_running:
                - Update job.work_rate based on the right above
                - Update job.timestamp_ETA = self.timestamp_now + (work_left / job.work_rate)

    - _assert_exec(<job>): Checks if job is runnable
    - _begin_exec(<job>)
        - Handle system mutations from the job start
        - Handle event logging
    - _end_exec(<job>)
        - Handle system mutations from the job completion
        - Handle event logging
    - _instant_exec(<job>):
        - Handle system mutations from the job start
        - Handle system mutations from the job completion
        - Handle event logging

    - _forward(): Sorts job_running, pops a job with earliest ETA
    - _cleanup(): Write a report and log it, stop the logger
    """
