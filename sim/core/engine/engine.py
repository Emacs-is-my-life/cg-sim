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
        super().__init__(obj_id, name, log)

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
        self.job_waiting: list[BaseJob] = []
        self.job_running: list[BaseJob] = []

        # Create subtracks for logging
        self.log.record(Log.subtrack(TrackID.Engine, self.id, self.name))

    """
    # TODO: Public Interfaces

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


    """
    # TODO: Private Interfaces

    - _complile(): Calls scheduler's compiler function
    - _layout(): Calles scheduler's layout function
    - _runtime(): Runs the simulation main loop
        - while True:
            # Advance simulation
            - retired_jobs = self._forward()   # retired_jobs is an array of jobs
            - for job in retired_jobs:
                - if job is ComputeJob and "LAST_NODE" in job.node.args:
                    - break

                # Handle job retirement
                - self._end_exec(job)

            # Update work progress for remaining jobs
            - for job in self.job_running:
                - job.work_done += self.time_elapsed * job.work_rate

            # Let scheduler make decisions (adding jobs to the job_queue)
            - self.sched.runtime(retired_job)   # Provide retired_job for scheduling decisions.
                                                # Scheduler already has read access to system(trace, hardware states)

            # Drain all runnable jobs from job_waiting to job_running, in FIFO manner
            - for job_w in self.job_waiting:
                # Check if this pending job is runnable
                - if not _assert_exec(job_w):
                    - if len(self.job_running) == 0:
                        # If the head job in job_waiting queue is not runnable,
                        # and there is no jobs(which can mutate system state) in job_running,
                        # then simulation will stay stuck forever.
                        - raise DEADLOCK_ERROR
                    - else:
                        - break

                # For job_w is runnable, run it
                - self._begin_exec(job_w)
                - self.job_running.append(job_w)

            # Update & Gather roofline performance of each hardware
            # Assign bandwidth to ongoing transfers using water fill algorithm
            - for job in self.job_running:
                - Update job.work_rate based on the right above
                - Update job.timestamp_ETA = self.timestamp_now + (work_left / job.work_rate)

    - _assert_exec(<job>): Checks if job is runnable      (job.assert_exec())
    - _duration_exec_begin(<job>)
        - Handle system mutations from the job start      (job.mut_begin())
        - Handle event logging                            (job.log_begin())
    - _deuration_exec_end(<job>)
        - Handle system mutations from the job completion (job.mut_end())
        - Handle event logging                            (job.log_end())
    - _instant_exec(<job>):
        - Handle system mutations from the job start      (job.mut_begin())
        - Handle system mutations from the job completion (job.mut_end())
        - Handle event logging                            (job.log_instant())

    - _forward(): Sorts job_running, pops a job with earliest ETA
    - _cleanup(): Write a report and log it, stop the logger
    """
