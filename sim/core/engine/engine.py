from typing import Any
import heapq

from sim.core import SimObject
from sim.core.log import Log, TrackID
from sim.core.trace import Trace


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
        self.sched = sched

        # Simulation states
        self.stage = "ASDF"        # Simulation Stage
        self.timestamp: float = 0  # Microseconds

        # TODO: After implementing Job
        # job_queue
        # job_running

        # Create subtracks for logging
        self.log.record(Log.subtrack(TrackID.Engine, self.id, self.name))

    # TODO: Methods to Implement
    """

    - request(<job>)
    - forward()
    """
