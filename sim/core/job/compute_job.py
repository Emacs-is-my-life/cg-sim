from sim.core.log import Log
from sim.core.trace import Node
from sim.core import System
from sim.hw.compute.common import BaseCompute

from .job import BaseJob

from .assertion.compute_assertion import assertion
from .mutation.compute_mutation import begin_mutation, end_mutation
from .logging.compute_logging import begin_log, end_log


class ComputeJob(BaseJob):
    """
    Computation Job
    """
    def __init__(self, compute_hw: BaseCompute, node: Node):
        work_total = node.compute_time_micros   # AU * microseconds
        super().__init__(work_total)
        self.running_on.append(compute_hw)
        self.node = node
        return

    def is_runnable(self, sys: System) -> bool:
        return assertion(self, sys)

    def begin_mut(self, sys: System) -> None:
        begin_mutation(self, sys)
        return

    def begin_log(self, log: Log, timestamp: float) -> None:
        begin_log(self, log, timestamp)
        return

    def end_mut(self, sys: System) -> None:
        end_mutation(self, sys)
        return

    def end_log(self, log: Log, timestamp: float) -> None:
        end_log(self, log, timestamp)
        return
