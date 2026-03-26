from ABC import ABC, abstractmethod
from typing import Any

from sim.core.trace import Trace
from sim.core.log import Log


class TraceLoader(ABC):
    def __init__(self, args: dict[str, Any]):
        self.args: dict[str, Any] = args
        return

    @abstractmethod
    def load(self) -> Trace:
        """
        Loads a trace from path given in args,
        convert it to Trace format for simulation
        """
        return
