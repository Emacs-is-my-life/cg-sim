from abc import ABC, abstractmethod
from typing import Any

from sim.core.log import Log

from .trace import Trace


class TraceLoader(ABC):
    def __init__(self, obj_id: int, name: str, log: Log, args: dict[str, Any]):
        self.id = obj_id
        self.name = name
        self.log = log
        self.args: dict[str, Any] = args
        return

    @abstractmethod
    def load(self) -> Trace:
        """
        Loads a trace from path given in args,
        convert it to Trace format for simulation
        """
        return
