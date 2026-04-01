from abc import abstractmethod
from typing import Any

from sim.core import SimObject, System
from sim.core.trace import Trace
from sim.core.log import Log
from sim.core.job import BaseJob


class BaseScheduler(SimObject):
    """Base class for schedulers"""

    def __init__(self, obj_id: int, name: str, log: Log, sys: System, args: dict[str, Any] | None = None):
        super().__init__(obj_id, name, log)
        self.sys: System = sys
        self.args: dict[str, Any] = args if args is not None else {}
        return

    @abstractmethod
    def compile(self, trace: Trace) -> None:
        pass

    @abstractmethod
    def layout(self, retired_jobs: list[BaseJob]) -> None:
        pass

    @abstractmethod
    def runtime(self, retired_jobs: list[BaseJob]) -> None:
        pass
