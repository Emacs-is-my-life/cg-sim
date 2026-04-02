from __future__ import annotations

from abc import abstractmethod
from typing import Any, TYPE_CHECKING

from sim.core.sim_object import SimObject
from sim.core.trace import Trace
from sim.core.log import Log
from sim.core.job import BaseJob
from sim.hw.storage.common import BaseStorage

if TYPE_CHECKING:
    from sim.core.system import System


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
    def layout(self, init_storage: BaseStorage) -> None:
        pass

    @abstractmethod
    def runtime(self, retired_jobs: list[BaseJob]) -> None:
        pass

    def log_counters(self) -> dict[str, Any] | None:
        """No counters to log"""
        return None

    def log_states(self) -> dict[str, Any] | None:
        """No states to log"""
        return None
