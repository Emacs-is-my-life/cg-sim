from abc import ABC, abstractmethod
from typing import Any

from sim.core.log import Log


class SimObject(ABC):
    """
    Represents an object in the simulation, whose states should be logged.
    Examples:
    - Trace
    - Scheduler
    - Hardwares
      - Compute(CPU, GPU, ...)
      - Memory(RAM, VRAM, SSD, ...)
    """

    def __init__(self, obj_id: int, name: str, log: Log):
        """Initialization"""
        self.id = obj_id
        self.name = name
        self.log = log
        return

    @abstractmethod
    def log_counters(self) -> dict[str, Any] | None:
        """
        Log its counter according to logging format

        Return: counters: dict[str, Any] | None
        """
        pass

    @abstractmethod
    def log_states(self) -> dict[str, Any] | None:
        """
        Log its state according to logging format

        Return: states: dict[str, Any] | None
        """
        pass
