from abc import ABC, abstractmethod
from typing import Any


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

    def __init__(self, id: int, name: str):
        """Initialization"""
        self.id: int = id
        self.name: str = name
        return

    # TODO: type update
    @abstractmethod
    def log_state(self) -> None:
        """Log its state according to logging format"""
        pass
