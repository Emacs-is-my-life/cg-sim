from abc import ABC, abstractmethod
from typing import Any

from sim.core.log import Log
from sim.hw.storage.common import BaseStorage

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

    @abstractmethod
    def placement(self, trace: Trace, storage: BaseStorage) -> None:
        """
        Do initial placement of tensors.

        Some tensors must be places in the storage unit(ex; weight tensor),
        unlike other tensors(immediates, KV cache, ...) which will be initialized
        as result of computations.

        Those tensors must be places into a storage(at least one) by the trace loader.
        """
        return
