from __future__ import annotations

from enum import Enum, auto
from abc import ABC
from typing import TYPE_CHECKING
import uuid
import fastuuid

if TYPE_CHECKING:
    from sim.hw.memory.common.base_memory import BaseMemory
    from sim.hw.storage.common.base_storage import BaseStorage


class DataRegionAccess(Enum):
    """
    Status of memory access to a MemoryRegion.
    There could be only one reader or write to a MemoryRegion.

    - IDLE: no reader or writer right now
    - READ: being read now
    - WRITE: being written now
    """

    IDLE = auto()
    BEING_READ = auto()
    BEING_WRITTEN = auto()


class DataRegion(ABC):
    """
    DataRegion represents a continuous chunk of data,
    where tensor can reside.
    """

    def __init__(self, hw: "BaseMemory | BaseStorage", num_pages: int, tensor_id: int):
        self.id: uuid.UUID = fastuuid.uuid4()
        self.hw: "BaseMemory | BaseStorage" = hw
        self.num_pages: int = num_pages
        self.tensor_id: int = tensor_id    # Data(tensor) stored in this region

        self.is_latest: bool = False       # Is this copy of tensor, up-to-date value?
        self.is_ready: bool = False        # Is value ready to be used? (not unitialized, or in the middle of data transfer)

        self.access_status: DataRegionAccess = DataRegionAccess.IDLE
        self.access_count: int = 0         # How many jobs are accessing this region now?
        return
