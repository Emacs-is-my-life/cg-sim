from enum import Enum, auto
from abc import ABC
import uuid
import fastuuid

from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage


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

    def __init__(self, hw: BaseMemory | BaseStorage, num_pages: int, tensor_id: int):
        self.id: uuid.UUID = fastuuid.uuid4()
        self.hw: BaseMemory | BaseStorage = hw
        self.num_pages: int = num_pages
        self.tensor_id: int = tensor_id    # Data(tensor) stored in this region

        self.is_latest: bool = False       # Is this copy of tensor, up-to-date value?
        self.is_ready: bool = False        # Is value ready to be used? (not in the middle of transfer)

        self.access_status: DataRegionAccess = DataRegionAccess.IDLE
        self.access_count: int = 0         # How many jobs are accessing this region now?
        return
