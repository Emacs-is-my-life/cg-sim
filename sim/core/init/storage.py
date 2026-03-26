from sim.hw.storage import *
from sim.hw.storage.common import BaseStorage


def LOAD_STORAGE_CLASS(storage_type: str, args: dict) -> BaseStorage:
    storage: BaseStorage = globals()[storage_type]
    return storage
