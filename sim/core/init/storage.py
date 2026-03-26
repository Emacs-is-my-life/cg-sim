from sim.hw.storage import *


def LOAD_STORAGE_CLASS(storage_type: str):
    storage: BaseStorage = globals()[storage_type]
    return storage
