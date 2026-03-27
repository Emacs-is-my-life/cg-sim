from sim.hw.storage import *


def LOAD_STORAGE_CLASS(storage_type: str):
    if not (storage_type in globals()):
        raise Exception(f"[init] Storage Hardware of type: {storage_type} does not exist.")

    storage = globals()[storage_type]
    return storage
