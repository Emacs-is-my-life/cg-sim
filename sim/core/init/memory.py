from sim.hw.memory import *


def LOAD_MEMORY_CLASS(memory_type: str):
    if not (memory_type in globals()):
        raise Exception(f"[init] Memory Hardware of type: {memory_type} does not exist.")

    memory = globals()[memory_type]
    return memory
