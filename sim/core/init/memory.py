from sim.hw.memory import *


def LOAD_MEMORY_CLASS(memory_type: str):
    memory: BaseMemory = globals()[memory_type]
    return memory
