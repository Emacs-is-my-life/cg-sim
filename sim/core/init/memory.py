from sim.hw.memory import *
from sim.hw.memory.common import BaseMemory


def LOAD_MEMORY_CLASS(memory_type: str, args: dict) -> BaseMemory:
    memory: BaseMemory = globals()[memory_type]
    return memory
