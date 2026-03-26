from sim.hw.compute import *
from sim.hw.compute.common import BaseCompute

def LOAD_COMPUTE_CLASS(compute_type: str, args: dict) -> BaseCompute:
    compute: BaseCompute = globals()[compute_type]
    return compute
