from sim.hw.compute import *


def LOAD_COMPUTE_CLASS(compute_type: str):
    compute: BaseCompute = globals()[compute_type]
    return compute
