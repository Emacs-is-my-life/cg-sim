from sim.hw.compute import *


def LOAD_COMPUTE_CLASS(compute_type: str):
    if not (compute_type in globals()):
        raise Exception(f"[init] Compute Hardware of type: {compute_type} does not exist.")

    compute = globals()[compute_type]
    return compute
