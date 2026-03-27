from sim.load import *


def LOAD_TRACE_CLASS(loader_type: str):
    if not (loader_type in globals()):
        raise Exception(f"[init] Trace loader of type: {loader_type} does not exist.")

    trace_loader_class = globals()[loader_type]
    return trace_loader_class
