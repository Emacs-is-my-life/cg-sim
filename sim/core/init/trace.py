from sim.load import *


def LOAD_TRACE_CLASS(loader_type: str):
    trace_loader_class = globals()[loader_type]
    return trace_loader_class
