from sim.load import *
from sim.core.trace import TraceLoader


def LOAD_TRACE_CLASS(loader_type: str, args: dict[str, Any]) -> TraceLoader:
    trace_loader_class = globals()[loader_type]
    return trace_loader_class(args)
