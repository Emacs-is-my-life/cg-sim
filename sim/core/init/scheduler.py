import importlib


def LOAD_SCHEDULER_CLASS(scheduler_type: str):
    # Live lookup so `restart_simulation(reload=True)` can hot-reload
    # user-edited scheduler code: after sys.modules invalidation, the
    # next import_module re-executes `sim/sched/__init__.py` (which
    # walks subpackages via pkgutil) and yields fresh class objects.
    pkg = importlib.import_module("sim.sched")
    scheduler = getattr(pkg, scheduler_type, None)
    if scheduler is None:
        raise Exception(f"[init] Scheduler of type: {scheduler_type} does not exist.")
    return scheduler
