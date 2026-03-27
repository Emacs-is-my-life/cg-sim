from sim.sched import *


def LOAD_SCHEDULER_CLASS(scheduler_type: str):
    if not (scheduler_type in globals()):
        raise Exception(f"[init] Scheduler of type: {scheduler_type} does not exist.")

    scheduler = globals()[scheduler_type]
    return scheduler
