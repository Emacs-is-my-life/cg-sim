from sim.sched import *


def LOAD_SCHEDULER_CLASS(scheduler_type: str):
    scheduler: BaseScheduler = globals()[scheduler_type]
    return scheduler
