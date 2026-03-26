from sim.sched import *
from sim.sched.common import BaseScheduler


def LOAD_SCHEDULER_CLASS(scheduler_type: str, args: dict) -> BaseScheduler:
    scheduler: BaseScheduler = globals()[scheduler_type]
    return scheduler
