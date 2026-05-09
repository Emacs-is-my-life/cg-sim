"""Bandwidth-aware admission scheduler.

A windowed-utilization variant of `jit_sim_prune`: instead of
admitting any job that fits its individual deadline, track an
aggregate H2D bandwidth utilization curve and reject any candidate
that would push some time window above ``bw_target``. Catches the
global over-commit that single-deadline FIFO admission misses.
"""

from .scheduler import BWAwareKnobs, solve_neutral, print_summary

__all__ = ["BWAwareKnobs", "solve_neutral", "print_summary"]
