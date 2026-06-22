"""Parametric-knapsack weight/KV residency scheduler (Path 1).

MILP-free macro residency selection: choose the resident (cold) set by
benefit-density ranking under a VRAM budget. Monotone in budget by
construction (continuous-knapsack exchange argument), so the budget→residency
curve is a nested family with a small number of breakpoints — no per-budget
re-solve. See DESIGN.md.
"""
from graph_modifiers.schedulers.ct_knapsack.scheduler import (  # noqa: F401
    solve_neutral,
    parametric_curve,
    print_summary,
)
