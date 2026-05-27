"""Interval-set math + percentile helper.

Used by stall/utilization analyses that need to compute unions and
gaps over `(begin, end)` timestamp pairs.
"""
from __future__ import annotations


def merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Sort and union-merge overlapping intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def union_length(intervals: list[tuple[float, float]]) -> float:
    """Total length covered by the union of `intervals`."""
    return sum(e - s for s, e in merge_intervals(intervals))


def percentile(sorted_values: list[float], p: float) -> float:
    """Linear-interpolation percentile (NumPy-style) on an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    return float(sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f))
