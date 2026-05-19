"""Inject a weight-streaming schedule into a cg-sim Trace.

Mirrors PyTorch's ``wrapper.py:_inject_weight_streaming_io`` but operates
on cg-sim's compute-graph trace instead of the inductor-generated Python
wrapper. The result is a Trace augmented with the schedule's prefetch
and evict ops as additional input-tensor dependencies on the original
kernel nodes — DAV's existing transfer logic then triggers RAM↔VRAM
moves at the right launch IDs without any scheduler changes.
"""

from .injector import inject_schedule_into_trace

__all__ = ["inject_schedule_into_trace"]
