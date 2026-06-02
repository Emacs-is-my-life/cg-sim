"""DiffusersGroupOffload — leaf-level group-offload scheduler (diffusers, use_stream).

Diffusers `group_offload` with offload_type='leaf_level', use_stream=True,
non_blocking=True, record_stream=False, low_cpu_mem_usage=False differs from
accelerate `cpu_offload` in exactly ONE execution dimension: H2D onloads run on a
dedicated side CUDA stream and OVERLAP compute (the next leaf is prefetched while
the current leaf computes), instead of accelerate's synchronous single-stream
serial load->compute->load.

That overlap is encoded in the TRACE, not the scheduler. The loader
(`PytorchOffloadLoader`, offload_variant auto-detected "diffusers") translates each
leaf's H2D `Memcpy HtoD (Pinned -> Device)` — recorded on a side stream (e.g.
stream 13/17/21), distinct from the compute stream (7) — into a transfer trigger
whose graph parent is the PREVIOUS side-stream Memcpy (`stream_order`), NOT the
compute kernel. So consecutive H2D transfers serialize among themselves (one side
stream => full pinned bandwidth, ~26.4 GB/s) while the compute kernels form their
own chain; the engine runs a ComputeJob (gpu) and the next TransferJob (memory
bandwidth) concurrently => the real prefetch overlap emerges for free. (Contrast
accelerate, where every node is on stream 7, so each Memcpy's stream_order parent
is the prior compute => strictly serial.)

Eviction is free-on-last-use: diffusers' `offload_()` with a stream is a pointer
swap back to the pre-pinned CPU master (NO device->host copy — verified: <=6 `Memcpy
DtoH` events in the entire trace, vs ~7.4k/20k H2D). The GPU weight is simply
dropped after its leaf computes. The loader's per-epoch `evict_after_node` (last gpu
use of each load) reproduces exactly that, and the recorded per-step re-stream
volume (22.18 GB SDXL / 121.10 GB SD3) by construction.

Everything else is identical to accelerate cpu_offload: the same-tid RAM->VRAM
TransferJob fired per trigger (node DONE on transfer retire), schedule-evict of
masters, refcount-free of intermediates (so VRAM peak tracks the real working set,
which for diffusers is activation-dominated), start-gated readiness, and the
multi-phase layout. So this is a thin subclass of `AccelerateCPUOffload`; see
`sim/sched/accelerate_cpu_offload/accelerate_cpu_offload.py` and
`docs/known_problems.md` for the executor mechanics and engine constraints.
"""

from __future__ import annotations

from sim.core.trace import Node
from sim.hw.common import DataRegionAccess
from sim.sched.accelerate_cpu_offload.accelerate_cpu_offload import AccelerateCPUOffload


class DiffusersGroupOffload(AccelerateCPUOffload):
    """Diffusers leaf-level group-offload. The H2D∥compute overlap and
    free-on-last-use weight eviction are carried by the faithfully-translated trace
    (side-stream stream_order + per-epoch evict), so the executor is inherited from
    `AccelerateCPUOffload`.

    ONE diffusers-specific override: skip pre-claiming ORPHAN dispatcher outputs.
    Diffusers eager traces allocate transient activation buffers via
    `aten::empty(device=cuda)` on the cpu thread (cross-device "dispatcher" outputs
    the base scheduler pre-claims in VRAM). The majority (SDXL: 4135 of 4780) have
    NO consumer in the rewritten graph — the framework writes/reads them through
    aliased views / untraced pointers, and the genuine gpu-produced activation that
    overwrites the same physical bytes is tracked separately. Pre-claiming AND
    holding these orphans double-counts VRAM and leaks (no consumer => the base
    refcount-free never fires): +33% peak, growing across the 4 steps. GROUND TRUTH:
    the kineto trace's own running `Total Allocated` peaks at ~129 MiB for the SDXL
    traced inference (NOT the 1513 MiB whole-process high-water, which is dominated
    by the out-of-scope VAE decode + warmup), matching the genuine-activation
    working set (~125 MiB) — the orphan buffers do NOT contribute to it. So skip
    pre-claiming orphan dispatcher outputs; still pre-claim the genuinely-consumed
    ones (cross-device values a later node reads, which refcount-free normally).
    """

    def _preclaim_dispatcher_outputs(self, node: Node) -> None:
        tm = self.sys.trace.tensor_map
        for tid in node.args.get("dispatcher_outputs") or []:
            # Orphan transient buffer: its bytes are already accounted by the genuine
            # activation that overwrites the same storage; holding it double-counts.
            if self._remaining_consumers.get(tid, 0) == 0:
                continue
            t = tm.get(tid)
            if t is None:
                continue
            home = self._home_memory(t)
            existing = home.space.get_by_tensor_id(tid)
            region = next((r for r in existing
                           if r.access_status == DataRegionAccess.IDLE), None)
            if region is None:
                if existing and self._release_dups:
                    self._suspect_dups.add(tid)
                region = self._claim(home, t)
                if region is None:
                    continue
            region.is_ready = True
            region.is_latest = True
