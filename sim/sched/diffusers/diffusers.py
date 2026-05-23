"""Simulator implementation of diffusers ``enable_group_offload`` (HuggingFace).

Algorithm (from ``docs/offload-schemes/diffusers_group-offload_use-stream-true.md``,
which documents ``apply_group_offloading(..., offload_type='block_level',
num_blocks_per_group=1, use_stream=True, non_blocking=False)`` — the same
combination used in ``tmp/diffusers_group-offload_sdxl-turbo.py``):

  Phase 1 — Attach (modeled at layout time):
    Every WEIGHT tensor is parked in the pageable CPU master copy (RAM).
    Loaded once via SSD->RAM, then never modified. No WEIGHTs are
    pre-loaded into VRAM. Non-WEIGHT INPUT/LEAF tensors are placed on
    their declared device — matches diffusers' view of activations
    (intermediates flow through VRAM during a block's forward and are
    freed by the caching allocator).

  Phase 2 — Per-block (modeled at runtime, per cgsim compute node):
    For every ready GPU compute node:
      - For each WEIGHT input that lacks a ready VRAM copy: claim a
        VRAM region and queue a RAM->VRAM transfer for *every member
        of the same offload group atomically*. Mark the group
        ``LOADING``; the compute is gated until the H2D retires
        (``_LOADED``).
      - For each non-WEIGHT input: standard "ensure resident on
        compute.memory" check.
      - On compute retire: attempt to release the VRAM region of the
        consumed weight's group (see "Per-retire eviction" below).
        The RAM master copy is *never* touched.

Group construction (spec §"offload_type=block_level"):
  Each top-level component (``unet``, ``vae``, ``text_encoder``, ...)
  is treated as an independent root. Within a component, only its
  *direct-child* ``ModuleList``/``Sequential`` becomes a matched
  container: each element of that list is one offload group
  (``num_blocks_per_group=1`` is forced by real diffusers whenever
  ``use_stream=True`` — spec line 81-83). Everything else in the
  component (non-list direct children plus nested ModuleLists invisible
  to the block-level scanner) collapses into a single "unmatched"
  group that lives loaded for the whole component's forward.

  For SDXL-Turbo this gives exactly 10 groups (spec §"Concrete count",
  table line 149-157):
    * 1 per fully-unmatched component: vae, text_encoder, text_encoder_2,
      and the unet "everything else" lump (mid_block + conv_in/out +
      time/add embeds + final norm/act).
    * 6 from the two matched ModuleLists under unet: down_blocks[0..2]
      and up_blocks[0..2].

  The trace's bundle CSV doesn't carry an ``isinstance(., ModuleList)``
  flag, so cgsim auto-detects matched lists by structural inference:
  any (component, direct-child) pair whose grandchildren paths are all
  integer-indexed is treated as a ModuleList. This is exactly what real
  diffusers' ``named_children() → isinstance(ModuleList/Sequential)``
  walk discovers on disk.

Per-retire eviction (modeling diffusers' post-forward eviction):
  Every WEIGHT-consumer retire — not just the DAG-final consumer —
  fires ``_release_group_vram`` for the consumed weight's offload
  group. A group whose members are referenced by K module invocations
  cycles H2D K times across the run, matching real ``group_offload``
  semantics where each block's post-forward eviction returns its
  tensors to ``offload_device``. ``_group_state[gid]`` flips back to
  ``_ABSENT`` and the next consumer in the next invocation re-triggers
  ``_stage_group``.

  Eviction is *gated* by ``_group_has_pending_consumer(gid)``: the
  scheduler scans ``ready_node_ids``, ``engine.job_waiting``, and
  ``engine.job_running`` for any node that still lists a member of
  the group in its ``input_tensors``. If any such consumer is queued
  or executing, the release is skipped (a later retire retries).
  This is what prevents freeing a weight whose H2D just landed for a
  GEMM already in ``job_waiting`` — the same correctness gate the
  Accelerate scheduler uses.

Knobs:
  - ``offload_type`` (default ``"block_level"``): ``"leaf_level"`` is
    not yet implemented; raises NotImplementedError. Block-level is
    the script-default for SDXL-Turbo.
  - ``use_stream`` (default True): in real diffusers this gates the
    secondary-stream prefetch and forces ``num_blocks_per_group=1``.
    cgsim's memory subsystem is always a separate hardware lane from
    the compute unit (transfers and compute always overlap when
    independent), so ``use_stream=True`` is the natural model and
    ``=False`` would only matter for fidelity around the false
    serialization the spec describes — also raises NotImplementedError
    until needed.
  - ``num_blocks_per_group`` (default 1): only honored under
    ``use_stream=True`` is forced-to-1 in real diffusers; we forbid
    other values to match.
  - ``components``: optional explicit list of top-level component
    names (``["unet", "vae", "text_encoder", "text_encoder_2"]``). If
    omitted, every distinct first dot-segment of any annotated
    ``module_path`` in the trace becomes a component.
  - ``block_modules``: optional dict ``{component: [direct-child
    names ...]}`` to force matching of non-ModuleList children (spec
    §"How to override the default matching"). Unused for SDXL-Turbo;
    pipeline-level entry doesn't forward it per spec line 139-143.

Notes vs the spec:
  - cg-sim already runs transfers on the memory subsystem in parallel
    with GPU compute on the compute unit, so the prefetch overlap
    ``use_stream=True`` is supposed to provide is *free* in the
    simulator's hardware model — no separate prefetch hook to wire
    up. Peak weight residency therefore tracks "currently-running
    block + the block being prefetched" — i.e. up to two block groups
    at once, which is the spec-correct ceiling under stream prefetch.
  - Speculative H2D prefetch: same as in Accelerate, a CPU dispatcher
    or CPU-compute that references a cuda-WEIGHT input kicks the
    RAM->VRAM load alongside its own work so downstream GPU kernels
    find the weight in flight. For diffusers this is exactly the
    "lazy prefetching hook" attached to the root module by
    ``apply_group_offloading`` (spec line 119) — same observable
    behavior, same wasted-bandwidth corner case when a CPU-only
    ``aten::view`` on a cuda-WEIGHT triggers a prefetch with no GPU
    consumer to follow.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any, TYPE_CHECKING

from sim.core.job import BaseJob, ComputeJob, TransferJob
from sim.core.log import Log
from sim.core.trace import Node, NodeHW, NodeStatus, Tensor, TerminalNode, Trace
from sim.hw.common import DataRegion, DataRegionAccess
from sim.hw.compute.common import BaseCompute
from sim.hw.memory.common import BaseMemory
from sim.hw.storage.common import BaseStorage
from sim.sched.common import BaseScheduler

if TYPE_CHECKING:
    from sim.core.system import System


# Per-WEIGHT residency states. CONTEXT / INPUT / LEAF tensors track
# residency via the cg-sim DataRegion's own (is_ready, is_latest,
# access_status) flags, so they don't need a parallel state machine.
_ABSENT = "ABSENT"    # WEIGHT lives in RAM only
_LOADING = "LOADING"  # RAM->VRAM transfer in flight
_LOADED = "LOADED"    # VRAM region ready


class DiffusersGroupOffload(BaseScheduler):
    """Implements diffusers ``apply_group_offloading`` (block-level, use_stream)."""

    # Non-WEIGHT tensors whose region must never be released.
    # WEIGHTs are intentionally excluded — they are evicted from VRAM
    # after each block's last consumer retires, while the RAM master
    # copy stays for the whole run.
    _PERMANENT_TYPES = frozenset({"INPUT", "LEAF"})

    # Suffix appended to a component name to denote its single
    # "unmatched lump" group (everything in the component that isn't a
    # direct-child ModuleList element).
    _UNMATCHED_SUFFIX = ":unmatched"

    # Page size used by every memory in cg-sim (see base_memory.py).
    _PAGE_SIZE_KB: int = 4

    def __init__(
        self,
        obj_id: int,
        name: str,
        log: Log,
        sys: System,
        args: dict[str, Any] | None = None,
    ):
        super().__init__(obj_id, name, log, sys, args)

        # ---------------- Topology discovery ----------------
        self.compute_by_name: dict[str, BaseCompute] = {}
        self.memory_by_name: dict[str, BaseMemory] = {}
        self.storage: BaseStorage | None = None

        for hw in sys.hw.values():
            if isinstance(hw, BaseCompute):
                self.compute_by_name[hw.name] = hw
            elif isinstance(hw, BaseMemory):
                self.memory_by_name[hw.name] = hw
            elif isinstance(hw, BaseStorage) and self.storage is None:
                self.storage = hw

        cpu_compute_name = self.args.get("cpu_compute", "cpu")
        cuda_compute_name = self.args.get("cuda_compute", "gpu0")
        self.cuda_device = str(self.args.get("cuda_device", "cuda:0")).lower()

        if cpu_compute_name not in self.compute_by_name:
            raise Exception(f"[DiffusersGroupOffload] CPU compute '{cpu_compute_name}' does not exist.")
        if cuda_compute_name not in self.compute_by_name:
            raise Exception(f"[DiffusersGroupOffload] CUDA compute '{cuda_compute_name}' does not exist.")

        self.cpu_compute: BaseCompute = self.compute_by_name[cpu_compute_name]
        self.cuda_compute: BaseCompute = self.compute_by_name[cuda_compute_name]
        self._ram: BaseMemory = self.cpu_compute.memory
        self._vram: BaseMemory = self.cuda_compute.memory

        # ---------------- Algorithmic knobs ----------------
        # offload_type: real diffusers supports "block_level" and
        # "leaf_level". The script-default for SDXL-Turbo is block_level
        # and that's the only path implemented; leaf_level would
        # require per-leaf grouping mirroring Accelerate's defaults.
        self.offload_type: str = str(self.args.get("offload_type", "block_level"))
        if self.offload_type != "block_level":
            raise NotImplementedError(
                f"[DiffusersGroupOffload] offload_type='{self.offload_type}' is not yet supported; "
                "only 'block_level' is implemented. Use the Accelerate scheduler "
                "with a per-leaf default for leaf-level semantics."
            )

        # use_stream: in real diffusers, True attaches a secondary
        # stream + lazy prefetch hook and forces num_blocks_per_group=1.
        # cgsim's memory subsystem is *always* a separate hardware lane
        # from compute, so transfers naturally overlap; we encode the
        # forcing rule here and refuse use_stream=False to avoid
        # silently modeling something other than the script's setting.
        self.use_stream: bool = bool(self.args.get("use_stream", True))
        if not self.use_stream:
            raise NotImplementedError(
                "[DiffusersGroupOffload] use_stream=False is not yet supported; "
                "the simulator's memory/compute parallelism already provides "
                "the stream overlap, so the False path would require explicit "
                "serialization that hasn't been modeled."
            )

        # num_blocks_per_group: real diffusers forces this to 1 whenever
        # use_stream=True (spec line 81-83). Refuse any other value.
        raw_nbpg = self.args.get("num_blocks_per_group", 1)
        try:
            nbpg = int(raw_nbpg)
        except (TypeError, ValueError):
            raise Exception(
                f"[DiffusersGroupOffload] num_blocks_per_group must be an int; got {raw_nbpg!r}."
            )
        if nbpg != 1:
            raise NotImplementedError(
                f"[DiffusersGroupOffload] num_blocks_per_group={nbpg} is incompatible with "
                "use_stream=True (real diffusers forces this to 1). Set to 1."
            )
        self.num_blocks_per_group: int = 1

        # components: optional explicit list of top-level component
        # names (the diffusers pipeline calls apply_group_offloading on
        # each component separately). If omitted, auto-discover from
        # the trace's module_path annotations.
        raw_components = self.args.get("components")
        if raw_components is None:
            self.components: list[str] = self._discover_components()
        else:
            if not isinstance(raw_components, (list, tuple)):
                raise Exception(
                    f"[DiffusersGroupOffload] components must be a list of strings; "
                    f"got {type(raw_components).__name__}."
                )
            self.components = [str(c) for c in raw_components]

        # invocation_gap_ns: temporal-cluster threshold for detecting
        # per-block invocation boundaries. Real diffusers' eviction
        # fires per ``module.forward()`` call, so cgsim needs to know
        # which consumers belong to which invocation. Consumer node
        # ``start_ns`` values cluster naturally: within one invocation
        # they're ~µs apart; between invocations (e.g. consecutive
        # ``num_inference_steps`` iterations of UNet) there's a gap of
        # tens to hundreds of ms while the rest of the pipeline runs.
        # Default 50 ms — large enough to swallow intra-block idle
        # spikes on the SDXL-Turbo single-step trace (sub-modules can
        # straddle 30+ ms gaps) while still cleanly separating
        # consecutive UNet inference steps (~200 ms apart at SDXL
        # speeds). Override per-trace if needed.
        self.invocation_gap_ns: int = int(self.args.get("invocation_gap_ns", 50_000_000))

        # block_modules: optional dict {component: [direct-child names]}
        # to force matching of non-ModuleList children. Mirrors real
        # diffusers' block_modules kwarg on apply_group_offloading. The
        # pipeline-level enable_group_offload doesn't forward this, so
        # it's empty for the standard SDXL-Turbo script.
        raw_block_mods = self.args.get("block_modules") or {}
        if not isinstance(raw_block_mods, dict):
            raise Exception(
                f"[DiffusersGroupOffload] block_modules must be a dict of "
                f"component -> [direct-child names]; got {type(raw_block_mods).__name__}."
            )
        self.block_modules_override: dict[str, list[str]] = {
            str(k): [str(v) for v in vs] for k, vs in raw_block_mods.items()
        }

        # ---------------- Trace bookkeeping ----------------
        self.pending_parent_count: dict[int, int] = {
            n.id: len(n.parent_nodes) for n in self.sys.trace.node_map.values()
        }
        self.ready_node_ids: deque[int] = deque(
            n.id for n in self.sys.trace.node_map.values()
            if self.pending_parent_count[n.id] == 0
        )

        # Per-tensor remaining-consumer counter (over the whole DAG).
        # WEIGHT VRAM regions are evicted when their *group's* total
        # counter hits zero; CONTEXT tensor regions are released across
        # all memories when this hits zero (unless permanent).
        self._remaining_consumers: dict[int, int] = {}
        for node in self.sys.trace.node_map.values():
            for tid in node.input_tensors:
                self._remaining_consumers[tid] = self._remaining_consumers.get(tid, 0) + 1

        # ---------------- Offload-group construction ----------------
        # tid -> group_id (string). For block_level + use_stream:
        #   * Each ModuleList element under a matched container becomes
        #     a group named "<component>.<list>.<index>".
        #   * Everything else in a component lumps into a single group
        #     named "<component>:unmatched".
        self._tensor_group: dict[int, str] = {}
        self._group_tids: dict[str, list[int]] = {}
        # (component, direct-child) pairs detected as ModuleLists. Used
        # for debuggability — see _build_offload_groups for the rule.
        self._matched_lists: set[tuple[str, str]] = set()
        self._build_offload_groups()

        # Per-group residency state (cuda-WEIGHT only; cpu-device
        # WEIGHTs stay in RAM forever). Eviction is driven by per-
        # invocation consumer counts (see _build_invocation_accounting),
        # not by DAG-wide consumer totals.
        self._group_state: dict[str, str] = {gid: _ABSENT for gid in self._group_tids}

        # Per-invocation consumer accounting. Populated by
        # _build_invocation_accounting after groups are constructed.
        # See that method for the temporal-clustering design.
        self._node_invocation_for_group: dict[int, dict[str, int]] = {}
        self._group_invocation_pending: dict[str, dict[int, int]] = {}
        self._group_active_invocation: dict[str, int] = {}
        self._build_invocation_accounting()

        # TransferJob.id -> list of WEIGHT tensor ids it is loading.
        self._inflight_load_tids: dict[Any, list[int]] = {}

        # Releases deferred because regions were BEING_READ at the time
        # we tried to release them; retried each tick. Keyed by group_id.
        self._pending_vram_group_releases: set[str] = set()
        self._pending_releases: set[int] = set()

        # Multi-phase layout: 0=claim, 1=ssd->ram, 2=ssd->vram, 3=done.
        self._layout_phase: int = 0

        # Outputs of dispatcher / alias nodes that the loader hid in
        # ``node.args["dispatcher_outputs"]``. Pre-claim on the home
        # memory before submission so downstream consumers find them.
        # Same as DAV — the loader-driven contract.
        self._tick_pending_dest_ids: set[int] = set()

        # ---------------- Explicit spec-state accumulators ----------------
        # cpu_state_dict_bytes / gpu_persistent_buffer_bytes are set
        # once at end of layout. _gpu_resident_bytes_cache is kept in
        # sync incrementally — see _on_weight_claimed/_released and
        # gpu_resident_bytes for the rationale (per-tick rescan over
        # all tensors was 87% of wall time on SDXL-Turbo).
        self.cpu_state_dict_bytes: int = 0
        self.gpu_persistent_buffer_bytes: int = 0
        self.peak_gpu_resident_bytes: int = 0
        self._gpu_resident_bytes_cache: int = 0
        return

    # ============================================================ groups
    def _discover_components(self) -> list[str]:
        """Auto-detect top-level components by collecting the first
        dot-segment of every annotated ``module_path``. Returns sorted
        for determinism."""
        components: set[str] = set()
        for node in self.sys.trace.node_map.values():
            mod = node.args.get("module")
            if not mod:
                continue
            path = mod.get("module_path")
            if not path:
                continue
            components.add(path.split(".", 1)[0])
        return sorted(components)

    def _detect_matched_lists(self) -> set[tuple[str, str]]:
        """For each (component, direct-child) pair seen in the trace,
        check whether the grandchildren are integer-indexed — the
        structural signature of a ``ModuleList``/``Sequential``.
        Returns the set of (component, child) tuples that qualify.

        Mirrors real diffusers' ``named_children() →
        isinstance(ModuleList | Sequential)`` walk: only direct children
        of the component root count (spec §"offload_type=block_level",
        "direct children only, not recursive"). Augmented by any
        ``block_modules_override`` the user passes through YAML.
        """
        by_comp_child: dict[tuple[str, str], set[str]] = defaultdict(set)
        for node in self.sys.trace.node_map.values():
            mod = node.args.get("module")
            if not mod:
                continue
            path = mod.get("module_path")
            if not path:
                continue
            parts = path.split(".")
            if len(parts) < 3:
                continue
            comp, child, grand = parts[0], parts[1], parts[2]
            if comp not in self.components:
                continue
            by_comp_child[(comp, child)].add(grand)

        matched: set[tuple[str, str]] = set()
        for (comp, child), grandchildren in by_comp_child.items():
            # A ModuleList's children are all integer-indexed. A mixed
            # set (some int, some named) wouldn't be a ModuleList in
            # PyTorch; rule it out.
            if grandchildren and all(g.isdigit() for g in grandchildren):
                matched.add((comp, child))

        for comp, children in self.block_modules_override.items():
            for child in children:
                matched.add((comp, child))
        return matched

    def _group_for_path(self, path: str) -> str:
        """Map a module path to its block_level offload group.

        Rule:
          * len(parts) ≥ 3 and (parts[0], parts[1]) is a matched
            ModuleList and parts[2] is the integer block index →
            ``"<comp>.<list>.<idx>"``.
          * Anything else (path under a component but not under a
            matched list) → ``"<comp>:unmatched"``.
        """
        parts = path.split(".")
        comp = parts[0]
        if (
            len(parts) >= 3
            and (comp, parts[1]) in self._matched_lists
            and parts[2].isdigit()
        ):
            return f"{comp}.{parts[1]}.{parts[2]}"
        return f"{comp}{self._UNMATCHED_SUFFIX}"

    def _build_effective_module_paths(self) -> dict[int, str]:
        """Map every node to its *effective* module_path.

        A node's effective path is its own ``args["module"]["module_path"]``
        when annotated; otherwise the path of the CPU dispatcher that
        launched it. This matches real diffusers' pre/post-forward-hook
        scope: a GPU kernel emitted during ``module.forward()`` has its
        ``cudaLaunchKernel`` CPU dispatcher as its submit-edge parent,
        and that dispatcher carries the module's ``module_path``.

        Climb strategy, in priority order:
          1. **Submit-edge parent** (matching ``correlation_id``):
             every GPU kernel emitted by ``cudaLaunchKernel`` shares a
             ``correlation_id`` with the CPU launch row in the PyTorch
             profile. Among ``parent_nodes``, we pick the one whose
             ``args['correlation_id']`` equals this node's — that's the
             true control-flow parent (vs. data-flow parents). Walk
             this chain until we hit an annotated ancestor.
          2. **Fallback BFS upstream** through any parent kind. Used
             for unannotated CPU helpers (e.g.
             ``aten::_has_compatible_shallow_copy_type``) that don't
             share a ``correlation_id`` with their launchers; their
             nearest annotated thread-order ancestor is the right
             owner.

        Why not BFS-all-parents from the start: the trace bundle's
        ``add_temporal_data_control_edges`` adds parent links from data
        producers to consumers, which crosses component boundaries
        (e.g. UNet's last output feeds VAE's first op, so a
        VAE-internal kernel's parent set includes UNet ancestors).
        Generic BFS happily walks into UNet and mislabels VAE weights.
        The correlation_id-first walk stays inside the current module's
        forward scope.
        """
        trace = self.sys.trace
        eff: dict[int, str] = {}
        for nid, node in trace.node_map.items():
            mod = node.args.get("module")
            if mod and mod.get("module_path"):
                eff[nid] = mod["module_path"]

        def _submit_parent(nid: int) -> int | None:
            node = trace.node_map[nid]
            corr = node.args.get("correlation_id")
            if not corr:
                return None
            for pid in node.parent_nodes:
                p_corr = trace.node_map[pid].args.get("correlation_id")
                if p_corr == corr:
                    return pid
            return None

        for nid in list(trace.node_map):
            if nid in eff:
                continue
            # (1) Correlation-id submit-edge chain.
            cur: int = nid
            steps = 0
            found_path: str | None = None
            while steps < 16:
                sp = _submit_parent(cur)
                if sp is None:
                    break
                if sp in eff:
                    found_path = eff[sp]
                    break
                cur = sp
                steps += 1
            if found_path is not None:
                eff[nid] = found_path
                continue

            # (2) Fallback: BFS through any parent kind.
            visited = {nid}
            queue: deque[int] = deque(trace.node_map[nid].parent_nodes)
            for pid in trace.node_map[nid].parent_nodes:
                visited.add(pid)
            while queue:
                cur = queue.popleft()
                if cur in eff:
                    found_path = eff[cur]
                    break
                for pid in trace.node_map[cur].parent_nodes:
                    if pid in visited:
                        continue
                    visited.add(pid)
                    queue.append(pid)
            if found_path is not None:
                eff[nid] = found_path
        return eff

    def _build_offload_groups(self) -> None:
        """Assign every cuda-device WEIGHT to a block_level group.

        Four passes:
          1. Detect matched ModuleLists (depends on self.components).
          2. Compute every node's *effective* module_path via correlation-
             id submit-edge climbing (see ``_build_effective_module_paths``).
          3. Collect consumer paths per cuda-WEIGHT, separately tracking
             GPU consumers (the kernels that actually read the weight
             data) and CPU consumers (metadata-only ops like
             ``aten::as_strided`` that don't touch the bytes).
          4. Decide the owning module: GPU-consumer paths win if any
             exist; otherwise fall back to CPU-consumer paths. Within
             the chosen set, pick the most-specific group via majority
             vote across paths, breaking ties by depth and "matched
             beats unmatched". WEIGHTs with no reachable consumers
             become ``"orphan:<tid>"`` singletons.

        Why prefer GPU consumers: PyTorch profiler tags storage-aliased
        tensors with whichever module's ``as_strided``/``view`` happened
        to alias into the same allocator slot first. That alias can
        live in a totally different component than the real owning
        Linear/Conv whose GEMM does the GPU read. The GPU consumer's
        annotation is the only signal that survives storage aliasing —
        all the metadata aliases agree about the slot, only one of them
        agrees about the data.
        """
        trace = self.sys.trace
        weight_tids = [
            tid for tid, t in trace.tensor_map.items()
            if t.args.get("tensor_type") == "WEIGHT"
            and str(t.args.get("device", "")).lower().startswith("cuda")
        ]

        # Pass 1.
        self._matched_lists = self._detect_matched_lists()

        # Pass 2.
        effective_path = self._build_effective_module_paths()

        # Pass 3.
        tid_gpu_paths: dict[int, list[str]] = defaultdict(list)
        tid_cpu_paths: dict[int, list[str]] = defaultdict(list)
        for nid, node in trace.node_map.items():
            path = effective_path.get(nid)
            if not path:
                continue
            dev = (node.args.get("device_type") or "").upper()
            bucket = tid_gpu_paths if dev == "CUDA" else tid_cpu_paths
            for tid in node.input_tensors:
                t = trace.tensor_map.get(tid)
                if t is None or t.args.get("tensor_type") != "WEIGHT":
                    continue
                bucket[tid].append(path)

        # Pass 4.
        for tid in weight_tids:
            paths = tid_gpu_paths.get(tid) or tid_cpu_paths.get(tid) or []
            if not paths:
                gid = f"orphan:{tid}"
            else:
                # Score by (vote_count, is_matched, path_depth). Vote
                # count wins first, so a true majority-component path
                # beats a single misrouted alias.
                counts: Counter[str] = Counter(paths)
                best_gid: str | None = None
                best_score: tuple[int, int, int] = (-1, -1, -1)
                for p, n in counts.items():
                    g = self._group_for_path(p)
                    is_matched = 1 if not g.endswith(self._UNMATCHED_SUFFIX) else 0
                    score = (n, is_matched, p.count("."))
                    if score > best_score:
                        best_score = score
                        best_gid = g
                gid = best_gid or f"orphan:{tid}"
            self._tensor_group[tid] = gid
            self._group_tids.setdefault(gid, []).append(tid)
        return

    # ============================================================ invocations
    def _build_invocation_accounting(self) -> None:
        """Detect per-group invocation boundaries via temporal
        clustering on consumer ``start_ns``, then build per-(group,
        invocation) consumer counts so eviction can wait for the
        *current* invocation to drain rather than thrashing the group
        on every consumer retire.

        Real diffusers fires post-forward eviction once per module
        invocation; for SDXL-Turbo with ``num_inference_steps=4`` that's
        4 evict cycles per UNet block, not 4 × N where N is the number
        of ops inside the block. Without per-invocation accounting,
        eager-evict-on-every-retire over-evicts by ~N× (we measured
        ~580× extra H2D bytes on SDXL-Turbo before this).

        Algorithm:
          1. For each group, gather the trace ``start_ns`` of every
             node that consumes any member tid.
          2. Sort, then split at gaps > ``invocation_gap_ns``. Each
             cluster = one invocation.
          3. Assign each consumer node ``invocation_index`` per group.
          4. Tally per-(gid, inv_idx) consumer counts as the eviction
             gate.

        Why not structural CPU module_path enter/exit instead of
        timestamps: the PyTorch profiler's CPU module annotations can
        bleed across threads — text_encoder_2 ops keep their old
        ``module_path`` tag when execution context shifts back to
        UNet, producing spurious enter/leave flips on every other
        CPU op. Timestamp clustering is robust to that noise as long
        as the gap threshold is larger than any intra-block stall.

        Edge case: a consumer node may belong to multiple groups (one
        op reading two weights from two ModuleLists). Each (gid,
        inv_idx) pair is tracked independently.
        """
        trace = self.sys.trace

        # tid -> gid lookup is already in self._tensor_group. Build
        # the reverse: gid -> (start_ns, consumer_node_id), one entry
        # per (node, gid) pair (deduped so a node touching two
        # members of the same group counts once).
        gid_consumers: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for nid, node in trace.node_map.items():
            seen_groups: set[str] = set()
            for tid in node.input_tensors:
                gid = self._tensor_group.get(tid)
                if gid is None or gid in seen_groups:
                    continue
                seen_groups.add(gid)
                start_ns = int(node.args.get("start_ns") or 0)
                gid_consumers[gid].append((start_ns, nid))

        for gid, events in gid_consumers.items():
            events.sort()  # by start_ns, then nid
            inv_idx = 0
            prev_ns: int | None = None
            pending: dict[int, int] = defaultdict(int)
            for st, nid in events:
                if prev_ns is not None and (st - prev_ns) > self.invocation_gap_ns:
                    inv_idx += 1
                prev_ns = st
                self._node_invocation_for_group.setdefault(nid, {})[gid] = inv_idx
                pending[inv_idx] += 1
            self._group_invocation_pending[gid] = dict(pending)
        return

    # ============================================================ spec-state accumulators
    @property
    def gpu_resident_bytes(self) -> int:
        """Current WEIGHT bytes resident on VRAM (spec state field).

        Returns the cached counter ``_gpu_resident_bytes_cache``, which
        is kept in sync incrementally via ``_on_weight_claimed`` /
        ``_on_weight_released`` rather than rescanned every tick. A
        full rescan over ``trace.tensor_map`` was 87% of total wall
        time on SDXL-Turbo (15k tensors × O(n) ``get_by_tensor_id``
        per tick) — the incremental form is O(1).
        """
        return self._gpu_resident_bytes_cache

    def _on_weight_claimed(self, tensor: Tensor) -> None:
        """Hook fired whenever a WEIGHT region is claimed on VRAM (via
        ``_claim_region`` from ``_stage_group`` or the singleton
        cpu-WEIGHT path)."""
        size_bytes = tensor.num_pages * self._PAGE_SIZE_KB * 1024
        self._gpu_resident_bytes_cache += size_bytes
        if self._gpu_resident_bytes_cache > self.peak_gpu_resident_bytes:
            self.peak_gpu_resident_bytes = self._gpu_resident_bytes_cache

    def _on_weight_released(self, tensor: Tensor) -> None:
        """Hook fired whenever a WEIGHT region is released from VRAM."""
        size_bytes = tensor.num_pages * self._PAGE_SIZE_KB * 1024
        self._gpu_resident_bytes_cache -= size_bytes

    def _on_group_load_started(self, node: Node, gid: str) -> None:
        """Called from ``_kick_cuda_weight_prefetch`` /
        ``_ensure_weight_loaded`` when a group transitions to LOADING
        on behalf of ``node``. Sets the group's active invocation
        index from the consumer's per-group invocation tag so
        subsequent retires of the same invocation drain the right
        counter."""
        inv = self._node_invocation_for_group.get(node.id, {}).get(gid)
        if inv is not None:
            self._group_active_invocation[gid] = inv

    def _on_weight_consumer_retired(self, node: Node, gid: str) -> None:
        """Decrement the per-invocation pending count for ``gid``;
        attempt eviction iff this drains the *current* invocation.
        Cross-group consumers (one node reading two ModuleLists) call
        this once per gid via the loop in ``_consume_inputs``."""
        inv_map = self._node_invocation_for_group.get(node.id) or {}
        inv = inv_map.get(gid)
        if inv is None:
            # Consumer wasn't picked up by clustering (rare — e.g. a
            # weight whose only consumer is itself the load trigger).
            # Fall back to immediate evict-attempt to preserve the
            # spec post-forward semantics.
            self._release_group_vram(gid)
            return
        bucket = self._group_invocation_pending.get(gid)
        if bucket is None:
            self._release_group_vram(gid)
            return
        remaining = bucket.get(inv, 0) - 1
        if remaining > 0:
            bucket[inv] = remaining
            return
        bucket.pop(inv, None)
        # Only evict if this was the active invocation; otherwise a
        # later invocation is already underway and will manage the
        # next eviction cycle.
        if self._group_active_invocation.get(gid) == inv:
            self._group_active_invocation.pop(gid, None)
        self._release_group_vram(gid)

    def _refresh_peak_gpu_resident(self) -> None:
        """No-op now that ``gpu_resident_bytes`` is incrementally
        maintained. Kept as a call site for clarity at the few places
        in ``runtime``/``_handle_transfer_retires`` where the peak
        might historically have been sampled — the actual peak update
        happens inside ``_on_weight_claimed``."""
        return

    # ============================================================ compile
    def compile(self, trace: Trace) -> None:
        return

    # ============================================================ topology helpers
    def _compute_for_node(self, node: Node) -> BaseCompute:
        # Terminal nodes are synthetic — keep them on CPU.
        if isinstance(node, TerminalNode):
            return self.cpu_compute
        if node.hw == NodeHW.GPU:
            return self.cuda_compute
        return self.cpu_compute

    def _home_memory_for_non_weight(self, tensor: Tensor) -> BaseMemory:
        """Where a non-WEIGHT tensor's home region lives. Driven by the
        tensor's declared device. WEIGHTs always live in RAM and are
        handled separately."""
        device = str(tensor.args.get("device", "cpu")).lower()
        if device.startswith("cuda"):
            return self._vram
        return self._ram

    @staticmethod
    def _region_readable(region: DataRegion) -> bool:
        return (
            region.is_ready
            and region.is_latest
            and region.access_status in (DataRegionAccess.IDLE, DataRegionAccess.BEING_READ)
        )

    def _find_free_page(self, memory: BaseMemory, num_pages: int) -> int | None:
        cursor = 0
        for region in memory.space._regions_by_page_idx_start.values():
            if region.page_idx_start - cursor >= num_pages:
                return cursor
            cursor = max(cursor, region.page_idx_end)
        if memory.space.num_total_pages - cursor >= num_pages:
            return cursor
        return None

    def _claim_region(self, memory: BaseMemory, tensor: Tensor):
        page_idx = self._find_free_page(memory, tensor.num_pages)
        if page_idx is None:
            self.sys.abort({
                "from": self.name,
                "error": "OOM",
                "msg": f"No free pages on {memory.name} for tensor {tensor.id} ({tensor.name}, {tensor.num_pages} pages).",
                "memory": {
                    "name": memory.name,
                    "used_pages": memory.space.num_used_pages,
                    "total_pages": memory.space.num_total_pages,
                },
            })
            return None
        return self.sys.claim(memory, tensor, page_idx)

    # ============================================================ layout
    def layout(self, init_storage: BaseStorage) -> bool:
        """Three-phase layout.

        Phase 0: Claim home regions.
          - Every WEIGHT tensor: RAM region (the CPU master copy).
            No VRAM stamp — that's the whole point of cpu_offload.
          - Every non-WEIGHT INPUT/LEAF tensor: region on its declared
            home memory (RAM or VRAM).

        Phase 1: One batched SSD->RAM transfer covering every WEIGHT
        and every cpu-homed initial tensor.

        Phase 2: One batched SSD->VRAM transfer for every cuda-homed
        non-WEIGHT initial tensor.

        SSD can only run one job at a time, so phases 1 and 2 must be
        separate layout-loop iterations (one transfer drains before
        the next is queued).
        """
        if self._layout_phase == 0:
            for tensor in self.sys.trace.tensor_map.values():
                ttype = tensor.args.get("tensor_type")
                if ttype == "WEIGHT":
                    if self._claim_region(self._ram, tensor) is None:
                        return True  # abort already signalled
                elif ttype in ("INPUT", "LEAF"):
                    home = self._home_memory_for_non_weight(tensor)
                    if self._claim_region(home, tensor) is None:
                        return True
            self._layout_phase = 1
            return False

        if self._layout_phase == 1:
            ssd_to_ram: list[tuple[DataRegion, DataRegion]] = []
            for tensor in self.sys.trace.tensor_map.values():
                ttype = tensor.args.get("tensor_type")
                if ttype not in ("WEIGHT", "INPUT", "LEAF"):
                    continue
                stor_regions = init_storage.space.get_by_tensor_id(tensor.id)
                if not stor_regions:
                    continue
                src = stor_regions[0]
                if ttype == "WEIGHT":
                    dest_regions = self._ram.space.get_by_tensor_id(tensor.id)
                else:
                    home = self._home_memory_for_non_weight(tensor)
                    if home is not self._ram:
                        continue  # SSD->VRAM goes in phase 2
                    dest_regions = home.space.get_by_tensor_id(tensor.id)
                if not dest_regions:
                    continue
                ssd_to_ram.append((src, dest_regions[0]))

            if ssd_to_ram:
                self.sys.transfer(ssd_to_ram)
            self._layout_phase = 2
            return False

        if self._layout_phase == 2:
            ssd_to_vram: list[tuple[DataRegion, DataRegion]] = []
            for tensor in self.sys.trace.tensor_map.values():
                ttype = tensor.args.get("tensor_type")
                if ttype not in ("INPUT", "LEAF"):
                    continue
                home = self._home_memory_for_non_weight(tensor)
                if home is not self._vram:
                    continue
                stor_regions = init_storage.space.get_by_tensor_id(tensor.id)
                if not stor_regions:
                    continue
                dest_regions = self._vram.space.get_by_tensor_id(tensor.id)
                if not dest_regions:
                    continue
                ssd_to_vram.append((stor_regions[0], dest_regions[0]))

            if ssd_to_vram:
                self.sys.transfer(ssd_to_vram)
            self._layout_phase = 3

            # Cache spec-section-8 set-once accumulators now that the
            # attach phase is complete.
            self.cpu_state_dict_bytes = sum(
                (r.page_idx_end - r.page_idx_start)
                for tid, t in self.sys.trace.tensor_map.items()
                if t.args.get("tensor_type") == "WEIGHT"
                for r in self._ram.space.get_by_tensor_id(tid)
            ) * self._PAGE_SIZE_KB * 1024
            self.gpu_persistent_buffer_bytes = sum(
                (r.page_idx_end - r.page_idx_start)
                for tid, t in self.sys.trace.tensor_map.items()
                if t.args.get("tensor_type") in self._PERMANENT_TYPES
                for r in self._vram.space.get_by_tensor_id(tid)
            ) * self._PAGE_SIZE_KB * 1024
            return True

        return True

    # ============================================================ residency
    def _find_latest_region(
        self, tensor_id: int, exclude_hw: BaseMemory | BaseStorage | None = None
    ) -> DataRegion | None:
        # Memory copies preferred over storage.
        for hw in self.sys.hw.values():
            if hw is exclude_hw or not isinstance(hw, BaseMemory):
                continue
            for region in hw.space.get_by_tensor_id(tensor_id):
                if self._region_readable(region):
                    return region
        for hw in self.sys.hw.values():
            if hw is exclude_hw or not isinstance(hw, BaseStorage):
                continue
            for region in hw.space.get_by_tensor_id(tensor_id):
                if self._region_readable(region):
                    return region
        return None

    def _idle_region_on(self, memory: BaseMemory, tensor_id: int) -> DataRegion | None:
        for region in memory.space.get_by_tensor_id(tensor_id):
            if region.access_status == DataRegionAccess.IDLE:
                return region
        return None

    def _pending_dest_ids(self) -> set[int]:
        ids: set[int] = set(self._tick_pending_dest_ids)
        for job in self.sys.engine.job_waiting:
            if isinstance(job, TransferJob):
                for _src, dst in job.batch:
                    ids.add(id(dst))
        for job in self.sys.engine.job_running:
            if isinstance(job, TransferJob):
                for _src, dst in job.batch:
                    ids.add(id(dst))
        return ids

    def _ensure_weight_loaded(
        self,
        tensor: Tensor,
        transfers: list[tuple[DataRegion, DataRegion]],
        loading_tids: list[int],
        node: Node | None = None,
    ) -> bool | None:
        """For a WEIGHT input of a GPU compute node, make sure a ready
        VRAM region exists. If a transfer is already in flight, return
        True (still wait). If we need to queue a new one, append to
        ``transfers`` / ``loading_tids`` and return True. Returns False
        only if we can't make progress (no idle dest etc.) so the node
        is re-queued for next tick.

        ``node`` is the consumer; when present, used to tag the group's
        active invocation so the matching ``_on_weight_consumer_retired``
        knows when this invocation has drained.

        Group-aware: staging operates on the tensor's whole offload
        group at once (under ``preload_module_classes``), so a single
        consumer can trigger an atomic load of every cuda-WEIGHT in
        the matching ancestor's subtree.
        """
        # If we already have a readable VRAM copy, we're done.
        for region in self._vram.space.get_by_tensor_id(tensor.id):
            if self._region_readable(region):
                return True

        gid = self._tensor_group.get(tensor.id)
        if gid is None:
            # cpu-device WEIGHT being consumed on a GPU compute: stage
            # it as a singleton, matching the pre-refactor behavior.
            # (Per spec these should be permanent-on-GPU buffers, but
            # the trace's reality may differ.)
            ram_regions = self._ram.space.get_by_tensor_id(tensor.id)
            if not ram_regions or not ram_regions[0].is_ready:
                return False
            dst = self._claim_region(self._vram, tensor)
            if dst is None:
                return False
            self._on_weight_claimed(tensor)
            transfers.append((ram_regions[0], dst))
            loading_tids.append(tensor.id)
            return True

        state = self._group_state.get(gid, _ABSENT)
        if state == _LOADING:
            return True
        if state == _LOADED:
            # Anomaly: group says LOADED but this member isn't readable
            # (early-return check above already missed). With the
            # all-or-nothing release in _release_group_vram this branch
            # is normally unreachable; resync state to ABSENT and let
            # _stage_group repair the missing members. Other members
            # whose regions are still readable are skipped inside
            # _stage_group.
            self._group_state[gid] = _ABSENT
        if node is not None:
            self._on_group_load_started(node, gid)
        return self._stage_group(gid, transfers, loading_tids)

    def _stage_group(
        self,
        gid: str,
        transfers: list[tuple[DataRegion, DataRegion]],
        loading_tids: list[int],
    ) -> bool:
        """Atomic: either every needs-staging member gets claimed+queued,
        or we touch nothing and return False. Two passes — first verify
        every member's RAM master is ready, then claim VRAM."""
        tensor_map = self.sys.trace.tensor_map
        to_stage: list[tuple[Tensor, DataRegion]] = []
        for member_tid in self._group_tids.get(gid, ()):
            member = tensor_map.get(member_tid)
            if member is None:
                continue
            if any(self._region_readable(r)
                   for r in self._vram.space.get_by_tensor_id(member_tid)):
                continue
            ram_regions = self._ram.space.get_by_tensor_id(member_tid)
            if not ram_regions or not ram_regions[0].is_ready:
                return False  # bail before claiming anything
            to_stage.append((member, ram_regions[0]))

        for member, src in to_stage:
            dst = self._claim_region(self._vram, member)
            if dst is None:
                # _claim_region already signalled OOM → abort path;
                # partial claims here are moot since the sim ends.
                return False
            self._on_weight_claimed(member)
            transfers.append((src, dst))
            loading_tids.append(member.id)

        if to_stage:
            self._group_state[gid] = _LOADING
        return True

    def _ensure_non_weight_resident(
        self, tensor: Tensor, memory: BaseMemory, transfers: list[tuple[DataRegion, DataRegion]]
    ) -> bool | None:
        """Standard "input must be readable on compute.memory" pattern.
        For non-WEIGHT tensors (CONTEXT / INPUT / LEAF) the home device
        is set by the trace; if the compute is on a different device we
        stage a transfer."""
        target_regions = memory.space.get_by_tensor_id(tensor.id)
        if any(self._region_readable(r) for r in target_regions):
            return True

        pending = self._pending_dest_ids()
        if any(
            (r.access_status == DataRegionAccess.BEING_WRITTEN) or (id(r) in pending)
            for r in target_regions
        ):
            # Already being filled; just wait.
            return True

        src = self._find_latest_region(tensor.id, exclude_hw=memory)
        if src is None:
            return False  # not produced yet

        # An IDLE existing region we can reuse, else claim new.
        dst = self._idle_region_on(memory, tensor.id)
        if dst is None:
            dst = self._claim_region(memory, tensor)
            if dst is None:
                return False
        transfers.append((src, dst))
        return True

    def _ensure_output_claimed(self, node: Node, memory: BaseMemory) -> bool:
        for tid in node.output_tensors:
            if tid in node.input_tensors:
                continue  # in-place / view: input region IS output region
            tensor = self.sys.trace.tensor_map[tid]
            regions = memory.space.get_by_tensor_id(tid)
            if any(r.access_status == DataRegionAccess.IDLE for r in regions):
                continue
            region = self._claim_region(memory, tensor)
            if region is None:
                return False
        return True

    def _outputs_free(self, node: Node, memory: BaseMemory) -> bool:
        pending = self._pending_dest_ids()
        for tid in node.output_tensors:
            if tid in node.input_tensors:
                continue
            regions = memory.space.get_by_tensor_id(tid)
            has_idle = False
            for r in regions:
                if r.access_status == DataRegionAccess.IDLE and id(r) not in pending:
                    has_idle = True
                    break
            if not has_idle:
                return False
        return True

    def _preclaim_dispatcher_outputs(self, node: Node) -> None:
        """Loader-driven contract: dispatcher nodes (e.g. cpu thread
        aten::empty producing a CUDA tensor) have their cross-device
        outputs stashed in ``node.args["dispatcher_outputs"]`` and
        removed from ``node.output_tensors``. We pre-claim those
        regions on the tensor's home memory so consumers see them
        ready/latest."""
        cross = node.args.get("dispatcher_outputs") or []
        for tid in cross:
            tensor = self.sys.trace.tensor_map.get(tid)
            if tensor is None:
                continue
            ttype = tensor.args.get("tensor_type")
            if ttype == "WEIGHT":
                # WEIGHTs already have their home RAM region from layout.
                home = self._ram
            else:
                home = self._home_memory_for_non_weight(tensor)
            target = self._idle_region_on(home, tid)
            if target is None:
                if self._remaining_consumers.get(tid, 0) <= 0 and ttype not in self._PERMANENT_TYPES:
                    continue
                target = self._claim_region(home, tensor)
                if target is None:
                    continue
            target.is_ready = True
            target.is_latest = True

    # ============================================================ submit loop
    def _parents_done(self, node: Node) -> bool:
        node_map = self.sys.trace.node_map
        return all(node_map[pid].status == NodeStatus.DONE for pid in node.parent_nodes)

    def _submit_transfer_batches(self, transfers: list[tuple[DataRegion, DataRegion]]) -> None:
        grouped: dict[tuple[int, int], list[tuple[DataRegion, DataRegion]]] = defaultdict(list)
        for src, dst in transfers:
            grouped[(src.hw.id, dst.hw.id)].append((src, dst))
        for batch in grouped.values():
            self.sys.transfer(batch)

    def _kick_cuda_weight_prefetch(self, node: Node) -> None:
        """For every offload group touched by ``node``'s WEIGHT inputs,
        kick a RAM->VRAM transfer if the group isn't already in flight
        or resident. Mirrors accelerate's "leaf module's forward starts
        -> H2D the weights" — even when the consumer node is the
        CPU-thread dispatcher whose own compute doesn't need them
        resident on VRAM. Doesn't gate the consumer; the transfer
        fires alongside.

        Group-aware: under ``preload_module_classes``, kicking on any
        member of a group stages the whole group atomically.
        """
        tensor_map = self.sys.trace.tensor_map
        seen_groups: set[str] = set()
        for tid in node.input_tensors:
            tensor = tensor_map.get(tid)
            if tensor is None:
                continue
            gid = self._tensor_group.get(tid)
            if gid is None or gid in seen_groups:
                continue
            seen_groups.add(gid)
            state = self._group_state.get(gid, _ABSENT)
            if state in (_LOADING, _LOADED):
                continue
            # If the group's members are already on VRAM and readable,
            # sync state to LOADED without staging.
            members = self._group_tids.get(gid, [])
            if members and all(
                any(self._region_readable(r)
                    for r in self._vram.space.get_by_tensor_id(mt))
                for mt in members
            ):
                self._group_state[gid] = _LOADED
                continue

            batch: list[tuple[DataRegion, DataRegion]] = []
            staged: list[int] = []
            self._on_group_load_started(node, gid)
            if not self._stage_group(gid, batch, staged):
                # Master copy not ready or OOM; try again next tick.
                continue
            if batch:
                job_id = self.sys.transfer(batch)
                self._inflight_load_tids[job_id] = staged
                for _src, dst in batch:
                    self._tick_pending_dest_ids.add(id(dst))

    def _submit_ready_nodes(self) -> bool:
        """One pass over ``ready_node_ids``; returns True iff anything
        was queued so the caller can re-drive (view-only retires cascade)."""
        submitted_any = False
        committed_per_compute: dict[BaseCompute, int] = {}
        self._tick_pending_dest_ids = set()

        num_ready = len(self.ready_node_ids)
        for _ in range(num_ready):
            node_id = self.ready_node_ids.popleft()
            node = self.sys.trace.node_map[node_id]
            if node.status != NodeStatus.TODO:
                continue
            if self.pending_parent_count[node_id] != 0 or not self._parents_done(node):
                self.ready_node_ids.append(node_id)
                continue

            compute = self._compute_for_node(node)

            # Alias / dispatcher nodes: pure pointer ops. The loader
            # gave them ``custom_deps``; the engine bypasses input/
            # output residency checks. The dispatcher itself does not
            # need its WEIGHT inputs resident on VRAM to run, but the
            # downstream GPU kernels do — fire the RAM->VRAM transfer
            # for any cuda-homed WEIGHTs alongside the dispatcher.
            if node.custom_deps:
                cap = getattr(compute, "max_concurrent_jobs", 1)
                if len(compute.job_running) + committed_per_compute.get(compute, 0) >= cap:
                    self.ready_node_ids.append(node_id)
                    continue
                self._kick_cuda_weight_prefetch(node)
                if node.args.get("dispatcher_outputs"):
                    self._preclaim_dispatcher_outputs(node)
                self.sys.compute(compute, node)
                committed_per_compute[compute] = committed_per_compute.get(compute, 0) + 1
                submitted_any = True
                continue

            cap = getattr(compute, "max_concurrent_jobs", 1)
            if len(compute.job_running) + committed_per_compute.get(compute, 0) >= cap:
                self.ready_node_ids.append(node_id)
                continue

            memory = compute.memory

            # 1. Output claiming first — needed before submit. If we
            # can't even claim outputs (OOM or busy), retry later.
            if not self._ensure_output_claimed(node, memory):
                self.ready_node_ids.appendleft(node_id)
                continue

            # 2. Per-input residency. WEIGHT inputs landing on a GPU
            # compute trigger RAM->VRAM staging that the compute will
            # block on; everything else uses the standard cross-device
            # residency machinery. CPU computes that happen to consume
            # a WEIGHT find it in RAM and don't trigger any transfer
            # via this path — the prefetch hook above handles their
            # GPU-side staging.
            transfers: list[tuple[DataRegion, DataRegion]] = []
            loading_tids: list[int] = []
            blocked = False
            tensor_map = self.sys.trace.tensor_map
            for tid in node.input_tensors:
                t = tensor_map[tid]
                ttype = t.args.get("tensor_type")
                if ttype == "WEIGHT" and memory is self._vram:
                    ok = self._ensure_weight_loaded(t, transfers, loading_tids, node=node)
                else:
                    ok = self._ensure_non_weight_resident(t, memory, transfers)
                if not ok:
                    blocked = True
                    break

            if blocked:
                self.ready_node_ids.append(node_id)
                continue

            # CPU compute nodes consume WEIGHT tensors via RAM (already
            # present from layout). But the downstream GPU kernels need
            # those weights on VRAM — kick the H2D prefetch alongside
            # this node's submission, same as the dispatcher path.
            if memory is self._ram:
                self._kick_cuda_weight_prefetch(node)

            # 3. Submit any new transfers; node compute waits another
            # tick until the transfer retires and the dest region is
            # readable.
            if transfers:
                weight_transfers: list[tuple[DataRegion, DataRegion]] = []
                other_transfers: list[tuple[DataRegion, DataRegion]] = []
                for src, dst in transfers:
                    if dst.hw is self._vram and src.hw is self._ram and dst.tensor_id in loading_tids:
                        weight_transfers.append((src, dst))
                    else:
                        other_transfers.append((src, dst))
                if weight_transfers:
                    job_id = self.sys.transfer(weight_transfers)
                    self._inflight_load_tids[job_id] = [d.tensor_id for _s, d in weight_transfers]
                if other_transfers:
                    self._submit_transfer_batches(other_transfers)
                for _src, dst in transfers:
                    self._tick_pending_dest_ids.add(id(dst))
                self.ready_node_ids.append(node_id)
                continue

            # 4. Outputs must also have no pending transfer landing on
            # their region; otherwise the compute's begin() would race.
            if not self._outputs_free(node, memory):
                self.ready_node_ids.append(node_id)
                continue

            self.sys.compute(compute, node)
            committed_per_compute[compute] = committed_per_compute.get(compute, 0) + 1
            submitted_any = True

        return submitted_any

    # ============================================================ retire
    def _handle_transfer_retires(self, retired_jobs: list[BaseJob]) -> None:
        for job in retired_jobs:
            if not isinstance(job, TransferJob):
                continue
            tids = self._inflight_load_tids.pop(job.id, None)
            if not tids:
                continue
            # Identify which groups this transfer satisfied. A group
            # only flips to LOADED when every member is readable on
            # VRAM — for singleton groups that's the trivial case; for
            # coalesced groups it guards against partial loads.
            touched_groups = {self._tensor_group.get(tid) for tid in tids}
            touched_groups.discard(None)
            for gid in touched_groups:
                if self._group_state.get(gid) != _LOADING:
                    continue
                members = self._group_tids.get(gid, [])
                if all(
                    any(self._region_readable(r)
                        for r in self._vram.space.get_by_tensor_id(mt))
                    for mt in members
                ):
                    self._group_state[gid] = _LOADED
        self._refresh_peak_gpu_resident()

    def _group_has_pending_consumer(self, gid: str) -> bool:
        """True iff any queued/in-flight node still has a member tid of
        ``gid`` in its input set. Used to gate eviction so we don't free
        a weight whose H2D is in flight for a kernel already in
        job_waiting, or whose owning kernel is currently executing."""
        members = self._group_tids.get(gid, ())
        if not members:
            return False
        members_set = set(members)
        trace = self.sys.trace
        for nid in self.ready_node_ids:
            node = trace.node_map[nid]
            if not members_set.isdisjoint(node.input_tensors):
                return True
        engine = self.sys.engine
        for job in engine.job_waiting:
            if isinstance(job, ComputeJob) and not members_set.isdisjoint(job.node.input_tensors):
                return True
        for job in engine.job_running:
            if isinstance(job, ComputeJob) and not members_set.isdisjoint(job.node.input_tensors):
                return True
        return False

    def _release_group_vram(self, gid: str) -> None:
        """All-or-nothing: if any member's VRAM region is still non-IDLE
        (BEING_READ / BEING_WRITTEN), defer the whole group; otherwise
        release every member and flip state to ABSENT. This prevents
        partial release from leaving group state stuck at _LOADED with
        a subset of members missing.

        Skips entirely if some queued/in-flight node still needs the
        group — eager eviction must not orphan an in-flight load that
        a kernel in job_waiting is depending on, nor pull the rug from
        a currently-executing kernel. A later consumer retire will
        re-attempt eviction when the queues drain."""
        if self._group_has_pending_consumer(gid):
            self._pending_vram_group_releases.discard(gid)
            return

        members = self._group_tids.get(gid, ())
        for member_tid in members:
            for region in self._vram.space.get_by_tensor_id(member_tid):
                if region.access_status != DataRegionAccess.IDLE:
                    self._pending_vram_group_releases.add(gid)
                    return

        tensor_map = self.sys.trace.tensor_map
        for member_tid in members:
            member = tensor_map.get(member_tid)
            for region in list(self._vram.space.get_by_tensor_id(member_tid)):
                self.sys.release(region)
                if member is not None:
                    self._on_weight_released(member)

        self._pending_vram_group_releases.discard(gid)
        self._group_state[gid] = _ABSENT

    def _release_non_weight_regions(self, tensor_id: int) -> None:
        deferred = False
        for hw in self.sys.hw.values():
            if not isinstance(hw, BaseMemory):
                continue
            for region in list(hw.space.get_by_tensor_id(tensor_id)):
                if region.access_status == DataRegionAccess.IDLE:
                    self.sys.release(region)
                else:
                    deferred = True
        if deferred:
            self._pending_releases.add(tensor_id)
        else:
            self._pending_releases.discard(tensor_id)

    def _consume_inputs(self, node: Node) -> None:
        tensor_map = self.sys.trace.tensor_map
        # Per-(node, gid) dedup: invocation accounting counts a node
        # once per group regardless of how many members it consumes,
        # so we have to decrement at the same granularity here.
        consumed_groups: set[str] = set()
        for tid in node.input_tensors:
            remaining = self._remaining_consumers.get(tid)
            if remaining is None:
                continue
            remaining -= 1
            tensor = tensor_map.get(tid)
            ttype = tensor.args.get("tensor_type") if tensor is not None else None
            gid = self._tensor_group.get(tid) if ttype == "WEIGHT" else None

            # WEIGHTs: evict per-invocation, modeling diffusers' per-
            # block post-forward eviction. We track which invocation
            # of each group this node belongs to (from offline temporal
            # clustering — see _build_invocation_accounting) and only
            # fire the release when the current invocation's consumer
            # count hits zero. Eager evict on every retire would
            # thrash a block group on every internal op; this defers
            # to the natural module-forward boundary.
            #
            # _release_group_vram's all-members-IDLE guard + pending-
            # consumer queue gate (_group_has_pending_consumer) then
            # cover the rare race where an in-flight H2D for the next
            # invocation has already arrived in job_waiting while the
            # current one's last consumer is retiring.
            if ttype == "WEIGHT" and tensor is not None:
                if gid is not None:
                    # Dedup per-(node, gid) to match the granularity
                    # that _build_invocation_accounting used when it
                    # counted pending consumers.
                    if gid not in consumed_groups:
                        consumed_groups.add(gid)
                        self._on_weight_consumer_retired(node, gid)
                else:
                    # cpu-device WEIGHT promoted to VRAM by a GPU consumer
                    # — released as a singleton, RAM master left alone.
                    for region in list(self._vram.space.get_by_tensor_id(tid)):
                        if region.access_status == DataRegionAccess.IDLE:
                            self.sys.release(region)
                            self._on_weight_released(tensor)

            if remaining > 0:
                self._remaining_consumers[tid] = remaining
                continue
            self._remaining_consumers.pop(tid, None)

            if tensor is None:
                continue

            if ttype == "WEIGHT":
                continue  # eviction already attempted above

            if ttype in self._PERMANENT_TYPES:
                continue  # INPUT / LEAF — held forever

            self._release_non_weight_regions(tid)

    def _release_dead_outputs(self, node: Node) -> None:
        """Producer outputs with no remaining consumer get released as
        soon as the producing op retires — mirrors PyTorch ref-counting.
        WEIGHTs are never node outputs in practice; skip."""
        for tid in node.output_tensors:
            if tid in node.input_tensors:
                continue
            if self._remaining_consumers.get(tid, 0) > 0:
                continue
            tensor = self.sys.trace.tensor_map.get(tid)
            if tensor is None:
                continue
            ttype = tensor.args.get("tensor_type")
            if ttype == "WEIGHT" or ttype in self._PERMANENT_TYPES:
                continue
            self._release_non_weight_regions(tid)

    def _retry_pending_releases(self) -> None:
        for gid in list(self._pending_vram_group_releases):
            self._release_group_vram(gid)
        for tid in list(self._pending_releases):
            self._release_non_weight_regions(tid)

    def _retire_computes(self, retired_jobs: list[BaseJob]) -> None:
        # Two passes: (1) promote every retired job's children to
        # ready_node_ids so the pending-consumer gate in eviction sees
        # them; (2) actually fire _consume_inputs / _release_dead_outputs.
        # Without this ordering the eviction in _consume_inputs runs
        # *before* the just-retired node's children have been promoted,
        # so _group_has_pending_consumer can't see them and evicts a
        # weight that a sibling op is about to consume — causing
        # massive H2D thrash inside a single block forward (we saw
        # ~580× extra ram->vram bytes on SDXL-Turbo before this).
        node_map = self.sys.trace.node_map
        for job in retired_jobs:
            if not isinstance(job, ComputeJob):
                continue
            for child_id in job.node.children_nodes:
                if child_id not in self.pending_parent_count:
                    continue
                if self.pending_parent_count[child_id] > 0:
                    self.pending_parent_count[child_id] -= 1
                child = node_map[child_id]
                if self.pending_parent_count[child_id] == 0 and child.status == NodeStatus.TODO:
                    self.ready_node_ids.append(child_id)

        for job in retired_jobs:
            if not isinstance(job, ComputeJob):
                continue
            self._consume_inputs(job.node)
            self._release_dead_outputs(job.node)

    # ============================================================ runtime
    def runtime(self, retired_jobs: list[BaseJob]) -> None:
        self._handle_transfer_retires(retired_jobs)
        self._retire_computes(retired_jobs)
        self._retry_pending_releases()

        while self._submit_ready_nodes():
            pass

        self._refresh_peak_gpu_resident()

        # Deadlock guard: nothing in flight, but nodes still TODO.
        if not self.sys.engine.job_running and not self.sys.engine.job_waiting:
            todo = [n for n in self.sys.trace.node_map.values() if n.status == NodeStatus.TODO]
            if todo:
                blocked = todo[0]
                self.sys.abort({
                    "from": self.name,
                    "error": "SCHEDULER_DEADLOCK",
                    "msg": "No runnable node and queues empty.",
                    "node": {
                        "id": blocked.id,
                        "name": blocked.name,
                        "parent_nodes": blocked.parent_nodes,
                        "input_tensors": blocked.input_tensors,
                        "output_tensors": blocked.output_tensors,
                    },
                })
        return
