# cg-sim — Session-carried context

Things a future Claude Code session should know before working in this
repo. Everything here is durable — not specific to a transient task.

## Where to look for offload-verification context

Reading order for a fresh session picking up offload-scheduler
verification work:

1. `README.md` — what cg-sim is and how to run it
2. **This file** (`CLAUDE.md`) — durable session-carried context
3. `docs/TODO.md` — the single offload-verification handoff doc:
   current direction (approach A, trace-driven), the D1-D8 divergence
   catalog, decision log, and a map of where each simulator mechanism
   lives in the repo. (This merges the former `TODO.md` and
   `docs/cg-sim_divergence_sources.md`.)

After those are read, you should be on the same page as where the
prior session left off. Read them in order before investigating any
verification gap or proposing new compensation logic.

## Faithfulness principle (read first)

cg-sim's purpose is to predict what would happen in a real run
without actually running it. If we make arbitrary adjustments to
fit the simulator's output to a reference trace, the simulator
stops being predictive — it just memorizes that trace.

Two responsibilities, two faithfulness contracts:

- **Loader's job**: faithfully translate the input trace into
  cg-sim's internal representation. Don't drop information, don't
  invent it. If the trace says a CPU node took 8 µs, that's 8 µs.
- **Scheduler's job**: faithfully model the real scheduler's
  behavior. If accelerate's `AlignDevicesHook.pre_forward` calls
  `aten::to` once per parameter, the simulator's scheduler should
  reflect that — not because it makes the wall time fit, but
  because that's what the real scheduler does.

Ad hoc adjustments (phantom nodes, latency knobs without physical
backing, deleting work from the trace, "fudge factors") are the
opposite of faithful. They should be **absolutely minimum** and
need a defensible justification any researcher in the field would
sign off on — e.g. "PCIe Gen4 ×16 pageable-source effective HtoD
throughput is well-known to be 13 GB/s after host-stack overhead",
not "13 GB/s lands inside the ±20% bound."

When a verification gap remains after honest modeling, that gap is
real data about the simulator's limits. Document the cause; don't
mask it. Examples of "honest documentation" already in this repo:
the SDXL bandwidth comment in this file (single global knob can't
model pinned + pageable simultaneously), and the
`AccelerateCpuOffload` tied-weight commit message (eager trace
records tied parameters as one tid; accelerate doesn't dedup).

## DiffusersGroupOffload scheduler

Lives at `sim/sched/diffusers_group_offload/`. Subclasses
`DeviceAwareVanillaAsync` and overrides only `compile()`; the DAV runtime
is reused as-is.

What it models: diffusers' `enable_group_offload(offload_type="block_level",
num_blocks_per_group=1, use_stream=True, record_stream=False)`. The
authoritative description of that runtime is
`docs/offload-schemes/diffusers_group-offload_use-stream-true.md` —
read it first if you need to change the scheduler.

How it works at compile time:

1. Reads `trace.args["module_hierarchy"]` (loaded from
   `module_hierarchy.json` by `sim/load/pytorch_profile/pytorch_profile.py`).
2. Direct-child `ModuleList`/`Sequential` grandchildren of each
   top-level pipeline component → **matched** groups (pinned H2D,
   pointer-swap eviction via `evict_after_node`).
3. Everything else under a component → per-component **unmatched**
   lump (pageable H2D + real DtoH bookends via `xfer_arrivals` /
   `d2h_xfer_arrivals`).
4. Patches `tensor.args["device"] = "cpu"` for every offload tid so
   layout places them in RAM, then re-runs `_init_xfer_states()`.
5. Re-runs `_build_arrival_index()` to refresh DAV's hint indexes
   (DAV's `__init__` already ran with empty trace.args).

`args.block_modules: dict[str, list[str]]` mirrors diffusers'
`block_modules=` knob — names additional non-ModuleList direct
children of a component to recurse into (Flux's `mid_block`, etc.).
SDXL/SD3 don't need it.

## PCIe bandwidth calibration

Read `docs/cg-sim_bandwidth_calibration.md` before changing
`memory_bandwidth_KBps` in any SDXL/SD3 YAML. TL;DR:

- The four SDXL/SD3 eager YAMLs are pinned to **13 000 000** (13 GB/s)
  to match the measured pageable HtoD/DtoH throughput from the
  reference SDXL group-offload trace (9.54 GB / 721 ms).
- LLM YAMLs (`pytorch-eager__llama-3-*__vanilla.yaml`,
  `pytorch-lazy__*`) keep **25 000 000** (25 GB/s). They are not
  transfer-bound; the calibration doesn't apply.
- cg-sim has a single global bandwidth knob shared by all
  TransferJobs. It cannot independently model pinned-source matched
  H2D (~25 GB/s real) and pageable-source unmatched H2D (~13 GB/s
  real). 13 GB/s is the smallest defensibly-grounded value and lands
  the SDXL e2e gap just inside the ±20% verification bound.
- If `tensor.args["transfer_path"] = "pinned" | "pageable"` ever gets
  added, split the bandwidth back into 25/13 and remove this
  compromise.

## Verification targets

For `examples/run/pytorch-eager__sdxl-turbo__diffusers_group_offload.yaml`:

| Metric | cg-sim | Target | Bound |
|---|---|---|---|
| e2e time | 2.092 s | 2.602 s (trace span) | ±20% |
| peak VRAM | 4.55 GB | 4.47 GB (manifest peak) | ±10% |

Targets come from
`examples/trace/diffusers-group-offload__sdxl-turbo__RTX4090/llama_bundle/`:
trace span = `max(end_ns) − min(start_ns)` over `runtime_nodes.csv`;
peak VRAM = `vram_peak_allocated_bytes` in `manifest.json`.

## SD3 verification — pending

`examples/trace/pytorch-eager__sd3__RTX4090/llama_bundle/runtime_nodes.csv`
is a 133-byte LFS pointer in this checkout. SD3 group-offload trace
(`examples/trace/diffusers-group-offload__sd3__RTX4090/`) doesn't
exist at all. To verify SD3:

```
git lfs pull   # needs a configured LFS remote — not present in
               # ephemeral remote-exec containers
```

The SD3 YAML (`examples/run/pytorch-eager__sd3__diffusers_group_offload.yaml`)
is ready; once the bundle materializes, `python3 main.py -i …` runs
end-to-end without code changes.

## Container quirks (remote-exec sessions)

The base image is missing many Python packages that the simulator
expects. Install before the first run:

```bash
pip install --no-deps orjson fastuuid sortedcontainers numpy \
  hydra-core omegaconf pydantic pydantic-core==2.46.4 \
  pydantic_settings annotated-types attrs referencing \
  jsonschema-specifications networkx polars ipython anyio mcp \
  starlette sse-starlette jsonschema httpx httpx-sse
```

`antlr4-python3-runtime` is special: pip's source-build fails in
this image. Use the 4.9.3 sources directly:

```bash
pip download antlr4-python3-runtime==4.9.3 -d /tmp/antlr
cd /tmp/antlr && tar -xzf antlr4-python3-runtime-4.9.3.tar.gz
cp -r antlr4-python3-runtime-4.9.3/src/antlr4 \
   /usr/local/lib/python3.11/dist-packages/
```

The default 4.13.x conflicts with omegaconf's bundled ATN.

## Commit hygiene

**Never commit anything under `examples/trace/`.** That directory is
LFS-managed (mostly); commits there belong to a separate trace-curation
workflow run by the repo owner, not to scheduler/engine code changes.

In particular: `git lfs pull` can leave files in a state where git
sees the working-tree (real content, MBs) differing from the committed
LFS pointer (134 bytes), and reports them as "modified". This is a
false positive — never `git add` those files. The stop-hook check will
nag about uncommitted changes; ignore that for trace files. There's
also a pre-existing repo bug where `examples/trace/sd3_offload_group/*`
files were committed as inline blobs containing LFS pointer text
*without* a matching `.gitattributes` LFS filter entry — those will
always show as modified in any session that ran `git lfs pull`. Fixing
that requires adding LFS filter entries and re-pointing the files;
not something to do incidentally.

When committing code changes, always stage explicit code paths
(`sim/...`, `examples/run/*.yaml`, `scripts/...`, `docs/...`,
`CLAUDE.md`, `AGENTS.md`) — never use `git add .` or `git add -A`.

## Git / commit signing

Commit signing in this environment is provided by the env-runner's
`code-sign` tool (`/tmp/code-sign` → `/opt/env-runner/environment-manager`).
It rejects commits with `"missing source"` unless the session itself
was launched with a GitHub source binding configured at the
platform level — adding a `git remote` from inside the container is
not sufficient.

If signing is unavailable and the user has explicitly authorized it,
bypass with `git -c commit.gpgsign=false commit …`.

To push, the container needs GitHub credentials. The clean way is
session-level GitHub integration (set up in the launching app). If
the user pastes a PAT into chat: use a temporary `GIT_ASKPASS`
helper script (`/tmp/.gh-askpass.sh` echoing an env var), then
`shred -u` the script and `unset` the env var. **Tell the user to
revoke the token immediately afterward** — chat transcripts leak.

Branch is currently `master`. If renamed on GitHub, future sessions
should mirror locally with `git branch -m master <new>` and
`git remote set-head origin -a`.
