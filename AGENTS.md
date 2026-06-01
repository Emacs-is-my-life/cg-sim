# cg-sim Agent Guide

This document is the operational reference for AI agents (Claude Code,
Codex, etc.) driving cg-sim through its MCP server. Humans see
[README.md](README.md) for the standard human-mode workflow.

# Running cg-sim via MCP

Run `main_agent.py` instead of `main.py`. It boots an MCP (Model Context
Protocol) server on its stdio and drives a loop so an agent can run the
simulator repeatedly — inspect state, resume execution, then `restart_simulation`
for a fresh run (optionally with a new input YAML), all via tool calls and
without restarting the process.

## One-time setup (Human must do)

Register cg-sim as an MCP server with your agent. For Claude Code:
```bash
# Recommended — no default config; the agent picks an input.yaml per run.
$ claude mcp add cg-sim-mcp -- python main_agent.py

# Alternative — pin a default config; the agent can still switch later.
$ claude mcp add cg-sim-mcp -- \
      python main_agent.py -i examples/run/llamacpp__llama-3-8B__flexinfer.yaml
```
`-i` is optional. The recommended form omits it, so the same MCP registration
serves every simulator config the agent might want — the first
`restart_simulation(input_path=..., overrides=...)` call builds the Simulator.
With `-i`, the named YAML is the default for the first run and the agent can
still switch on any subsequent run via `restart_simulation(input_path=...)`.

Substitute absolute paths for both `python` and the input file if your agent
launches the server from a different working directory. Start a new agent
session after registering — tool lists are snapshotted at session start.

> The MCP server also advertises this same debugging surface to the
> connected LLM client via its connection-time instructions (in
> `sim/core/debug/agent_server.py`, `_SERVER_INSTRUCTIONS`). Keep this
> document and that string in sync.

## Tool surface

The server exposes eight tools:
- `list_breakpoints` — return all `BREAK_*` flags and their On/Off status.
- `toggle_breakpoint(name)` — flip a flag.
- `start_simulation` — release the simulator (it blocks at startup so the agent
  can configure breakpoints first). **Blocking**: returns once the first
  breakpoint fires or the run finishes, with the resulting state in the
  response.
- `current_state` — re-read the current state without releasing the worker.
  Only needed if `start_simulation` / `continue_simulation` returns
  `timed_out=True`. The state shape is `{at_breakpoint, breakpoint,
  variables, tip, simulation_finished}`.
- `execute(code)` — run Python against the parked breakpoint's namespace.
  A bare last expression has its value echoed (REPL-style); see *Breakpoint
  namespace* and *Debugger methods* below for what's in scope.
- `continue_simulation` — resume the simulator. **Blocking**: returns once
  the next breakpoint fires or the run finishes, with the resulting state
  in the response.
- `restart_simulation(input_path=None, overrides=None, reload=True)` — tear
  down the just-finished simulator and build a fresh one. Also the *first*
  call when the server was launched without `-i` (no default config) —
  pass `input_path=...` to build the initial Simulator. Otherwise only
  callable after `simulation_finished=true` (or before the first run when
  a default `-i` was supplied). Pass `input_path` to switch the YAML
  config (required on first construction when no `-i` was given;
  sticky otherwise). Pass `overrides` (a list of Hydra-style strings
  like `["scheduler.args.prefetch_window=8"]`) to apply CLI-equivalent
  config overrides — see *Config overrides at restart* below for the
  full semantics. With `reload=True` (default), drops the whole
  `sim.*` tree (except the MCP daemon harness) from `sys.modules` so
  source edits anywhere — schedulers, hardware, loaders, and framework
  core — are picked up; see *Hot-reloading code* below. **Blocking**:
  returns once the new Simulator is constructed.
- `shutdown` — end the agent session and exit the process. If the simulator is
  parked at a breakpoint, releases it first so the current run drains cleanly.

## Typical session

0. (Only if the server was registered without `-i`) `restart_simulation(
   input_path="examples/run/<config>.yaml", overrides=[...])` to build the
   first Simulator. With `-i`, this is unnecessary on the first run.
1. `list_breakpoints` → see available flags.
2. `toggle_breakpoint("BREAK_BEFORE_COMPILE_STAGE")` → enable the ones you want.
3. `start_simulation` → blocks; response carries the new state.
4. `execute("trace.node_map")` (or any other inspection/mutation).
5. `continue_simulation` → blocks; response carries the new state.
6. Repeat 4–5 until the response reports `simulation_finished=true`.
7. `restart_simulation()` (optionally with a new `input_path`) to go again
   from step 1, or `shutdown` to exit the process.

## Stage breakpoints

The five coarse `BREAK_*` flags fire once per simulator stage:
- `BREAK_BEFORE_COMPILE_STAGE` — before the scheduler's compile pass.
- `BREAK_AFTER_COMPILE_STAGE` — after compile, before layout.
- `BREAK_AFTER_LAYOUT_STAGE` — after initial tensor placement.
- `BREAK_IN_RUNTIME_STAGE` — master switch enabling the per-Node/per-Job
  breakpoints below.
- `BREAK_AFTER_RUNTIME_STAGE` — after the runtime loop exits.

## Per-Node / per-Job breakpoints

To break on a *specific* compute graph node or job during the runtime stage,
set a `BREAK_AT_JOB_*` flag on the Node or Job object itself via `execute`.
Typical flow: enable `BREAK_AFTER_COMPILE_STAGE`, inspect `trace.node_map` to
find the node you care about, arm the flag, then `continue_simulation` — the
run will stop again when that node's job hits the chosen lifecycle point.

Available flags (settable on any `Node` or `BaseJob`; Jobs inherit the flag
from their Node at creation):
- `BREAK_AT_JOB_SUBMITTED` — when the job is enqueued onto `job_waiting`.
- `BREAK_AT_JOB_HEAD` — when the job reaches the head of `job_waiting`.
- `BREAK_AT_JOB_DISPATCHED` — when the job moves into `job_running`.
- `BREAK_AT_JOB_RETIRED` — when the job completes and is retired.

When one fires, the breakpoint name reports as
`break_in_runtime_stage[JOB_<PHASE>]` and the namespace exposes `job`
(the triggering job), `timestamp_now`, `job_waiting`, `job_running`, `hw`,
and `trace`. Example:
```python
# at break_after_compile_stage
target = next(n for n in trace.node_map.values() if n.name == "Qcur-16")
target.BREAK_AT_JOB_RETIRED = True
# continue_simulation → next stop will be break_in_runtime_stage[JOB_RETIRED]
# with job.node is target
```
Flags persist across `continue_simulation` calls, so disarm them
(`target.BREAK_AT_JOB_RETIRED = False`) once you're done if you want the run
to finish without stopping again.

## Per-Node execution hooks (hook_pre_run / hook_post_run)

Two optional callables on every `Node` let user code run *around* that
node's runtime execution, without editing the engine or the scheduler:

- `node.hook_pre_run: Callable[[System], None] | None` — invoked just
  *before* the node's ComputeJob calls `begin(...)`. Mutations made
  here (tensor sizes, hardware state, trace bookkeeping) are visible
  to `begin()` itself, so they can affect the *current* node's
  compute/transfer ETA estimate.
- `node.hook_post_run: Callable[[System], None] | None` — invoked just
  *after* the node's ComputeJob retires, *before* `BREAK_AT_JOB_RETIRED`
  parks the run and before the terminal-node check. At that breakpoint
  the observer therefore sees post-hook mutations already applied —
  symmetric with `BREAK_AT_JOB_DISPATCHED`, which sees the pre-hook
  mutations. Mutations here propagate to downstream nodes, but the
  just-retired job's recorded duration is already final.

Both receive a single argument — the `System` object (same as
`engine.sys`) — which exposes `sys.trace` (`node_map`, `tensor_map`)
and `sys.hw` (`dict[str, BaseHardware]`). The engine, debugger, and
triggering job are intentionally **not** passed: hooks are meant to
manipulate workload/hardware state, not framework internals.

Only `ComputeJob`s trigger hooks; `TransferJob`s do not. When unset
(`None`), the cost is one null-check per ComputeJob dispatch/retire.

**Intended uses.** Trace/workload manipulation at simulation time —
e.g., dynamic weight sparsity (shrink an FFN tensor a few layers
ahead), runtime quantization changes (mutate transfer cost
mid-execution), KV-cache eviction simulation, fault injection.

**Attaching from an MCP session.** Hooks are just attributes; set them
via `execute(...)` at any breakpoint where the target Node is
reachable. Closure-capture anything else you need (e.g. `debug` for
breadcrumb logging) — only `sys` is passed at fire time:
```python
# At break_after_compile_stage:
execute("""
ffn_tid   = <tensor_id of layer-9 FFN weight>
trigger   = trace.node_map[<layer-5 node_id>]

def shrink_layer9_ffn(sys, _dbg=debug):
    t = sys.trace.tensor_map[ffn_tid]
    _dbg.record({"hook": "shrink", "tid": ffn_tid,
                 "old_pages": t.num_pages, "new_pages": t.num_pages // 2})
    t.num_pages //= 2

trigger.hook_post_run = shrink_layer9_ffn
""")
continue_simulation
```
Hooks persist across `continue_simulation` until you clear them
(`trigger.hook_post_run = None`) or `restart_simulation` rebuilds the
Trace.

**Scheduler-owned hooks for state-coupled mutations.** Schedulers
commit layout decisions at compile/layout stage (slot sizes, page
indices, prefetch plans). An *externally-attached* hook that mutates
`tensor.num_pages` mid-run does **not** re-derive any of that —
LlamaCppFlexInfer still reserves the original slot, and the resulting stall
numbers will be subtly wrong rather than visibly broken. For
state-coupled experiments, attach hooks from inside a custom
scheduler's `__init__` or `compile()` so the *same* scheduler that
mutates state also reads it back in its `runtime()` callback. The
engine-level mechanism is identical; only the attach site changes.

**Pitfalls.**
- **Exceptions propagate.** A buggy hook is not caught; with
  `BREAK_ON_EXCEPTION` on (default for MCP), the run lands at
  `break_on_exception[<Type>]` with `exception_origin` pointing to the
  hook line. Fail-loud, not silent.
- **No `TransferJob` variant** — intentional. The two natural
  surrogates: producer's `hook_post_run` fires the instant the output
  tensor exists on its source device (*before* any transfer); the
  consumer's `hook_pre_run` fires after every required input transfer
  has completed (a ComputeJob isn't runnable until they have).
- **No auto-logging.** Capture `debug` by closure at attach time and
  call `debug.record(...)` from inside the hook if you want a
  breadcrumb in the log file.

## Abort breakpoint (BREAK_ON_ABORT, default On for MCP)

Generic soft-failure safety net. **Every** abort path in the simulator
funnels through `Engine._log_abort(args)`:

- engine-internal deadlocks and invalid-job submissions
- scheduler-issued `sys.abort(...)` (LlamaCppVanilla, LlamaCppFlexInfer, DeviceAwareVanillaAsync, Stub, your scheduler)
- job assertion failures (`transfer_assertion.py`, `compute_assertion.py`)
- mutation invariant breaks (`claim_mutation.py`)
- any future caller of `sys.abort()` or `_log_abort()` you add — covered
  automatically

With `BREAK_ON_ABORT` enabled (the MCP default; Off for human-mode
runs), `_log_abort` fires the `break_on_abort` breakpoint *after*
logging the reason but *before* `signal_abort=True` tears the run down.

The namespace adds:

- `abort_args` — the dict that was logged. Read `abort_args['msg']`
  and `abort_args['from']` for the reason; specific aborts add detail
  keys (e.g. runtime deadlock puts the stuck job's identity in
  `abort_args['job']`).
- `abort_stack` — full call chain at the abort site (from
  `inspect.stack()`). The breakpoint's natural frame is
  `Engine._log_abort`, which is *not* where the abort decision was
  actually made — that's a few frames up (scheduler, assertion,
  mutation, etc.). Use `abort_stack` to navigate:
  ```python
  # See the call chain:
  execute("""
  [(f.filename.rsplit('/', 1)[-1], f.lineno, f.function)
   for f in abort_stack[:8]]
  """)
  # → e.g. [('engine.py', 422, '_log_abort'),
  #         ('system.py', 114, 'abort'),
  #         ('vanilla.py', 65, 'runtime'),    ← decision frame
  #         ('engine.py', 280, '_runtime'),
  #         ...]
  # Inspect the decision frame's locals:
  execute("dict(abort_stack[2].frame.f_locals)")
  ```
- Standard runtime locals: `engine`, `timestamp_now`, `job_waiting`,
  `job_running`, `hw`, `trace`.

Continuing from the breakpoint lets the normal abort flow proceed
(`signal_abort=True`; run ends with `simulation_success=false`). For
fail-fast batch runs that should not pause on aborts,
`toggle_breakpoint("BREAK_ON_ABORT")` to disable.

## Exception breakpoint (BREAK_ON_EXCEPTION, default On for MCP)

Hard-failure counterpart of `BREAK_ON_ABORT`. When an uncaught
exception propagates out of `engine.run()` — bug in your scheduler,
HW model, trace loader, or any user code — `Simulator.run`'s
outermost handler:

1. Writes a `SIMULATION_EXCEPTION` log entry (type, message, full
   traceback) so post-mortem reading is symmetric with the soft-abort
   case.
2. Fires `break_on_exception[<ExceptionType>]` if enabled. The
   exception type is in the breakpoint name itself —
   `current_state.breakpoint` reads `"break_on_exception[KeyError]"`
   or `"[AttributeError]"`, no `execute` call needed for triage.
3. Ends the run gracefully (no re-raise), so `restart_simulation`
   works the same as after a clean finish or an abort.

The namespace adds:

- `exception` — the caught `BaseException` object. Read
  `str(exception)`, `exception.args`, `exception.__cause__`, etc.
- `exception_origin` — `{file, line, function}` pointing to the
  raise site (the deepest traceback frame). Surfaced top-level so
  the agent can `execute("exception_origin")` and immediately see
  where to look.
- `exception_stack` — list of traceback objects walked from
  `exception.__traceback__`. `[0]` is the outermost frame
  (engine.run); `[-1]` is the failing frame. Same navigation pattern
  as `abort_stack`, but sourced from `__traceback__` (the live stack
  has unwound by the time the exception is caught):
  ```python
  # Quick view of the chain:
  execute("""
  [(f.tb_frame.f_code.co_filename.rsplit('/', 1)[-1],
    f.tb_lineno, f.tb_frame.f_code.co_name)
   for f in exception_stack]
  """)
  # Inspect the failing frame's locals:
  execute("dict(exception_stack[-1].tb_frame.f_locals)")
  ```
- Standard: `debug`, `engine`.

Performance: the wrapping `try/except` in `Simulator.run` is zero-cost
on the success path in Python 3.11+ (compile-time exception tables,
no setup bytecode). The full handler cost is paid only when an
exception actually propagates.

To let exceptions propagate up to `main_agent.py` instead of pausing,
`toggle_breakpoint("BREAK_ON_EXCEPTION")` to disable. Logging and
graceful-end behavior remain — only the breakpoint stop is gated.

## Breakpoint namespace

The exact set is in `current_state.variables` for the current breakpoint.
Every breakpoint always binds at least:
- `debug` — the Debugger (see *Debugger methods* below).
- `engine` — the Engine; exposes `engine.sched` (scheduler), `engine.sys`,
  `engine.signal_abort`, `engine.job_stats`, and other internal state.

Most breakpoints also bind `trace` and `hw`. Runtime breakpoints additionally
bind `timestamp_now`, `job`, `job_waiting`, `job_running`.

The namespace **persists across `execute` calls within one breakpoint** —
locals you assign survive until the next `continue_simulation`, then are
cleared.

## Debugger methods / accessors

Reachable from inside `execute` (since `debug` is in scope):
- `debug.record(dict_args)` — write a JSON-serializable dict into the
  simulation log file under track Engine → Debug. Use this to leave
  breadcrumbs that survive after the agent session ends. Raises if
  `dict_args` isn't JSON-serializable.
- `debug.args` — free-form scratchpad dict (`{}` at start of each run).
  Stash whatever you want to carry across breakpoints: findings, working
  hypotheses, intermediate values, a list of nodes you've already
  inspected. Per-run state — cleared on `restart_simulation`. Use
  `debug.record(...)` for persistent log entries instead.
- `debug.break_lambda` — custom-predicate runtime breakpoint. Assign a
  callable `(engine, sys) -> bool`; once per runtime-loop tick (after
  retiring jobs, before progress update), it's evaluated and a return
  value of `True` (strictly) fires `break_in_runtime_stage[LAMBDA]`. No
  master flag needed — just setting the field enables it. Auto-clears if
  the predicate raises. At the LAMBDA breakpoint, `job` is not bound (no
  triggering Job); other runtime locals (`engine`, `timestamp_now`,
  `job_waiting`, `job_running`, `hw`, `trace`) are. Example:
  ```python
  debug.break_lambda = lambda engine, sys: (
      len(engine.job_running) > 8 and engine.timestamp_now > 1_000_000
  )
  # next continue_simulation stops the first tick the predicate is True
  ```
  Set to `None` to disable.
- `debug.help()` — re-print the current breakpoint context (banner +
  variables table + tip) to stdout, captured in `output`. Redundant with
  `current_state` for agents but available for a textual dump.
- `debug.engine` — same Engine reachable as the `engine` variable.
- `debug.log_path` — absolute `Path` of the simulation log file (a
  single Chrome-trace JSON object `{"traceEvents": [...]}`, finalized
  in `Simulator.run()`'s `finally` block — i.e. *after*
  `simulation_finished=True` is reported). So grab the path during a
  breakpoint and read the file out-of-band once the run ends:
  ```python
  execute("str(debug.log_path)")
  # Then, from your shell (or another process) after simulation_finished:
  # $ python3 -c "import json,sys; \
  #     print(json.load(open(sys.argv[1]))['traceEvents'][-3:])" \
  #     <path>
  ```
  See *Reviewing your breadcrumbs* under *Workflow: Writing a new
  Scheduler* for the full post-mortem pattern.

## Config overrides at restart

`restart_simulation(overrides=[...])` accepts a list of Hydra-style override
strings — the same syntax as the CLI overrides documented under
*Overriding input.yaml config from the command line* in
[README.md](README.md). This lets the agent sweep
scheduler / hardware / trace knobs across runs in a single session without
editing the YAML and without the caveats of mutating already-constructed
objects (which only catches fields read live; values consumed inside
`__init__` — derived tables, scheduler caches — won't be re-derived by a
`hw['ram'].memory_size_KB = ...` poke).

```python
# Sweep prefetch_window in a single MCP session:
restart_simulation(overrides=["scheduler.args.prefetch_window=8"])
restart_simulation(overrides=["scheduler.args.prefetch_window=16"])

# Override a list element by index:
restart_simulation(overrides=["hardware.memory.0.args.memory_size_KB=10485760"])

# Combine multiple overrides; combine with a new input YAML:
restart_simulation(
    input_path="examples/run/llamacpp__llama-3-8B__vanilla.yaml",
    overrides=["+debug=on", "logger.args.log_level=3"],
)
```

Semantics:
- `overrides=None` (default) — keep the previous overrides. Sticky across
  restarts and initialized from whatever extra CLI args were passed to
  `main_agent.py` at startup.
- `overrides=[]` — explicitly clear all overrides.
- `overrides=[...]` — replace with the supplied list.

The applied list is echoed back as `overrides` in the response so the
agent can verify. Invalid override strings raise during construction
(Hydra surfaces them as `OverrideParseException` or similar) and the
session lands in `CONSTRUCT_FAILED`; recover by calling
`restart_simulation` again with corrected `overrides`.

## Hot-reloading code (entire simulator, including core)

`restart_simulation(reload=True)` (the default) drops the simulator's
modules from `sys.modules` before rebuilding the Simulator, so the agent
can edit source files on disk and pick up the changes on the next run
without restarting the process.

In scope (reloaded) — **the entire `sim.*` tree**:
- `sim/sched/<impl>/...` — scheduler implementations.
- `sim/hw/<type>/<impl>/...` — compute, memory, storage models.
- `sim/load/<impl>/...` — trace loaders.
- `sim/.../common/...` — base classes / ABCs.
- `sim/core/...` — framework core: engine, system, job, trace, log,
  simulator, and the Debugger.

Out of scope (spared — restart the agent process to pick up edits here):
- `sim/core/debug/agent_runner.py` and `sim/core/debug/agent_server.py`
  — the live MCP daemon harness. The running process *is* executing this
  code and holds a live `AgentSession`/`Phase` across the reload
  boundary, so it cannot reload itself out from under its own stack.
- `main_agent.py` — the entry point (not a `sim.*` module).

**Why reload core at all?** Reloading core *together* with everything
else keeps class and enum identities consistent. A partial reload — core
spared, hw/sched reloaded — splits a shared identity such as the
`DataRegionAccess` enum across two module instances: un-reloaded core
mutation code stamps regions with one enum object while a
freshly-reloaded scheduler compares against another, so
`region.access_status == DataRegionAccess.IDLE` is silently `False` and
the scheduler deadlocks on its own regions. Reloading the whole tree
removes the split by construction. (Base classes reload alongside their
subclasses, so `isinstance` stays correct without the `*.common`
carve-out earlier versions needed.)

Mechanism: the `LOAD_*_CLASS` functions in `sim/core/init/` do live
`importlib.import_module` lookups for sched/hw/load; `main_agent._construct`
re-fetches `Simulator` via `importlib.import_module("sim.core.simulator")`
so the core subtree re-executes too. Both rely on the `sys.modules`
eviction performed by `agent_server._hot_reload_user_modules`.

The response includes `reloaded_modules` (count of evicted `sys.modules`
entries) so the agent can confirm the reload happened — expect a larger
count than pre-core-reload builds, since core modules are now included.
Pass `reload=False` to preserve current class identities (compare
back-to-back runs of identical code without paying the re-import cost).

Example:
```python
# 1. Edit sim/sched/llamacpp_flexinfer/flexinfer.py — or sim/core/engine/engine.py.
# 2. From the agent loop, after simulation_finished:
restart_simulation(reload=True)
# 3. The next start_simulation runs against the freshly-imported code.
```

## Environment knobs

- `CG_SIM_BREAKPOINTS` — comma-separated `BREAK_*` flag names to pre-enable
  before the server starts (alternative to calling `toggle_breakpoint`).
  Re-applied to each fresh Debugger built by `restart_simulation`, so it
  acts as a persistent default across runs in one agent session.
- `CG_SIM_AGENT_MODE` — set automatically by `start_agent_server`
  (called from `main_agent.py`) to suppress the interactive
  `welcome_prompt` (whose `input()` would otherwise race the MCP server
  for stdin). Do not set this manually.

# Workflow: Writing a new Scheduler

A worked example that ties the debugging surface to a real authoring
workflow. The same pattern applies to writing new hardware models or
trace loaders; the file layout below changes accordingly.

## 0. Scaffold

Copy the simplest existing scheduler as a starting point and rename:
```bash
cp -r sim/sched/llamacpp_vanilla sim/sched/myscheduler
# In sim/sched/myscheduler/__init__.py: re-export your renamed class.
# In sim/sched/myscheduler/myscheduler.py: rename `class LlamaCppVanilla(...)` →
#   `class MyScheduler(BaseScheduler)`.
```
Point an input YAML at it:
```yaml
scheduler:
  type: "MyScheduler"
  args: { ... }
```
Register `cg-sim` as an MCP server (see *One-time setup* above) and connect.

## 1. First run — surface construction errors

```
toggle_breakpoint("BREAK_BEFORE_COMPILE_STAGE")
start_simulation
```
If `Simulator.__init__` blows up (bad import, wrong base class, missing
required `args` key, …), the failure surfaces here. Fix in source.

## 2. Inspect what your scheduler is handed

At `break_before_compile_stage`, the namespace exposes `trace`, `hw`,
and `engine`:
```python
execute("len(trace.node_map), len(trace.tensor_map)")
execute("[(n.id, n.name) for n in list(trace.node_map.values())[:5]]")
execute("list(hw.keys())")
```
Confirm your scheduler's `compile(trace)` is seeing what you expect.

## 3. Iterate without restarting the agent

After editing `sim/sched/myscheduler/myscheduler.py` on disk:
```
restart_simulation()   # reload=True is the default
start_simulation       # runs the freshly-imported class
```
No need to disconnect or re-add the MCP server. See *Hot-reloading
code* above for what reloads (the whole `sim.*` tree, including core)
and what's spared (the MCP daemon harness).

## 4. Trace a single node through runtime

When one specific node misbehaves (wrong device, never dispatched,
retires too early/late), enable the runtime master switch and arm a
per-Node flag at the compile-boundary breakpoint:
```python
execute("""
target = next(n for n in trace.node_map.values() if n.name == 'Qcur-16')
target.BREAK_AT_JOB_DISPATCHED = True
target.BREAK_AT_JOB_RETIRED = True
""")
toggle_breakpoint("BREAK_IN_RUNTIME_STAGE")
continue_simulation
```
The next stop is `break_in_runtime_stage[JOB_DISPATCHED]` with `job`
bound to that node's job.

## 5. Custom stop conditions

For "wake me up when X holds" — pressure thresholds, queue overflow,
suspicious mismatches — set `debug.break_lambda`:
```python
execute("""
debug.break_lambda = lambda engine, sys: (
    len(engine.job_running) > 8 and engine.timestamp_now > 1_000_000
)
""")
continue_simulation
```
Strict-`True` only; auto-clears if it raises.

## 6. Leave breadcrumbs as you investigate

Two persistence levels:
```python
# In-memory, per-run (cleared on restart):
execute("debug.args['hypothesis'] = 'tensor X always evicted on tick Y'")

# To the log file, persists after the run:
execute("debug.record({'decision': 'preferred device 0', 'tensor_id': tid})")
```

## 7. Reviewing your breadcrumbs

Two natural review paths:

**In-session, while still parked at a breakpoint.** Use `debug.args`
as your working notebook — it's in-memory and readable at any
breakpoint in the same run:
```python
# during the run:
execute("debug.args.setdefault('odd-tensors', []).append(tid)")
# later, at break_after_runtime_stage:
execute("debug.args['odd-tensors']")
```

**Post-mortem, after the run ends.** Records written with
`debug.record(...)` go to `debug.log_path` (Chrome-trace JSON:
`{"traceEvents": [...]}`). The file is finalized only after the
writer thread closes it (during `Simulator.run()`'s `finally`),
which happens *after* `simulation_finished=True` is reported to the
agent — by which point the agent no longer has a breakpoint to
`execute` from. So the log is best read **out-of-band**: print
`debug.log_path` at any breakpoint, then read the file from your
shell, a separate process, or after `shutdown`.
```python
# Grab the path before the run finishes:
execute("print(debug.log_path)")
# After shutdown (or via another shell during the run):
# $ jq '.traceEvents[] | select(.name == "DEBUG_MSG") | .args' \
#       /path/to/result.json
```
If you need cross-run persistence, copy the file aside between runs
or change `result_path` in the YAML between runs.

## 8. A/B against a known-good scheduler

Three reference implementations ship today:
- `LlamaCppVanilla` (no offload) and `LlamaCppFlexInfer`
  (memory-saving) for `llamacpp` traces.
- `DeviceAwareVanillaAsync` for `pytorch-eager` /
  `pytorch-lazy` traces — vanilla scheduler with async H2D streams
  that routes each Node to the compute device its kineto record
  came from (`node.hw`).
- `Stub` (in `sim/sched/generic_stub/`) — no-op scheduler.
  `compile`/`layout` succeed so the simulation reaches every
  pre-runtime breakpoint, but `runtime` aborts on entry. Useful
  for loading a config whose intended scheduler isn't on this
  branch and inspecting the compiled Trace.

Switch input YAMLs mid-session:
```
restart_simulation(input_path="examples/run/llamacpp__llama-3-8B__vanilla.yaml")
```
Then re-run with the same breakpoints to compare. `debug.record(...)`
breadcrumbs end up in each run's own log file.

## Common scheduler pitfalls

Two safety nets are On by default *for MCP sessions* (Off for human-mode
runs of `main.py`) — one for each failure mode:
- `BREAK_ON_ABORT` catches every abort path through `Engine._log_abort`
  — engine deadlocks, scheduler-issued `sys.abort(...)`, assertion
  failures. Agent lands at `break_on_abort` with `abort_args` and live
  runtime state.
- `BREAK_ON_EXCEPTION` catches uncaught exceptions from anywhere under
  `engine.run()` — bugs in your scheduler code, AttributeError from a
  missing attr, KeyError from a typo, anything you forget to handle.
  Agent lands at `break_on_exception[<ExceptionType>]` with
  `exception_origin` (file/line/function of the raise site) and
  `exception_stack` (full chain from `__traceback__`).

- Returning early from `compile/layout/runtime` without setting state
  on the trace → engine sees an empty job pipeline, simulation ends
  before it starts. **No abort fires** — this is a silent
  no-progress case; check `trace.node_map` and `job_waiting` at
  `break_after_compile_stage` / `break_after_layout_stage` instead.
- Submitting a non-`TransferJob` during the layout phase →
  `break_on_abort`. `abort_args['msg']` reads "Scheduler can only
  submit TransferJob in layout phase."
- Cycle or missing dependency between nodes → `break_on_abort` with
  `abort_args['msg']` = "Deadlock detected." Inspect
  `job_waiting[0].is_runnable(engine.sys)` and walk
  `job_waiting[0].node.parent_nodes` to see which dependency is
  unmet. For runtime-phase deadlocks, `abort_args['job']` carries the
  stuck job's identity.
- Scheduler explicitly calling `sys.abort({'msg': '...'})` (the
  natural pattern for "I detected an unrecoverable condition") → also
  `break_on_abort`, with `abort_args` being exactly the dict you
  passed.
- An uncaught exception anywhere — typo, missing attribute, wrong
  type — → `break_on_exception[<Type>]`. Read `exception_origin` for
  the raise site (file/line/function), inspect
  `exception_stack[-1].tb_frame.f_locals` for what the failing code
  was doing.

# Why agents should prefer the analysis scripts

`engine.job_stats` at `break_after_runtime_stage` gives raw totals
(compute time, transfer time, byte counts) but cannot distinguish a
prefetcher that overlaps transfer with compute (good) from one that
doesn't (bad) — both report the same `transfer_total_time`. The
analysis scripts in `scripts/analysis/` derive the quantities that
actually answer those questions, so reach for them whenever raw
totals are not enough. See [README.md](README.md) for the script
catalog and calling convention.

During parameter sweeps, set `logger.args.result_path` to a distinct
path per run (via the `overrides` arg of `restart_simulation`) so
each run's log is preserved for post-mortem.

# Output directory conventions

Every artifact an agent produces — simulator traces, analyses, plots —
goes under `output/<experiment-setup>/`. One experiment, one tree. No
loose files in `output/` root, no `tmp/sweeps/`, no ad-hoc paths in
`tmp/`. The sweep runner already enforces this layout; standalone
agent runs should match it.

## Layout

```
output/<experiment-setup>/
  experiment.yaml           # manifest: base config, overrides, git SHA, timestamp
  summary.csv               # one row per cell, joined metrics (sweeps only)
  sim_results/
    <run-id>.json           # raw cg-sim trace (Chrome-trace JSON)
    <run-id>.command.txt    # exact `main.py` invocation + overrides
    <run-id>.stdout.log
    <run-id>.stderr.log
  analysis/
    <run-id>/
      prefetch_quality/     # CSVs + meta.json from scripts/analysis/*.py --out
      link_utilization/
  plots/
    <plot-name>.png         # from scripts/visualization/*.py --out
    <plot-name>.html        # interactive timelines
```

Plus, refreshed at the end of every sweep:

```
output/latest -> <most-recent-experiment-setup>/   # convenience symlink
```

## Naming rules

- **`<experiment-setup>`** — kebab-case identifier, one per logical
  experiment. For sweep runners, pass it explicitly
  (`flexinfer-mem-sweep`, `prefetch-window-tuning`). For one-off runs
  of a stock YAML, the YAML basename is the natural default
  (`llamacpp__llama-3-8B__flexinfer` for
  `examples/run/llamacpp__llama-3-8B__flexinfer.yaml`).
- **`<run-id>`** — the cell label in sweeps (`4GB`, `flexinfer-pw3`,
  `vanilla`). For single-shot runs of a YAML use `result` (so the
  artifact reads as `sim_results/result.json`, not the misleading
  `default.json`). Filesystem-safe: ASCII letters/digits/underscores/dashes only.
- **`<plot-name>`** — script stem by default
  (`plot_metric_vs_param.png`). Override with `--out` when producing
  multiple plots of the same script in one experiment.

## When the user asks you to run experiments

1. **Pick a setup name** that describes the experiment in one
   hyphenated phrase. If the user names it, use that. Otherwise
   default to `<base-config-basename>` for single runs or
   `<scheduler>-<knob>-sweep` for sweeps.
2. **Drive the run** through the sweep runner if there's more than
   one cell, or through `main.py` with a `logger.args.result_path`
   override otherwise. Either path must land JSONs under
   `output/<setup>/sim_results/<run-id>.json`.
3. **Run analyses with `--out output/<setup>/analysis/<run-id>/<analysis-name>/`**
   so each analysis's CSVs + `meta.json` land in the canonical place.
   The sweep runner does this for you; manual `python3
   scripts/analysis/*.py` calls need an explicit `--out`.
4. **Render plots with `--out output/<setup>/plots/<plot-name>.png`**
   (or `.html`). When `--out` is omitted, the viz scripts default to
   `<experiment-root>/plots/<script-stem>.png`, so this often happens
   automatically — but be explicit when generating more than one plot
   per script.
5. **Mention the artifact path** in your final summary so the user
   can `ls output/<setup>/`. Don't drop files in the repo root, in
   `tmp/`, or in `output/` directly.

## What goes where, by tool

| Producer | Writes to |
|---|---|
| `main.py` / `main_agent.py` | `sim_results/<run-id>.json` (via `logger.args.result_path`) |
| `scripts/experiments/sweep_memory.py`, `compare_schedulers.py` | full experiment tree, manifest, `summary.csv` |
| `scripts/analysis/*.py --out PATH` | a single `<analysis-name>/` subdir (CSVs + `meta.json`) |
| `scripts/visualization/*.py --out PATH` | a single PNG/HTML file under `plots/` |

## Manifest (`experiment.yaml`)

Each sweep writes a manifest at the experiment root capturing
everything needed to reproduce it: base config path, common
overrides, per-cell overrides, git SHA at run time, UTC timestamp,
free-form description. Treat this file as authoritative provenance —
do not edit it after the run; rerun the experiment instead.

## Use `tmp/`, not `output/`, for scratch

`output/` is for artifacts the user will browse, share, or version
between experiments. `tmp/` is for scratch: throwaway MCP-driven
runs, exploratory dumps, intermediate files you'll delete in the
same session. Both are in `.gitignore`. When in doubt, put it in
`output/` — disambiguating later is harder than overcleaning now.

# Simulator internals reference

The sections above cover the *operational* surface (MCP tools,
breakpoints, output layout). The sections below cover the
*internals* — what the engine actually does each tick, what
contracts a Scheduler / Hardware model / TraceLoader is expected
to honor, and the non-obvious behaviors that bite first-time
contributors. Read these before writing a new Scheduler or Hardware
class; the MCP debugging surface is great for *finding* bugs but
will not tell you which invariants you've violated.

## Module dependency layers

Bottom-up, so that importing higher layers cannot create cycles:

```
log/, hw/common/data_region.py                 (no sim.* deps)
  └─ sim_object.py, trace/, hw/common/base_hardware.py
       └─ hw/memory/common/, hw/storage/common/
            └─ job/{assertion,mutation,logging}/, hw/compute/common/
                 └─ engine/, simple hardware (sim/hw/<type>/<impl>/)
                      └─ system.py, debug/
                           └─ sched/common/, sched/<impl>/, load/<impl>/
                                └─ core/init/, simulator.py
                                     └─ main.py, main_agent.py
```

Two patterns to note:
- **Aggregator `__init__.py`s.** `sim/sched/__init__.py`,
  `sim/hw/<type>/__init__.py`, and `sim/load/__init__.py` all do
  `pkgutil.iter_modules(__path__)` + `import_module` to auto-discover
  subpackages and re-export anything in `module.__all__`. **A new
  scheduler / loader / hardware impl is picked up automatically
  once it's importable and lists its class in its own `__all__`** —
  no central registry to edit.
- **Live-lookup loaders.** `sim/core/init/{compute,memory,storage,
  trace,scheduler}.py` each define a single `LOAD_*_CLASS(name)`
  that calls `importlib.import_module("sim.X")` then `getattr`.
  Intentionally re-imports every call so `restart_simulation(reload=True)`
  picks up source edits to sched/hw/load without restarting the agent
  process. The core subtree is refreshed the same way — `main_agent._construct`
  re-fetches `Simulator` via `importlib.import_module("sim.core.simulator")`
  after the eviction — so core edits take effect on the next run too.

## Discrete-event engine: how time moves

cg-sim is **event-driven**, not tick-driven. The clock (`engine.timestamp_now`)
**only advances** in two places, both inside `engine.py:_runtime_forward`:

1. `self.time_elapsed = job.timestamp_ETA - self.timestamp_now`
2. `self.timestamp_now = job.timestamp_ETA`

Both jump straight to the next-earliest-ETA job's finish time. There
is no `time += 1` loop; idle gaps where nothing happens are skipped
without cost. `compile` and `layout` advance time by exactly 10 µs each
at the end of their stage as a visual marker in the Chrome-trace log;
they otherwise model zero wall time.

The corollary: **the entire runtime loop is purely reactive**. Each
iteration responds to one or more job retirements at the same instant.
Schedulers never get a "time step" callback — they get a `runtime(retired_jobs)`
callback after some jobs have just finished, with `retired_jobs`
possibly empty (e.g. on the first tick when `job_running` is empty
but `job_waiting` has entries that need to be drained first).

## The three stages, in order

`engine.run()` (`engine.py:89`) walks three stages and calls back
into the Scheduler at each. After each stage the corresponding
`BREAK_AFTER_*_STAGE` debugger flag, if set, fires.

### Compile (`engine.py:149`)
Just `sched.compile(trace)`. Free pass for the scheduler to mutate
the trace (insert deps, attach hooks, tag NodeHW, build internal
state). No job submission expected. Time advances by 10 µs at the
end as a marker.

### Layout (`engine.py:176`)
**Logging is disabled for the entire layout phase** (`self.log.on = False`).
Layout transfer Begin/End events therefore *do not appear* in the
Chrome-trace JSON. Their effects on `MemorySpace.peak_num_used_pages`
*do* land in the final `SIMULATION_RESULT` because mutation is
independent of logging. If you need layout events visible, set
`log.on = True` from a debug hook or inside the scheduler's `layout()`.

The phase iterates:

```
finished = False
while not finished:
    finished = sched.layout(init_storage)   # returns True when done
    drain job_waiting → job_running         # every job set to ETA = now
    retire all                              # zero wall-time
```

Layout invariants:

- **Only `TransferJob` may be submitted.** Anything else triggers
  the abort `"Scheduler can only submit TransferJob in layout phase."`
  (`engine.py:163`).
- **Every submitted job must be immediately runnable.** The drain loop
  HOL-blocks (just like runtime) but with no in-flight jobs to wait
  for, head-of-line block → `"Deadlock detected."` abort.
- **All retires happen at the same `timestamp_now`** because
  `timestamp_ETA = now` is set explicitly for every dispatched job.
  Layout transfers are *instantaneous* in simulated time — you
  cannot use the layout phase to charge SSD load time against the
  wall-clock budget. Charge it in runtime instead.
- **`init_storage` is hardcoded to the first storage** in `Simulator.__init__`
  (`simulator.py:148`: `init_storage = hw[next(iter(hw))]`). The
  insertion order is the YAML order. Multi-storage configs always
  initial-place onto `hardware.storage[0]`.

Because `sched.layout` is iterated, you can return `False` to be
called again — useful for multi-phase placement, e.g. DAV's
SSD → DRAM → VRAM staging where the SSD's `can_run` allows only one
concurrent job (`device_aware_vanilla_async.py:492`).

### Runtime (`engine.py:271`)
One iteration of the loop = **one observable simulated instant**.
The exact order of operations is load-bearing and not obvious from
a casual read:

```
loop:
  1. Log per-hw counters/states (driven by Log.level).
  2. _runtime_forward():
     - Peek the top of `job_running` and the top of `job_fixed_latency`,
       advance timestamp_now to the earlier of the two ETAs.
     - If both are inf (nothing initialized), return [] (waiting on
       newly-dispatched jobs to get an ETA later in this tick).
     - Drain `job_running` cohort at timestamp_now. A TransferJob
       with fixed_latency_micros > 0 is *not* retired here: its
       work_done is pinned to work_total, its ETA is set to
       `now + fixed_latency_micros`, and it moves to a separate
       `job_fixed_latency` heap. Non-TransferJobs (and zero-fl
       transfers) retire normally.
     - Drain `job_fixed_latency` cohort at timestamp_now. Each pop
       is a real retire — the latency window is up.
  3. For each retired job (in order):
     - ComputeJob: call node.hook_post_run(sys), if set.
     - Fire BREAK_AT_JOB_RETIRED if armed on the job/node.
     - **If the retired job's node is a TerminalNode → return from _runtime.**
       (Co-retired jobs *after* the terminal in the cohort are
       skipped — their hooks and break-retired flags do not fire.)
  4. Evaluate debug.break_lambda(engine, sys) if set. Auto-clears
     on exception.
  5. For each job in job_running: update_progress(time_elapsed).
  6. sched.runtime(retired_jobs).                ← scheduler reacts
  7. Drain job_waiting head-first into job_running (FIFO):
     - If head is_runnable(sys) → pre-run hook, dispatch, fire
       BREAK_AT_JOB_DISPATCHED.
     - If head NOT runnable and job_running is *empty* → abort
       "Deadlock detected."
     - If head NOT runnable and job_running has work → **break out
       of the drain loop**. Other waiting jobs do not get a chance.
  8. update_running_jobs(): recompute every running job's ETA
     against current bandwidth contention (water-filling for
     transfers, max_work_rate for compute).
  9. heapify(job_running).
```

Two consequences worth flagging:

**Job submission ordering matters** (see *FIFO HOL blocking* below).

**`sched.runtime` runs *after* progress update**, so if you compare
`job.work_done` between consecutive `runtime` calls you can see
exactly how much work each job did in the just-elapsed interval.

## Job lifecycle and timestamps

`BaseJob` (`job/job.py:15`) carries four lifecycle timestamps; each
field is set exactly once and then is permanent:

| Field | Set by | Meaning |
|---|---|---|
| `timestamp_queued` | `engine.submit` | Enters `job_waiting` |
| `timestamp_at_head` | `engine._runtime`, first time head | Reached front of queue (= `now` then) |
| `timestamp_begin` | `BaseJob.begin` | Dispatched to `job_running` |
| `timestamp_end` | `BaseJob.end` | Retired |

`timestamp_at_head − timestamp_queued` = queue wait. `timestamp_begin − timestamp_at_head`
= head-of-line stall (zero unless the head sat unrunnable for a
tick). `timestamp_end − timestamp_begin` = effective execution time.
Schedulers that want to budget stall vs. execution should read these
fields; the `prefetch_quality.py` analysis script (see README) does
this decomposition by transfer type.

`timestamp_ETA` is the heap key, recomputed every tick by
`update_running_jobs`. A job for which no ETA has been computed yet
sits in the heap with `timestamp_ETA = float("inf")`; `_runtime_forward`
detects this and bails (returns empty `retired_jobs`).

The two synchronous job types — `ClaimJob` and `ReleaseJob` — **never go
through `engine.submit`**. `System.claim` and `System.release`
construct, assert, mutate, log, and discard them inline; they never
appear in `job_waiting` or `job_running`. `BaseJob.__lt__`'s
heap-tie-break by `self.id` (a random UUID) means **same-ETA cohort
retire order is non-deterministic between runs** — relevant for
anything bit-reproducible.

## The four Job subclasses

Each subclass plugs three callbacks (assertion / mutation / logging)
via the matching file in `sim/core/job/{assertion,mutation,logging}/`.
The pattern: `is_runnable` → `begin_mut` → `begin_log` → (engine
ticks update `work_done`) → `end_mut` → `end_log`.

| Job | Synchronous? | `work_total` | Use |
|---|---|---|---|
| `ComputeJob` | No (queued) | `node.compute_time_micros` (AU·µs) | Run a Node on a compute hw |
| `TransferJob` | No (queued) | sum of `4 * min(src,dest).num_pages` KB across batch | Batch copy between two hardwares |
| `ClaimJob` | **Yes** (sys.claim) | 0 | Allocate a `DataRegion` |
| `ReleaseJob` | **Yes** (sys.release) | 0 | Free a `DataRegion` |

`TransferJob`'s **batch must be src-hw → dest-hw single-pair**: every
`(src_region, dest_region)` in the batch must have `src.hw == batch[0][0].hw`
and `dest.hw == batch[0][1].hw`, else the assertion aborts. To
issue cross-hardware transfers in one scheduler call, group by hw
pair and submit one job per pair (DAV does this in
`_submit_transfer_batches`).

`TransferJob.fixed_latency_micros` is taken from whichever of
`hw_from` / `hw_to` is a `BaseStorage` with a non-zero
`fixed_latency_micros` (defaults to 0, set in `SimpleSSD.__init__`).
It models per-op overhead (think NAND command setup). The engine
implements it as a two-phase retire backed by **two separate heaps**:
`job_running` (bandwidth phase) and `job_fixed_latency` (latency
phase). When the bw phase ends, the job migrates between the heaps
rather than being re-pushed; it's invisible to the water-filling
bandwidth allocator (`update_transfer_jobs`) and to `update_progress`
during the latency window, but it still occupies the hw's
`job_running` list, so `can_run` continues to return False for
additional jobs on that hw.

## Why FIFO HOL blocking matters for schedulers

`job_waiting` is a `collections.deque`; `engine._runtime` drains it
strictly head-first. If `job_waiting[0]` is not runnable this tick,
the drain loop **breaks immediately** without trying any subsequent
job (`engine.py:362-393`). The drain resumes next tick.

This means **the order in which a scheduler calls `sys.compute(...)` /
`sys.transfer(...)` directly determines the order in which jobs are
considered for dispatch**, and a poorly-placed transfer at the head
can stall otherwise-runnable computes sitting behind it for an entire
simulated instant.

Practical patterns:

- If you submit a `TransferJob` and a `ComputeJob` that depends on
  the transfer in the same `runtime()` call, submit the *transfer
  first* — the compute will sit in the queue with `is_runnable=False`
  (its input tensor isn't on the compute's memory yet) and you'd
  HOL-block the entire pipeline.
- If you submit several `ComputeJob`s for different compute devices
  (CPU + GPU), submit them **interleaved**, not all-CPU-first then
  all-GPU. SimpleCPU only admits one job (`can_run` returns
  `len(job_running) == 0`); if you queue [CPU_A, CPU_B, GPU_X],
  CPU_B fails `can_run`, HOL breaks, and GPU_X never dispatches
  this tick. DAV solves this with a `committed_per_compute` counter
  that caps commits to one-per-hw per tick
  (`device_aware_vanilla_async.py:_submit_ready_nodes_core`).
- Head-of-line block + empty `job_running` = `"Deadlock detected."`
  abort. So a scheduler that runs out of dispatched work AND has an
  unrunnable head ends the run. The abort message includes the head
  job's identity in `abort_args["job"]`.

The flip side: **the FIFO never reorders**. You cannot push a job
to the back of the queue once submitted. If you need a different
order, build your own dispatcher in the scheduler and submit one
job at a time.

## Bandwidth model: weighted water-filling

`engine/update_transfer.py:update_transfer_jobs` runs every tick to
re-allocate bandwidth across all running `TransferJob`s. The
algorithm:

1. Start every active job at bandwidth 0.
2. Sum the *capacities* (`hw.max_work_rate()`) of each hw touched
   by any active transfer.
3. Find the smallest bandwidth increment that would saturate some
   hw. Increment every active job's bandwidth by that delta.
4. Freeze every job that touches any newly-saturated hw.
5. Repeat with the remaining active jobs until no further growth
   is possible.

Two consequences:

- Jobs share bandwidth **equally** unless they're freed up by a
  shared bottleneck. A 1-byte transfer and a 1-GB transfer get the
  same bandwidth on the same link until one of them is done.
- `hw.max_work_rate()` is **queried fresh every tick**. A custom
  hw model can return different capacities for different running-job
  shapes — `SimpleSSD` does this, returning bandwidth interpolated
  from an IO-size curve (`simple_ssd.py:_get_bandwidth_KBps`).
  Compute hardware can do the same for clock-throttling models, etc.

Compute jobs bypass water-filling — `update.py:update_running_jobs`
just asks each compute hw for `max_work_rate()` and assigns it to
the job directly. SimpleCPU/SimpleGPU return a fixed `modifier` so
each running ComputeJob gets full rate (GPU's `max_concurrent_jobs > 1`
effectively models infinite parallelism with constant per-job
throughput).

## The DataRegion state machine (= cache coherence model)

This is the trickiest piece. `DataRegion` (`hw/common/data_region.py`)
is the unit of memory occupancy and the unit of cache coherence.
Each region tracks:

- `tensor_id` — which logical tensor it currently holds.
- `is_latest` — is this copy of the tensor up-to-date?
- `is_ready` — are the bytes actually present (post-init, not
  mid-transfer)?
- `access_status` — `IDLE | BEING_READ | BEING_WRITTEN`.
- `access_count` — number of jobs currently reading.

Lifecycle:

```
fresh region (post-claim):  is_ready=False, is_latest=False, IDLE
loaded from storage:        is_ready=True,  is_latest=True,  IDLE
being read by a transfer:   ............... is_latest=...,   BEING_READ (count++)
being written (compute/xfer dest): is_ready=False, is_latest=True, BEING_WRITTEN
                                            ^^ flipped to True only on the chosen
                                               output region by compute_mutation
```

**Global invalidation on write.** When a `ComputeJob` begins, for
each "real output" tensor (in `output_tensors` but not also in
`input_tensors`), `compute_mutation.invalidate(sys, tid)` walks
**every memory and storage in `sys.hw`** and sets every region for
that tensor's `is_latest = False` (`job/mutation/utils.py:invalidate`).
Then the *chosen* output region (on the compute's local memory) is
re-marked `is_latest = True`. So after a compute, exactly one region
across the entire system carries the latest copy of each written
tensor. Other devices holding that tensor see `is_latest = False`
and the engine's transfer assertion will refuse to source from them.

**Transfers inherit `is_latest`.** When a transfer begins, the dest
region's `is_latest` is set to the *src* region's current
`is_latest` (`transfer_mutation.py:26`). Copying from a stale source
produces a stale destination. Schedulers must not pick stale sources
unless they're sure no fresher copy exists.

**`access_status` is the locking primitive.** A region with
`access_status == BEING_WRITTEN` cannot be read or written by anyone
else (assertion fails). `BEING_READ` allows additional readers (the
count tracks them) but no writers. `IDLE` allows either. Every
mutation pair must balance: every `access_count++` has a matching
`access_count--`, and the region returns to `IDLE` when count drops
to 0. Bugs here typically present as "deadlock detected" with a
stuck job whose `is_runnable` check fails on a region that should
have been freed.

The `compute_assertion.py` data-dependency check (`compute_assertion.py:54-87`)
requires every input tensor to have **at least one** region on
`hw.memory` that is simultaneously `is_ready`, `is_latest`, and
`access_status ∈ {IDLE, BEING_READ}`. Outputs need a region that is
`IDLE`. A scheduler that doesn't claim outputs ahead of time will
HOL-block waiting for one to appear.

## Custom dependencies: an all-or-nothing escape hatch

A Node with `node.custom_deps != []` opts out of the engine's
built-in control + data dependency check **entirely** —
`compute_assertion.py:38` runs only the custom predicates, skipping
the parent-DONE check, the input-residency check, and the output-IDLE
check. The hardware admission check (`hw.can_run` + NodeHW match)
still runs.

This is powerful but easy to get wrong. The DAV scheduler uses it
for alias / dispatcher / python-released nodes; the typical pattern
is to add a `NodeDoneDep` per parent so the control check is
re-implemented without the data check:

```python
for parent_id in node.parent_nodes:
    node.custom_deps.append(NodeDoneDep(parent_id))
```

This effectively says "no data residency requirement, just wait for
parents to be DONE." If you forget to add the parent deps, the node
becomes immediately runnable regardless of graph ordering — usually
producing wrong results before a visible crash.

**Side effect to know about:** `compute_mutation.begin_mutation`
still iterates `node.input_tensors` and `node.output_tensors`
unconditionally. If a custom_deps node has inputs/outputs that
aren't actually resident, `begin_mutation` just doesn't find them
and `job.input_regions`/`output_regions` stay empty — but no crash.
`end_mutation` then iterates those (empty) lists and does nothing.
This is intentional: alias nodes have data tensors that physically
exist on a different memory, and engine-level region bookkeeping
shouldn't touch them. But it means a Node with `custom_deps` set
*also* needs its `output_tensors` cleaned out of cross-device
tensors (DAV moves them to `node.args["dispatcher_outputs"]` and
pre-claims them in the scheduler).

Shipped `CustomDep` subclasses (`trace/custom_dep.py`):
`NodeDoneDep`, `TensorAtHWDep` (via `args["custom_dep_tag"]` on hw),
`MinTimestampDep`, `LambdaDep`. Subclass `CustomDep` for anything
reusable; `LambdaDep` is the escape-hatch for one-offs (not
serializable to logs in a useful way).

## The `System` API (scheduler → engine boundary)

`System` (`sim/core/system.py`) is the only interface a scheduler
should use to talk to the engine. Six calls plus two signals:

| Call | Sync? | Returns | Notes |
|---|---|---|---|
| `compute(hw, node, args=None)` | submits ComputeJob | `uuid.UUID` (job id) | Sets `node.status = WAITING`. Engine will dispatch later. |
| `transfer(batch, args=None)` | submits TransferJob | `uuid.UUID` | `batch: list[tuple[DataRegion, DataRegion]]`, all src.hw same, all dest.hw same. |
| `claim(hw, tensor, page_idx_start=-1)` | **synchronous** | `DataRegion \| None` | Asserts + mutates inline. Returns None on failure (with abort already signalled). |
| `find(hw, tensor)` | synchronous | `list[DataRegion]` | All regions on `hw` holding `tensor` (or `tensor_id` int). |
| `release(region)` | **synchronous** | `None` | Asserts (region must be IDLE) + mutates inline. Aborts on busy region. |
| `end_stage()` | signals engine | — | Drops `EngineSignal.END_STAGE` (largely unused in current schedulers). |
| `abort(args)` | signals engine | — | Logs the abort and sets `signal_abort = True`. Tear-down happens on next engine tick. |

**`abort` does not raise**. It logs and signals; the caller's code
keeps running until it returns. Schedulers calling `sys.abort()` from
inside `runtime` should typically `return` immediately afterward to
avoid building more state on a doomed run.

`sys.claim` and `sys.release` happen "for free" in simulated time —
they neither advance the clock nor enter the engine queue. They
are how a scheduler manipulates the memory layout in the middle of
runtime without spending budget on it. The Chrome-trace log shows
them as instant events (`event_instant`, `ph: "i"`).

## Hardware admission protocol

Every `BaseHardware` implements two methods that the engine calls:

- `can_run(job)` — return True iff this hw can accept another job
  *right now*. Used in assertion phase to gate dispatch.
- `max_work_rate()` — return the per-µs rate this hw can deliver
  *given the current `job_running` list*. Compute hw: AU/µs.
  Memory/storage hw: KB/µs. Returns 0 when idle.

Both are called every tick (`can_run` in `is_runnable`,
`max_work_rate` in `update_running_jobs`). They should be cheap.

The `job_running` list on a hardware is **separate from** the
engine's `job_running` heap. `BaseHardware.run(job)` appends to
`hw.job_running`; `BaseHardware.retire(job)` removes by id (O(n)
rebuild — fine for small lists). The engine's heap is the
scheduling structure; the per-hw list is the admission state. Jobs
running on multiple hardwares (e.g. TransferJob spans src and dest)
appear in both hardwares' lists.

## Trace structure conventions

A `Trace` (`sim/core/trace/trace.py`) is `{node_map, tensor_map, args}`.
Loaders are free to attach arbitrary data to `trace.args` — it's the
side-channel for scheduler-specific hints that don't fit the Node /
Tensor schema. Examples already in use:

- `trace.args["start_gated_edges"]` — `[(parent_id, child_id), ...]`
  (PytorchProfile)
- `trace.args["xfer_arrivals"]`, `d2h_xfer_arrivals`,
  `evict_after_node`, `evictable_tensor_ids` — populated by
  `graph_modifiers.inject_schedule` for WS-style experiments

`Trace.__init__` runs `map_check` which:
- Verifies every tensor referenced by a node's `input_tensors` /
  `output_tensors` exists in `tensor_map`.
- **Requires the last node in `node_map` (insertion order) to be a
  `TerminalNode`.** Without one, the simulation has no exit condition.

`Node.compute_time_micros` is the entire computational budget for
the node — there's no separate `cost(input_size, hw_speed)` formula.
A trace loader either records measured durations (llamacpp records,
PyTorch profiler kineto records) or hand-models them.

## Initialization order in `Simulator.__init__`

(`sim/core/simulator.py:99`) Order matters because each step
depends on prior steps' state:

```
1. Hydra config parse
2. Log (start daemon writer thread)
3. Debugger (welcome_prompt() in human mode if cfg["debug"])
4. Trace loader class + Trace.load()
5. Storage hardware (one or more)
6. trace_loader.placement(trace, hw[hardware.storage[0]])
       ↑ ALWAYS placed on the first storage in YAML order;
         that storage gets initial_placement=True.
7. Memory hardware
8. Compute hardware (consumes hw[c_cfg["args"]["memory"]] by name)
9. Validate custom_dep_tag uniqueness + TensorAtHWDep references
10. System(trace, hw)
11. Engine(...)  (scheduler = None initially)
12. Scheduler  (engine.sched = sched after construction)
13. SIM_CONFIG dump (resolved Hydra config + id_map + git metadata)
```

The Scheduler is constructed **last** and receives `System` (with
the trace, hw, engine all wired). It can read `sys.hw` and walk
`sys.trace` in its `__init__`. It cannot submit jobs in `__init__`
— the engine hasn't started yet, so anything submitted would land
in `job_waiting` before stages begin (technically fine, but
unconventional; use `compile` or `layout`).

If any step raises, `Simulator.__init__`'s `try/except` calls
`log.stop()` and re-raises. The agent-mode driver
(`main_agent.py:_construct`) catches that, prints the traceback to
stderr, and transitions the session to `CONSTRUCT_FAILED`.

## Surprises and gotchas (read these once)

These are the things that don't match a first-pass reading of the
code, listed so a new contributor doesn't relearn them by losing a
day to each.

1. **Layout disables logging.** Transfer events fired during
   `engine._layout` are not in the Chrome-trace JSON. Peak-memory
   stats *are* still accumulated. If you need layout events
   visible for an analysis, the loader's `placement()` runs before
   `log.start()` so storage-region claims won't log either; you'd
   need to defer the placement to the layout stage and re-enable
   `log.on` from inside the scheduler.

2. **TerminalNode return short-circuits the retire cohort**
   (`engine.py:301-302`). The `return` happens inside the for-loop
   iterating `retired_jobs`. If multiple jobs co-retire at the same
   instant as the TerminalNode and the TerminalNode comes first in
   the list, the cohort's `hook_post_run` and `BREAK_AT_JOB_RETIRED`
   handlers are skipped for everything after it. In practice
   TerminalNode usually retires alone (compute_time_micros=0,
   parents must be DONE), but if you set TerminalNode's compute time
   non-zero, be aware.

3. **Heap tie-break is by `uuid.uuid4()`** (`job/job.py:47`). Two
   jobs with the same ETA retire in non-deterministic order between
   runs. Sweeps measuring small differences should expect noise on
   the order of single-tick reorderings.

4. **`engine.submit` is FIFO, no priority, no reordering.** Once a
   job is in `job_waiting`, the only way it changes position is by
   moving to `job_running` (head only). A scheduler that needs
   priority semantics must implement its own ready queue and submit
   one job at a time. See *FIFO HOL blocking* above.

5. **`claim` / `release` are synchronous and bypass the queue.**
   They run inside the scheduler's call frame; they cannot be
   batched. The Chrome-trace log shows them as instant events with
   `ph: "i"`. They never appear in `job_waiting`/`job_running`.

6. **Hot reload covers the whole `sim.*` tree, including core.**
   `restart_simulation(reload=True)` reloads schedulers, hardware,
   loaders, base classes, AND `sim/core/...` (engine, system, job,
   trace, log, Debugger). The only spared modules are the live MCP
   daemon harness, listed in
   `agent_server.py:_HOT_RELOAD_SPARED_PREFIXES`:
   `sim.core.debug.agent_runner` and `sim.core.debug.agent_server` —
   plus `main_agent.py` itself. Edits to *those* need an agent-process
   restart; everything else takes effect on the next run. Reloading
   core together with hw/sched is what keeps shared class/enum
   identities consistent (see *Hot-reloading code* above).

7. **Initial-placement storage is the first `BaseStorage` in
   YAML order**, not the first hw of any type
   (`simulator.py:148-160`). All initial tensors land on
   `hardware.storage[0]`. Multi-storage configs cannot direct
   different tensors to different storages at load time — the
   scheduler must move them in the layout stage.

8. **`debug.args` is per-Debugger, not per-run.** Construction
   rebuilds the Debugger, so it's empty on every fresh run. Use
   `debug.record(...)` to land structured records in the log file
   if you want post-run persistence.

9. **TransferJob fixed-latency lives in a second heap.** A transfer
   whose source or destination is a `BaseStorage` with non-zero
   `fixed_latency_micros` retires in two phases: the bandwidth phase
   runs in `engine.job_running` (normal water-filling), then the job
   migrates to `engine.job_fixed_latency` for the latency window
   before final retire. The migrated job is invisible to
   `update_progress` and the bandwidth allocator during the window,
   so neither sees a "phantom" bandwidth consumer. If you walk job
   state during runtime, remember `job_fixed_latency` is a third
   queue alongside `job_waiting` and `job_running`.

10. **Logging level is a config knob and a perf knob.**
    `Level.COUNTER` and `Level.STATE` events fire **every runtime
    tick** for every hw with a running job (`engine.py:274-281`).
    A long-running trace at `log_level=3` (STATE) writes hundreds
    of MB of trace JSON. Default is `EVENT` (1).

## TransferJob fixed-latency: how the two phases retire

`Engine._runtime_forward` (`engine.py:239`) splits a TransferJob's
lifecycle into two heaps. While the bandwidth phase runs, the job
lives in `engine.job_running` alongside other running jobs and is
subject to the water-filling allocator. When bw work completes, the
job migrates to a separate heap, `engine.job_fixed_latency`, with
its `timestamp_ETA` rewritten to `now + fixed_latency_micros` and
its `work_done` pinned to `work_total`. It stays there until the
latency window elapses, then drains and retires normally:

```python
while self.job_running and (self.job_running[0].timestamp_ETA == self.timestamp_now):
    job = heapq.heappop(self.job_running)
    if isinstance(job, TransferJob) and job.fixed_latency_micros > 0.0:
        job.work_done = job.work_total
        job.timestamp_ETA = self.timestamp_now + job.fixed_latency_micros
        heapq.heappush(self.job_fixed_latency, job)
    else:
        job.end(self.log, self.sys, self.timestamp_now)
        retired_jobs.append(job)

while self.job_fixed_latency and (self.job_fixed_latency[0].timestamp_ETA == self.timestamp_now):
    job = heapq.heappop(self.job_fixed_latency)
    job.end(self.log, self.sys, self.timestamp_now)
    retired_jobs.append(job)
```

The "next event time" is the min of the two heap tops. The
fixed-latency phase therefore costs zero bandwidth (the job is
invisible to `update_transfer_jobs` while it's in
`job_fixed_latency`) but the underlying hw's `job_running` list
still contains the job, so `can_run` keeps returning False for
additional jobs on that hw until the latency window finishes and
`BaseJob.end → end_mutation` removes it.

Verified on 2026-05-27 with an MCP breakpoint test: a 1-page (4 KB)
SSD→RAM transfer at the SimpleSSD `read_io_curve_KBps` setting
`[4, 80000]` (= 0.08 KB/µs) retires in exactly 100 µs — 50 µs bw +
50 µs `fixed_latency_micros`.

The deadlock check in `_runtime` correctly considers both queues:
head-of-line blocking with `job_running` empty *and*
`job_fixed_latency` empty is a real deadlock; with either non-empty
the engine waits for the next event instead.

## Writing a new Scheduler (internals-side recipe)

The *operational* workflow (scaffolding, debugging, A/B against
known-good) is in *Workflow: Writing a new Scheduler* above. Here's
the contract-side recipe — what the engine actually expects from
your three methods.

### `__init__(self, obj_id, name, log, sys, args)`

- Call `super().__init__(obj_id, name, log, sys, args)`. Don't
  shadow `self.args` or `self.sys`.
- Walk `sys.hw.values()`. By convention, you'll want references to
  one or more `BaseCompute`, `BaseMemory`, `BaseStorage` instances.
  Convention is "use `isinstance(hw, BaseCompute)`" rather than
  string-matching on hw type.
- You can read `sys.trace.node_map`, `tensor_map`, `args` here.
- You **cannot submit jobs** here (the engine hasn't started). If
  you need to set up state from the trace, do it here; if you need
  to submit jobs, do it in `compile` or `layout`.

### `compile(self, trace)`

- One call, before layout. Mutate the trace as you like: insert
  control edges (`add_parent_node` / `add_child_node`), attach
  hooks (`node.hook_pre_run`, `node.hook_post_run`), tag NodeHW,
  pre-compute scheduler-private state.
- **Do not submit jobs from compile.** The engine doesn't drive
  retirement here; submitted jobs would sit in `job_waiting`
  unprocessed until runtime, at which point any FIFO assumptions
  break.

### `layout(self, init_storage) -> bool`

- Iterated by the engine. Return `True` when initial placement is
  complete; return `False` to be called again after the current
  batch of submitted transfers retires.
- **Only `TransferJob` may be submitted.** Use `sys.transfer(batch)`.
  Mix of `sys.claim` and `sys.transfer` is fine — claims are
  synchronous, transfers queue.
- Every submitted transfer must be immediately runnable
  (src is_ready+is_latest+IDLE-or-BEING_READ, dest IDLE). The
  layout drain loop has no retry; if a job is not runnable, it
  aborts with "Deadlock detected."
- All retires happen at the same `timestamp_now` — transfers are
  instantaneous in layout. Don't try to model SSD wall-clock cost
  here; do it in runtime.
- Multi-phase layouts (claim home regions → SSD→DRAM → DRAM→VRAM)
  use a phase counter inside `self` and return False between phases.
  Engine drains, then comes back for the next phase. DAV is the
  canonical example.

### `runtime(self, retired_jobs)`

- Called once per engine tick, after the tick's jobs have retired.
- `retired_jobs` includes both ComputeJob and TransferJob retires
  from this tick. Walk it to update per-job-id state, refcounts,
  prefetch trackers, etc.
- Submit new work: `sys.compute(...)`, `sys.transfer(...)`,
  `sys.claim/release(...)` synchronously. Order of submission
  becomes the FIFO order in `job_waiting` — see *FIFO HOL blocking*.
- The engine WILL re-call you next tick. Don't try to drain everything
  in one tick; submit what's runnable now and let the engine pace.
- If you detect an unrecoverable condition, call
  `sys.abort({"from": self.name, "msg": "<reason>", ...})` and
  `return`. The engine will tear down gracefully and the agent
  lands at `break_on_abort` with your dict in `abort_args`.

### Patterns to copy

- **DAG walking with refcounts**: `pending_parent_count`, `ready_node_ids`,
  decrement on retire, enqueue when count hits zero. See
  `device_aware_vanilla_async.py:143-150` and the retire loop.
- **Prefetch slots with frozen layout**: pin some tensors, allocate
  slots for the rest, walk the prefetch pointer ahead of the
  compute pointer. See `llamacpp_flexinfer/flexinfer.py:layout` and
  the `dyn_slots` array.
- **Multi-phase layout for SSD-constrained traces**: phase counter
  in `self`, return False between phases. See
  `device_aware_vanilla_async.py:_layout_phase`.
- **Side-channel info from the loader**: pull from
  `sys.trace.args["<key>"]`. Don't invent a new field on Trace;
  the args dict is the supported extension point.

### Anti-patterns to avoid

- **Don't mutate Node parents/children inside `runtime`.** The
  engine reads `node.parent_nodes` for the control-dependency check.
  Mid-run mutation produces "deadlock detected" aborts that look
  like a graph error but are actually a scheduler bug. If you need
  per-tick gating beyond the static graph, use `custom_deps`.
- **Don't submit a transfer and a compute that depends on it
  back-to-back without checking the head.** The compute will
  HOL-block the queue if the transfer is still in flight. Either
  submit the compute on a *later* tick (after the transfer retires
  and shows up in `retired_jobs`), or attach a `custom_deps` predicate
  that gates on the tensor's residency.
- **Don't issue >1 ComputeJob to the same single-slot device per
  tick.** SimpleCPU/SimpleSSD admit one. The second job HOL-blocks
  everything queued after it. Interleave or cap commits per device
  per tick.
- **Don't re-claim a region without releasing the old one.** Memory
  occupancy is tracked region-by-region; the old region holds onto
  pages until `sys.release` is called. Peak-memory metrics will
  blow up.
- **Don't trust `node.compute_time_micros` blindly.** It may be 0
  (alias/dispatcher nodes after loader rewrites; TerminalNode), it
  may carry per-node probe-effect compensation (PytorchProfile
  loader), it may be hand-modeled. Treat it as the authoritative
  budget but don't *also* try to charge transfer cost into it.

## Writing a new Hardware model

Concrete hardware lives in `sim/hw/<type>/<impl>/`. `<type>` is one
of `compute`, `memory`, `storage`. Each impl is a Python package
with at least an `__init__.py` (re-exports the class via `__all__`)
and the source file.

```
sim/hw/compute/my_gpu/
  __init__.py     # __all__ = ["MyGPU"]; from .my_gpu import MyGPU
  my_gpu.py
```

Subclass the right base — `BaseCPU` / `BaseGPU` / `BaseNPU` /
`BaseMemory` / `BaseStorage`. The framework calls only two methods:

```python
def can_run(self, job: BaseJob) -> bool: ...
def max_work_rate(self) -> float: ...
```

Other methods you may want to override:

- `log_counters()` / `log_states()` — what shows up under the
  Counter / State tracks in the Chrome-trace JSON.
- `__init__` — pull args from the YAML's `hardware.<type>[i].args`
  dict. Compute hw also receives a `memory: BaseMemory` (the
  Simulator wires it from `args["memory"]`).

Once your `__all__` is set, the pkgutil aggregator picks it up
automatically; reference it from a YAML as
`hardware.compute[0].type: "MyGPU"`. No central registry to update.

### Common patterns

- **Concurrency cap**: read `args["max_concurrent_jobs"]` (cap N
  concurrent jobs). `SimpleGPU` and `SimpleVRAM` do this.
- **IO-size curve**: `read_io_curve_KBps` / `write_io_curve_KBps`
  for storage. `SimpleSSD._get_bandwidth_KBps` interpolates.
- **Tagged hardware for `TensorAtHWDep`**: set
  `args["custom_dep_tag"]: "<unique-string>"` in the YAML; the tag
  must be unique across all hw (Simulator validates this). Then a
  `TensorAtHWDep("...", tag)` on a Node gates on this hw.

### What the engine assumes

- `BaseHardware.run(job)` is called by mutation; you do not call it.
- `BaseHardware.retire(job)` is called by mutation; you do not call it.
- `BaseHardware.job_running` is the admission state. Read it from
  `can_run` and `max_work_rate`. Do not mutate it directly.
- `args` is preserved on the instance for cross-cutting framework
  reads (`custom_dep_tag`, etc.).

## Writing a new Trace Loader

Concrete loaders live in `sim/load/<impl>/`. Like hardware, just a
Python package with `__all__` set.

Subclass `TraceLoader` and implement two methods:

```python
def load(self) -> Trace: ...
def placement(self, trace: Trace, storage: BaseStorage) -> None: ...
```

`load()` parses your source files (whatever path/CSV/DOT/binary
format you have) and builds `node_map` + `tensor_map`. Return a
`Trace(self.id, self.name, self.log, node_map, tensor_map, args=...)`.

`placement()` runs after `load()` and is handed the *first* storage
device. Convention: claim a `StorageRegion` for every tensor that
exists at startup (`WEIGHT`, `INPUT`, `LEAF`, `KVCACHE`); set
`is_ready=True, is_latest=True` on each. Intermediates are not
placed — they're produced at runtime.

### What to set on each Node

| Field | Required? | Notes |
|---|---|---|
| `id` | yes | Unique int. Conventional to use `len(node_map)` at insert. |
| `name` | yes | Free-form. Schedulers may regex-match. |
| `compute_time_micros` | yes | Pure execution time. Use `0.0` for "free" nodes (aliases, terminal). |
| `parent_nodes` / `children_nodes` | yes | Control deps. Use `add_parent_node`/`add_child_node`. |
| `input_tensors` / `output_tensors` | yes | Data deps. Use `add_input_tensor` / `add_output_tensor`. |
| `hw` | optional | `NodeHW.CPU | NodeHW.GPU | NodeHW.NPU` by default. Beware the wrong-hw check has a hole (see gotchas). |
| `custom_deps` | optional | For exotic dependencies. List of `CustomDep`. |
| `hook_pre_run` / `hook_post_run` | optional | Callables for trace-time mutation. |
| `args["step"]` | convention | Per-step index for multi-step (decoding) traces. |
| `args` (other) | free-form | Loader/scheduler-specific hints. |

### What to set on each Tensor

| Field | Required? | Notes |
|---|---|---|
| `id` | yes | Unique int. |
| `name` | yes | Free-form. |
| `size_bytes` | yes | Used to compute `num_pages` (4 KB pages, 64 B aligned). |
| `args["tensor_type"]` | convention | One of `WEIGHT`, `INPUT`, `LEAF`, `KVCACHE`, `INTERMEDIATE`. |
| `args["device"]` | (PyTorch traces) | `"cpu"` or `"cuda:N"` — DAV scheduler routes by this. |
| `args` (other) | free-form | Loader/scheduler-specific hints. |

### Loader-side preprocessing patterns to know

- **Lifetime-aware storage aliasing** (PytorchProfile): tensors
  sharing `storage_id` with overlapping lifetimes are merged into a
  single cgsim tensor; sequential reincarnations stay separate (so
  peak-memory accounting reflects the caching allocator's slot
  reuse). See `pytorch_profile.py:_apply_storage_aliasing`.
- **Alias/dispatcher node annotation**: nodes that are pure views
  or cross-device dispatchers get `custom_deps=[NodeDoneDep(p) for
  p in parents]` so the engine bypasses inappropriate residency
  checks. See `pytorch_profile.py:_annotate_alias_dispatcher_deps`.
- **Start-gated edges**: kineto `submit` edges from a `submit`-role
  node into a `gpu_runtime` kernel are pulled out of the control
  graph and stashed in `trace.args["start_gated_edges"]`. The
  scheduler enforces them as "parent must have STARTED" rather than
  DONE. See `pytorch_profile.py:_is_start_gated_edge`.
- **Implicit-input detection**: if a tensor has no real producer in
  the trace (or all producers are aliases/views), retype it as
  `INPUT` so it gets initial-placed at layout. See
  `pytorch_profile.py:_mark_implicit_inputs`.

The takeaway: loaders are not just "convert file format X to cg-sim
format" — they're also the right place to encode workload-domain
knowledge (cache-allocator behavior, async-launch semantics, view
aliasing) that the engine and scheduler shouldn't have to rediscover.

## Quick "where does X live" map

When tracking down behavior, these are the files to read first:

- **"Why didn't my scheduler dispatch?"** → `compute_assertion.py`
  (or `transfer_assertion.py`). The `is_runnable` decision.
- **"Why is `is_latest` False?"** → `compute_mutation.py:invalidate`.
  Some other compute wrote the same tensor.
- **"Why did the engine abort?"** → `engine.py:_log_abort`. Single
  choke point. Walk `abort_stack` to find the real caller.
- **"Why is the ETA weird?"** → `engine/update.py` (compute) and
  `engine/update_transfer.py` (water-filling). The latter is
  particularly subtle.
- **"What does the scheduler get at each stage?"** → `debug/debug.py:break_*_stage`.
  Each `_Symbol` declares what binds into the breakpoint REPL.
- **"What does the MCP server expose?"** → `debug/agent_server.py`.
  Tools forward to `Debugger.agent_*` methods.
- **"How is the YAML wired?"** → `sim/core/simulator.py`. Reads
  `cfg["logger"]`, `cfg["trace"]`, `cfg["hardware"]["storage"]`,
  `cfg["hardware"]["memory"]`, `cfg["hardware"]["compute"]`,
  `cfg["scheduler"]`. `+debug=on` adds `cfg["debug"]`.
