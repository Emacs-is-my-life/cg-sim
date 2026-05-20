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
      python main_agent.py -i examples/run/llamacpp_llama-3-8B_flexinfer.yaml
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
  full semantics. With `reload=True` (default), drops user-editable
  modules from `sys.modules` so source edits are picked up — see
  *Hot-reloading user code* below. **Blocking**: returns once the new
  Simulator is constructed.
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
FlexInfer still reserves the original slot, and the resulting stall
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
- scheduler-issued `sys.abort(...)` (Vanilla, FlexInfer, your scheduler)
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
    input_path="examples/run/llamacpp_llama-3-8B_vanilla.yaml",
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

## Hot-reloading user code

`restart_simulation(reload=True)` (the default) drops user-editable modules
from `sys.modules` before rebuilding the Simulator, so the agent can edit
source files on disk and pick up the changes on the next run without
restarting the process.

In scope (reloaded):
- `sim/sched/<impl>/...` — scheduler implementations.
- `sim/hw/<type>/<impl>/...` — compute, memory, storage models.
- `sim/load/<impl>/...` — trace loaders.

Out of scope (preserved across reload, by design):
- `sim/.../common/...` — base classes / ABCs. These keep their identity
  so `isinstance(...)` checks in framework code (e.g. `sim/core/engine/`)
  remain consistent against freshly-built instances.
- `sim/core/...` — framework code. Edits here will *not* take effect on
  reload; restart the agent process instead.

Mechanism: the `LOAD_*_CLASS` functions in `sim/core/init/` do live
`importlib.import_module` lookups, so the next call after `sys.modules`
invalidation re-executes each subtree's `__init__.py` (the pkgutil
aggregator) and the user-edited leaf files.

The response includes `reloaded_modules` (count of evicted `sys.modules`
entries) so the agent can confirm the reload happened. Pass `reload=False`
to compare back-to-back runs of identical code without paying the
re-import cost.

Example:
```python
# 1. Edit sim/sched/flexinfer/flexinfer.py on disk.
# 2. From the agent loop, after simulation_finished:
restart_simulation(reload=True)
# 3. The next start_simulation runs against the freshly-imported FlexInfer.
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
cp -r sim/sched/vanilla sim/sched/myscheduler
# In sim/sched/myscheduler/__init__.py: re-export your renamed class.
# In sim/sched/myscheduler/myscheduler.py: rename `class Vanilla(...)` →
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
user code* above for the spared `*.common.*` invariant.

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

The two reference implementations are `Vanilla` (no offload) and
`FlexInfer` (memory-saving). Switch input YAMLs mid-session:
```
restart_simulation(input_path="examples/run/llamacpp_llama-3-8B_vanilla.yaml")
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
  (`llamacpp_llama-3-8B_flexinfer` for
  `examples/run/llamacpp_llama-3-8B_flexinfer.yaml`).
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
