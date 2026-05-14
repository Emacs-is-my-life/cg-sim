# Introduction
## What is cg-sim?
**cg-sim**(Compute Graph Simulator) is a testbed for testing various data(weight, intermediate, KV cache, ...) placement strategies for ML workload.  
This simulator has following characteristics:  

- Input:
  - Compute graph trace
  - Hardware configuration
  - Scheduler policy
- Output:
  - Performance statatistics like execution time, peak memory usage, etc
  - Detailed execution log in Chrome tracing format for pipeline analysis
- Discrete event system simulator for fast speed
- Batteries included but replacable:
  - `cg-sim/load/`: Write your own trace loader for your favorite ML framework
  - `cg-sim/hw/`: Write your own hardware(compute, memory, storage) model
  - `cg-sim/sched/`: Write your own scheduler policy

## How to use
### Installation
```bash
git clone https://github.com/Emacs-is-my-life/cg-sim.git
cd cg-sim
uv pip install docs/requirements.txt

# graphviz should be installed separately
# Use your system package manager: apt / yum / pacman / guix / ...
```

### Run a Simulation
```bash
# python main.py -i <path-to-input.yaml>
python main.py -i examples/llama3-flexinfer/input.yaml           # Normal run
python main.py -i examples/llama3-flexinfer/input.yaml +debug=on # Debugging mode
```

### Debugging
#### For Human
Append `+debug=on` flag at the end, when running `main.py`  
You can set breakpoints at various points of simulator lifecycle  
IPython REPL session will launch for interactive inspection & manipulation  

#### For Agent (LLM)
Run `main_agent.py` instead of `main.py`. It boots an MCP (Model Context
Protocol) server on its stdio and drives a loop so an agent can run the
simulator repeatedly — inspect state, resume execution, then `restart_simulation`
for a fresh run (optionally with a new input YAML), all via tool calls and
without restarting the process.

##### One-time setup
Register cg-sim as an MCP server with your agent. For Claude Code:
```bash
claude mcp add cg-sim-debugger -- \
    python main_agent.py -i <path-to-input.yaml>
```
The `-i` path supplies the default input; the agent can switch to a different
YAML on any subsequent run via `restart_simulation(input_path=...)`.
Substitute absolute paths for both `python` and the input file if your agent
launches the server from a different working directory. Start a new agent
session after registering — tool lists are snapshotted at session start.

##### Tool surface
The server exposes eight tools:
- `list_breakpoints` — return all `BREAK_*` flags and their On/Off status.
- `toggle_breakpoint(name)` — flip a flag.
- `start_simulation` — release the simulator (it blocks at startup so the agent
  can configure breakpoints first). **Blocking**: returns once the first
  breakpoint fires or the run finishes, with the resulting state in the
  response.
- `current_state` — re-read the current state without releasing the worker.
  Only needed if `start_simulation` / `continue_simulation` returns
  `timed_out=True`.
- `execute(code)` — run Python against the parked breakpoint's namespace.
  A bare last expression has its value echoed (REPL-style); the namespace
  persists across calls within one breakpoint and exposes the variables
  advertised by `current_state` plus `debug` (the Debugger).
- `continue_simulation` — resume the simulator. **Blocking**: returns once
  the next breakpoint fires or the run finishes, with the resulting state
  in the response.
- `restart_simulation(input_path=None)` — tear down the just-finished simulator
  and build a fresh one. Only callable after `simulation_finished=true` (or
  before the first run). Pass `input_path` to switch the YAML config.
  **Blocking**: returns once the new Simulator is constructed.
- `shutdown` — end the agent session and exit the process. If the simulator is
  parked at a breakpoint, releases it first so the current run drains cleanly.

##### Typical session
1. `list_breakpoints` → see available flags.
2. `toggle_breakpoint("BREAK_BEFORE_COMPILE_STAGE")` → enable the ones you want.
3. `start_simulation` → blocks; response carries the new state.
4. `execute("trace.node_map")` (or any other inspection/mutation).
5. `continue_simulation` → blocks; response carries the new state.
6. Repeat 4–5 until the response reports `simulation_finished=true`.
7. `restart_simulation()` (optionally with a new `input_path`) to go again
   from step 1, or `shutdown` to exit the process.

##### Per-Node / per-Job breakpoints
The five `BREAK_*` stage flags from `list_breakpoints` are coarse — they fire
once per simulator stage. To break on a *specific* compute graph node or job
during the runtime stage, set a `BREAK_AT_JOB_*` flag on the Node or Job
object itself via `execute`. Typical flow: enable `BREAK_AFTER_COMPILE_STAGE`,
inspect `trace.node_map` to find the node you care about, arm the flag, then
`continue_simulation` — the run will stop again when that node's job hits the
chosen lifecycle point.

Available flags (settable on any `Node` or `BaseJob`):
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

##### Environment knobs
- `CG_SIM_BREAKPOINTS` — comma-separated `BREAK_*` flag names to pre-enable
  before the server starts (alternative to calling `toggle_breakpoint`).

# Codebase Overview
## Simulator core
- `cg-sim/core/`: Base directory for simulator core
  - `cg-sim/core/log/`: In charge of logging simulator events and messages
  - `cg-sim/core/trace/`: Data structure that represents a workload(compute graph), which is a combination of Nodes and Tensors
  - `cg-sim/core/init/`: Initialization logic to import trace, intialize logger, hardwares and scheduler for simulation run 
  - `cg-sim/core/job/`: Represents jobs. Scheduler requests job, hardware models calculate how much would it take, then engine simulates time advance
  - `cg-sim/core/engine/`: Core engine that interacts with hardware & scheduler, asserts actions they do, and processes discrete events
  - `cg-sim/core/debug/`: Debugging infrastructure
  
## Trace Importer
- `cg-sim/load/`: Base directory for trace importers
  - `cg-sim/load/llamacpp/`: llama.cpp trace importer code

## Hardware models
- `cg-sim/hw/`: Base directory for hardware models
  - `cg-sim/hw/common/`: Common component and logic for hardwares
  - `cg-sim/hw/compute/`: Computation hardware like CPU, GPU or NPU
  - `cg-sim/hw/memory/`: Memory unit like VRAM, RAM
  - `cg-sim/hw/storage/`: Storage unit like SSD
  
Implement your own hardware model in `cg-sim/hw/<hardware-type>/<hardware-name>/`.

## Scheduler
- `cg-sim/sched/`: Base directory for scheduler, orchestrating hardwares for trace execution
  - `cg-sim/sched/common/`: Common component and logic for schedulers
  - `cg-sim/sched/vanilla/`: No-offload policy, which keeps all tensors in memory, gives up if its impossible.
  - `cg-sim/sched/flexinfer/`: Scheduler implementing FlexInfer(https://dl.acm.org/doi/10.1145/3721146.3721961) policy for memory saving
  - And more to come...
  
Implement your own scheduler logic in `cg-sim/sched/<scheduler-name>/`.

## Others
- `docs/`: More detailed documentation
- `examples/`: Example input
- `scripts/`: Scripts for running simulations, plotting graphs, etc
- `main.py`: Simulator entry point
