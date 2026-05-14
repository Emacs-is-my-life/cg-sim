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
python main.py -i <path-to-input.yaml>
```

### Debugging
#### For Human
Append `+debug=on` flag at the end, when running `main.py`  
You can set breakpoints at various points of simulator lifecycle  
IPython REPL session will launch for interactive inspection & manipulation  

#### For Agent (LLM)
Append `+debug_agent=on` flag at the end, when running `main.py`  
Instead of launching an IPython REPL, the simulator boots an MCP (Model Context
Protocol) server on its stdio. Any MCP-capable agent can then drive
breakpoints, inspect state, and resume execution via tool calls.

##### One-time setup
Register cg-sim as an MCP server with your agent. For Claude Code:
```bash
claude mcp add cg-sim-debugger -- \
    python main.py -i <path-to-input.yaml> +debug_agent=on
```
Substitute absolute paths for both `python` and the input file if your agent
launches the server from a different working directory. Start a new agent
session after registering — tool lists are snapshotted at session start.

##### Tool surface
The server exposes six tools:
- `list_breakpoints` — return all `BREAK_*` flags and their On/Off status.
- `toggle_breakpoint(name)` — flip a flag.
- `start_simulation` — release the simulator (it blocks at startup so the agent
  can configure breakpoints first).
- `current_state` — return the active breakpoint name, variable table, tip,
  and whether the simulation has finished. Poll this to detect a breakpoint.
- `execute(code)` — run Python against the parked breakpoint's namespace.
  A bare last expression has its value echoed (REPL-style); the namespace
  persists across calls within one breakpoint and exposes the variables
  advertised by `current_state` plus `debug` (the Debugger).
- `continue_simulation` — resume the simulator from the current breakpoint.

##### Typical session
1. `list_breakpoints` → see available flags.
2. `toggle_breakpoint("BREAK_AFTER_COMPILE_STAGE")` → enable the ones you want.
3. `start_simulation` → release the gate.
4. Poll `current_state` until `at_breakpoint=true`.
5. `execute("trace.node_map")` (or any other inspection/mutation).
6. `continue_simulation` → resume.
7. Repeat 4–6 until `current_state` reports `simulation_finished=true`.

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
