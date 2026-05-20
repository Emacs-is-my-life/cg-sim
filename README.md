# cg-sim
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
  - `sim/load/`: Write your own trace loader for your favorite ML framework
  - `sim/hw/`: Write your own hardware(compute, memory, storage) model
  - `sim/sched/`: Write your own scheduler policy

# How to use

## Installation
```bash
$ git clone https://github.com/Emacs-is-my-life/cg-sim.git
$ cd cg-sim
$ pip install --user -r docs/requirements.txt

# graphviz should be installed separately
# Use your system package manager: apt / yum / pacman / guix / ...

# Install cg-sim-mcp for agentic-run
$ claude mcp add cg-sim-mcp -- python main_agent.py
```

## Run cg-sim (For Human)
Simulator configs live in `examples/run/`; the heavy trace files they
consume live in `examples/trace/`. Each config references its trace
bundle by a relative path (`../trace/<trace_dir>/...`), so the two
directories travel together but can be swapped independently.

```bash
# python main.py -i <path-to-input.yaml>
$ python main.py -i examples/run/llamacpp_llama-3-8B_flexinfer.yaml           # Normal run
$ python main.py -i examples/run/llamacpp_llama-3-8B_flexinfer.yaml +debug=on # Debugging mode
```

Currently shipped configs under `examples/run/`:
- `llamacpp_llama-3-8B_example-cpu.yaml` — llama.cpp CPU trace, `ExampleCPU` scheduler
- `llamacpp_llama-3-8B_vanilla.yaml` — llama.cpp CPU trace, `Vanilla` scheduler
- `llamacpp_llama-3-8B_flexinfer.yaml` — llama.cpp CPU trace, `FlexInfer` scheduler
- `pytorch_eager_llama-3-8B_vanilla.yaml` — PyTorch eager GPU trace (Llama-3 8B), `DeviceAwareVanillaAsync`
- `pytorch_eager_sdxl-turbo_vanilla.yaml` — PyTorch eager GPU trace (SDXL-turbo), `DeviceAwareVanillaAsync`
- `pytorch_lazy_llama-3-8B_vanilla.yaml` — PyTorch lazy (Inductor) GPU trace (Llama-3 8B), `DeviceAwareVanillaAsync`
- `pytorch_lazy_sdxl-turbo_vanilla.yaml` — PyTorch lazy (Inductor) GPU trace (SDXL-turbo), `DeviceAwareVanillaAsync`

### Overriding input.yaml config from the command line
`main.py` parses extra positional args with [Hydra](https://hydra.cc/)'s override
syntax, so any leaf in the input YAML can be overridden without editing the file.
Dotted paths address nested keys; integer indices address list elements.

```bash
# Override a scalar
$ python main.py -i examples/run/llamacpp_llama-3-8B_flexinfer.yaml \
      scheduler.args.prefetch_window=8

# Override a list element by index (hardware.memory is a list)
$ python main.py -i examples/run/llamacpp_llama-3-8B_flexinfer.yaml \
      hardware.memory.0.args.memory_size_KB=10485760
```

Useful for parameter sweeps from a shell loop — `scripts/sim_run/flexinfer.sh`
does exactly this, sweeping `hardware.memory.0.args.memory_size_KB` and
redirecting each run's output via `logger.args.result_path`:
```bash
$ python main.py -i "$INPUT_CFG" \
      logger.args.result_path="${result}" \
      hardware.memory.0.args.memory_size_KB="${kb}"
```

The `+debug=on` form above is the same mechanism (the leading `+` adds a key
that doesn't exist in the YAML); see Hydra's
[override grammar](https://hydra.cc/docs/advanced/override_grammar/basic/) for
the full syntax (append `+`, force-override `++`, delete `~`, etc.).

### Debugging
Append `+debug=on` flag at the end, when running `main.py`  
You can set breakpoints at various points of simulator lifecycle  
IPython REPL session will launch for interactive inspection & manipulation  

At every breakpoint, the banner table lists the available commands:
- `debug.help()` — re-print this breakpoint's context (banner + variables + tip).
- `debug.record(dict)` — write a JSON-serializable dict into the simulation
  log under track Engine → Debug. Survives the run.
- `debug.args` — free-form scratchpad dict (`{}` at start of each run). Stash
  notes/findings here as you navigate. Cleared on simulator restart; for
  persistent records use `debug.record(...)`.
- `debug.break_lambda` — assign a `(engine, sys) -> bool` callable to fire
  `break_in_runtime_stage[LAMBDA]` whenever the predicate returns `True`
  (strictly). Evaluated once per runtime-loop tick (after retiring jobs,
  before progress update). Set to `None` to disable. Auto-clears on raise.
  Example: `debug.break_lambda = lambda engine, sys: engine.timestamp_now > 1_000_000`.
- `debug.log_path` — absolute Path of the simulation log file (where
  `debug.record()` writes). Handy for post-mortem inspection after the run.
- `exit()` — continue simulator execution.

`BREAK_ON_ABORT` and `BREAK_ON_EXCEPTION` are Off by default in human
mode. If you enable either, see the *Abort breakpoint* and *Exception
breakpoint* sections in [AGENTS.md](AGENTS.md) — the variable names
(`abort_stack`, `exception_stack`) and frame-inspection idioms
(`abort_stack[i].frame.f_locals`, `exception_stack[i].tb_frame.f_locals`)
apply identically inside the IPython REPL.

## Run cg-sim (Using Agent)
If you'd rather drive the simulator through an AI coding agent
(Claude Code, Codex, …) than the CLI, cg-sim ships an MCP (Model
Context Protocol) server (`main_agent.py`) that exposes the
simulator as an interactive breakpoint-debugging session the agent
navigates end-to-end — inspect state at any stage, sweep config
knobs across runs, hot-reload edited scheduler code, all driven by
chat.

### One-time setup
Register the MCP server with your agent. For Claude Code:
```bash
$ claude mcp add cg-sim-mcp -- python main_agent.py
```
(also covered under *Installation* above). Start a **new** agent
session after registering — tool lists are snapshotted at session
start. Substitute absolute paths for both `python` and the repo if
your agent launches the server from a different working directory.
For other agents, follow their MCP registration docs and point the
server command at `python main_agent.py` in this repo.

### What to ask the agent
Once registered, describe what you want in plain English; the agent
picks the right MCP tool calls. Example prompts:

- *"Sweep the FlexInfer scheduler on the llama-3 8B (llama.cpp)
  trace across 4 GB, 6 GB, 8 GB VRAM. Report peak memory and
  exposed-transfer stall per run."*
- *"At `break_after_compile_stage`, find node `Qcur-16` and break
  when its job retires. Show me what's in `job_running` at that
  moment."*
- *"My new scheduler in `sim/sched/myscheduler/` aborts at runtime
  with 'Deadlock detected' — walk the abort stack and tell me
  which dependency is unmet."*
- *"A/B `Vanilla` against `FlexInfer` on the llama-3 8B (llama.cpp)
  trace with the same hardware config. Summarize the wall-time
  speedup and where it comes from."*

The agent's full tool surface and workflow recipes live in
[AGENTS.md](AGENTS.md) — useful as a reference if you want to
suggest specific breakpoints or commands, or to learn what's
possible.

## Run cg-sim-mcp (For Agent)
**Agents reading this README:** the operational reference for
driving cg-sim via MCP — tool surface, breakpoint catalog, per-Node
hooks, abort/exception handling, config overrides, hot-reload
semantics, and a worked scheduler-authoring workflow — lives in
[AGENTS.md](AGENTS.md). Start there.

# Analyzing simulation results

Every `main.py` / `main_agent.py` run writes a Chrome-trace JSON event
log to `logger.args.result_path` (also reachable as `debug.log_path`
at any breakpoint). Post-run analysis scripts live in
`scripts/analysis/` and consume that log. Plotting recipes that pair
with their output live in `scripts/visualization/`.

### Convention for `scripts/analysis/*.py`
Every script in this directory follows the same shape so it can be
invoked uniformly — by humans on the CLI, by agents via `Bash`, or by
other Python code via `import`:

- **Signature:** `def main(log_path: Path, *script_specific_args) -> None`
  — first positional argument is the Chrome-trace JSON path; any
  script-specific knobs (hardware names, thresholds, …) follow as
  additional positional arguments with parsed Python types (`str`,
  `int`, …). The function prints human-readable results to stdout and
  returns `None`.
- **CLI:** `python scripts/analysis/<script>.py <log_path> [extra args...]`
  — the `if __name__ == "__main__":` block parses `sys.argv` into the
  `main()` signature, prints a usage line and `sys.exit(2)`s on misuse.
- **Importable:** because `main()` takes parsed Python values rather
  than `argv`, agent code can `from parse_stall_time import main` and
  call `main(Path(result_path), "cpu", "ram")` directly after a run
  finishes, with no subprocess hop.
- **Optional structured output (for downstream plotting):** Scripts that
  produce tabular results MAY also write CSVs + a `meta.json` to a
  directory, so that `scripts/visualization/` recipes can replot
  without re-parsing the log.
  - Signature gains a keyword-only `out_dir: Path | None = None`. When
    `None`, write nothing (stdout summary still prints).
  - CLI exposes it as `--out DIR`. A bare `--out` (no value) resolves
    to the default `tmp/analysis/<script_stem>/<log_path.stem>/`.
  - Tables are CSV with stable column names: time in microseconds with
    `_us` suffix (`ts_us`, `dur_us`, `end_us`, `begin_us`), bytes in
    `size_KB`, rates in `rate_KBps`, identifiers as `node_id`,
    `tensor_id`, `hw_name`/`src_name`/`dest_name`.
  - `meta.json` carries run config: `log_path`, hardware names,
    `runtime_start_us`, `runtime_span_us`, plus any script knobs.
  - Stdout summary is unchanged — disk output is a side-effect.

When adding a new analysis script, copy `parse_stall_time.py` as a
template and keep the same structure. Use the helpers in
`scripts/analysis/common/` so the column conventions are enforced
rather than re-stated:

  * `common.events` — `load_events`, `find_runtime_start`,
    `parse_compute_jobs`, `parse_transfer_jobs`, dataclasses
  * `common.intervals` — `merge_intervals`, `union_length`, `percentile`
  * `common.io` — `default_out_dir`, `write_meta`, `write_table`,
    `parse_out_flag`

### Convention for `scripts/visualization/*.py`
Visualization scripts consume the structured output of an analysis
script (or a sweep) and render a figure. They follow the analysis
convention with two substitutions: `log_path` → `in_dir`, stdout →
figure file.

- **Signature:** `def main(in_dir: Path, *script_specific_args, out_path: Path | None = None) -> None`.
  `in_dir` is the analysis script's `out_dir` *or* a sweep dir
  containing `summary.csv`. If `out_path` is `None`, the script
  writes to `<in_dir>/<script_stem>.png` (or `.html` for Plotly).
- **CLI:** `python scripts/visualization/<script>.py <in_dir> [extra args...] [--out PATH]`.
- **Importable:** same rationale — Python callers pass parsed values.
- **Shared helpers** live in `scripts/visualization/common/`:
  * `common.style` — Okabe-Ito palette and paper-friendly matplotlib defaults
  * `common.io` — `read_meta`, `read_table`, `load_summary`,
    `parse_out_path_flag`, `default_viz_out_path`
- **Library choice:** matplotlib for paper figures (line, CDF, stacked
  bar, heatmap), Plotly for interactive timelines / Gantt views.

### Convention for `scripts/experiments/*.py`
Experiment runners produce a config matrix, run cg-sim per cell, call
analysis helpers per cell, and aggregate metrics into a sweep
directory.

- **Output layout:**
  ```
  tmp/sweeps/<experiment_name>/
      summary.csv                # one row per cell, joined metrics
      <cell_label>/
          result_log.json
          stdout.log, stderr.log, command.txt
          prefetch_quality/      # analysis --out side-output
          link_utilization/
          reuse_distance/
  ```
- **Shared driver** is `scripts/experiments/sweep/`:
  * `Cell(label, params, overrides)` describes a single run.
  * `run_cell(base_yaml, cell, sweep_dir)` invokes `python main.py`
    with the Hydra overrides + a forced `logger.args.result_path`
    pointing into the cell directory.
  * `collect_metrics(log_path, cell_dir, compute_hw, memory_hw, ...)`
    runs the analysis scripts on the log and returns aggregate metrics
    extracted from their `meta.json`s.
  * `write_summary(sweep_dir, results, param_keys)` builds
    `summary.csv` from the metric union.
- **Visualization pairing:** `summary.csv` is the input contract for
  `plot_metric_vs_param.py` and `plot_time_breakdown.py`.

### Current scripts
- `parse_stall_time.py <log.json> <compute_hw_name> <memory_hw_name>`
  — exposed-transfer stall via interval-set arithmetic:
  `|union(TRANSFER_JOB intervals into memory_hw) \ union(COMPUTE_JOB
  intervals on compute_hw)|`, restricted to runtime-stage events.
  This is the prefetcher-quality metric — wall-time speedup of a
  memory-saving scheduler comes from shrinking *this*, not from
  shrinking raw `transfer_total_time`. Example:
  ```
  python scripts/analysis/parse_stall_time.py tmp/flexinfer_result.json cpu ram
  ```

# Codebase Overview
## Simulator core
- `sim/core/`: Base directory for simulator core
  - `sim/core/log/`: In charge of logging simulator events and messages
  - `sim/core/trace/`: Data structure that represents a workload(compute graph), which is a combination of Nodes and Tensors
  - `sim/core/init/`: Initialization logic to import trace, intialize logger, hardwares and scheduler for simulation run 
  - `sim/core/job/`: Represents jobs. Scheduler requests job, hardware models calculate how much would it take, then engine simulates time advance
  - `sim/core/engine/`: Core engine that interacts with hardware & scheduler, asserts actions they do, and processes discrete events
  - `sim/core/debug/`: Debugging infrastructure
  
## Trace Importer
- `sim/load/`: Base directory for trace importers
  - `sim/load/llamacpp/`: llama.cpp trace importer code

## Hardware models
- `sim/hw/`: Base directory for hardware models
  - `sim/hw/common/`: Common component and logic for hardwares
  - `sim/hw/compute/`: Computation hardware like CPU, GPU or NPU
  - `sim/hw/memory/`: Memory unit like VRAM, RAM
  - `sim/hw/storage/`: Storage unit like SSD
  
Implement your own hardware model in `sim/hw/<hardware-type>/<hardware-name>/`.

## Scheduler
- `sim/sched/`: Base directory for scheduler, orchestrating hardwares for trace execution
  - `sim/sched/common/`: Common component and logic for schedulers
  - `sim/sched/vanilla/`: No-offload policy, which keeps all tensors in memory, gives up if its impossible.
  - `sim/sched/flexinfer/`: Scheduler implementing FlexInfer(https://dl.acm.org/doi/10.1145/3721146.3721961) policy for memory saving
  - And more to come...
  
Implement your own scheduler logic in `sim/sched/<scheduler-name>/`.

## Others
- `docs/`: More detailed documentation
- `examples/`: Example inputs, split into two sibling subdirectories
  so configs stay light while bulky traces can be swapped or omitted
  independently:
  - `examples/run/`: Simulator config YAMLs (one per
    framework × workload × scheduler combo). Each YAML's trace
    fields use `../trace/<trace_dir>/...` relative paths, so the
    pair is portable as long as `run/` and `trace/` remain siblings.
    Invoke with `python main.py -i examples/run/<config>.yaml`.
  - `examples/trace/`: Heavy trace bundles consumed by configs in
    `examples/run/`. One subdirectory per workload trace (e.g.
    `llamacpp_CPU_llama-3-8B-Q8/`,
    `pytorch_eager_GPU-llama-3-8B/llama_bundle/`,
    `pytorch_lazy_GPU-sdxl-turbo/llama_bundle/`). PyTorch bundles'
    `manifest.json` references its own siblings (`runtime_*.csv`,
    `step_*_compute_graph.dot`) by bare filenames — they resolve
    relative to the manifest, so the bundle directory is self-contained.
- `scripts/`: Helper scripts, organized by purpose:
  - `scripts/sim_run/`: Bash drivers that launch `main.py` under various
    configs (e.g. `flexinfer.sh` sweeps memory sizes).
  - `scripts/sim_test/`: MCP / debugger tests that drive `main_agent.py`
    end-to-end (`test_mcp_*.py`).
  - `scripts/analysis/`: Post-run log analysis (e.g. `parse_stall_time.py`,
    `prefetch_quality.py`, `link_utilization.py`, `reuse_distance.py`,
    `schedule_diff.py`). Shared helpers in `common/`.
  - `scripts/experiments/`: Sweep runners (`sweep_memory.py`,
    `compare_schedulers.py`). Shared subprocess driver in `sweep/`.
  - `scripts/visualization/`: Plotting recipes for the analysis output
    (`plot_metric_vs_param.py`, `plot_time_breakdown.py`, `plot_cdf.py`,
    `plot_miss_curve.py`, `plot_timeline.py`, `plot_results.gp`). Shared
    style/io in `common/`.
- `main.py`: Simulator entry point
