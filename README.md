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
$ python main.py -i examples/run/llamacpp__llama-3-8B__flexinfer.yaml           # Normal run
$ python main.py -i examples/run/llamacpp__llama-3-8B__flexinfer.yaml +debug=on # Debugging mode
```

Currently shipped configs under `examples/run/` (filename is
`<loader>__<workload>__<scheduler>.yaml`, double-underscore separated):
- `llamacpp__llama-3-8B__vanilla.yaml` — llama.cpp CPU trace, `LlamaCppVanilla` scheduler
- `llamacpp__llama-3-8B__flexinfer.yaml` — llama.cpp CPU trace, `LlamaCppFlexInfer` scheduler
- `pytorch-eager__llama-3-3B__vanilla.yaml` — PyTorch eager GPU trace (Llama-3 3B), `DeviceAwareVanillaAsync`
- `pytorch-eager__llama-3-8B__vanilla.yaml` — PyTorch eager GPU trace (Llama-3 8B), `DeviceAwareVanillaAsync`
- `pytorch-eager__sd3__vanilla.yaml` — PyTorch eager GPU trace (Stable Diffusion 3), `DeviceAwareVanillaAsync`
- `pytorch-eager__sdxl-turbo__vanilla.yaml` — PyTorch eager GPU trace (SDXL-turbo), `DeviceAwareVanillaAsync`
- `pytorch-lazy__llama-3-3B__vanilla.yaml` — PyTorch lazy (Inductor) GPU trace (Llama-3 3B), `DeviceAwareVanillaAsync`
- `pytorch-lazy__llama-3-8B__vanilla.yaml` — PyTorch lazy (Inductor) GPU trace (Llama-3 8B), `DeviceAwareVanillaAsync`
- `pytorch-lazy__sd3__vanilla.yaml` — PyTorch lazy (Inductor) GPU trace (Stable Diffusion 3), `DeviceAwareVanillaAsync`
- `pytorch-lazy__sdxl-turbo__vanilla.yaml` — PyTorch lazy (Inductor) GPU trace (SDXL-turbo), `DeviceAwareVanillaAsync`

### Overriding input.yaml config from the command line
`main.py` parses extra positional args with [Hydra](https://hydra.cc/)'s override
syntax, so any leaf in the input YAML can be overridden without editing the file.
Dotted paths address nested keys; integer indices address list elements.

```bash
# Override a scalar
$ python main.py -i examples/run/llamacpp__llama-3-8B__flexinfer.yaml \
      scheduler.args.prefetch_window=8

# Override a list element by index (hardware.memory is a list)
$ python main.py -i examples/run/llamacpp__llama-3-8B__flexinfer.yaml \
      hardware.memory.0.args.memory_size_KB=10485760
```

Useful for parameter sweeps from a shell loop — `scripts/sim_run/flexinfer.sh`
does exactly this against the `llamacpp__llama-3-8B__flexinfer.yaml` config,
sweeping `hardware.memory.0.args.memory_size_KB` and redirecting each run's
output via `logger.args.result_path`:
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
knobs across runs, hot-reload edited code (schedulers, hardware, even framework core), all driven by
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

- *"Sweep the LlamaCppFlexInfer scheduler on the llama-3 8B (llama.cpp)
  trace across 4 GB, 6 GB, 8 GB VRAM. Report peak memory and
  exposed-transfer stall per run."*
- *"At `break_after_compile_stage`, find node `Qcur-16` and break
  when its job retires. Show me what's in `job_running` at that
  moment."*
- *"My new scheduler in `sim/sched/myscheduler/` aborts at runtime
  with 'Deadlock detected' — walk the abort stack and tell me
  which dependency is unmet."*
- *"A/B `LlamaCppVanilla` against `LlamaCppFlexInfer` on the llama-3 8B (llama.cpp)
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
  than `argv`, agent code can `from prefetch_quality import main` and
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

When adding a new analysis script, copy `link_utilization.py` as a
template (the simplest of the existing analyses with the full
`--out` + `meta.json` shape) and keep the same structure. Use the
helpers in `scripts/analysis/common/` so the column conventions are
enforced rather than re-stated:

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

- **Output layout:** see [AGENTS.md](AGENTS.md) "Output directory
  conventions" for the canonical spec. Sweeps land in
  `output/<experiment-setup>/` with `sim_results/`, `analysis/<run-id>/`,
  `plots/`, plus `summary.csv` and `experiment.yaml`.
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
- `prefetch_quality.py <log.json> <compute_hw> <memory_hw> [--out [DIR]]`
  — the centerpiece. Four views from one log pass: stall blame
  (which transfer caused each compute-gap), prefetch slack (how
  early each transfer arrived relative to its consumer), transfer
  phase decomposition (queue_wait / head_wait / xfer), and
  per-module rollup of attributed stall.
- `link_utilization.py <log.json> <memory_hw> [window_us] [--out [DIR]]`
  — bandwidth-side view: busy fraction, achieved aggregate rate,
  per-source breakdown for the offload link.
- `cpu_offload_timeline.py <log.json> [--filter weight|any] [--out DIR]`
  — per-tensor TRANSFER/RELEASE swimlane (CSV + interactive Plotly
  HTML) for validating accelerate/cpu_offload-style schedulers.
- `extract_sim_metrics.py <log.json> [...]` — one-line summary
  (`simulation_time_s`, `peak_vram0_GB`) per `SIMULATION_RESULT`
  event. Suitable for condensing a directory of sweep results into
  a quick table.
- `sim_summary.py <log.json>` — streaming summary that tolerates a
  truncated trace (e.g. a still-running or killed sweep), pulled
  via regex without loading the whole JSON.
- `trace_inspect/` — pre-sim trace introspection helpers
  (`budget.py`, `cp_analysis.py`, `cpu_leaf_types.py`,
  `op_distribution.py`); these read trace bundles directly rather
  than cg-sim result logs.

## Calibrating PyTorch traces against profiler probe effect

When a PyTorch trace is recorded with kineto attached, every
`cpu_leaf` op's `duration_ns` is inflated by per-op observer overhead
plus workload-context effects (cache pollution, allocator pressure,
profiler buffer contention). Replaying such a trace inflates cg-sim
e2e wall time, most severely in eager mode where most cpu_leafs sit
on the simulator's critical path. The full thesis and evidence are in
`docs/eager-lazy-probing-effect.md` and
`docs/sim_real-run_comparison.md`.

Two scripts under `scripts/tool/` produce the calibration data the
loader uses to compensate:

- `scripts/tool/kineto_probe_microbench.py` — **runs on the PyTorch
  profiling host**, NOT the cg-sim simulation host. Measures per-call
  wall time for the top cpu_leaf ops with profiler off vs on. Writes
  `overhead_results.txt` next to itself. Copy that file back to the
  cg-sim repo (same directory) if running on a different machine.

- `scripts/tool/generate_probe_effect_tables.py` — **runs on the
  cg-sim host**. Reads `overhead_results.txt` and each trace's
  `runtime_nodes.csv`, computes
  `probe_effect_ns = trace_median(duration_ns) − microbench_probed_ns`
  per op_name, and writes `probe_effect_table.csv` to each trace
  directory (sibling of `llama_bundle/`).

When the `PytorchProfile` loader sees `probe_effect_table.csv` next
to a bundle, it subtracts `probe_effect_ns` from each matching
cpu_leaf's `duration_ns` at load time (clamped to ≥ 0). Missing file
= no correction, fully backward-compatible. To re-calibrate after
re-recording a trace or upgrading PyTorch/kineto, re-run the two
scripts above.
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
  - `sim/load/llamacpp/`: llama.cpp trace importer (CPU-only GGML
    profiling records → cg-sim Trace).
  - `sim/load/pytorch_profile/`: PyTorch profiler trace importer.
    Consumes the `llama_bundle/` files (`runtime_nodes.csv`,
    `runtime_edges.csv`, `pytorch_runtime_tensors.csv`,
    `module_hierarchy.json`, `manifest.json`,
    `step_0_compute_graph.dot`) emitted by the upstream
    `pytorch-source` profiler. Annotates each Node with its hosting
    `torch.nn.Module` path and the device the op ran on; applies
    kineto probe-effect compensation when a sibling
    `probe_effect_table.csv` is present (see *Calibrating PyTorch
    traces against profiler probe effect* above).

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
  - `sim/sched/llamacpp_vanilla/` (`LlamaCppVanilla`): No-offload policy for llama.cpp traces — keeps all tensors in memory, aborts if it doesn't fit.
  - `sim/sched/llamacpp_flexinfer/` (`LlamaCppFlexInfer`): Implements the FlexInfer (https://dl.acm.org/doi/10.1145/3721146.3721961) memory-saving policy for llama.cpp traces.
  - `sim/sched/device_aware_vanilla_async/` (`DeviceAwareVanillaAsync`): Vanilla scheduler for PyTorch-profile traces — routes each Node to the compute device its kineto record came from (`node.hw`), with async H2D transfer streams.
  - `sim/sched/generic_stub/` (`Stub`): No-op scheduler — spins up a simulation for inspection at the compile/layout breakpoints but aborts at runtime. Useful when the intended scheduler for a config is not yet implemented on this branch.
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
    `examples/run/`. One subdirectory per workload trace, named
    `<loader>__<workload>__<host>` (e.g.
    `llamacpp__llama-3-8B-Q8__datai/`,
    `pytorch-eager__llama-3-8B__RTX4090/llama_bundle/`,
    `pytorch-lazy__sdxl-turbo__RTX4090/llama_bundle/`,
    `accelerate-cpu-offload__llama-3-8B__RTX4090/llama_bundle/`).
    PyTorch bundles' `manifest.json` references its own siblings
    (`runtime_*.csv`, `pytorch_runtime_tensors.csv`,
    `module_hierarchy.json`, `step_*_compute_graph.dot`,
    `compiled_*` files for Inductor-mode bundles) by bare filenames
    — they resolve relative to the manifest, so the bundle
    directory is self-contained. A sibling `probe_effect_table.csv`
    next to `llama_bundle/`, if present, is consumed by the
    PyTorchProfile loader for kineto probe-effect compensation.
- `scripts/`: Helper scripts, organized by purpose:
  - `scripts/sim_run/`: Bash drivers that launch `main.py` under various
    configs (e.g. `flexinfer.sh` sweeps memory sizes; `example.sh`
    is a minimal single-run driver).
  - `scripts/sim_test/`: MCP / debugger tests that drive `main_agent.py`
    end-to-end (`test_mcp_debugger.py`, `test_mcp_breakonabort.py`,
    `test_mcp_breakonexception.py`, `test_mcp_breaklambda.py`,
    `test_mcp_hotreload.py`, `test_mcp_overrides.py`,
    `test_mcp_prerun_restart.py`, `test_mcp_fragile.py`).
  - `scripts/analysis/`: Post-run log analysis
    (`prefetch_quality.py`, `link_utilization.py`,
    `cpu_offload_timeline.py`, `extract_sim_metrics.py`,
    `sim_summary.py`). Shared helpers in `common/`. The
    `trace_inspect/` subdirectory holds pre-sim trace
    introspection tools (`budget.py`, `cp_analysis.py`,
    `cpu_leaf_types.py`, `op_distribution.py`) that read
    trace bundles directly rather than cg-sim result logs.
  - `scripts/experiments/`: Sweep runners (`sweep_memory.py`,
    `compare_schedulers.py`). Shared subprocess driver in `sweep/`.
  - `scripts/visualization/`: Plotting recipes for the analysis output
    (`plot_metric_vs_param.py`, `plot_time_breakdown.py`, `plot_cdf.py`,
    `plot_timeline.py`). Shared style/io in `common/`.
  - `scripts/tool/`: One-shot calibration utilities — see
    *Calibrating PyTorch traces against profiler probe effect*
    above (`kineto_probe_microbench.py`,
    `generate_probe_effect_tables.py`).
  - `scripts/sim_sweep_script.py`: Top-level sweep driver for the
    pytorch-source 0521 profiling results — runs vanilla and
    HF-offload-mode simulations and aggregates a metrics table.
- `main.py`: Simulator entry point (human / CLI use).
- `main_agent.py`: MCP server entry point (agent use); see
  [AGENTS.md](AGENTS.md).
