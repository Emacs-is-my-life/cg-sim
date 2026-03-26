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

## How to use?
```bash
git clone https://github.com/Emacs-is-my-life/cg-sim.git
cd cg-sim
uv pip install docs/requirements.txt
python main.py --input gogo --output gaga -o "It's not complete yet"
```

# Codebase Overview
## Simulator core
- `cg-sim/core/`: Base directory for simulator core
  - `cg-sim/core/log/`: In charge of logging simulator events and messages
  - `cg-sim/core/trace/`: Data structure that represents a workload(compute graph), which is a combination of Nodes and Tensors
  - `cg-sim/core/init/`: Initialization logic to import trace, intialize logger, hardwares and scheduler for simulation run 
  - 'cg-sim/core/job/`: Represents jobs. Scheduler requests job, hardware models calculate how much would it take, then engine simulates time advance
  - `cg-sim/core/engine/`: Core engine that interacts with hardware & scheduler, asserts actions they do, and processes discrete events
  
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
  - `cg-sim/sched/flexgen/`: Scheduler implementing FlexGen(https://dl.acm.org/doi/abs/10.5555/3618408.3619696) policy for memory saving
  
Implement your own scheduler logic in `cg-sim/sched/<scheduler-name>/`.

## Others
- `docs/`: More detailed documentation
- `examples/`: Example input
- `main.py`: Simulator entry point
