# RTX4090 Environment (e2e time, peak VRAM)
## PyTorch Mode
- Lazy (Inductor): Does compute graph optimizations, like kernel fusing
- Eager: Runs PyTorch in eager mode.

## Run Type
- Normal Run: Ordinary pytorch execution
- Profiled Run: Pytorch run with profiler(kineto, trace observer) attached. Causes large overhead.
- cg-sim Replay: Using pytorch traces obtained from Profiled Run, replay that trace in cg-sim

## Result
| PyTorch Mode    | Run Type      | llama-3-3B     | llama-3-8B    | sd3           | sdxl-turbo     |
|-----------------|---------------|----------------|---------------|---------------|----------------|
| Lazy (Inductor) | Normal Run    | 0.137s, 6.1 GB | 0.283s, 15 GB | 0.889s, 15 GB | 0.180s, 6.6 GB |
| Lazy (Inductor) | Profiled Run  | 0.587s, 6.2 GB | 0.673s, 15 GB | 1.658s, 17 GB | 0.798s, 8 GB   |
| Lazy (Inductor) | cg-sim Replay |                |               |               |                |
| Eager           | Normal Run    | 0.159s, 6.1 GB | 0.307s, 15 GB | 1.193s, 15 GB | 0.185s, 6.6 GB |
| Eager           | Profiled Run  | 1.16s, 6.2 GB  | 1.382s, 15 GB | 3.271s, 16 GB | 1.023s, 7 GB   |
| Eager           | cg-sim Replay |                |               |               |                |
