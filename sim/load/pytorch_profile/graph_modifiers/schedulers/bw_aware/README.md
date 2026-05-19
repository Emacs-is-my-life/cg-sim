# `bw_aware`

`bw_aware` is a conservative admission scheduler that limits how much
H2D bandwidth the schedule is allowed to consume. It uses the same
per-use reload-job view as `jit_sim_prune`, but the key knob is not a
channel count. The key knob is a bandwidth target:

```text
Admit reloads only while the planned H2D traffic stays within a chosen
fraction of the iteration's transfer capacity.
```

This makes the scheduler useful when a purely deadline-based greedy
pass admits jobs that individually fit but collectively leave too
little bandwidth headroom for runtime jitter, launch anchoring, or
simulator details.

## Problem View

For every tensor use, the scheduler builds a reload job:

- the job can start after the tensor's previous use,
- it must finish before the next consumer,
- its duration is based on H2D latency and bandwidth,
- its value can be based on size, memory-time saved, or simply start
  order.

The scheduler then asks:

```text
Which reloads should be admitted if we only want to spend a controlled
fraction of the H2D lane over this iteration?
```

The answer is a subset of reload jobs. Admitted jobs allow evictions.
Dropped jobs cause the tensor to remain resident across that interval,
or to cold-start if the missing job was the first touch.

## Objective

`bw_aware` does not minimize peak VRAM explicitly. Its goal is to
control the aggressiveness of streaming:

- higher `bw_target` admits more reloads and usually lowers peak VRAM,
- lower `bw_target` admits fewer reloads and usually lowers stall risk.

You can think of `bw_target` as a throttle. It trades memory savings for
transfer slack.

## Admission Model

The scheduler computes an aggregate byte budget:

```text
budget_bytes = bw_target * h2d_bandwidth * iteration_wall_time
```

During admission, each accepted reload consumes part of this budget.
Once admitting another job would exceed the budget, the job is rejected.

The scheduler also applies a local FIFO deadline check. Even if there
is aggregate budget left, a job is not admitted unless it can finish
before its consumer in the modeled lane order.

These two tests play different roles:

- the global budget prevents overcommitting the whole iteration,
- the local deadline check prevents obviously late transfers.

## Ordering And Value

By default, jobs are processed in ALAP start order through the
`start_ns` value model. That makes the pass close to the time-ordered
behavior of `jit_sim_prune`.

Other value models change the admission priority:

- `tensor_size` keeps large tensors first,
- `dram_size` favors large memory-time savings,
- `dram_density` favors memory-time saved per transfer time,
- `uniform` treats all candidates equally.

When the budget is tight, this priority matters. It determines which
tensors get the limited streaming capacity and which tensors remain
resident.

## Relationship To `jit_sim_prune`

`jit_sim_prune` asks whether a job fits in a modeled channel schedule.
`bw_aware` asks whether admitting the job keeps the schedule under a
budgeted fraction of the H2D lane.

At `bw_target = 1.0`, the scheduler is most aggressive. As
`bw_target` decreases, it sheds reloads earlier and leaves more margin.
This makes it a useful base scheduler for `sim_loop`, which can search
for the most aggressive bandwidth target that still respects a measured
stall budget.

## Strengths

- Gives a simple continuous knob for streaming aggressiveness.
- Often more stable than admitting every locally feasible reload.
- Fast enough to sweep over many `bw_target` values.
- Can prioritize by time order or by memory value.

## Limitations

- The byte budget is global and coarse. It does not fully model every
  burst or anchoring effect.
- It is greedy and priority-dependent.
- It controls H2D pressure indirectly rather than solving for peak VRAM
  directly.
- A low target may keep too many tensors resident and leave memory
  savings on the table.

## Main Knobs

- `--bw-target`: fraction of H2D capacity the scheduler may spend.
- `--value-model`: candidate admission order.
- `--no-drop-infeasible`: keep jobs with negative individual slack for
  debugging.
- `--lock`: keep selected graph inputs resident.

## When To Use

Use `bw_aware` when you want a quick schedule with a clear
aggressiveness knob. It is especially useful when `jit_sim_prune` is
too eager and causes simulation stalls, or when you want `sim_loop` to
automatically search for a good memory-stall tradeoff.
