# verify_workset — run every driver on the target GPU

## Why this step exists

forge-loop treats `--driver` as a **protected file**: its optimizing agent may
not modify it. Every conclusion forge-loop reaches about correctness and speed
therefore rests on a driver this package wrote, and it has no way to check that
driver itself.

A driver that is subtly wrong does not fail loudly. It sends the optimization
somewhere plausible and useless, and the report at the end looks fine. So each
driver is executed here, on the hardware it will run on, before it is handed
over.

## What it does

Copies the worksets to the allocated node, then per operator, inside the serving
container:

```
python scripts/forge_driver.py                                  -> SNR / allclose
5 x  python scripts/forge_driver.py --warmup 10 --iters 20 --bench-mode
```

Five groups of twenty iterations is `temp/mission.md` 3.2.7 — "5次加权平均，每次
运行loop 10次以上取平均" — in the shape `assets/lib/bench_stats.py` computes.
That module is imported by this task and by `check_workset_runs`, so producer
and validator cannot disagree about what the weighted average means.

## What it needs

`--var jobid=<slurm job>` and `--var gpu_node=<hostname>`, neither of which has
a default. It reaches the node with `srun --overlap`, which joins the existing
allocation rather than queueing for a new one.

It does **not** need model weights, and it does not need the GLM-5.3 engine
image. A single operator is measured, not a served model, and torch, triton,
tilelang and aiter are all in the sglang base image already.

## Watch out

**One operator's failure does not end the task.** It is recorded against that
operator and the rest continue. `check_workset_runs` then decides whether enough
measured. Exiting on the first failure would throw away the evidence for the
others — and a program task's stdout is discarded on success, so anything worth
keeping has to be written into the handoff rather than printed.

**The measured time is not comparable to the profile's `avg_us`.** This is one
operator alone on the machine; the profile's figure is a serving-time average
over mixed batch sizes with everything else competing. `forge_task.yaml`'s
`targets.baseline_wall_ms` takes the number from here, not from the profile.
