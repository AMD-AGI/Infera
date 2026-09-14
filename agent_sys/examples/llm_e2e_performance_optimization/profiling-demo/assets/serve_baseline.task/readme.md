# serve_baseline

Bring GLM-5.3-Flash up in MIX (aggregated) mode on one 8×MI355X node with decode
CUDA graphs ON and the torch-profiler control plane OFF, prove it serves, and
hand back a `reproducible` record of the deployment.

A program body, not an AI one. Every step is a fixed command whose output has to
be byte-comparable between rounds for the measurements to mean anything.

## What it runs

`entry.sh` sets the round (`baseline`, graphs on, profiling off) and calls
`assets/serve/round.sh`, which runs on the login node and reaches the compute
node through `srun --overlap` into an already-running Slurm allocation. The
scripts it drives there — `mix_up.sh`, `mix_worker.sh`, `mix_smoke.sh`,
`reset_gpus.sh` — are the ones the manual walk-through validated on 2026-08-31,
carried across unmodified. `round.sh` maps the package's `PD_*` variables onto
the names those scripts read rather than rewriting them, so what runs here is
what was tested.

## The engine recipe is not a tuning surface

`mix_worker.sh` carries AMD's correctness-validated recipe for this model
(cookbook #36608): TP 8, DSA tilelang backends, BF16 KV cache, triton MoE runner,
`SGLANG_USE_AITER=1`, and no speculative decoding — MTP/EAGLE is explicitly not
validated for GLM-5.3-Flash on ROCm. The only knobs this package turns are the
two that define a round.

## Three things that differ from the reference scripts

Each of these was found by running the reference version and watching it fail or
do damage, and each is documented at its site:

- **etcd is on 12379.** These nodes run a Kubernetes control plane whose own etcd
  holds 2379 over TLS on both localhost and the node IP. `mix_up.sh` port-checks
  before it starts anything.
- **`reset_gpus.sh` does not kill everything holding a KFD handle.** On a Slurm
  GPU node that set includes `slurmstepd`, which holds one for the step's cgroup;
  the reference version killed one on its first run. The gate is now VRAM
  returning to baseline, and only plausible engine leftovers are killed.
- **The trace output directory is created host-side.** The engine container runs
  as root, so a directory it creates on the bind mount cannot later be written by
  the user running the analysis.

## Cost

A cold start reads 306 GB of FP8 weights off NFS at about 921 MB/s — measured 819
seconds to a serving health endpoint. A second bring-up in the same session is
served from page cache and took 243 seconds, because the node has 3 TB of RAM.
Budget a quarter of an hour, and do not read silence as a hang.
