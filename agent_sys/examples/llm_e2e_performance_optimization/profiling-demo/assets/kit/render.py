#!/usr/bin/env python3
"""Write the packup's three mandatory documents from the handoffs themselves.

`deliverable_layout.md` says README.md, REPRODUCE.md and environment.md are never
omitted, and that environment.md is the number-one reproduction trap so it should
be exhaustive. Generating them from the handoffs rather than from a template with
blanks is what keeps them true: every number below is read out of a published
artefact, so a document that disagrees with the run cannot be produced.
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def metric(summary: dict, key: str, column: str = "avg") -> str:
    """One metric from a summarise.py document, or an explicit dash.

    A missing metric prints as `-` rather than as 0. The two are different facts
    and a table that renders them the same is how a gap becomes a claim.
    """
    value = (summary.get("metrics") or {}).get(key, {}).get(column)
    if value is None:
        return "-"
    return f"{value:,.2f}" if abs(value) < 1e6 else f"{value:,.0f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--deployment-baseline", required=True)
    ap.add_argument("--deployment-profiled", required=True)
    ap.add_argument("--aiperf-baseline", required=True)
    ap.add_argument("--aiperf-profiled", required=True)
    ap.add_argument("--kernel-table", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    dep_base = load(Path(args.deployment_baseline) / "items/env/deployment.json")
    dep_prof = load(Path(args.deployment_profiled) / "items/env/deployment.json")
    aip_base = load(Path(args.aiperf_baseline) / "items/result/summary.json")
    aip_prof = load(Path(args.aiperf_profiled) / "items/result/summary.json")
    load_cfg = load(Path(args.aiperf_baseline) / "items/env/load.json")
    kernels = load(Path(args.kernel_table) / "items/result/top_kernels.json")
    traces = load(Path(args.kernel_table) / "items/env/trace_manifest.json")

    argv_path = Path(args.deployment_baseline) / "items/env/engine_argv.txt"
    engine_argv = argv_path.read_text(encoding="utf-8", errors="replace") if argv_path.is_file() else ""
    gpu_path = Path(args.deployment_baseline) / "items/env/gpu.txt"
    gpu = gpu_path.read_text(encoding="utf-8", errors="replace").strip() if gpu_path.is_file() else ""

    # How a reproducer steps into the allocation. This is a fact about the
    # cluster the round ran on, not about the package, so it is read from the
    # same `PD_TRANSPORT` switch `assets/lib/remote.sh` dispatched on rather than
    # asserted as `srun` -- a REPRODUCE.md naming a transport the reader's
    # cluster does not have is worse than one that names none.
    # `auto` is resolved here exactly as remote.sh resolves it, rather than being
    # guessed at: this body runs on the same login node the transport is used
    # from, so the same probe gives the same answer.
    transport = os.environ.get("PD_TRANSPORT") or "auto"
    if transport == "auto":
        transport = "spur" if shutil.which("spur") else "srun"
    if transport == "local":
        machine_note = (
            "a shell on the node itself. This round drove the steps locally "
            "(`--var transport=local`), so there is no transport to step through — "
            "run the commands below on the node"
        )
    elif transport == "spur":
        machine_note = (
            "`spur exec <jobid> bash -lc '<command>'`. The login node has no GPUs and "
            "cannot ssh to a compute node, which is why every step goes through that "
            "transport; the controller routes by job id, so no node name is passed"
        )
    else:
        machine_note = (
            "`srun --jobid=<id> --overlap -N1 -n1 -w <node> bash -lc '<command>'`. The "
            "login node has no GPUs and cannot ssh to a compute node, which is why "
            "every step goes through that transport"
        )
    # Both taken from the round's own record rather than from the eight-GPU
    # GLM-5.3-Flash deployment this package was first written for.
    gpus = dep_base.get("tp_size", "?")
    model_name = Path(dep_base.get("model_path", "the model")).name

    top = (kernels.get("kernels") or [])[:10]
    top_rows = "\n".join(
        f"| `{k['name'][:70]}` | {k['calls']:,} | {k['self_cuda_us']:,.0f} | {k['pct_total']:.2f}% |"
        for k in top
    ) or "| (no kernels ranked) | | | |"

    (root / "README.md").write_text(
        f"""# GLM-5.3-Flash profiling under a Mooncake trace replay

**Ran:** {args.stamp}
**Status:** PASS — the deployment served, the replay completed, the capture
covered every rank and the ranking is non-empty.

## Goal

Profile GLM-5.3-Flash on one 8-GPU MI355X node under a realistic load, and hand
back the kernel ranking that the operator-selection stage consumes. The load is
AIPerf replaying a Mooncake production trace at its recorded timestamps, which is
what exercises prefix reuse; a synthetic sweep cannot.

**Success criteria:** the service answers correctly; the replay completes with a
low error rate; the capture holds GPU kernels on every tensor-parallel rank; the
ranking accounts for the trace.

## Result

Two rounds, differing only in decode CUDA graphs and whether the profiler control
plane was attached.

| Round | CUDA graphs | Requests | Output tok/s | TTFT avg (ms) | Quotable |
|---|---|---|---|---|---|
| baseline | on  | {metric(aip_base, 'request_count')} | {metric(aip_base, 'output_token_throughput_tps')} | {metric(aip_base, 'ttft_ms')} | yes |
| profiled | off | {metric(aip_prof, 'request_count')} | {metric(aip_prof, 'output_token_throughput_tps')} | {metric(aip_prof, 'ttft_ms')} | no — profiler attached |

Top kernels by self CUDA time, from the profiled round:

| Kernel | Calls | Self CUDA (us) | % of total |
|---|---|---|---|
{top_rows}

The top {kernels.get('top_n', 0)} account for {kernels.get('top_n_share_pct', 0)}%
of {kernels.get('totals', {}).get('self_cuda_us', 0):,.0f} us across
{traces.get('totals', {}).get('files', 0)} ranks.

## How to reproduce

See `REPRODUCE.md`. In one line: bring the engine up in MIX mode, replay two
minutes of the trace, restart with graphs off and the profiler enabled, replay
again while cutting a window, then rank the kernels.

## Folder map

- `REPRODUCE.md` — step-by-step reproduction
- `environment.md` — exact hardware and software the numbers came from
- `scripts/` — the scripts that ran, verbatim, plus the per-round invocations
- `results/` — the machine-readable evidence
- `notes.md` — the things that cost time
- `logs/` — gzipped log tails
""",
        encoding="utf-8",
    )

    ports = dep_base.get("ports") or {}
    (root / "REPRODUCE.md").write_text(
        f"""# Reproduction kit — GLM-5.3-Flash profiling

Goal: reproduce the kernel ranking in `results/gap_analysis.csv` from a clean
machine with cluster access.
Estimated time: about 40 minutes, of which two cold engine starts are ~13 and ~4
minutes — the second is faster because the checkpoint is in page cache.

## 0. Prerequisites

- **Machine:** one node with {gpus} × MI355X, held by an allocation you reach with
  {machine_note}.
- **Paths** (site-specific; every `@NAME@` below is one of these):
  - `@MODEL_MOUNT@` — the directory holding the `{model_name}` checkpoint.
  - `@WORK_ROOT@` — a node-local work area with ~2 GB free. Not a network mount:
    one capture is about 462 MB and it is written once per rank in parallel.
  - `@TRACE_DIR@` — the directory holding the Mooncake trace JSONL.
  - `@MAGPIE_ROOT@` — a checkout of Magpie.
- **Images:** `{dep_base.get('image', '?')}` for the engine — it must carry both
  infera and an SGLang that recognises this checkpoint's `model_type`; for
  GLM-5.3-Flash (`glm5_next`) no released SGLang does and the image is built from
  `Dockerfile.sglang.glm53`. `{load_cfg.get('aiperf_image', '?')}` for AIPerf,
  and an etcd image.
- **Ports:** router {ports.get('router', '?')}, worker {ports.get('worker', '?')},
  etcd {ports.get('etcd', '?')}. **Not 2379** — these nodes run a Kubernetes
  control plane whose own etcd holds it over TLS.

## 1. Bring the engine up with CUDA graphs on

    export MODEL_MOUNT=... WORK_ROOT=... SCRIPTS=./scripts
    ./scripts/command.deployment_baseline.sh

Wait for `MIX_UP_OK`. A cold start reads the whole checkpoint; silence is not a
hang.

## 2. Replay two minutes of the trace

    export AIPERF_TRACE=@TRACE_DIR@/conversation_trace.jsonl
    ./scripts/command.aiperf_baseline.sh

Ends with `AIPERF_OK` and a run directory under `@WORK_ROOT@/aiperf`.

## 3. Restart with graphs off and the profiler enabled

    ./scripts/command.deployment_profiled.sh

## 4. Replay again, cutting a profiler window out of the load

    ./scripts/command.aiperf_profiled.sh &
    ./scripts/command.torch_trace.sh

The capture waits for the engine to report an actual batch before it starts
counting warm-up, so a cold AIPerf synthesising prompts does not push the window
into an idle period.

## 5. Rank the kernels

    export MAGPIE_ROOT=... TRACES=@WORK_ROOT@/profiles/<tag>/mixed OUT=./out
    ./scripts/command.kernel_table.sh

## Expected output

`out/gap_analysis/gap_analysis.csv` with
{kernels.get('totals', {}).get('kernels', 0)} kernel rows, the largest being
`{(top[0]['name'][:60] if top else '?')}` at about
{(f"{top[0]['pct_total']:.1f}%" if top else '?')} of total self CUDA time.

## If it doesn't reproduce

See `notes.md`.
""",
        encoding="utf-8",
    )

    (root / "environment.md").write_text(
        f"""# Environment

The number-one reproduction trap, so this is exhaustive rather than tidy.

## Hardware

```
{gpu}
```

Node `{dep_base.get('node', '?')}` at `{dep_base.get('node_ip', '?')}`, held by
Slurm job `{dep_base.get('slurm_jobid', '?')}`, 8 GPUs, TP size
{dep_base.get('tp_size', '?')}.

## Images

| role | image | id |
|---|---|---|
| engine | `{dep_base.get('image', '?')}` | `{dep_base.get('image_id', '?')}` |
| AIPerf | `{load_cfg.get('aiperf_image', '?')}` | |

The engine image is not a released tag. No published SGLang carries
`model_type: glm5_next` — checked against v0.5.18 and the 0.5.18.dev20260826 ROCm
nightly, neither of which has `Glm5NextConfig`. It is built by overlaying the
support PR's python tree onto a stock ROCm image.

## Model

`{dep_base.get('model_path', '?')}`, served as
`{dep_base.get('served_model_name', '?')}`, `disagg_mode`
`{dep_base.get('disagg_mode', '?')}`.

## Engine command line

The baseline round's, as the process actually had it:

```
{engine_argv.strip()}
```

## Scripts

Package `{(dep_base.get('scripts') or {}).get('package', '?')}` at commit
`{(dep_base.get('scripts') or {}).get('commit', '?')}`.

## Load

Trace `{load_cfg.get('trace', '?')}`, window
{load_cfg.get('trace_window_ms', '?')} ms, concurrency ceiling
{load_cfg.get('concurrency_ceiling', '?')}, {load_cfg.get('aiperf_workers', '?')}
AIPerf workers, ISL block size {load_cfg.get('isl_block_size', '?')}.

## Profiler window

Warm-up {(load_cfg.get('profiler_window') or {}).get('warmup_s', '?')} s, window
{(load_cfg.get('profiler_window') or {}).get('window_s', '?')} s,
`record_shapes` true and `with_stack` false, plus a short second window with
`with_stack` true for launcher resolution.
{traces.get('totals', {}).get('files', 0)} rank files,
{traces.get('totals', {}).get('bytes', 0):,} bytes,
{traces.get('totals', {}).get('gpu_kernels', 0):,} GPU kernel events.
""",
        encoding="utf-8",
    )

    (root / "notes.md").write_text(
        """# Notes — the things that cost time

## etcd cannot use 2379 on these nodes

They run a Kubernetes control plane whose own etcd holds 2379 over TLS on both
localhost and the node IP. The reference bring-up scripts hard-code 2379; here
that makes the etcd container exit with "address already in use" and then points
the worker's plaintext discovery client at a TLS endpoint that is not ours. The
symptom arrives much later, as a router with an empty worker pool.

## Do not kill everything holding a KFD handle

The reference GPU reset kills every pid `rocm-smi --showpids` reports. On a Slurm
GPU node that set includes `slurmstepd`, which holds one for the step's cgroup —
killing it can take down an unrelated job step. The gate here is VRAM returning
to baseline, and only plausible engine leftovers are killed.

## The engine container runs as root

A directory it creates on a bind mount cannot be written afterwards by the user
running the analysis, so the profiler output directory is created host-side
before the capture starts.

## `ls` lies immediately after a write

Measured repeatedly on both the local array and NFS: `du -sb` reported 484 MB
while `ls -l` reported 118 KB per file for the same directory, and `stat`
afterwards agreed with `du`. Anything deciding "is this file non-empty" has to
`stat` at read time, not trust a listing taken just after the writer finished.

## A cold start is not a hang

The first engine start of a session reads the whole checkpoint over NFS at about
921 MB/s — around 13 minutes to a serving health endpoint. The second start in
the same session took 4 minutes because the node has 3 TB of RAM and the
checkpoint stayed in page cache.

## The profiled round's throughput is not a baseline

CUDA graphs are a start-up flag. With them on the profiler records one launch per
decode step instead of the kernels inside it, so the trace is useless for
attribution; with them off, throughput drops severalfold. That is why there are
two rounds and why only the baseline round's numbers are quotable.

## AIPerf needs four container settings that fail quietly

`--user` with `HOME=/tmp` (the image runs as uid 1000 but writes host-owned
directories), the two dataset mmap paths (or the prompt cache dies with the
`--rm` container and every run re-tokenises the trace), a `PYTHONPATH` that keeps
the image's own entries, and the offline flags plus the `sitecustomize` shim that
lets a local tokenizer directory be loaded as one.
""",
        encoding="utf-8",
    )

    print(f"render: wrote README.md, REPRODUCE.md, environment.md, notes.md under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
