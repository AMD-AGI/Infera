# `profiling-demo` — profile GLM-5.3-Flash under a Mooncake trace replay

The profiling stage of `llm_e2e_performance_optimization`. It deploys
GLM-5.3-Flash on one MI355X node, drives it with AIPerf replaying a Mooncake
production trace, cuts a torch-profiler window out of the running load, and ranks
the kernels with Magpie.

`DESIGN.md` is the design and the record of what was decided and why.
`temp/manual/` holds the manual walk-through that validated the whole chain
before any of it was wired into agent_sys, including the measured numbers this
package's comments cite.

## Status

**Complete and runnable.** All six leaves are wired. Run twice on 2026-08-31,
both times with seven tasks succeeded, seven handoffs valid and six validator
verdicts PASS: 18 minutes reusing the deployment already up, and 24 minutes from
cold with no development aids set.

```
final  serve_baseline: succeeded     handoff deployment_baseline: valid   check_service_live:   PASS
final  run_baseline:   succeeded     handoff aiperf_baseline:     valid   check_aiperf_report:  PASS
final  serve_profiled: succeeded     handoff deployment_profiled: valid   check_service_live:   PASS
final  run_profiled:   succeeded     handoff aiperf_profiled:     valid   check_aiperf_report:  PASS
                                     handoff torch_trace:         valid   check_trace_coverage: PASS
final  kernel_scan:    succeeded     handoff kernel_table:        valid   check_kernel_table:   PASS
final  packup:         succeeded     handoff profile_packup:      valid   check_packup_shape:   PASS
```

What the cold run produced — the numbers worth quoting, because the reused-
deployment run measured a warm radix cache (see below): 346 requests replayed per
round; 631 output tokens/s with graphs on against 380 with graphs off; eight trace
ranks totalling 365 MB and 1,096,288 GPU kernel events; 159 kernels ranked, of
which the top 25 account for 89.2% of 138.6 seconds of aggregate self CUDA time.
The largest single kernel is AITER's `cross_device_reduce_2stage` at 22.9% — a
TP-8 all-reduce holding nearly a quarter of the engine's GPU time, which is a
communication cost rather than a kernel to optimise in isolation.

The one piece designed and not written is `check_reproduces`, the validator that
hands a fresh Claude Code session the packup and sees whether it reproduces. It is
the only place in this package where an agent earns its place; everything else is
a fixed command whose output has to be byte-comparable between rounds.

## Running it

Needs Python 3.12 or newer. `agent_sys` declares `requires-python = ">=3.10"` but
cannot be imported below 3.12; see
`../temp/bugs/001-requires-python-3.10-but-fails-below-3.12.md`. The repository's
own `.venv` is 3.12 and works.

Needs a Slurm allocation that is already holding the node, because the login node
has no GPUs and cannot ssh to a compute node:

```bash
sbatch --parsable --partition=Compute-DCPT --nodelist=<node> --nodes=1 \
  --ntasks-per-node=1 --exclusive --time=09:00:00 --job-name=hold \
  --wrap='srun --ntasks=1 sleep infinity'
```

Then:

```bash
export PATH="<repo>/.venv/bin:$PATH"
AGENT_SYS_NO_PERMISSIONS=1 agent-sys run \
  --package agent_sys/examples/llm_e2e_performance_optimization/profiling-demo \
  --var jobid=<the job id> \
  --var node=smci355-ccs-aus-n04-33 \
  --var node_ip=10.235.192.139 \
  --var model_path=/apps/qiongzhu/models/GLM-5.3-Flash-FP8
```

Four variables carry no default. A Slurm job id, which node it holds, that node's
IP and where the weights are, are facts about one allocation on one cluster; a
default would be one machine's answer shipped as everyone's, and would go stale
the moment the job ends. Omit one and the load fails naming the file, the line and
the variable.

Everything else defaults: `image`, `etcd_image`, `served_name`, `tp`,
`router_port`, `worker_port`, `etcd_port`, `work_root`. They are listed in
`shared.yaml`, which is the only place this package may name its own variables.

`AGENT_SYS_NO_PERMISSIONS=1` is required by the mission this package is developed
under. The `permissions.grants` blocks are written anyway — they are the answer to
"what does this touch", and writing them now means turning enforcement on later is
a configuration change rather than an archaeology exercise.

## Prerequisites that are not part of the package

Three one-time setup steps, all validated on 2026-08-31 and recorded in
`temp/manual/FINDINGS.md`:

1. **Build the engine image.** No released SGLang carries `model_type:
   glm5_next` — checked against v0.5.18 and the 0.5.18.dev20260826 ROCm nightly,
   neither of which has `Glm5NextConfig`. Build from
   `examples/glm53flash-demo/patches/Dockerfile.sglang.glm53` with the repository
   root as context; it took 9m25s on the node and asserts three times that the
   PR #36507 overlay reached the bytecode the interpreter imports.
2. **Pull the AIPerf image**: `nvcr.io/nvidia/ai-dynamo/aiperf:0.12.0`.
3. Nothing for the weights. Staging them to local NVMe was in the design and was
   dropped after measurement: `/apps` sustains 921 MB/s single-stream on this
   node, so a copy would read the same 306 GB over the same mount and then write
   it again, and the node's 3 TB of RAM keeps the checkpoint in page cache after
   the first load anyway.

## The shape

```
main                    non-leaf, no agent
 ├── serve_baseline     → deployment_baseline    check_service_live    completeness/strong
 ├── run_baseline       → aiperf_baseline        check_aiperf_report   completeness/strong
 ├── serve_profiled     → deployment_profiled    check_service_live
 ├── run_profiled       → aiperf_profiled        check_aiperf_report
 │                      → torch_trace            check_trace_coverage  completeness/strong
 ├── kernel_scan        → kernel_table           check_kernel_table    usability/strong
 └── packup             → profile_packup         check_packup_shape    completeness/strong   is_end
```

A chain, not a fan-out, and two edges are worth explaining:

- `serve_profiled` depends on **`run_baseline`**, not on `serve_baseline`. Its
  first act is to tear the baseline deployment down, so the edge has to mean "the
  baseline has been measured" — wiring it to `serve_baseline` would let agent_sys
  schedule it alongside `run_baseline` and destroy the deployment mid-measurement.
- `run_profiled` produces **two** handoffs. The profiler window has to fall inside
  the load window; as two sibling tasks agent_sys would schedule them concurrently
  with nothing to synchronise them.
- `run_profiled` also cuts **two profiler windows** into one `torch_trace`: the
  measurement window without Python stacks, and a short one with them. See *The
  stack window* below.

Six of the seven kinds are `reproducible` and `profile_packup` is `code`. Neither
choice was aesthetic. `structured_text`'s optional items are exactly `text.json` /
`text.yaml` / `text.xml` / `schema` and `handoff/content.py:check_items` rejects
any top-level item the type never declared — so a handoff carrying directories
cannot use it. `profile_packup` is `code` for the reason `single_real_task` gives:
laying a packup into `reproducible` renames `results/` to `items/result` and leaves
`REPRODUCE.md` with no item to be, which destroys the thing `check_packup_shape`
exists to check.

## What it costs

A cold start reads 306 GB of FP8 weights off NFS: 819 seconds to a serving health
endpoint, measured. A second bring-up in the same session is served from page
cache and took 243 seconds. Do not read silence as a hang — a program body's
stdout is not streamed, and `agent_sys` prints nothing between dispatch and the
end of a phase.

A whole run is 24 minutes from cold, measured, and 18 when the baseline
deployment is reused.

`PD_REUSE_DEPLOYMENT=1` in the environment lets `serve_baseline` adopt a
deployment that is already serving instead of restarting it.

**It is a development aid, and it changes the measurement. Do not quote a number
from a run that used it.** A deployment that has already served this trace has the
trace's prefixes in its radix cache, and a Mooncake replay expands `hash_ids` into
real token blocks — so prefix hit rate decides how much prefill there is to do.
Measured across two runs of the same configuration on the same trace: 630 output
tokens/s and 25.9 s mean TTFT from cold, against 1,004 tokens/s and 484 ms when
the deployment was reused, with the all-reduce kernel's share of GPU time moving
from 22.9% to 9.9%. `temp/ARTIFACTS.md` has both sets.

What it *is* safe against is coming up in the wrong round: `round.sh` records the
engine's **observed** argv and the router's actual invocation, and
`check_service_live` decides the round from those rather than from anything the
task declares about itself. `serve_profiled` ignores the variable unconditionally.

## The stack window

`kernel_scan` names the Python frame that launched each ranked kernel, and
publishes it as a `launcher` block per row of `items/result/text.json`. That is
the only evidence which answers "which source file do I edit" for a symbol that
is a compilation artefact — `main_kernel` is what TileLang names every kernel it
generates, 3.29% of GPU time under one meaningless name in the sample profile,
and no amount of searching for that string can locate anything.

The frames come from a **second, short profiler window** taken with
`with_stack` on, right after the measurement window and inside the same load.
Two windows rather than one, because stacks are expensive and change nothing that
is measured. Profiling one workload twice in the engine image on
smci355-ccs-aus-n04-29, 2026-09-01:

| | `with_stack: false` | `with_stack: true` |
|---|---|---|
| uncompressed | 228,553 B | 2,996,700 B (13.1×) |
| gzipped | 14,843 B | 245,167 B (16.5×) |
| `python_function` events | 0 | 9,565 |
| `kernel` events / total kernel time | 48 / 4057 us | 48 / 4057 us |

The last row is the one that decides the design: identical kernel counts and
identical kernel time, so stacks cost bytes and nothing else. At this package's
measured 60.5 MB per rank for a 15 s window, stacks across the measurement window
would be roughly 1 GB per rank and 8 GB for the round — while resolution needs a
handful of launches per kernel, votes over three probes, and reads two rank
files. So `stack_window_s` defaults to 3 and `stack_ranks` to 2.

`--var stack_window_s=0` skips the window. Pair it with
`--var min_launchers_in_top_n=0`, or `check_kernel_table` will fail the round for
the missing frames — which is the point of the floor.

**A frame is published as a placeholder plus a relative path**, never as an
absolute one, and `path_form` says how exact the relative part is. torch strips
the longest matching `sys.path` entry from a frame path, so one capture yields
both shapes:

```
aiter/ops/triton/softmax.py(10): softmax                          -> sys_path_relative
/sgl-workspace/sglang/python/sglang/srt/utils/common.py(3341): ... -> container_absolute
```

`/sgl-workspace/aiter` is on `sys.path` so aiter's frames arrive relative;
sglang is installed **editable**, so `sys.path` carries a finder hook instead of
the real directory, nothing matches, and its frames arrive absolute. That is not
an edge case to tolerate — the GLM-5.3-Flash tree *is* the editable PR #36507
overlay, so the frames this pipeline most wants are exactly the ones that arrive
absolute.

A `sys_path_relative` path carries an ambiguity nothing here can settle:
`aiter/ops/x.py` is equally consistent with a `sys.path` entry of
`/sgl-workspace/aiter` and one of `/sgl-workspace`. So this package reports what
it saw and the consumer binds it — `analyze-demo`'s `identify.bind_launcher`
tests candidates against the repository it indexed, because it can stat a file
and a producer inside a container cannot.

## What `handoff` will and will not seal

Two of this package's shapes are dictated by the publication seal rather than
chosen, and both are worth knowing before adding a handoff here.

**A content file should not name an absolute path outside a small allow-list —
and nothing enforces that at publication.** The rule is right, because a record
of one machine's afternoon is not transferable, but it is this package's
discipline rather than the store's admission check.

`handoff/store.py` does not call `locality.check`, in either `seal` or `put`;
both call sites carry `# locality.check — NOT CALLED. User-ruled 2026-08-31`
and the reason, which is that the shape heuristic read an HTTP access-log line
as a filesystem path on an artefact whose brief *required* that line — 97% false
positive on a real kit. `ROADMAP.md` §6.4 carries the rebuild at P2 and keeps
`locality.py` intact, so re-wiring it is one line. **`handoff/protocols.py:294`
still describes the check as part of publication and is stale on that.**

Our own numbers agree with the ruling rather than contradicting it: over one
round's logs, 818 absolute paths, of which **817** were container-internal
paths, HTTP routes in an access log, and an etcd key prefix. A check firing on
those would have been wrong 817 times.

So the discipline is kept and enforced *here*, on the producing side, which is
where it belongs anyway — a producer knows which of its paths are site roots and
a shape heuristic cannot:

- site roots are replaced by `@NAME@` placeholders on the way in, by
  `assets/lib/redact.py`, which then applies the allow-list and shape rule
  **itself** and fails naming anything it was not given a placeholder for. Since
  the seal does not check, this script is the only thing standing between a
  local path and a published artefact — treat it as load-bearing, not as a
  belt-and-braces duplicate;
- raw logs are published gzipped, which keeps the bytes exactly where substituting
  them would corrupt the one artefact whose value is being faithful;
- the handoffs carry `command` rather than `script` — a real bring-up script names
  its own log file under a temp directory and cannot pass.

The placeholder is `@NAME@` and not `${NAME}` because `}` is not in the shape
rule's lookbehind class, so `${TASK_PACKAGE}/assets/serve/mix_up.sh` offers
`/assets/serve/mix_up.sh` as a fresh candidate. `@` is in the class. That is a
property of `locality.py`'s regex, which `redact.py` reuses — so it still
applies here even though the seal never runs it.

**`script`, `command` and `entry` items must be executable.** `agent/gate.py`
copies the version out and checks the mode. **This one is enforced**, unlike the
locality rule above, and a missing `chmod +x` costs the whole graph while
reporting only a timeout. It is also a good constraint: it pushed `command` from
a transcript into a runnable script, and writing it as one is what made it clean
under the locality rule, because a script that takes its site paths as shell
variables has no absolute path to leak.

## Known gaps

- **No teardown step.** If the graph fails partway the engine container keeps the
  GPUs. `assets/serve/mix_up.sh` starts with an idempotent teardown and a VRAM
  gate, so rerunning is safe; recovering without rerunning is manual
  (`docker rm -f glm53_mix glm53_mix_etcd`).
- **No `resources` block.** The leaf legitimately wants eight GPUs, but the CLI
  composition root declares no pools, so naming one would name a pool that does
  not exist. Nothing here reserves anything; the Slurm allocation is what keeps
  two runs from colliding.
- **`repos: [infera, sglang]` is declared and read by nothing.** It is where the
  schema says such a fact goes; `env_mgr` does not act on it today.
- **The bodies run staged scripts by absolute path**, which needs the run root to
  be on a filesystem the compute node also mounts. `$HOME` here is NFS from the
  same server, so it works. `assets/lib/remote.sh` asserts it rather than assuming
  it, so moving `--demo-root` to local disk fails with the reason rather than with
  "No such file".
- **`check_reproduces` is not written.** The design has it as the one AI-shaped
  check in this package; `check_packup_shape` counts substance but cannot tell
  whether the commands work.
- **A shared script may not be named after a closure.** `spec_loader` resolves a
  body by matching the closure's name against filenames under `assets/`, so
  `assets/kit/packup.sh` and `assets/packup.task/entry.sh` were two candidates for
  `packup`'s entry and the load refused to guess. The shared one is `assemble.sh`.
- **Run roots are shared between packages.** The default `--demo-root` is one
  directory for every package, and `agent-sys run --clean` from a concurrent run
  removes everything in it. This package's own runs go to a separate root.
- **`kernel_scan` needs numpy and PyYAML in the compute node's system
  `python3`.** `assets/analyze/megapie.sh` runs `python3 -m Magpie` on the node
  directly, and Magpie needs both. That held on `smci355-ccs-aus-n04-33`; it does
  not hold on `smci355-ccs-aus-n04-29`, where the system interpreter has neither
  (`ModuleNotFoundError: No module named 'numpy'`). Nothing in the package checks
  for it, so the symptom is a Magpie exit code partway through the step. The
  launcher resolution beside it has no such dependency — `launchers.py` and
  `assets/lib/trace_stream.py` are standard library only.
- **Three spec names still collide with `analyze-demo`'s**, down from five.
  `spec_loader/registry.py` keys on `name` alone, so a name held by two specs
  that are not byte-identical is `SpecInconsistent` at load time. Each package
  loads with no problems on its own; the collision only fires when both go into
  one registry, which is what joining the graphs requires.

  `analyze-demo` renamed its `packup` task to `pack_analyze` and its
  `check_packup_shape` validator to `check_analyze_packup_shape`. Those two were
  plain name clashes — each package had independently picked the obvious word for
  its own last step, and the two tasks share no input kind, output kind, agent or
  body. **This package keeps `packup` and `check_packup_shape`.**

  What remains is `kernel_table`, `check_kernel_table` and `main`, and renaming is
  the wrong answer to all three: the first two are one kind declared twice, and
  `main` is fixed by `cli/main.py`. They resolve by deletion and by merging the
  roots, which is what the join does anyway — verified, the merged package loads
  37 specs with zero problems. The *content* of `kernel_table` is already
  compatible in both directions: `kernel_scan` writes the consumer's field names
  and every row, and `analyze-demo`'s `assets/lib/kernel_table.py` reads either
  layout. `analyze-demo/README.md`, section "Joining up with profiling-demo", has
  the steps.
