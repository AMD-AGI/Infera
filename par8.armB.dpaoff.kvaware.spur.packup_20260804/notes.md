# Notes — gotchas, wrong turns, and what the failed sibling arm established

## 1. EP_DECOUPLE — turning off DP-attention silently turned off expert parallelism

**What.** The leg script gated `--ep-size` and `--enable-dp-attention` on the same
`if [ "$DPA" = "1" ]`.

**Why it matters.** GLM-5.2 is a MoE. `--ep-size` selects **expert** parallelism;
`--enable-dp-attention` selects **attention** parallelism. They are different
axes. With them coupled, `DPA=0` also collapsed the MoE from ep8 to the TP
default — so a run billed as "DPA off" would in fact differ in the attention
layout *and* the expert-dispatch collective, and any latency delta could not be
attributed to either.

**How it was resolved.** `DP_ARGS=(--ep-size "$TP")` initialised outside the
branch; only `--dp-size` and `--enable-dp-attention` remain inside it. Verified on
the live command line: the DPA-off prefill leg carries `--ep-size 8` with no
`--dp-size`.

**Context.** Not discovered here. The spur nodpa kit
(`../agenticbench.mtp.nodpa.packup_20260802/notes/nodpa_design.md` §2) found and
named this. It was re-applied because this workspace's leg script descends from
the *Case A* kit, which never ran DPA-off and therefore never had the bug
triggered.

## 2. `--chunked-prefill-size` is GLOBAL, and DPA divides it — the 8× trap

**What.** `--chunked-prefill-size` is a **global** per-step token budget. sglang
divides it by `dp_size` *only* when DP-attention is on (`server_args.py:4902` — a
division, not a clamp).

**Why it matters.** At dp8, a requested 65,536 resolves to `8192` per rank while
remaining 65,536 machine-wide. The script's old `DPA=0` branch hardcoded 8,192 —
which, with no division happening, is 8,192 **globally**, i.e. ⅛ the work per
step. Flipping DPA off would therefore *also* have cut the global chunk 8×. The
engine's own warning text reads like a clamp, which is why this took the nodpa kit
two attempts and a rerun to pin down.

**How it was resolved.** `CHUNK` defaults to `ISL*TP` unconditionally, and this
arm passes `CHUNK=65536` explicitly on top.

**Context — the two reference kits disagree, and this is not resolved.**

| kit | cluster | conc | `CHUNK` | reasoning |
|---|---|---|---|---|
| nodpa | spur | 1 | 65,536 | match the *global* budget |
| par8 | vultr | 24 | 16,384 | "near the measured sweet spot" |

Neither matches *per-forward* work (that would be 8,192). This run follows nodpa
because it is the first-hand result **on this cluster**. The consequence is that
this arm does 8× the per-forward prefill work of the vultr par8 run — one of two
uncontrolled variables in the cross-cluster comparison in `README.md`, and a
reason not to read that comparison as a measurement.

## 3. DPA-off cannot boot at the DPA arm's `mem-fraction-static`

**What.** `GMU=0.80` — the value the DPA-on legs use — does not start a DPA-off
prefill leg. It dies with
`HSA_STATUS_ERROR_OUT_OF_RESOURCES … Fatal Python error: Aborted`.

**Why.** Read the diagnostic off the log rather than assuming: `token usage: 0.04`
at the time, i.e. the KV pool is **empty**. This is not KV exhaustion — it is
**activation** memory. Without DP-attention, one rank computes attention over the
whole chunk instead of its 1/8 slice, so the transient peak is far larger at
identical chunk size.

**How.** `GMU=0.70` on this arm. It booted first try, ready in **403 s**, zero
faults across the run.

**Context — the direction is counter-intuitive and worth stating explicitly.**
Prefill activation OOM is fixed by **LOWERING** `mem-fraction-static`, the
*opposite* of the decode-side retract fix (raise it, for more KV room). Diagnose
by phase:

| phase | symptom | direction |
|---|---|---|
| decode | retract / `get_cpu_copy NotImplementedError` | **RAISE** |
| prefill | `HSA_STATUS_ERROR_OUT_OF_RESOURCES` / `Aborted` at low token usage | **LOWER** |

This is the third independent confirmation of that rule on this stack (Case A
0.88→0.80; nodpa 0.80→0.70; this arm 0.80→0.70). It is a property of the
deployment, reported as a result, not papered over.

## 4. What arm A's crash established — and why this kit ships alone

The paired arm (**DPA on + round-robin routing**) is not in this kit. It OOMed
60 seconds into its measurement window. That failure produced a real finding,
recorded here because it is the reason arm B stands alone and because it will not
be re-derivable once the logs age out.

**The finding: prefill DP-rank utilisation is set by the routing policy.**

Prefill batches served, per DP rank, over comparable windows:

| | DP0 | DP1 | DP2 | DP3 | DP4 | DP5 | DP6 | DP7 |
|---|---|---|---|---|---|---|---|---|
| earlier acceptance run (**kv-aware**, DPA on) | 1,405 | 1,497 | 1 | 1 | 1 | 1 | 1 | 1 |
| **arm A (round-robin**, DPA on) | 30 | 9 | 8 | 19 | 19 | 17 | 6 | 11 |

Under kv-aware, six of eight prefill ranks did **one batch each** across a full
run. Under round-robin, all eight are in play immediately. This settles an open
question from the acceptance run, where `infera_policy_cache_view_size` showed
prefill dp0/dp1 populated and dp2..dp7 at zero, and two hypotheses were live:

- **A** — dp2..dp7 are not computing at all.
- **B** — all 8 compute, but only 2 ranks' KV events reach the router.

Neither, quite. The ranks *are* alive and *do* receive work when the router sends
it; kv-aware simply concentrates on the ranks that already hold the prefix, and
the concentration is self-reinforcing (no traffic → empty cache view → never the
cheapest candidate → no traffic).

**The cost, also measured.** In the seconds before arm A's abort, **4–5 DP ranks
were prefilling concurrently**, against 1–2 under kv-aware. Each holds its own
chunk's activations. At `mem-fraction-static 0.80` (56 GB left outside the static
reservation) that peak does not fit:

```
:0:rocdevice.cpp:3582 … HSA_STATUS_ERROR_OUT_OF_RESOURCES … Available Free mem : 52 MB
Fatal Python error: Aborted
```

with `token usage: 0.05`, `#running-req: 0` — the KV pool empty, i.e. the same
activation-memory signature as §3, arrived at from the opposite direction.

**So: round-robin on a DPA prefill leg needs a lower `mem-fraction-static` than
kv-aware does.** That is a deployment coupling between the *routing policy* and
an *engine memory knob*, which is not obvious from either component's
documentation.

**Why arm A is not here.** It was restarted at `GMU=0.70` immediately, but the
24 h allocation expired (`ExitCode=143:0`, SIGTERM) during the ~12 min cold start.
That expiry was an operator error — the remaining window could not fit a restart
plus a 4,007 s run, and the wall clock should have been extended *before*
restarting rather than after.

## 5. Errors in this run were client-side timeouts, not server failures

25 errors against 2,907 requests. Classified from the driver log by raw-byte
phrase grep:

| phrase | count |
|---|---|
| `timed out` | **25** |
| `failed: HTTP <code>` | 0 |

All 25 are `asyncio.TimeoutError` against the driver's own
`aiohttp.ClientTimeout(total=240)` ceiling. Zero HTTP failures, zero engine
faults. At a p99 output length of 10,249 tokens and TPOT p99 of 35 ms, a single
long generation legitimately exceeds 240 s — these are the tail, not a defect.

> **Grep the driver log with `-a` and by phrase, not by line.** The progress bar
> uses `\r` overwrite, so error prints land on the *same physical line* as the
> bar. Line-oriented `grep`/`tail` will appear to show no errors at all. Use
> `tr -d '\000' < log | grep -aoE 'timed out|failed: HTTP [0-9]+'`.

## 6. Traps that cost time here (or elsewhere) and are cheap to avoid

**A bad node's `/tmp` fails the build with a permissions error.** One node
(`crsuse2-m2m-149`) had `/tmp` as `drwxr-xr-x` instead of `drwxrwxrwt`. The build
died at `mkdir /tmp/dockercfg: permission denied`, which reads like a script bug.
It is a node defect. Check before building:

```bash
spur exec <job> bash -c 'ls -ld /tmp'      # want drwxrwxrwt
```

**Never background a long docker client inside `spur exec`.** The exec namespace
teardown kills it, even under `nohup`/`setsid`. The build runs in the
**foreground** of the exec (and the *login-side* driver is what gets backgrounded);
leg launches use `docker exec -d` with a staged env file.

**`export DOCKER_CONFIG=/tmp/dockercfg` before every docker call.** Docker 29 has
no classic builder and its buildx plugin discovery fails on the node's root-owned
default config.

**Server logs contain binary bytes.** Pipe through `strings` — a bare `grep` will
report "binary file matches" and show nothing.

**Never probe a PD leg's own port from outside its container** — it hangs. Health
checks go through the leg's own container, and clients go through the router.

**Never grep an appended log for readiness.** These logs accumulate across runs;
a grep for `ready to roll` matches a *previous* run's line within seconds. Poll
the HTTP endpoint instead (`scripts/ab_wait.sh`).

**`--dashboard-mode` is mandatory and has nothing to do with a dashboard.**
`summary.json`, `metrics.jsonl` and `metadata.json` are all written inside
`if dashboard_mode and benchmark_name and data_dir:`. Without the flag the run
completes, prints a full report, exits 0 — and persists nothing. That has cost one
run on this stack already.

**Pass no load knobs on the bench CLI.** The YAML is the single source of truth;
`--initial-sessions` on the command line silently shadows it, and two arms would
then no longer share a workload.
