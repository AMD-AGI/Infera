# Notes — the crash, the fixes, and the traps

## 1. The main finding: routing policy and `mem-fraction-static` are coupled

**What.** This arm's first attempt used `--mem-fraction-static 0.80` — the value
every DPA-on leg on this stack had used, including the acceptance run of this same
branch and image. It aborted **60 seconds** into the measurement window:

```
:0:rocdevice.cpp:3582 … HSA_STATUS_ERROR_OUT_OF_RESOURCES … Available Free mem : 52 MB
Fatal Python error: Aborted
```

> **The crash log itself did not survive** — the GMU-0.70 restart reused the same
> log tag and truncated it in place. Everything quoted in this section was read
> off the live log at the time and is recorded faithfully, but it is **not
> independently checkable from this kit**. See `logs/README.md`. The conclusion is
> corroborated by this run's own working config (0.70) and by the measured 8-way
> spread in `analysis/routing_distribution.md`.

**Why — read the diagnostic, don't assume it.** At the moment of the abort:

```
Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 0,
  token usage: 0.05, #running-req: 0, #queue-req: 0
```

`token usage: 0.05` — **the KV pool is 95 % empty**. This is not KV exhaustion. It
is activation memory.

**The mechanism is the routing change itself.** Counting distinct DP ranks issuing
`Prefill batch` per second in the run-up to the abort:

```
12:44:51 -> 5 ranks    12:44:57 -> 4 ranks    12:45:02 -> 3 ranks
12:44:52 -> 4 ranks    12:44:59 -> 4 ranks    12:45:04 -> 3 ranks   <- abort
12:44:54 -> 4 ranks    12:45:01 -> 4 ranks
```

**4–5 ranks prefilling concurrently.** Under kv-aware only 1–2 ever are, because
kv-aware concentrates the shared prefix on the ranks that already hold it (see
§2). Each concurrently-prefilling rank holds its own 8,192-token chunk's
activations, and the ~56 GB left outside a 0.80 static reservation does not fit
4–5 of them at once.

**How it was resolved.** `GMU=0.70`.

| | attempt 1 | attempt 2 |
|---|---|---|
| `mem-fraction-static` | 0.80 | **0.70** |
| avail mem after pool | 56.4 GB | **85.2 GB** (+51 %) |
| `max_total_num_tokens` | 2,939,264 | 2,387,200 (−19 %) |
| outcome | abort at t+60 s | **4,006.7 s, 0 faults** |

The 19 % KV-pool cost is free in practice: peak `token usage` on this workload is
~0.05, so the pool was never the binding resource.

**Context — the direction is counter-intuitive.** Prefill activation OOM is fixed
by **LOWERING** `mem-fraction-static`, the *opposite* of the decode-side retract
fix (raise it, for more KV room). Diagnose by phase:

| phase | symptom | direction |
|---|---|---|
| decode | retract / `get_cpu_copy NotImplementedError` | **RAISE** |
| prefill | `HSA_STATUS_ERROR_OUT_OF_RESOURCES` at low `token usage` | **LOWER** |

Fourth independent confirmation on this stack: Case A 0.88→0.80 (DPA, 48 conc);
nodpa 0.80→0.70 (DPA off, conc 1); arm B 0.80→0.70 (DPA off, 24 conc); this arm
0.80→0.70 (DPA on, **round-robin**, 24 conc).

**The new part** is the last one. The first three were all explained by "one rank
holds a whole chunk instead of its 1/8 slice" (DPA off). This arm has DPA **on**,
so each rank *does* hold only its slice — and it still OOMs, because *more ranks
are active at once*. Same resource, different route to exhausting it. **The
coupling is between the routing policy and an engine memory knob**, and neither
component's documentation mentions the other.

## 2. What the spreading settles about kv-aware

Prefill batches per DP rank, same branch / image / workload family, differing only
in routing policy:

| routing | DP0 | DP1 | DP2 | DP3 | DP4 | DP5 | DP6 | DP7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kv-aware (acceptance run) | 1,405 | 1,497 | **1** | **1** | **1** | **1** | **1** | **1** |
| **round-robin (this run)** | 515 | 495 | 508 | 493 | 508 | 502 | 475 | 499 |

After the acceptance run, two hypotheses were live for the dp2..dp7 zeros:

| | claim | verdict |
|---|---|---|
| A | those ranks are not computing at all — DPA is not truly 8-way | **refuted** |
| B | all 8 compute, but only 2 ranks' KV events reach the router | **refuted as stated** |

Neither. The ranks are alive and take ~500 batches each the moment the router
sends work. The mechanism is a **self-reinforcing affinity loop** in the cost
function `w·(request_blocks − hits) + active_blocks`: a rank with an empty cache
view has zero hits → maximal cost → never the cheapest candidate → no traffic →
stores no blocks → view stays empty. dp0/dp1 won the first few picks and the loop
closed behind them.

**This is the policy working as designed**, not a bug — 89 % of every par8 request
is a shared prefix, and concentrating it maximises hit rate. But it means the
*effective* prefill parallelism under kv-aware on a nested-prefix workload is set
by the first few picks, not by `dp_size`. Worth knowing before sizing a
deployment.

Full treatment with the router's own pick log in
`analysis/routing_distribution.md`.

## 3. Why round-robin is what exercised the kvd storage tier

Every prior run on this branch recorded `gets_total = 0`: the GPU radix cache
served everything and nothing was ever read back from host. This run:

```
gets_total 14,864    hits_total 14,864    sets_total 199,085    evictions 121,835
host_bytes 84.6 GB   long_bytes 297 GB
```

Because round-robin spreads the shared prefix across 8 ranks rather than
concentrating it on 2, no single rank's GPU radix holds the whole hot path, so the
L3 host tier is genuinely consulted — 14,864 times, hitting every time. `host_bytes`
also exceeded the 64 GB `--max-bytes` cap, so spillover into the long tier was
live.

An unintended but real benefit: **the routing change is what turned kvd from
"proven wired" into "proven exercised."** If the goal is to stress the storage
tier, concentrating routing works against you.

## 4. EP_DECOUPLE and the global-chunk trap (inherited, still load-bearing)

Both are in `patches/0001-…diff`. Neither is tuning; each removes a way that
flipping `DPA` would silently change a *second* variable. They matter to this arm
because it is the DPA-**on** half of a pair whose other half is DPA-off — the fixes
must be in place on both or the comparison is meaningless.

- **`--ep-size` outside the DPA branch.** GLM-5.2 is a MoE; `--ep-size` selects
  *expert* parallelism while `--enable-dp-attention` selects *attention*
  parallelism. Gating both on one `if` means `DPA=0` also collapses the MoE. Both
  arms now carry `--ep-size 8`.
- **`--chunked-prefill-size` matched at the GLOBAL level.** sglang divides it by
  `dp_size` only under DPA (`server_args.py:4902` — a division, not a clamp). This
  arm requests 65,536 and resolves to 8,192/rank; arm B requests 65,536 and
  resolves to 65,536 (no division). **Global budgets match; per-forward work does
  not.** That is the deliberate choice — see `patches/README.md`, where the two
  reference kits' disagreement on this point is recorded rather than resolved.

## 5. Operational traps hit during this arm

**A 12-hour wall clock was too short and cost the first attempt.** Arm A's restart
at `GMU=0.70` was launched ~30 min before the allocation expired; the 12 min cold
start plus a 4,007 s run could not fit. The allocation died with `ExitCode=143:0`
mid-boot. **Extend the wall clock *before* restarting, not after** — or size it for
`build + bring-up + run + one full retry` from the start. This rerun used 24 h.

**A bad node's `/tmp` fails the build with a permissions error.** One node
(`crsuse2-m2m-149`) had `/tmp` as `drwxr-xr-x` instead of `drwxrwxrwt`. The build
died at `mkdir /tmp/dockercfg: permission denied`, which reads like a script bug.
Check before building:

```bash
spur exec <job> bash -c 'ls -ld /tmp'      # want drwxrwxrwt
```

**`spur exec` does not parse a shell command string, and does not forward stdin.**
Collecting the environment needed the third form:

```bash
spur exec $J bash -s < collect_env.sh          # -> empty file (stdin not forwarded)
spur exec $J "bash /path/collect_env.sh"       # -> nsenter: failed to execute
spur exec $J bash -c "bash /path/collect_env.sh"   # -> works
```

**`docker exec … gzip -c` through `spur exec` corrupts the stream.** Pulling
`/tmp/router.log` out of the container as gzip produced a file that `zcat` rejected
with "not in gzip format". Base64 in between fixes it:

```bash
spur exec $J bash -c 'docker exec $CTR bash -c "gzip -c /tmp/router.log | base64 -w0"' \
  | tr -d '\n' | base64 -d > router.log.gz
```

**Never background a long docker client inside `spur exec`** — the exec namespace
teardown kills it even under `nohup`/`setsid`. The build runs in the *foreground*
of the exec; leg launches use `docker exec -d` with a staged env file.

**`export DOCKER_CONFIG=/tmp/dockercfg` before every docker call.** Docker 29 has
no classic builder and its buildx plugin discovery fails on the node's root-owned
default config.

**`WARNING: engines still present after 40s` on a clean boot is a false alarm.**
`ab_boot.sh`'s wait loop counts matching processes *after* `docker exec -d` has
already started the new leg. Verify by timestamp (`ps -eo pid,lstart,args`) rather
than assuming a stale process.

**Server logs contain binary bytes** — pipe through `strings`.

**Never probe a PD leg's own port from outside its container** — it hangs.

**Never grep an appended log for readiness** — these logs accumulate across runs
and a grep for `ready to roll` matches a *previous* run within seconds. Poll HTTP
(`scripts/ab_wait.sh`).

**`--dashboard-mode` is mandatory and has nothing to do with a dashboard.**
`summary.json`, `metrics.jsonl` and `metadata.json` are all written inside
`if dashboard_mode and benchmark_name and data_dir:`. Without it the run completes,
prints a full report, exits 0 — and persists nothing.

**Pass no load knobs on the bench CLI.** The YAML is the single source of truth;
`--initial-sessions` on the command line silently shadows it and the two arms would
no longer share a workload.

## 6. What this pair cannot tell you

The two arms differ in **two** variables — prefill DP-attention *and* routing
policy — because that is what was specified. Arm B is faster on every latency
percentile and carries 25 % more throughput, but **that gap cannot be split
between the two changes from this data.**

The 2×2 has two empty cells: (DPA on + kv-aware) at par8 load, and (DPA off +
round-robin). The first is closest to the acceptance run but at Case A load, not
par8's. Neither exists as a par8-load measurement.

One prior single-variable result bears on the DPA half: the spur nodpa kit
measured DP-attention costing **1.65–1.93× TTFT at concurrency 1**. Direction
consistent; magnitude not transferable at 24× the load.
