# Reproduce — lat1

Ordered and copy-pasteable. Read `environment.md` first for what must exist
before step 0.

Total wall clock: **~50 min from a cold cluster** — ~6 min bring-up per leg (cold
start is *not* a hang), ~8 min correctness, ~6 min probe, ~33 min measured run.
**~45 min if the deployment is already up**, which is how this run was actually
done: lat1 reused the Case A legs untouched, 21 h into the same containers.

Throughout: `P` = job **24300** / `crsuse2-m2m-253` / `10.245.157.89` (prefill,
etcd, router, kvd), `D` = job **24301** / `crsuse2-m2m-236` / `10.245.146.87`
(decode), `W` = `/shared_nfs/yihou_agbench_mtp`.

> **`export DOCKER_CONFIG=/tmp/dockercfg` before every docker call.** Docker 29's
> buildx plugin discovery needs it.
>
> **Never background a long docker client inside `spur exec`** — the exec
> namespace teardown kills it. Stage a script and `docker exec -d` it instead.
>
> **`ssh` to compute nodes is blocked** and the error lies
> (`Permission denied (publickey)`; the real cause is an `AllowUsers` whitelist).
> Use `spur exec`.

---

## ⚠️ Read this first — the one thing unique to reproducing lat1

**Give this run a `random_seed` no other run on the same engine has used.**

The driver's fresh-content seed is the *run-local* `request_id`
(`agent_throughput.py:2187`), and the shared-base seed is a hard-coded constant
(`:2054`). Neither varies across runs. So two runs sharing `random_seed` draw the
same input lengths → the same cached/fresh split → **byte-identical prompts in
the same order** → the second run replays the first into a warm radix tree and
measures cache hits instead of compute.

This is not hypothetical: the first lat1 attempt inherited Case A's `1337` and
returned **cache hit ~100 %** against a configured 0.89, with 14–62 uncached
tokens per request. It would have reported a latency floor roughly 3× too good.

The kit ships `random_seed: 20260802` (full) and `2026080299` (probe). **If you
re-run either against an engine that has already seen those seeds, change them**
— and keep the probe's distinct from the full run's, or the probe warms the full
run's cache and you reintroduce the defect one level down.

Full mechanism, including how it was localised to the GPU radix tree rather than
kvd: `notes/notes.lat1.md` § "Defect 1".

---

## 0. Prerequisites

- Two spur nodes held with 8 GPUs each (`sbatch -p amd-spur -q amd-burst-qos -N1 -G8`).
- The image built **on both nodes** from branch `yihou.dev.glm52.merged.experiment`
  @ `e56e975`, **with the two uncommitted working-tree changes applied** — see
  `environment.md` § "Uncommitted working-tree state". Without them kvd + long
  prompts GPU-fault on gfx950 (`xnack-`, no page-migration fallback).
- `Optimus-AgenticBench` @ **`1cf01cb`** (branch
  `fix/realistic-profile-session-driver`, **not `main`** — main under-loads
  silently) installed editable into `/shared_nfs/yihou_agentbench/venv`.
- Model weights at `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`.

```bash
mkdir -p $W/{scripts,results,logs}
cp scripts/* $W/scripts/
cp patches/apply_p1v3.py $W/scripts/
cp spec/lat1_full.yaml spec/lat1_probe.yaml \
   /home/yihou/dev/git/infera.merge.liying.kv.mtp/work.agenticbench.mtp/workloads/
```

> `run_lat1.sh` resolves its workload as
> `…/work.agenticbench.mtp/workloads/lat1_<variant>.yaml`. Either put the YAMLs
> there (above) or edit `WL=` in the script.

## Steps 1–6 — bring-up: identical to the Case A kit

lat1 changed **no server flag**. Rather than duplicate it, follow
`../agenticbench.mtp.caseA.packup_20260801/REPRODUCE.md` steps 1–6 verbatim:

1. `start_ctr.sh` (both nodes) — 8-assertion bytecode gate, expect `BYTECODE_GATE OK`
2. **Patch the decode leg with `GLM52_P1V3` — MANDATORY.** Without it the run
   dies with `Expected lengths.size(0) == B` under MTP draft-extend. Confirm
   `md5sum` = `632f17acd38737459b43f830ee60ee89` first, apply, then verify the
   **loaded** module reports 3 occurrences (not the file — stale `__pycache__`
   has invalidated an experiment in this tree before).
3. `boot.sh prefill 262144 1 0 <tag>` / `boot.sh decode 262144 0 1 <tag>`.
   **`--mem-fraction-static` on prefill must be 0.80** — 0.88 aborts the leg with
   `HSA_STATUS_ERROR_OUT_OF_RESOURCES` under these prompt lengths.
4. The gate table — every row, before any measured window.
5. `router.sh` → `{"status":"ok","active_workers":2}`. Never probe a PD leg's own
   port; it hangs.
6. `correctness.py` with the model's own sampling (temp 1.0 / top_p 0.95).
   `temperature: 0` + MTP is indistinguishable from KV corruption.

**If the deployment is already up** (as it was here), verify instead of rebooting:

```bash
spur exec 24300 bash -lc 'curl -s http://10.245.157.89:8190/health'
# expect: {"status":"ok","active_workers":2}
spur exec 24301 docker exec agbench_mtp python3 -c \
  "import sglang.srt.layers.attention.dsa.dsa_indexer as m, inspect; print(inspect.getsource(m).count('GLM52_P1V3'))"
# expect: 3
```

## 7. kvd baseline

```bash
spur exec 24300 docker exec agbench_mtp python3 -m infera.kvd.statctl \
  --socket /tmp/kvd/kvd.sock > $W/results/lat1_kvd_before.json
```

## 8. Dry-run the profile — free, no server contact

```bash
cd /home/yihou/dev/git/Optimus-AgenticBench
/shared_nfs/yihou_agentbench/venv/bin/python3 -m agent.agent_throughput \
  --mode preview --model glm5.2-mxfp4 \
  --workload-config .../work.agenticbench.mtp/workloads/lat1_full.yaml \
  --tokenizer /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
```

Expect `turns_per_session` to realize **1.0 / 1.0 / 1.0** and the input/output
triples to match spec.

> **It will emit three warnings, and all three are false here.** They read:
> `max_inflight=1 throttles even at zero latency`, `max_sessions=1 is reached
> even at zero latency`, `initial_sessions=1 is far from the steady-state range
> 23-58`. They come from `run_profile_preview`'s open-loop Little's-law model
> (`:3925`), which does **not** know about the `active_sessions < max_sessions`
> gate at `:2509`. It extrapolates N = 1.0 births/s × 1 turn × 33 s = 33. The
> gate blocks every one of those spawns. Empirically falsified in step 9:
> `in_flight` and `sessions_active` both stay pinned at 1.

## 9. Probe — 6 min, and it earns its keep

```bash
nohup bash scripts/run_lat1.sh probe lat1_probe > $W/lat1_probe.console.log 2>&1 &
```

Three gates, all of which must pass before spending 33 minutes:

```bash
D=$(ls -d $W/bench/lat1_probe/lat1_probe/*/ | tail -1)
/shared_nfs/yihou_agentbench/venv/bin/python3 scripts/lat1_analyze.py "$D" | head -20
```

| gate | expect | what a failure means |
|---|---|---|
| `in_flight max` / `sessions_active max` | **1 / 1** | the concurrency-1 guarantee broke; this is not the experiment |
| `cache actual` vs `ideal` | **0.889 / 0.890**, efficiency ≈ 1.00 | ~1.00 actual = **seed collision**, see the warning at the top |
| artifacts | `summary.json` + `metrics.jsonl` + `metadata.json` | `--dashboard-mode` missing |

The probe caught both defects in this kit. The second — `random_seed` above
`2**32-1`, which numpy rejects outright — would otherwise have killed a 33-minute
window at startup.

## 10. The run — 33 min

```bash
nohup bash scripts/run_lat1.sh full lat1_full > $W/lat1_full.console.log 2>&1 &
```

`--dashboard-mode` is set inside the script and is **mandatory**: `summary.json`,
`metrics.jsonl` and `metadata.json` are all written inside
`if dashboard_mode and benchmark_name and data_dir:` (`agent_throughput.py:2067`).
Without it the run completes, prints a full report, exits 0 — and persists
nothing.

Live abort criteria (the inverse of Case A's):

* `In-flight` ever **> 1**, or `Sessions` ever **> 1/1** → the guarantee broke.
* Cache hit in the live line reading far above 89 % → seed collision; kill it.

Expect ~124 completed requests at ~16 s each. Checking in-flight cheaply:

```bash
tail -c 200 $W/lat1_full.console.log
```

## 11. Post-run — kvd, faults, acceptance

```bash
spur exec 24300 docker exec agbench_mtp python3 -m infera.kvd.statctl \
  --socket /tmp/kvd/kvd.sock > $W/results/lat1_kvd_after.json

for j in 24300:armB_prefill 24301:armB_decode; do
  spur exec ${j%%:*} bash -lc "strings $W/logs/${j##*:}.log | grep -cE \
    'HSA_STATUS_ERROR|Fatal Python error|Expected lengths.size|Memory access fault|Scheduler hit an exception|Traceback'"
done
# expect: 0 and 0

spur exec 24301 bash -lc "strings $W/logs/armB_decode.log | grep -oE 'accept len: [0-9.]+' \
  | tail -4000 | awk '{print \$3}' | sort -n \
  | awk '{a[NR]=\$1;s+=\$1} END{printf \"n=%d mean=%.3f p50=%.2f p90=%.2f\n\",NR,s/NR,a[int(NR*0.5)],a[int(NR*0.9)]}'"
# expect: mean ~2.85
```

`accept len` comes from the **decode leg's log**, not the driver: the driver's
`new_acceptance_lengths` averages SSE chunk sizes and undercounts whenever a
chunk boundary misses a verify step (2.2 vs 2.846 here). The engine counter is
authoritative.

**kvd `gets` will not move.** Prefill kvd only fetches on a radix-tree miss, and
at 89 % planned hit with a single session the in-GPU tree serves everything. Only
`sets` climbs (+3,792 here). Same as Case A; a property of the workload, not a
defect.

## 12. Analysis

```bash
/shared_nfs/yihou_agentbench/venv/bin/python3 scripts/lat1_analyze.py \
  $W/bench/lat1_full/lat1_full/<TIMESTAMP> \
  --json $W/results/lat1_ladders.json
```

Prints the ladders **with their sample count**, the concurrency-guarantee check,
and the TTFT-vs-input-length curve plus its least-squares fit. Expect
**R² ≈ 0.956** and a marginal rate near 34,000 tok/s of presented prompt.

The cross-run figures — the bin-matched Case A vs lat1 penalty table and the
Case A `R² = 0.0016` fit — come from a **self-contained 40-line snippet inlined
in `analysis/lat1_latency_floor.md`** (§ "Reproduce both tables above"). It reads
only this kit plus `../agenticbench.mtp.caseA.packup_20260801/results/metrics.jsonl.gz`,
takes no arguments, and prints both tables. Verified by extracting and executing
it verbatim from the document:

```bash
python3 -c "
import re
blk = re.search(r'\`\`\`python\n(.*?)\`\`\`',
                open('analysis/lat1_latency_floor.md').read(), re.S).group(1)
exec(blk)"
# Case A (N~44)  TTFT_ms =     8681 +   6.56/ktok   R2=0.0016
# lat1  (N=1)    TTFT_ms =     -319 +  29.33/ktok   R2=0.9563
```

## Verification — what "done" means

| gate | expected | actual |
|---|---|---|
| `in_flight` max / `sessions_active` max | 1 / 1 | **1 / 1** |
| cache actual / ideal | ≈ 0.889 / 0.890 | **0.8897 / 0.8899** |
| completed / errors | ~124 / 0 | **124 / 0** |
| engine faults, both legs | 0 | **0** |
| retractions, both legs | 0 | **0** |
| `accept len` (engine) | 2.1–3.0 | **2.846** |
| TTFT-vs-length R² | > 0.9 | **0.9563** |
| artifacts | 3 files | **3** |
