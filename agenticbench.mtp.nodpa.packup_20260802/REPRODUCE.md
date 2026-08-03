# Reproduce — noDPA

Ordered and copy-pasteable. Read `environment.md` first for what must exist
before step 0.

Total wall clock from nothing: **~2 h for one arm** — ~15 min to hold two nodes,
~12 min to build the image on both (in parallel), ~12 min bring-up per leg, ~10
min correctness, ~6 min probe, ~33 min measured run.

**Both arms: ~2 h 50 min.** The second arm only needs a prefill reboot (~8 min),
a fresh probe (~6 min) and its own 33-min window; containers, image, decode leg,
etcd and kvd all carry over.

Throughout: `P` = job **28490** / `crsuse2-m2m-231` / `10.245.150.172` (prefill,
etcd, router, kvd), `D` = job **28485** / `crsuse2-m2m-276` / `10.245.152.249`
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

## ⚠️ Read this first — the three things unique to reproducing noDPA

### 1. `DPA=0` alone does NOT isolate DP-attention. It moves four things.

The stock leg script gated four behaviours behind one `if`. Only one is the
variable. `scripts/glm52_leg_spur_mtp.sh` in this kit is the **corrected** version;
if you use the branch's own, you will measure something else:

| | stock `DPA=0` | this kit | disposition |
|---|---|---|---|
| `--dp-size` / `--enable-dp-attention` | dropped | dropped | **the variable** |
| `--ep-size 8` | **dropped** | **kept** | MoE expert parallelism — a different axis; dropping it makes this a 2-variable run |
| `--chunked-prefill-size` | **8192 hardcoded** | caller-pinned | see below — the trap |
| `--enable-prefill-delayer`, `SGLANG_DP_USE_GATHERV` | dropped | dropped | DPA's own machinery; correctly leaves with it |

### 2. The chunk size trap — `--chunked-prefill-size` is GLOBAL and DPA divides it

`server_args.py:4902` is a **division**, not a clamp:

```python
if self._resolved().enable_dp_attention:
    self.chunked_prefill_size = self.chunked_prefill_size // self.dp_size
    logger.warning(f"DP attention is enabled. The chunked prefill size is adjusted to "
                   f"{self.chunked_prefill_size} to avoid MoE kernel issues.")
```

So lat1 at dp8 requested 65,536, its `server_args=` reads **8,192 — per rank** —
and the machine-wide budget is still **65,536**. With DPA off there is no
division: whatever you pass is the global budget.

**Pass `CHUNK=65536` on this arm.** That is what matches lat1's machine. Passing
8,192 "because that is what lat1's `server_args=` shows" gives ⅛ the per-step
budget — and this run made exactly that mistake, measured a full window on it,
and had to rerun (see `notes/notes.nodpa.md` §3).

| arm | pass `CHUNK=` | `server_args=` | **global tokens/step** |
|---|---:|---:|---:|
| lat1 (DPA on, dp8) | 65536 | 8192 (per rank) | **65,536** |
| **noDPA MAIN** | **65536** | 65536 | **65,536** ✓ |
| noDPA chunk-control | 8192 | 8192 | 8,192 (⅛) |

**Decode does not enter this.** A PD decode leg runs no prefill, so its
`chunked_prefill_size` is inert on every arm — it is printed in `server_args=`
anyway, which is what made an earlier version of this kit treat it as matched.

Cross-check that needs no flag reasoning at all — `#new-token` in the prefill log:
the MAIN arm reaches **65,536** while the control arm's ceiling *is* 8,192.
Batches above 8,192 are impossible under an 8,192 budget, so this proves the
larger global budget from the engine's own counters:

```bash
zcat logs/chunk65536_prefill_tail6000.log.gz | grep -oE '#new-token: [0-9]+' \
  | awk '{print $2}' | sort -n | tail -1     # -> 65536
zcat logs/chunk8192_prefill_tail6000.log.gz  | grep -oE '#new-token: [0-9]+' \
  | awk '{print $2}' | sort -n | tail -1     # -> 8192
```

### 3. This arm cannot boot at the DPA arm's `mem-fraction-static`

`GMU=0.80` dies with:

    HSA_STATUS_ERROR_OUT_OF_RESOURCES ... Available Free mem : 254 MB
    Fatal Python error: Aborted

`token usage: 0.04` at the time — the KV pool is empty, so this is **activation**
memory, not KV exhaustion. Without dp-attention one rank computes attention over
the whole 8192-token chunk instead of its 1/8 slice. **Use `GMU=0.70`.** This is a
result, not a workaround — see `analysis/nodpa_vs_lat1.md`.

---

## 0. Prerequisites

- Two spur nodes with 8 GPUs each. Expect `JobHoldMaxRequeue` on bad nodes —
  catch the node each held job landed on and `--exclude` it:

```bash
sbatch --parsable -p amd-spur -q amd-burst-qos -N1 -G8 -t 24:00:00 hold.sh
# then verify a REAL hold; a job can show RUNNING for one poll and still requeue:
spur exec <job> true && echo HELD
```

- `Optimus-AgenticBench` @ **`1cf01cb`** (branch
  `fix/realistic-profile-session-driver`, **not `main`** — main under-loads
  silently) installed editable into `/shared_nfs/yihou_agentbench/venv`.
- Model weights at `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`.

```bash
mkdir -p $W/{scripts,results,logs,build}
cp scripts/* $W/scripts/
cp spec/nodpa_full.yaml spec/nodpa_probe.yaml \
   /home/yihou/dev/git/infera.merge.liying.kv.mtp/work.agenticbench.mtp/workloads/
```

> `run_nodpa.sh` resolves its workload as
> `…/work.agenticbench.mtp/workloads/nodpa_<variant>.yaml`. Either put the YAMLs
> there (above) or edit `WL=` in the script.

## 1. Build the image on BOTH nodes

Apply `patches/0001-dockerfile-rocm-hicache-hostalloc.patch` to a clean `e56e975`
checkout first, plus `patches/sglang_rocm/` — without them kvd + long prompts
GPU-fault on gfx950 (`xnack-`, no page-migration fallback).

```bash
bash scripts/stage_source.sh                 # worktree -> $W/build/src.tar
nohup bash scripts/build_image.sh 28490 &    # ~12 min, parallel
nohup bash scripts/build_image.sh 28485 &
# wait for both:
cat $W/build/build_crsuse2-m2m-{231,276}.status   # expect: ok sha256:...
```

> `stage_source.sh`'s exclude regex must skip the `agenticbench.*` packup dirs or
> the build context balloons (18 MB vs 12 MB here). Fixed in this kit's copy.

## 2. Containers + the bytecode gate

```bash
bash scripts/start_ctr.sh 28490 prefill      # expect BYTECODE_GATE OK
bash scripts/start_ctr.sh 28485 decode       # expect BYTECODE_GATE OK
```

8 assertions read from **freshly compiled bytecode**, not source — a stale
`__pycache__` running unpatched code has invalidated experiments on this stack
twice.

## 3. `GLM52_P1V3` on the decode leg — MANDATORY

Without it the run dies with `Expected lengths.size(0) == B` under MTP
draft-extend.

```bash
spur exec 28485 bash -c "export DOCKER_CONFIG=/tmp/dockercfg
docker exec agbench_mtp bash -c '
  F=/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
  md5sum \$F        # expect 632f17acd38737459b43f830ee60ee89
  cp \$F /tmp/dsa_indexer.py.orig
  python3 $W/scripts/apply_p1v3.py'"

# verify the LOADED module, not the file -- stale __pycache__ has burned this tree before
spur exec 28485 docker exec agbench_mtp python3 -c \
  "import sglang.srt.layers.attention.dsa.dsa_indexer as m, inspect; print(inspect.getsource(m).count('GLM52_P1V3'))"
# expect: 3
```

## 4. etcd + kvd

```bash
bash scripts/start_services.sh 28490 prefill 10.245.150.172   # etcd + kvd
bash scripts/start_services.sh 28485 decode  10.245.152.249   # kvd only
```

## 5. Boot the legs — the one command that encodes the whole experiment

```bash
# PREFILL: DPA OFF (the variable), chunk matched to lat1's GLOBAL 65536, GMU lowered
DPA=0 CHUNK=65536 GMU=0.70 bash scripts/boot.sh prefill 262144 1 0 nodpa

# DECODE: unchanged from lat1 -- DPA on, MTP on. CHUNK is inert here (no prefill
# on a PD decode leg); passed only so both legs take one uniform command.
DPA=1 CHUNK=65536 bash scripts/boot.sh decode 262144 0 1 nodpa
```

To reproduce the **chunk-control arm** instead (`results/chunk8192_ARM/`), pass
`CHUNK=8192` on prefill. Everything else is identical. Running both is what
demonstrates the chunk effect is nil — and therefore what licenses reading the
lat1-vs-noDPA delta as a DPA effect.

Weight load is ~7 min (prefill) / ~12 min (decode, extra EAGLE draft model). A
cold start is **not** a hang.

**Verify from `server_args=`, never from the launch command** — the engine
overrides things:

```bash
spur exec 28490 bash -lc "strings $W/logs/nodpa_prefill.log | grep -oE \
  'enable_dp_attention=[A-Za-z]+|chunked_prefill_size=[0-9]+|ep_size=[0-9]+|mem_fraction_static=[0-9.]+' | head -4"
# expect: enable_dp_attention=False  chunked_prefill_size=65536  ep_size=8  mem_fraction_static=0.7
#   (chunk-control arm: chunked_prefill_size=8192)
```

## 6. Router + gate table

```bash
bash scripts/router.sh          # expect {"status":"ok","active_workers":2}
bash scripts/gate.sh nodpa      # NOTE the tag -- default is g1 and reads STALE logs
```

The tag matters: `gate.sh` with no argument reads `$W/logs/g1_*.log`, which on a
reused workspace is a previous run's file and will report a previous run's
`dp_size=8`. This bit during this very run.

Rows that must hold, and the two that **prove the variable**:

| row | prefill | decode |
|---|---|---|
| `ready to roll` | 1 | 1 |
| `Memory access fault` / `Scheduler hit an exception` / `Traceback` | 0 | 0 |
| **`dp_size=8`** | **0** ← the variable | 1 |
| **`scheduler_DP` procs** | **0** ← the variable | 8 |
| `infera-kvd adapter` | 25 | 0 (by design) |
| `speculative EAGLE` | 0 | 1 |
| `MC_FORCE_TCP` | 0 | 0 |
| `Errno 98` **after** the ready line | 0 | 0 |

Grep logs through `strings` (they contain binary bytes). Never probe a PD leg's
own port; it hangs — go through the router.

## 7. Correctness

```bash
/shared_nfs/yihou_agentbench/venv/bin/python3 scripts/correctness.py http://10.245.150.172:8190
```

Uses the model's own sampling (temp 1.0 / top_p 0.95). **`temperature: 0` + MTP is
indistinguishable from KV corruption** — do not "fix" the test by setting it to 0.

Expect short factual **4/4**. Needle was **2/5 cold, 5/5 warm** here. Do not sink
time into it: the discriminator is whether the *failing depths move* between runs
and whether every depth returns its exact 7-digit value at least once. Both held,
which is sampling variance — the same signature Case A documented (3/5 then 4/5).

## 8. kvd baseline

```bash
spur exec 28490 docker exec agbench_mtp python3 -m infera.kvd.statctl \
  --socket /tmp/kvd/kvd.sock > $W/results/nodpa64_kvd_before.json
```

## 9. Probe — 6 min, and it earned its keep again

```bash
nohup bash scripts/run_nodpa.sh probe nodpa64_probe > $W/nodpa64_probe.console.log 2>&1 &
```

```bash
D=$(ls -d $W/bench/nodpa64_probe/nodpa64_probe/*/ | tail -1)
/shared_nfs/yihou_agentbench/venv/bin/python3 scripts/lat1_analyze.py "$D" | head -20
```

| gate | expect | what a failure means |
|---|---|---|
| `in_flight max` / `sessions_active max` | **1 / 1** | the concurrency-1 guarantee broke; this is not the experiment |
| `cache actual` vs `ideal` | ~0.83 rising to 0.889 | **~1.00 = seed collision**; a *low* value on a cold engine is normal (radix tree still filling) |
| artifacts | `summary.json` + `metrics.jsonl` + `metadata.json` | `--dashboard-mode` missing |

The probe caught the seed-range defect here (see below), which would otherwise
have killed a 33-minute window at startup.

### Seeds — the defect that repeats

`random_seed` must be **≤ 2**32-1** (numpy seeds `np.random` directly and rejects
larger with `ValueError: Seed must be between 0 and 2**32 - 1`). This run's first
probe died on `20260802002` ≈ 2.03e10. Shipped values: **`2026080211`** (MAIN full)
and **`2026080212`** (MAIN probe), both ≈ 2.03e9 and both valid. The retained
chunk-control arm used `2026080201` / `2026080202`.

They must also be distinct from each other **and** from every seed the engine has
already seen (lat1 used `20260802` / `2026080299`, Case A `1337`). The driver's
fresh-content seed is the run-local `request_id` (`agent_throughput.py:2187`) and
the shared-base seed is a constant (`:2054`), so two runs sharing `random_seed`
replay byte-identical prompts into a warm radix tree and measure cache hits
instead of compute. Full mechanism: `../agenticbench.mtp.lat1.packup_20260802/notes/notes.lat1.md`.

## 10. The run — 33 min

```bash
nohup bash scripts/run_nodpa.sh full nodpa64_full > $W/nodpa64_full.console.log 2>&1 &
```

`--dashboard-mode` is set inside the script and is **mandatory**: `summary.json`,
`metrics.jsonl` and `metadata.json` are all written inside
`if dashboard_mode and benchmark_name and data_dir:` (`agent_throughput.py:2067`).
Without it the run completes, prints a full report, exits 0 — and persists nothing.

Live abort criteria:

* `In-flight` ever **> 1**, or `Sessions` ever **> 1/1** → the guarantee broke.
* Cache hit reading far above 89 % → seed collision; kill it.

Expect **~115 completed requests at ~17 s each** on the MAIN arm (the
chunk-control arm ran 175 at ~11 s — it drew a lighter input distribution, which
is exactly why the bin-matched table rather than the raw p50 is the result).

```bash
tail -c 200 $W/nodpa64_full.console.log
```

## 11. Post-run — kvd, faults, acceptance

```bash
spur exec 28490 docker exec agbench_mtp python3 -m infera.kvd.statctl \
  --socket /tmp/kvd/kvd.sock > $W/results/nodpa64_kvd_after.json

for j in 28490:prefill 28485:decode; do
  spur exec ${j%%:*} bash -lc "strings $W/logs/nodpa_${j##*:}.log | grep -cE \
    'HSA_STATUS_ERROR|Fatal Python error|Expected lengths.size|Memory access fault|Scheduler hit an exception|Traceback'"
done
# expect: 0 and 0

# retractions -- grep for a NON-ZERO count, not the word
spur exec 28485 bash -lc "strings $W/logs/nodpa_decode.log | grep -c '#retracted-req: [1-9]'"
# expect: 0   (a bare `grep -ci retract` matches the routine '#retracted-req: 0' field and reads 1829)

spur exec 28485 bash -lc "strings $W/logs/nodpa_decode.log | grep -oE 'accept len: [0-9.]+' \
  | tail -4000 | awk '{print \$3}' | sort -n \
  | awk '{a[NR]=\$1;s+=\$1} END{printf \"n=%d mean=%.3f p50=%.2f p90=%.2f\n\",NR,s/NR,a[int(NR*0.5)],a[int(NR*0.9)]}'"
# expect: mean ~2.85 (MAIN arm)
```

`accept len` comes from the **decode leg's log**, not the driver: the driver's
acceptance averages SSE chunk sizes and undercounts (2.0 vs 2.934 here). The
engine counter is authoritative.

**kvd `gets` barely moves** (MAIN +1,424 against `sets` +9,796; the control arm
was +0). Prefill kvd only fetches on
a radix-tree miss, and at 89 % planned hit with a single session the in-GPU tree
serves everything. Same as lat1; a property of the workload, not a defect.

## 12. Analysis

```bash
/shared_nfs/yihou_agentbench/venv/bin/python3 scripts/lat1_analyze.py \
  $W/bench/nodpa64_full/nodpa64_full/<TIMESTAMP> \
  --json $W/results/nodpa64_ladders.json
```

(The analyzer is lat1's, unmodified and deliberately so — using the same tool on
both arms removes it as a source of difference. Its banner says "lat1 analysis";
that is cosmetic.)

The cross-run tables — bin-matched TTFT and both fits — come from a
**self-contained snippet inlined in `analysis/nodpa_vs_lat1.md`**
(§ "Reproduce both tables above"). It reads only this kit plus
`../agenticbench.mtp.lat1.packup_20260802/results/metrics.jsonl.gz`, takes no
arguments, and prints both tables.

## Verification — what "done" means

| gate | expected | actual |
|---|---|---|
| `BYTECODE_GATE`, both nodes | OK | **OK** |
| `enable_dp_attention`, prefill | **False** | **False** |
| `scheduler_DP` procs, prefill / decode | 0 / 8 | **0 / 8** |
| `chunked_prefill_size` global, MAIN arm | 65536 | **65536** |
| `ep_size`, both arms | 8 | **8** |
| `in_flight` / `sessions_active` max | 1 / 1 | **1 / 1** |
| cache actual / ideal, MAIN | ≈0.889 / 0.890 | **0.8893 / 0.8899** |
| completed / errors, MAIN | ~115 / ≤1 | **115 / 1** |
| engine faults, both legs | 0 | **0** |
| retractions, both legs | 0 | **0** |
| `accept len` (engine) | 2.1–3.0 | **2.846** (MAIN) / 2.934 (control) |
| TTFT-vs-length R² (outliers excluded) | > 0.8 | **0.8324** (MAIN) / 0.9747 (control) |
| artifacts | 3 files | **3** |
