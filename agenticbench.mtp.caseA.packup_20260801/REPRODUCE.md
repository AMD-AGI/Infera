# Reproduce

Ordered and copy-pasteable. Read `environment.md` first for what must exist
before step 0.

Total wall clock: **~85 min** — ~6 min bring-up per leg (cold start is *not* a
hang), ~8 min correctness, ~67 min measured run.

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

## 0. Prerequisites

- Two spur nodes held with 8 GPUs each (`sbatch -p amd-spur -q amd-burst-qos -N1 -G8`).
- The image built **on both nodes** from branch `yihou.dev.glm52.merged.experiment`
  @ `e56e975`, **with the two uncommitted working-tree changes applied** — see
  `environment.md` § "Uncommitted, and load-bearing". Without them kvd + long
  prompts GPU-fault on gfx950.
- `Optimus-AgenticBench` @ **`1cf01cb`** (branch
  `fix/realistic-profile-session-driver`, **not `main`** — main under-loads
  silently) installed editable into `/shared_nfs/yihou_agentbench/venv`.
- Model weights at `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`.

Stage the scripts:

```bash
cp scripts/* $W/scripts/
cp patches/apply_p1v3.py $W/scripts/
```

## 1. Containers and services

```bash
bash scripts/start_ctr.sh          # both nodes; runs the 8-assertion bytecode gate
bash scripts/start_services.sh     # etcd on prefill; kvd daemon on both
```

`start_ctr.sh` asserts from **freshly compiled bytecode** that the image carries
the ROCm hicache host-alloc fix, mooncake early-send, and the three DSA patches —
plus the behavioural check that `ALLOC_MEMORY_FUNCS["cuda"] is
alloc_with_pin_memory`. A build log saying a patch printed success is not the
same as the interpreter running patched code. Gate on `BYTECODE_GATE OK`.

## 2. ⚠️ Patch the decode leg — MANDATORY

**Without this the run dies within ~13 minutes** with
`Expected lengths.size(0) == B` in the DSA indexer under MTP draft-extend.
Reproduced twice on the vultr sibling cluster.

```bash
spur exec 24301 bash -c '
  docker exec agbench_mtp cp /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py /tmp/dsa_indexer.py.orig
  docker cp '"$W"'/scripts/apply_p1v3.py agbench_mtp:/tmp/apply_p1v3.py
  docker exec agbench_mtp python3 /tmp/apply_p1v3.py'
# expect: patched OK - GLM52_P1V3 occurrences: 3
```

First confirm the image matches what the patch anchors on:

```bash
spur exec 24301 docker exec agbench_mtp md5sum \
  /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
# expect: 632f17acd38737459b43f830ee60ee89
```

Clear bytecode and verify the **loaded** module, not the file on disk — a stale
`__pycache__` has invalidated an experiment in this tree before:

```bash
spur exec 24301 docker exec agbench_mtp bash -c '
  find /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/__pycache__ -name "dsa_indexer*" -delete
  python3 -c "import sglang.srt.layers.attention.dsa.dsa_indexer as m, inspect; print(inspect.getsource(m).count(\"GLM52_P1V3\"))"'
# expect: 3
```

## 3. Boot both legs

**`--mem-fraction-static` on prefill must be 0.80.** The default 0.88 aborts the
leg with `HSA_STATUS_ERROR_OUT_OF_RESOURCES` under Case A's prompt lengths. This
is baked into `scripts/glm52_leg_spur_mtp.sh`; do not override it upward.

```bash
bash scripts/boot.sh prefill 262144 1 0 armB    # ctx kvd=1 mtp=0
bash scripts/boot.sh decode  262144 0 1 armB    # ctx kvd=0 mtp=1
bash scripts/wait_ready.sh 1800
# expect: BOTH READY after ~320s
```

Cold start is 5–6 min (weights → tilelang JIT → DP cudagraph capture). It is not
a hang.

**If you are restarting after a crash, restart BOTH legs together** and poll all
16 GPUs to VRAM 0 % first — restarting one orphans the other's c10d state, and
`boot.sh` waits for process teardown but *not* for VRAM release.

## 4. Gate — every row, before any measured window

```bash
for j in 24300:armB_prefill 24301:armB_decode; do
  spur exec ${j%%:*} bash $W/scripts/gate_armB.sh $W/logs/${j##*:}.log
done
```

| row | prefill | decode |
|---|---|---|
| `ready to roll` | 1 | 1 |
| `Memory access fault` / `HSA_STATUS_ERROR` / `Fatal Python error` | 0 | 0 |
| `Traceback` / `Scheduler hit an exception` | 0 | 0 |
| `Expected lengths.size` | 0 | 0 |
| `infera-kvd adapter connected` | **8** | 0 (by design) |
| `AiterCustomAllreduce` | 0 | 0 |
| `All-reduce call path: NCCL` | 16 | 16 |
| `MC_FORCE_TCP` | 0 | 0 |
| `mem_fraction_static` | **0.8** | 0.85 |
| `context_length` | 262144 | 262144 |
| `speculative_algorithm` | None | **EAGLE** |
| `Errno 98` **after** the ready line | 0 | 0 |
| `avail mem` after memory pool | **284 GB** | 284 GB |

`Errno 98` is checked *after* the ready line specifically: a `--kv-snapshot-port`
collision lets a leg log `ready to roll` and then die during etcd registration —
healthy-looking, never registered.

Logs contain binary bytes; the gate script pipes through `strings`. Plain `grep`
will report `Binary file matches` and count nothing.

## 5. Router

```bash
bash scripts/router.sh
# expect: {"status":"ok","active_workers":2} and both roles registered
```

**Never probe a PD leg's own port from outside its container — it hangs.** Go
through the router.

## 6. Correctness — with the model's own sampling

```bash
/shared_nfs/yihou_agentbench/venv/bin/python3 \
  scripts/correctness.py http://10.245.157.89:8190 120000
```

Expect `short=4/4`. The needle is 3–5 out of 5.

> **`temperature: 0` + MTP is indistinguishable from KV corruption.** GLM-5.2's
> `generation_config.json` specifies **temperature 1.0 / top_p 0.95**; greedy
> decoding sends the model into repetition on a long prompt and EAGLE amplifies
> it. `correctness.py` uses the official sampling.
>
> **A needle miss is not automatically a bug.** Run it twice. If the *failing
> depths move between runs* and every depth returns its exact needle at least
> once, it is sampling variance (`finish=length` repetition), not corruption.
> That is what happened here: 3/5 then 4/5, disjoint failures. Do not sink time
> into it beyond the second run.

## 7. kvd baseline

```bash
spur exec 24300 docker exec agbench_mtp python3 -m infera.kvd.statctl \
  --socket /tmp/kvd/kvd.sock > $W/results/armB_kvd_before.json
```

## 8. The run — 67 min

```bash
nohup bash scripts/run_bench.sh full caseA_armB \
  > $W/logs/caseA_armB.driver.log 2>&1 &
```

`--dashboard-mode` is baked into `run_bench.sh` and is **mandatory**:
`summary.json`, `metrics.jsonl` and `metadata.json` are all written inside
`if dashboard_mode and benchmark_name and data_dir:` (`agent_throughput.py:1674`).
Without it the run completes, prints a full report to stdout, exits 0, and
persists **nothing**.

**Abort live if** in-flight pins at 48, or live sessions climb monotonically
toward 128, or `active_workers` drops to 1. Any of these means backpressure —
not your config — is setting the load.

Watch:

```bash
bash $W/scripts/probe_armB.sh          # driver line + alive check
spur exec 24300 bash $W/scripts/legfault.sh $W/logs/armB_prefill.log
```

Healthy mid-run looks like: in-flight 8–36, sessions oscillating (births ≈
deaths), `faults=0 retract_nonzero=0`.

> **`#retracted-req: 0` matches a naive `grep -c retract`.** Count
> `'#retracted-req: [1-9]'`, not the field name, or you will report thousands of
> retractions that did not happen.

## 9. Collect

```bash
spur exec 24300 docker exec agbench_mtp python3 -m infera.kvd.statctl \
  --socket /tmp/kvd/kvd.sock > $W/results/armB_kvd_after.json

spur exec 24301 bash $W/scripts/acclen.sh      # MTP acceptance ladder

for j in 24300:armB_prefill 24301:armB_decode; do
  spur exec ${j%%:*} bash $W/scripts/legfault.sh $W/logs/${j##*:}.log
done   # want faults=0 on both
```

Artifacts land in
`$W/bench/caseA_armB/caseA_armB/<timestamp>/{summary,metadata}.json` +
`metrics.jsonl`.

## 10. Reproduce the analysis

Every percentile in `analysis/` recomputes from `results/metrics.jsonl.gz`:

```bash
zcat results/metrics.jsonl.gz | python3 -c "
import sys,json,math
def P(a,p):
    a=sorted(a); k=(len(a)-1)*p/100.0; lo,hi=math.floor(k),math.ceil(k)
    return a[lo] if lo==hi else a[lo]+(a[hi]-a[lo])*(k-lo)
t=[]
for line in sys.stdin: t+=json.loads(line).get('new_ttfts') or []
print([round(P(t,p)*1000) for p in [1,5,10,25,50,75,90,95,99,99.5,99.9]])"
```

Split ramp/sustain at `elapsed_seconds < 400`.

---

## Expected result

| | |
|---|---|
| duration | 4,007 s |
| requests | 2,881 sent / 2,811 completed |
| success | **0.9757** (SLA 0.97 ✓) |
| TTFT p50 / p90 / p99 (sustain) | 6,733 / **18,877** / 33,097 ms (SLA p90 ≤ 30,000 ✓) |
| TPOT p50 | 17.9 ms |
| cache hit | 88.82 % actual / 88.99 % ideal |
| MTP acceptance | 2.736 (engine), 4.00 on 3.0 % of batches |
| in-flight peak | 44 / 48 |
| errors | 39, **all** client `aiohttp.ClientTimeout(total=240)` — 0 HTTP failures, 0 engine faults |

## If the prefill leg aborts with `HSA_STATUS_ERROR_OUT_OF_RESOURCES`

`--mem-fraction-static` is too **high**. Lower it (0.80 → 0.75), do not raise it.
Confirm the diagnosis first: the scheduler lines just before the abort will read
`token usage: 0.01–0.05`, i.e. the KV pool is empty and this is activation
memory, not KV exhaustion. Full mechanism in `notes/notes.config.md`.

## If the decode leg dies with `Expected lengths.size(0) == B`

You skipped step 2, or the patch did not reach the running process. Check the
**loaded** module (not the file), then relaunch the decode leg.
