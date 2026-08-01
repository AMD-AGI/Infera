# Reproduce

Ordered and copy-pasteable. Read `environment.md` first for what must exist before
step 0.

Total wall clock: **~90 min** — ~15 min bring-up (cold start is 5–8 min per leg and
is *not* a hang), ~17 min probe, ~67 min full run.

Throughout: `J` = jump host `root@149.28.124.225`, `P` = chi2879 (prefill),
`D` = chi2867 (decode), `W` = `/mnt/vast/c_huggingface/bench_20260801`.

> **Shared-node rules, non-negotiable.** The slurm hold on both nodes belongs to
> `yeandy-debug`. **Never `scancel`.** Kill only your own `bench_run` container and
> your own sglang processes. Do not prune images. Do not fill disks.

---

## 0. Stage

```bash
W=/mnt/vast/c_huggingface/bench_20260801
ssh root@149.28.124.225 "mkdir -p $W/{scripts,results,logs}"
scp scripts/* root@149.28.124.225:$W/scripts/
```

Bench repo + venv + workload (once):

```bash
ssh root@149.28.124.225 bash -s <<EOF
cd $W
git clone -b fix/realistic-profile-session-driver <Optimus-AgenticBench> agbench
cd agbench && git checkout 1cf01cb          # NOT main -- main under-loads silently
python3 -m venv $W/venv && $W/venv/bin/pip install -e .
cp agent/workloads/glm52_crxx_caseA.fix.yaml $W/caseA.yaml
EOF
# then edit $W/caseA.yaml: tokenizer -> /mnt/vast/xiaobo/models/GLM-5.2-MXFP4
# (the shipped /path/to/GLM-5.2-MXFP4 placeholder ABORTS the run)
```

## 1. Reset both nodes

```bash
ssh root@149.28.124.225 "ssh chi2879 'cd $W && ROLE=prefill MY_IP=10.2.122.10 ETCD=1 bash scripts/reset_node.sh'"
ssh root@149.28.124.225 "ssh chi2867 'cd $W && ROLE=decode  MY_IP=10.2.122.44        bash scripts/reset_node.sh'"
```

Gate on `PORT_ACTIVE: 8` in the output. Fewer means the libionic injection failed and
mooncake will silently fall back to TCP — the run then "works" while measuring nothing.

## 2. Launch the legs

```bash
# prefill: MTP off, gmu 0.80 (see Phase 1 patches/0002 -- prefill OOM is fixed by
# LOWERING mem-fraction-static, the opposite of the decode retract fix)
ssh root@149.28.124.225 "ssh chi2879 'cd $W && ROLE=prefill MY_IP=10.2.122.10 ETCD_IP=10.2.122.10 MTP=0 TAG=p4 bash scripts/start_leg.sh'"

# decode: MTP ON, gmu 0.85
ssh root@149.28.124.225 "ssh chi2867 'cd $W && ROLE=decode  MY_IP=10.2.122.44 ETCD_IP=10.2.122.10 MTP=1 TAG=p6 bash scripts/start_leg.sh'"
```

Cold start is **5–8 min** (weights → tilelang JIT → DP cudagraph capture).

## 3. ⚠️ Patch the decode leg — MANDATORY

**Without this the run dies within 13 minutes.** Stock `merged-e` crashes the decode
leg on `Expected lengths.size(0) == B` in the DSA indexer under MTP draft-extend.
Reproduced twice. See `notes/notes.dsa.mtp.crash.md`.

```bash
ssh root@149.28.124.225 "ssh chi2867 '
  docker exec bench_run cp /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py /tmp/dsa_indexer.py.orig
  docker cp $W/scripts/apply_p1v3.py bench_run:/tmp/apply_p1v3.py
  docker exec bench_run python3 /tmp/apply_p1v3.py'"
# expect: patched OK - GLM52_P1V3 occurrences: 3
```

Clear bytecode and verify the **loaded** module, not the file on disk — stale
`__pycache__` has invalidated an experiment in this tree before:

```bash
ssh root@149.28.124.225 "ssh chi2867 'docker exec bench_run bash -c \"
  find /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/__pycache__ -name dsa_indexer\* -delete
  python3 -c \\\"import sglang.srt.layers.attention.dsa.dsa_indexer as m,inspect; print(inspect.getsource(m).count(chr(71)+chr(76)+chr(77)+chr(53)+chr(50)+chr(95)+chr(80)+chr(49)+chr(86)+chr(51)))\\\"\"'"
# expect: 3
```

Then **relaunch the decode leg** (step 2's decode line) so the patched module is the
one actually running.

## 4. Router

```bash
ssh root@149.28.124.225 "ssh chi2879 'cd $W && BACKEND=rust bash scripts/start_router.sh'"
```

## 5. Verify all five features BEFORE spending a measured window

A green run that proves nothing is the default outcome here. Each feature has a
positive signal that goes red if absent:

```bash
ssh root@149.28.124.225 "ssh chi2879 'cd $W && bash scripts/feature_proof.sh'"
```

| feature | expected |
|---|---|
| PD | `/v1/workers` shows `disagg_mode` prefill **and** decode, both `active` |
| DPA | `dp_size: 8` on both; 8 `sglang::scheduler_DP*` per node |
| RDMA | 0 `MC_FORCE_TCP`, 0 mooncake init failures |
| MTP | `accept len` in the decode log, in the 1.5–3.9 band — **4.00 is BAD** (repetition loop) |
| kvd | `statctl` gets/hits rising with **sets flat** |
| kv-aware | all 8 ranks picked on both legs, prefill skewed / decode uniform |

Health check — **never probe a leg's own port directly, it hangs**:

```bash
ssh root@149.28.124.225 "curl -sf -m10 http://10.2.122.10:8100/health"
# {"active_workers":2,"status":"ok"}
```

## 6. Offline preview (no GPU, no server)

```bash
ssh root@149.28.124.225 "cd $W/agbench && $W/venv/bin/python -m agent.agent_throughput \
  --workload-config $W/caseA.yaml --mode preview --model glm5.2-mxfp4 \
  --tokenizer /mnt/vast/xiaobo/models/GLM-5.2-MXFP4"
```

Expect `E[turns]=9.64`, `E[input]=86,012 tok`, offered rate `0.96 req/s`, and one
warning: *`max_inflight=48` throttles once E2E exceeds ~32 s*. Watch that one.

## 7. Probe run — 17 min

Purpose is CASE_AB_GUIDE Step 3: measure E2E so the birth rate can be re-solved.

```bash
ssh root@149.28.124.225 "ssh chi2879 'docker exec bench_run python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock'" \
  > /tmp/caseA_probe.kvd_before.json

ssh root@149.28.124.225 "cd $W && TAG=probe RAMP=400 SUSTAIN=600 setsid bash scripts/caseA_run.sh < /dev/null"
```

Read off the summary: mean E2E, actual cache hit, in-flight peak.
Ours: **E2E 17.8 s, cache 88.9 %, in-flight max 18 / 48.**

### Then do the Step-3 arithmetic — and expect NOT to change anything

    new_session_rate = N_target / (E[turns] x (measured_E2E + E[delay]))
                     = 32 / (9.50 x (17.8 + 18.0)) = 0.0941 /s

Shipped is `0.10`; a 6 % move is inside noise, so **leave it**.

> ⚠️ **The probe will show N≈17, not 32. Do not "fix" this by raising the rate.**
> It is tail censoring — a 1,000 s window cannot realize a 240 s p99 inter-turn
> delay or a 103-turn session. At 4,000 s the population rises on its own
> (we measured 22 → 36 across quarters). Doubling the rate to force N=32 lands
> the full run near N≈70, pins `max_inflight`, and backpressure — not your
> config — sets the load. That is exactly how the spur run lost a 20-min window.

## 8. Full run — 67 min

```bash
ssh root@149.28.124.225 "ssh chi2879 'docker exec bench_run python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock'" \
  > $W/results/caseA_full.kvd_before.json

ssh root@149.28.124.225 "cd $W && TAG=full RAMP=400 SUSTAIN=3600 setsid bash scripts/caseA_run.sh < /dev/null"
```

`--dashboard-mode` is baked into `caseA_run.sh` and is **mandatory** — without it
nothing structured is persisted and the run is unrecoverable once the terminal
scrolls.

**Abort live if** in-flight pins at 48, or live sessions climb monotonically toward
`max_sessions=128`, or `active_workers` drops to 1.

Watch:

```bash
ssh root@149.28.124.225 'tail -c 400 '$W'/logs/caseA_full.log | tr "\r" "\n" | tail -2
curl -sf -m10 http://10.2.122.10:8100/health'
```

## 9. Collect

```bash
ssh root@149.28.124.225 "ssh chi2879 'docker exec bench_run python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock'" \
  > $W/results/caseA_full.kvd_after.json

# artifacts land in $W/results/caseA_full/caseA_full/<timestamp>/
#   summary.json  metadata.json  metrics.jsonl
```

Scan both engine legs for faults (**`strings`, never plain `grep`** — server logs
contain binary bytes):

```bash
for n in "chi2879 p4_prefill" "chi2867 p6_decode"; do set -- $n
  ssh root@149.28.124.225 "ssh $1 'strings $W/logs/$2.log | grep -icE \"Expected lengths.size|abort|OUT_OF_RESOURCES|Traceback\"'"
done   # want 0 0
```

## 10. Reproduce the analysis

Every number in `analysis/` recomputes from `results/metrics.jsonl.gz`:

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

---

## Expected result

| | |
|---|---|
| duration | 4,006 s |
| requests | 2,988 sent / 2,952 completed |
| success | **0.988** (SLA 0.97 ✓) |
| TTFT p50 / p90 / p99 | 4,378 / 8,940 / 16,492 ms (SLA p90 ≤ 30,000 ✓) |
| TPOT p50 | 14.8 ms |
| cache hit | 89.2 % actual / 89.0 % ideal |
| MTP acceptance | 2.04 per-request mean |
| kvd | 452 gets / 452 hits / 0 misses |
| errors | 18, **all** client `aiohttp.ClientTimeout(total=240)` — 0 HTTP failures, 0 engine faults |

## If it crashes at 2–13 minutes with `Expected lengths.size(0) == B`

You skipped step 3, or the patch did not reach the running process. Check the
**loaded** module (not the file), then relaunch the decode leg. To capture the
mechanism yourself, relaunch with `DSA_ROWS=1` and look for the fatal shape:

    mode=IDLE q_fp8=(1,32,128) q_offset=2 ntnp=0 agree=False lengths=(2,) -> mqa_q=(1,32,128)
                        ^ 1 row handed to the kernel        ^ against 2 lengths entries
