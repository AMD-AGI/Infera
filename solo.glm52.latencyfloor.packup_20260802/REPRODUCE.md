# Reproduce

Ordered and copy-pasteable. Read `environment.md` first for what must exist
before step 0.

**Total wall clock: ~55 min** on a live deployment (~37 min run + ~18 min
checks), or **~70 min** from cold (add ~15 min bring-up; cold start is 5–8 min
per leg and is *not* a hang).

Throughout: `J` = jump host `root@149.28.124.225`, `P` = chi2879 (prefill),
`D` = chi2867 (decode), `W` = `/mnt/vast/c_huggingface/bench_20260801`.

> **Shared-node rules, non-negotiable.** The slurm hold on both nodes belongs to
> `yeandy-debug`. **Never `scancel`.** Kill only your own `bench_run` container
> and your own sglang processes. Do not prune images. Do not fill disks.
>
> **Never bare-`pkill -f infera.kvd`** — `-f` is a regex and `.` is a wildcard,
> so it matches the engine's own `--infera-kvd-socket` argument and kills the
> engine too. Use `scripts/restart_kvd.sh`.

---

## Path A — the deployment is already up (what we did)

This run reused the exact deployment Case A had been measured on, without
restarting anything. If `/health` returns `active_workers: 2`, skip to step 3.

## Path B — from cold

Steps 0–2 below are **identical to the Case A kit**; they are repeated here so
this file is self-contained. Full rationale for every flag lives in
`../caseA.glm52.fullfeature.packup_20260801/REPRODUCE.md`.

### 0. Stage

```bash
W=/mnt/vast/c_huggingface/bench_20260801
ssh root@149.28.124.225 "mkdir -p $W/{scripts,results,logs}"
scp scripts/* root@149.28.124.225:$W/scripts/
scp spec/solo.yaml root@149.28.124.225:$W/solo.yaml
```

Bench repo + venv (once):

```bash
ssh root@149.28.124.225 bash -s <<EOF
cd $W
git clone -b fix/realistic-profile-session-driver <Optimus-AgenticBench> agbench
cd agbench && git checkout 1cf01cb          # NOT main -- main under-loads silently
python3 -m venv $W/venv && $W/venv/bin/pip install -e .
EOF
```

`spec/solo.yaml` already carries the real tokenizer path
(`/mnt/vast/xiaobo/models/GLM-5.2-MXFP4`) — unlike the shipped Case A yaml,
whose `/path/to/GLM-5.2-MXFP4` placeholder **aborts the run**.

### 1. Reset both nodes

```bash
ssh root@149.28.124.225 "ssh chi2879 'cd $W && ROLE=prefill MY_IP=10.2.122.10 ETCD=1 bash scripts/reset_node.sh'"
ssh root@149.28.124.225 "ssh chi2867 'cd $W && ROLE=decode  MY_IP=10.2.122.44        bash scripts/reset_node.sh'"
```

Gate on `PORT_ACTIVE: 8`. Fewer means the libionic injection failed and mooncake
will silently fall back to TCP — the run then "works" while measuring nothing.

### 2. Launch the legs, patch decode, relaunch

```bash
ssh root@149.28.124.225 "ssh chi2879 'cd $W && ROLE=prefill MY_IP=10.2.122.10 ETCD_IP=10.2.122.10 MTP=0 TAG=p4 bash scripts/start_leg.sh'"
ssh root@149.28.124.225 "ssh chi2867 'cd $W && ROLE=decode  MY_IP=10.2.122.44 ETCD_IP=10.2.122.10 MTP=1 TAG=p6 bash scripts/start_leg.sh'"
```

**The decode leg MUST be patched** (`patches/0004`) or it dies within minutes:

```bash
ssh root@149.28.124.225 "ssh chi2867 '
  docker exec bench_run cp /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py /tmp/dsa_indexer.py.orig
  docker cp $W/scripts/apply_p1v3.py bench_run:/tmp/apply_p1v3.py
  docker exec bench_run python3 /tmp/apply_p1v3.py'"
# expect: patched OK - GLM52_P1V3 occurrences: 3
```

Then relaunch the decode leg so the patched module is the one running, and start
the router:

```bash
ssh root@149.28.124.225 "ssh chi2879 'cd $W && BACKEND=rust bash scripts/start_router.sh'"
```

---

## 3. ⚠️ Patch the DRIVER — MANDATORY for this experiment

**Without this the run produces no E2E and no TPOT ladder** — the two things it
exists to measure. See `patches/0005`.

```bash
W=/mnt/vast/c_huggingface/bench_20260801
ssh root@149.28.124.225 "
  cd $W
  cp agbench/agent/agent_throughput.py agbench/agent/agent_throughput.py.orig
  python3 scripts/apply_solo_metrics.py $W/agbench/agent/agent_throughput.py
  find $W/agbench -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
"
# expect: patched OK - SOLO_M1 occurrences: 8 (want 8)
```

Verify the **loaded** module, not the file on disk — stale `__pycache__` has
invalidated an experiment in this tree before:

```bash
ssh root@149.28.124.225 "cd $W/agbench && $W/venv/bin/python -c \"
import agent.agent_throughput as m, inspect
print('SOLO_M1 in loaded module:', inspect.getsource(m).count('SOLO_M1'))
M = m.BenchMetrics()
print('has actual_e2es:', hasattr(M,'actual_e2es'), '| has actual_tpots_aligned:', hasattr(M,'actual_tpots_aligned'))\""
# expect: 8  /  True True
```

`solo_run.sh` also hard-gates on this and refuses to start if the marker is
absent.

## 4. Verify the deployment before spending a window

```bash
ssh root@149.28.124.225 "curl -sf -m10 http://10.2.122.10:8100/health"
# {"active_workers":2,"status":"ok"}

# the engine patch must be in the LOADED module
ssh root@149.28.124.225 "ssh chi2867 'docker exec bench_run python3 -c \"
import sglang.srt.layers.attention.dsa.dsa_indexer as m, inspect
print(\\\"P1V3:\\\", inspect.getsource(m).count(\\\"GLM52_P1V3\\\"))\"'" 2>/dev/null | tail -1
# expect: P1V3: 3
```

Never probe a leg's own port directly — it hangs. Always go through the router.

The full six-feature proof (PD / DPA / RDMA / MTP / kvd / kv-aware) is
`scripts/feature_proof.sh`; it is unchanged from the Case A kit and worth running
if the deployment is cold.

## 5. Offline preview — validate the config parses as intended

No GPU, no server. This is where a broken workload file is caught cheaply.

```bash
ssh root@149.28.124.225 "cd $W/agbench && $W/venv/bin/python -m agent.agent_throughput \
  --workload-config $W/solo.yaml --mode preview --model glm5.2-mxfp4 \
  --tokenizer /mnt/vast/xiaobo/models/GLM-5.2-MXFP4"
```

**The line that matters:**

    turns_per_session   1.0   1.0   1.0   1.0    1/1/1

mean/p50/p90/p99 all exactly 1.0. Input and output triples must match Case A's
(`E[input] = 86,012 tok`).

> **Three warnings are EXPECTED and must be ignored:** *"max_inflight=1 throttles
> even at zero latency"*, *"max_sessions=1 is reached even at zero latency"*,
> *"initial_sessions=1 is far from the steady-state range 23-58"*. Preview models
> load with an open-loop Little's-law formula and does not know `max_sessions`
> gates the spawn. The real loop does. Step 7 verifies this empirically.

## 6. Optional shakeout — 3 min

Cheap insurance that the new arrays actually land before committing 37 minutes.

```bash
ssh root@149.28.124.225 "cd $W && TAG=shake RAMP=60 SUSTAIN=120 setsid bash scripts/solo_run.sh < /dev/null"
```

## 7. The run — 37 min

```bash
ssh root@149.28.124.225 "ssh chi2879 'docker exec bench_run python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock'" \
  > $W/results/solo_full.kvd_before.json

ssh root@149.28.124.225 "cd $W && TAG=full RAMP=400 SUSTAIN=1800 setsid bash scripts/solo_run.sh < /dev/null"
```

`--dashboard-mode` is baked into `solo_run.sh` and is **mandatory** — without it
nothing structured is persisted and the run is unrecoverable once the terminal
scrolls.

Watch (`In-flight` must never read 2):

```bash
ssh root@149.28.124.225 'tail -c 300 '$W'/logs/solo_full.log | tr "\r" "\n" | tail -2'
```

**Abort live if** `In-flight` shows 2, or the driver prints
`Hit max_inflight` — either means the session accounting is not doing what this
experiment assumes, and the numbers would be uninterpretable.

## 8. Collect

```bash
ssh root@149.28.124.225 "ssh chi2879 'docker exec bench_run python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock'" \
  > $W/results/solo_full.kvd_after.json

# artifacts land in $W/results/solo_full/solo_full/<timestamp>/
#   summary.json  metadata.json  metrics.jsonl
```

Two gates that must both pass:

```bash
# (a) the safety valve never fired
ssh root@149.28.124.225 "grep -c 'Hit max_inflight' $W/logs/solo_full.log"    # want 0

# (b) no engine faults DURING THE WINDOW. Scope by timestamp -- these logs are
# appended all day and contain earlier runs' crashes. Use `strings`, never plain
# grep: server logs contain binary bytes.
for n in "chi2879 p4_prefill" "chi2867 p6_decode"; do set -- $n
  ssh root@149.28.124.225 "ssh $1 'strings $W/logs/$2.log \
    | grep -E \"^\\[2026-08-01 16:[0-4][0-9]\" \
    | grep -icE \"Expected lengths.size|Aborted|OUT_OF_RESOURCES|Traceback|exception\"'"
done   # want 0 0
```

> A whole-file grep returns non-zero even on a perfect run: the `server_args=`
> dump contains the substring `abort_on_priority_when_disabled`, and the decode
> log holds two `Aborted by AbortReq` lines from the earlier Case A window.
> Scope by timestamp or you will chase a ghost.

## 9. Reproduce the analysis

Every table in `analysis/solo_latency.md` recomputes from the raw arrays.
Nothing is read from `summary.json`.

```bash
python3 scripts/solo_analyze.py results/metrics.jsonl.gz --phase sustain
```

The solo-vs-Case-A comparison (the kit's headline finding) needs the sibling
Case A kit present at `../caseA.glm52.fullfeature.packup_20260801/`:

```bash
python3 scripts/compare_vs_caseA.py
```

Spot-check by hand:

```bash
zcat results/metrics.jsonl.gz | python3 -c "
import sys,json,math
def P(a,p):
    a=sorted(a); k=(len(a)-1)*p/100.0; lo,hi=math.floor(k),math.ceil(k)
    return a[lo] if lo==hi else a[lo]+(a[hi]-a[lo])*(k-lo)
t=[];e=[]
for line in sys.stdin:
    r=json.loads(line)
    if r.get('phase')!='sustain': continue
    t+=r.get('new_ttfts') or []; e+=r.get('new_e2es') or []
print('n=',len(t))
print('ttft p50/p90/p99', [round(P(t,p)*1000) for p in (50,90,99)])
print('e2e  p50/p90/p99', [round(P(e,p)*1000) for p in (50,90,99)])"
# n= 102
# ttft p50/p90/p99 [1563, 2899, 4472]
# e2e  p50/p90/p99 [4124, 48140, 121243]
```

---

## Expected result

| | |
|---|---|
| duration | 2,200 s (ramp 400 + sustain 1800) |
| requests | 145 total, **102 in the measured window** |
| errors | **0** |
| in_flight distinct values | **{0, 1}** — never 2 |
| `Hit max_inflight` count | **0** |
| duty cycle | 96.4 % |
| TTFT p50 / p90 / p99 | 1,563 / 2,899 / 4,472 ms |
| TPOT p50 / p90 / p99 | 10.68 / 12.75 / 14.65 ms (min 7.78) |
| E2E p50 | 4,124 ms |
| MTP acceptance mean | 2.069 (engine log `accept len` 2.20–2.52 agrees) |
| cache hit | 88.95 % per-request mean / 89.0 % ideal |
| kvd | **+0 gets**, +1,122 sets |

## If it doesn't reproduce

- **`In-flight: 2` appears** → the workload file was not the one loaded. Check
  the driver's `Applied: N parameters` line and that no CLI flag overrode
  `max_sessions`. `solo_run.sh` deliberately passes none of the load knobs.
- **No `new_e2es` key in metrics.jsonl** → SOLO_M1 did not reach the running
  process. Check the *loaded* module (step 3), not the file.
- **Decode leg dies at 2–13 min on `Expected lengths.size(0) == B`** → the P1V3
  engine patch is missing. See `patches/0004` and `notes/notes.dsa.mtp.crash.md`.
- **Sample count far below 102** → check the output-length draw; a run that
  happens to sample several 17K-token generations spends its window on a handful
  of requests. This is real variance at n≈100, not a fault.
