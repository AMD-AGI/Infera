# REPRODUCE

Top-to-bottom, copy-pasteable. Every script referenced lives in `scripts/`.
Cold start is **6–9 min per leg** (weights → tilelang JIT → cudagraph capture) —
that is not a hang. Total wall clock: **~25 min** bring-up + **45 min**
measurement (3 × 900 s + inter-point overhead).

```bash
# Constants used throughout. Run everything from the spur LOGIN node
# (crs-m2m-cpu-spur-012) — /shared_nfs and the spur CLI are local there.
W=/shared_nfs/yihou_agentx_caseA
PREFILL_JOB=35748 ; PREFILL_IP=10.245.145.242   # crsuse2-m2m-268
DECODE_JOB=35749  ; DECODE_IP=10.245.152.60     # crsuse2-m2m-288
CTR=agx_caseA
```

> **Node substitution is expected**, since spur allocations are ephemeral. If
> you change nodes: the IPs change, `ETCD_IP` must be the **prefill** node, and
> the router follows the prefill node. See step 1 — **screen candidates before
> committing to them**.

---

## 0. Prerequisites

- **`export DOCKER_CONFIG=/tmp/dockercfg` before every docker call.** Docker
  29's buildx plugin discovery needs it.
- **Never background a long docker client inside `spur exec`** — the exec
  namespace teardown kills it even under `nohup`/`setsid`. Either run it in the
  foreground of the exec, or stage a script and `docker exec -d` it.
- `ssh` to compute nodes is blocked; the `Permission denied (publickey)` error
  is misleading. Use `spur exec`.
- `sbatch --output` is **silently ignored**; logs go to `~/spur-<jobid>.out`.
- Model weights at `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`.
- Build context `build/src.tar` (in this kit) staged to `$W/build/src.tar`.

## 1. Hold two nodes — and SCREEN them

**`-G8` reserves a scheduling slot, not exclusive GPU memory.** Other tenants
can and do occupy the GPUs on a node you hold. Screening on the image alone
cost this experiment two dead legs (see `notes.md` Trap 1).

```bash
cat > $W/hold.sh <<'EOF'
#!/bin/bash
echo "held: $(hostname) job=$SLURM_JOB_ID gpus=$ROCR_VISIBLE_DEVICES"
sleep 43200
EOF

for i in $(seq 1 8); do
  sbatch --parsable -p amd-spur -q amd-burst-qos -N1 -G8 -t 12:00:00 \
         --job-name=agx_$i $W/hold.sh
done
sleep 35 ; squeue -u "$USER" -o "%i %j %T %N"
```

Screen every RUNNING candidate — **both** the image and free VRAM:

```bash
bash scripts/screen.sh <jobid>
#   crsuse2-m2m-268 img=0 maxUsedGB=0     <- GPUs free, needs a build
#   crsuse2-m2m-276 img=1 maxUsedGB=278   <- has image, GPUs FULL -> reject
```

Keep two nodes with **`maxUsedGB` near 0**; `scancel` the rest. Confirm each is
a real hold: `spur exec <job> true` must exit 0 (a job can show `RUNNING` for
one poll and then requeue).

## 2. Build the engine image on both nodes (parallel, ~4 min)

Skip only if a node already carries `infera/engine-sglang:final-pr` **and** its
GPUs are free. Do **not** substitute `merged-mtp` — it fails the P1V3 gate.

```bash
nohup bash scripts/build_image.sh $PREFILL_JOB > $W/build/drv_pf.log 2>&1 &
nohup bash scripts/build_image.sh $DECODE_JOB  > $W/build/drv_dc.log 2>&1 &
wait ; cat $W/build/status_*        # expect: ok <sha256>
```

## 3. Start the containers — the bytecode gate is the first real check

```bash
IMG=infera/engine-sglang:final-pr CTR=$CTR bash scripts/start_ctr.sh $PREFILL_JOB prefill
IMG=infera/engine-sglang:final-pr CTR=$CTR bash scripts/start_ctr.sh $DECODE_JOB  decode
```

**Gate on `BYTECODE_GATE OK`** (9 assertions, read from freshly compiled
`.pyc`s — not the `.py` source). Expect also `GPUGATE True 8` and
`RDMA PORT_ACTIVE in container: 1` (**1 is correct** — that is `mlx5_0`; a
reading of 9 means the ionic devices are being counted, which is the vultr
fabric).

If the gate reports `_p1v2_rows pyc_hits=0 MISSING`, you are on `merged-mtp`.
Either rebuild (step 2) or apply the patch — see `patches/README.md` §1.

## 4. etcd + kvd

`ETCD` starts on the **prefill** node only. Both nodes get their own kvd daemon;
these are two separate stores, not two views of one.

```bash
CTR=$CTR bash scripts/start_services.sh $PREFILL_JOB prefill $PREFILL_IP
CTR=$CTR bash scripts/start_services.sh $DECODE_JOB  decode  $DECODE_IP
# expect: "etcd up" (prefill only) and "kvd socket OK" on both
```

## 5. Launch both legs

`env_prefill.sh` carries the load-bearing pair **`DPA=0 CHUNK=65536 GMU=0.70`**
(see `environment.md` for why the chunk is not a free parameter).

```bash
# edit the three IPs in env_prefill.sh / env_decode.sh first if nodes changed
spur exec $PREFILL_JOB bash -c "docker exec -d $CTR bash $W/env_prefill.sh"
spur exec $DECODE_JOB  bash -c "docker exec -d $CTR bash $W/env_decode.sh"
```

Wait for readiness by **timestamp-scoped** log read — never grep an appended log
for readiness, it matches the previous run:

```bash
for i in $(seq 1 20); do
  P=$(strings $W/logs/prefill.log | grep -ac "server is fired up")
  D=$(strings $W/logs/decode.log  | grep -ac "server is fired up")
  echo "$(date -u +%H:%M:%S) prefill=$P decode=$D"
  [ "$P" -ge 1 ] && [ "$D" -ge 1 ] && break
  sleep 30
done
```

Decode takes longer than prefill — it loads the EAGLE draft model too.

## 6. Router — kv-aware, on the PREFILL node

```bash
PJOB=$PREFILL_JOB PIP=$PREFILL_IP CTR=$CTR bash scripts/router.sh 8190
# -> {"status":"ok","active_workers":2}   <- 2, not 1
```

## 7. Verify every feature BEFORE spending 45 minutes

A green run that proves nothing is the default outcome here.

| feature | command | expected |
|---|---|---|
| PD | `curl .../v1/workers` | prefill `dp_size: null` + decode `dp_size: 8`, both `active` |
| prefill DPA **off** | `docker exec $CTR ps -eo args \| grep '[l]aunch_server'` | **no** `--enable-dp-attention`, **no** `--dp-size`; **has** `--ep-size 8`, `--chunked-prefill-size 65536` |
| decode DPA **on** | same on the decode node, `grep -oE 'scheduler_DP[0-9]+' \| sort -u \| wc -l` | **8** |
| RDMA | `strings <leg>.log \| grep -cE 'MC_FORCE_TCP\|GID is NULL'` | **0** on both |
| MTP | `strings decode.log \| grep -oE 'accept len: [0-9.]+'` | 1.5–3.9 band. **4.00 is BAD** (repetition loop) |
| kvd | `docker exec $CTR python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock` | all-zero before the run |

**Server logs contain binary bytes** — always `strings <log> | grep`, never bare
`grep`. And `grep -ic error` over a leg log returns ~1,500 false hits from the
`server_args=` dump alone; exclude `server_args|aiter|fused_moe`.

Smoke test through the router (cross-node PD proof):

```bash
spur exec $PREFILL_JOB bash -c "curl -sf -m180 http://$PREFILL_IP:8190/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{\"model\":\"glm5.2-mxfp4\",
  \"messages\":[{\"role\":\"user\",\"content\":\"What is 17 multiplied by 23? Answer with the number only.\"}],
  \"max_tokens\":200,\"temperature\":1.0,\"top_p\":0.95}'"
# -> coherent, finish_reason: stop
```

## 8. Stage the customer benchmark, VERBATIM

```bash
cd /path/to/MAD && git fetch origin pull/173/head:pr173
mkdir -p $W/bench
for f in replay_caseA.sh gen_caseA_conformance.py verify_caseA.py MANIFEST.md README.md; do
  git show pr173:scripts/AgentX_CaseA/$f > $W/bench/$f
done
git show pr173:scripts/AgentX_CaseA/caseA_conformance_corpus.tar.gz \
  > $W/bench/caseA_conformance_corpus.tar.gz
chmod +x $W/bench/replay_caseA.sh

md5sum $W/bench/replay_caseA.sh   # MUST be 7cde1afc627c7e4868eac0fd13741baa
cd $W/bench && tar xzf caseA_conformance_corpus.tar.gz && ls corpus/*.json | wc -l   # 200
python3 verify_caseA.py corpus    # 13/13 axes within band
```

Stage the tokenizer **inside** `$W/bench` — the customer script mounts only that
directory:

```bash
mkdir -p $W/bench/tokenizer
for f in tokenizer.json tokenizer_config.json config.json generation_config.json chat_template.jinja; do
  cp /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4/$f $W/bench/tokenizer/
done
```

## 9. Build the aiperf image (on the node that will drive the load)

```bash
mkdir -p $W/imgbuild && cp scripts/Dockerfile.aiperf $W/imgbuild/Dockerfile
bash scripts/build_aiperf.sh $PREFILL_JOB
cat $W/imgbuild/status_*        # expect: ok
```

## 10. Capture the pre-run baseline

```bash
for pair in "$PREFILL_JOB prefill" "$DECODE_JOB decode"; do
  set -- $pair
  spur exec $1 bash -c "docker exec $CTR python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock" \
    > $W/results/kvd_before_$2.json
done
```

## 11. Run — 3 × 900 s

`run_caseA.sh` invokes the customer's script **unmodified**; every site-specific
value is environment. **`OUT` must stay inside `$HERE`** — see
`patches/README.md` §2 for what happens otherwise (a completed run recorded as
`FAILED`).

```bash
spur exec $PREFILL_JOB bash -c \
  "export DOCKER_CONFIG=/tmp/dockercfg; bash $W/run_caseA.sh"
```

Defaults: `CONCS="8 16"`, `DUR=900`, `URL=http://$PREFILL_IP:8190`,
`SERVED=glm5.2-mxfp4`. **900 is the scenario's enforced minimum**
(`inferencex_agentx_mvp.py`) — the customer script's own default of 300 is
rejected by the validator. **conc=1 is unsupported** by the agentic-replay
scenario (its warmup builds one trajectory lane per concurrency and the
warmup-credit count floors to 0 at a single lane).

The C=2 point in `results/c2/` was produced by the same aiperf invocation with
`--concurrency 2` and `--output-artifact-dir $W/bench/smoke_art`.

## 12. Capture the after-state

```bash
for pair in "$PREFILL_JOB prefill" "$DECODE_JOB decode"; do
  set -- $pair
  spur exec $1 bash -c "docker exec $CTR python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock" \
    > $W/results/kvd_after_$2.json
done

python3 scripts/analyze.py $W/bench/results/c8/art $W/bench/results/c16/art
```

Results land in `$W/bench/results/{summary.csv,c8/art/,c16/art/}`.

---

## Success criteria vs. measured

The task brief (`spec/task_brief.md`) set **no numeric target** — it asked to
successfully run the customer's Case-A bench against an infera deployment on
CruSoe and produce an analysed result. Against that, and against the
operational bars this stack uses:

| bar | source | measured | verdict |
|---|---|---|---|
| customer bench runs to completion | the brief | 3/3 points, 900 s each | **PASS** |
| customer script unmodified | the brief | md5 matches the PR blob | **PASS** |
| deployment is ours, on CruSoe | the brief | spur 2-node PD, prefill DPA off | **PASS** |
| corpus conformance | `verify_caseA.py` | **13/13 axes** | **PASS** |
| `submission_valid` | aiperf scenario | **true** on all three points | **PASS** |
| engine faults | method | **0** on both legs | **PASS** |
| MTP not degenerate | method (4.00 = repetition loop) | mean **3.06**, p50 2.98 | **PASS** (see `notes.md` §5 on the 9.1 % ceiling batches) |
| RDMA, no silent TCP fallback | method | **0** occurrences, both legs | **PASS** |

No latency SLA was specified for this run. For reference, the infera Case-A
workload's own bars are `ttft_p90_ms: 30000` and `success_rate: 0.97`; C=8's
TTFT p90 of 23,775 ms clears the former, and C=16's 42,484 ms does not — but
**those bars belong to a different workload** (different driver, different
session model) and are not a criterion here.

> **Both drivers are closed-loop** — infera's declares it (`initial_sessions` /
> `max_inflight`), the customer's implements it inside the agentic-replay
> strategy (see `notes.md` Trap 11). So TTFT here is a function of the offered
> load, which is itself a function of server speed. Quote it with the measured
> arrival rate attached (`scripts/pd_bottleneck.py --jsonl`), never bare.

---

## 13. Arm 2 — prefill DPA ON, re-derived kv-aware weights

Arm 1 (steps 1-12) runs prefill **DPA off**. Arm 2 flips that one axis, at C=8
only. Analysis: [`analysis/dpa8_arm.md`](analysis/dpa8_arm.md).

### 13a. Capture the router log BEFORE restarting it

`/tmp/router.log` lives **inside the container** and is truncated by the next
`router*.sh`. It holds the per-decision `picked= cache_hits= active_blocks=`
lines — the only direct evidence of why the policy chose a rank. This run lost
them by restarting first.

```bash
spur exec $PREFILL_JOB bash -c \
  "docker exec $CTR gzip -c /tmp/router.log" > $W/results/router_arm1.log.gz
```

### 13b. Restart the prefill leg with DPA=8

**`GMU=0.70`, not 0.80.** 0.80 is what the spur Case-A kit records as good and
it **crashes here** — see `notes.md` Trap 6.

```bash
spur exec $PREFILL_JOB bash -c '
  docker exec '"$CTR"' bash -c "pkill -9 -f sglang.launch_server; pkill -9 -f infera.engine.sglang; true"
  for i in $(seq 1 20); do
    n=$(docker exec '"$CTR"' bash -c "ps aux | grep -E \"launch_server|infera.engine\" | grep -v grep | wc -l")
    [ "$n" -eq 0 ] && { echo drained; break; }; sleep 2
  done'

# scripts/env_prefill_dpa8.sh:  DPA=1  CHUNK=65536  GMU=0.70
spur exec $PREFILL_JOB bash -c "docker exec -d $CTR bash $W/env_prefill_dpa8.sh"
```

Wait for `server is fired up` in `$W/logs/prefill_dpa8.log`, then confirm the
axis actually flipped — and that the engine divided the chunk:

```bash
spur exec $PREFILL_JOB bash -c "docker exec $CTR ps -eo args | grep '[l]aunch_server'" \
  | grep -oE '\-\-dp-size [0-9]+|\-\-enable-dp-attention|\-\-mem-fraction-static [0-9.]+'
# -> --mem-fraction-static 0.70 --dp-size 8 --enable-dp-attention

spur exec $PREFILL_JOB bash -c "docker exec $CTR ps -eo args | grep -oE 'scheduler_DP[0-9]+' | sort -u | wc -l"
# -> 8
```

The resolved per-forward chunk shows up in the leg log as `#new-token: 8192`
(65536 ÷ 8), versus 65536 in arm 1. That is `server_args.py:4902` at work.

### 13c. Router — kv-aware with weights derived from arm 1's data

```bash
PJOB=$PREFILL_JOB PIP=$PREFILL_IP CTR=$CTR bash scripts/router_tuned.sh 8190
# --router-policy kv-aware --kv-prefill-overlap-weight 5.0 --kv-decode-overlap-weight 1.0
```

5.0/1.0 rather than the 20.0/2.0 default: the defaults assume a ~0.89 hit rate,
this workload realizes 0.66. The derivation is in `analysis/dpa8_arm.md`.

### 13d. Start the per-rank sampler, THEN the bench

The sampler must be running before the load starts — `prefill_batches` is
cumulative, and a late start silently loses the ramp.

```bash
nohup bash scripts/sample_ranks.sh > $W/logs/sampler.log 2>&1 &
sleep 3
nohup spur exec $PREFILL_JOB bash -c \
  'export DOCKER_CONFIG=/tmp/dockercfg; OUT='"$W"'/bench/results_dpa8 CONCS=8 bash '"$W"'/run_caseA.sh' \
  > $W/logs/run_dpa8.log 2>&1 &
```

Watch balance live; it drifts, so read it more than once:

```bash
python3 scripts/rank_balance.py            # every few minutes
```

When the run ends: `pkill -f sample_ranks.sh`, then

```bash
python3 scripts/analyze.py $W/bench/results_dpa8/c8/art
python3 scripts/rank_balance.py > analysis/rank_balance_dpa8.txt
```

### Arm 2 measured

| C=8 | arm 1 (DPA off) | arm 2 (DPA on) |
|---|---:|---:|
| TTFT p50 / p90 | **6,698 / 23,775 ms** | 13,578 / 32,506 ms |
| output throughput | **258.5 tok/s** | 220.2 tok/s |
| server cache read/prompt | 66.1 % | 66.2 % |
| `submission_valid` | true | true |
| engine faults | 0 | 0 |
| per-rank max/min | n/a (1 target) | **5.49x**, CV 0.494 |

DPA-on is worse on every latency/throughput axis at C=8. **Not a clean
ablation** — DPA and the resolved chunk size are coupled by the ÷`dp_size`.

---

## 14. Arm 3 — decode MTP OFF (which turns the decode radix cache ON)

**These are one switch, not two.** SGLang rejects
`--disaggregation-decode-enable-radix-cache` under `--speculative-algorithm`, so
`infera/engine/sglang/args.py:261-278` only appends it when MTP is absent.
Asking for "MTP off" and "decode radix cache on" is the same request here.

### 14a. Rescue the router log, then tear BOTH legs down properly

```bash
spur exec $PREFILL_JOB bash -c \
  "docker exec $CTR bash -c 'cp /tmp/router.log $W/logs/router.<prev-arm>.log'"
```

Then kill the **whole** engine family — the wrapper alone leaves the subprocess
tree holding all the VRAM (Trap 9):

```bash
for J in $PREFILL_JOB $DECODE_JOB; do
  spur exec $J bash -c "docker exec $CTR bash -c \
    \"pkill -9 -f sglang.launch_server; pkill -9 -f 'sglang::'; \
      pkill -9 -f infera.engine.sglang; true\""
done
# GATE: do not relaunch until BOTH read ~0
for J in $PREFILL_JOB $DECODE_JOB; do
  spur exec $J bash -c 'rocm-smi --showmeminfo vram | grep -oE "Used Memory \(B\): [0-9]+" \
    | grep -oE "[0-9]+$" | sort -rn | head -1'
done
```

### 14b. Relaunch — prefill back to arm 1, decode with MTP off

Arm 3 isolates the decode side, so **prefill and the router must return to the
arm-1 baseline** (that is the only C=8 point it is comparable against).

```bash
spur exec $PREFILL_JOB bash -c "docker exec -d $CTR bash $W/env_prefill.sh"       # DPA=0
sleep 20
spur exec $DECODE_JOB  bash -c "docker exec -d $CTR bash $W/env_decode_nomtp.sh"  # MTP=0
```

`env_decode_nomtp.sh` carries `EXTRA_ARGS="--num-reserved-decode-tokens 256"`.
**That pin is load-bearing:** `RESERVED_TOK` lives inside the leg script's
`MTP_ARGS` block, so dropping MTP drops the flag and SGLang's default (**512**)
silently takes over — a 2x change attributed to nothing.

### 14c. Verify the switch actually flipped (bytecode-grade, not source-grade)

```bash
strings $W/logs/decode_nomtp.log | grep -oE "spawning sglang subprocess.*" | head -1 \
  | grep -oE "(--disaggregation-decode-enable-radix-cache|--num-reserved-decode-tokens [0-9]+|--speculative-algorithm [A-Z]+)" | sort -u
strings $W/logs/decode_nomtp.log | grep -oE "Tree cache initialized: [^\"]{0,110}" | sort | uniq -c
strings $W/logs/decode_nomtp.log | grep -c "accept len"
```

| check | expected |
|---|---|
| spawned argv | **has** `--disaggregation-decode-enable-radix-cache`, **has** `--num-reserved-decode-tokens 256`, **no** `--speculative-algorithm` |
| tree cache | `8 ... impl=RadixCache` (arm 1: `impl=ChunkCache`) |
| `accept len` count | **0** (arm 1: 5,564 batches) |

> A `KV cache is forced as chunk cache for decode server` WARNING still appears
> **once**, early. That is the *wrapper's* first parse, before the flag is
> appended. Judge by the spawned subprocess's `server_args`
> (`disable_radix_cache=False`, `disaggregation_decode_enable_radix_cache=True`)
> and by the 8 `impl=RadixCache` lines — not by that warning.

### 14d. Router back to the arm-1 weights, then run

```bash
PJOB=$PREFILL_JOB PIP=$PIP CTR=$CTR bash scripts/router.sh 8190   # 20.0 / 2.0
# driver runs on the HOST -- the engine container has no docker client (Trap 10)
nohup spur exec $PREFILL_JOB bash -c \
  'export DOCKER_CONFIG=/tmp/dockercfg; OUT='"$W"'/bench/results_nomtp CONCS=8 bash '"$W"'/run_caseA.sh' \
  > $W/logs/run_nomtp2.log 2>&1 &
```

The nohup target must be writable **by you** — a path left over from a
`docker exec -d` is root-owned and the redirect dies before the driver starts.

### Arm 3 measured

| C=8 | arm 1 (MTP + ChunkCache) | arm 3 (no MTP + RadixCache) |
|---|---:|---:|
| **TTFT p50 / p90** | 6,698 / 23,775 ms | **4,224 / 16,999 ms** |
| **ITL p50 / p90** | **13.26 / 18.49 ms** | 22.21 / 24.86 ms |
| E2E p50 / p90 | **13,874 / 38,774 ms** | 14,375 / 69,078 ms |
| **output throughput** | **258.5 tok/s** | 211.0 tok/s |
| server cache read/prompt | 50.8 % | 49.5 % |
| `submission_valid` | true | **true** |
| engine faults | 0 | **0** |
| **arrival rate** (closed-loop OUTPUT) | **0.257 req/s** | **0.196 req/s** (−24 %) |

### 14e. Always run the bottleneck + arrival analysis before reading the table

```bash
python3 scripts/pd_bottleneck.py \
  --prefill $W/logs/prefill.log --decode $W/logs/decode_nomtp.log \
  --jsonl $W/bench/results_nomtp/c8/art/profile_export.jsonl \
  --decode-cap 256 --label arm3
```

`--decode-cap` takes the **DP-adjusted** `max_running_requests` — the *second*
value printed in `server_args` (256), not the 2048 passed on the CLI.

**TTFT improved because the load fell, not because the server sped up.** The
driver is closed-loop (`notes.md` Trap 11), so arm 3's slower decode (ITL +68 %,
arm 1's accept length was 3.06) slowed lane turnover and cut arrival by 24 %,
draining prefill's queue from 2.00 to 0.50. Both arms are **prefill-bound**;
transfer is **exonerated** (state 2 shallow, zero TCP fallback); decode sat at
**0.1–0.3 %** of its running cap with zero prealloc and zero retracts.

Full treatment: [`analysis/nomtp_radix_arm.md`](analysis/nomtp_radix_arm.md).
Raw analyser output:
[`analysis/pd_bottleneck_arm1_vs_arm3.txt`](analysis/pd_bottleneck_arm1_vs_arm3.txt).
