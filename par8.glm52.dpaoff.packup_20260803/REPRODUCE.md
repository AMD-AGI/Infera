# REPRODUCE — par8

Top-to-bottom. Every script referenced lives in `scripts/`. Cold start is
**5–8 min per leg** (weights → tilelang JIT → cudagraph capture) — that is not a
hang. Total wall clock: ~25 min bring-up + **67 min** measurement.

```bash
# Constants used throughout
JUMP=root@149.28.124.225
W=/mnt/vast/c_huggingface/bench_20260801
PREFILL_NODE=chi2835 ; PREFILL_IP=10.2.122.78
DECODE_NODE=chi2879  ; DECODE_IP=10.2.122.10
```

> **Node substitution is allowed but not free.** Any two MI355X nodes with the
> `merged-e` image work. If you change nodes: the IPs above change, `ETCD_IP`
> must be the *prefill* node, and the router URL follows the prefill node. Check
> `PORT_ACTIVE` on both (see step 1) — chi2879 ran this experiment with **7/8**
> rails, which is recorded, not desired.

---

## 0. Prerequisites

- **Slurm holds belong to `yeandy-debug` on both nodes. Never `scancel`.** Kill
  only your own `sglang.launch_server` / `infera` processes and the `bench_run`
  container.
- Root disk: `docker save` of this image materialises **78.6 GB** in
  `/var/lib/docker/tmp` on the *source* before streaming. Need ≥ 90 GB free
  there if you must transfer the image; run **one transfer at a time**.
- Driver staged at `$W/agbench` (branch `fix/realistic-profile-session-driver`
  @ `1cf01cb`, **not main**) with `SOLO_M1` applied, venv at `$W/venv`.

## 1. Stage scripts and the workload

```bash
# from this kit's directory
for f in scripts/*; do scp "$f" $JUMP:$W/scripts/; done
scp spec/par8.yaml $JUMP:$W/par8.yaml
ssh $JUMP "md5sum $W/par8.yaml"     # expect 78e4badf107178f64c6d45a85674f2cb
```

Apply **CHUNK_PASSTHROUGH** (skip if `start_leg.sh` already contains it — the
patch is idempotent and prints `already patched`):

```bash
ssh $JUMP "cd $W/scripts && cp start_leg.sh start_leg.sh.bak_prechunk_\$(date +%Y%m%d-%H%M) \
  && python3 patch_leg_chunk.py start_leg.sh"
# -> patched OK - CHUNK_PASSTHROUGH occurrences: 1
```

## 2. Reset both nodes

`reset_node.sh` tears down **only our own** containers (`bench_run`,
`merged_run`), waits for the GPUs to release, starts a fresh container, checks
RDMA, and starts a cold kvd daemon. `ETCD=1` only on the prefill node.

```bash
ssh $JUMP "ssh $PREFILL_NODE 'cd $W && ROLE=prefill MY_IP=$PREFILL_IP ETCD=1 bash scripts/reset_node.sh'"
ssh $JUMP "ssh $DECODE_NODE  'cd $W && ROLE=decode  MY_IP=$DECODE_IP  ETCD=0 bash scripts/reset_node.sh'"
```

Expect `PORT_ACTIVE: 8 (want 8)`. **chi2879 reports 7** (`ionic_5` down) — the
leg script auto-excludes it; record it rather than fixing it.

## 3. Launch the prefill leg — DPA OFF, CHUNK 16384

**`CHUNK=16384` is the load-bearing argument of this step.** sglang divides
`chunked_prefill_size` by `dp_size` only under DPA (`server_args.py:4902`), so
without it the leg silently runs 8192 per forward.

```bash
ssh $JUMP "ssh $PREFILL_NODE 'cd $W && \
  ROLE=prefill MY_IP=$PREFILL_IP ETCD_IP=$PREFILL_IP \
  DPA=0 CHUNK=16384 MTP=0 TAG=q1 bash scripts/start_leg.sh'"
```

## 4. Launch the decode leg — DPA ON, MTP ON

```bash
ssh $JUMP "ssh $DECODE_NODE 'cd $W && \
  ROLE=decode MY_IP=$DECODE_IP ETCD_IP=$PREFILL_IP \
  MTP=1 TAG=q4 bash scripts/start_leg.sh'"
```

Wait for readiness — **never grep an appended log for readiness, it matches the
previous run**; scope by timestamp:

```bash
ssh $JUMP "ssh $DECODE_NODE 'strings $W/logs/q4_decode.log | grep -a \"server is fired up\" | tail -1'"
```

## 5. Apply GLM52_P1V3 to the decode leg — then RELAUNCH

**Without this the decode leg crashes mid-run** with
`Expected lengths.size(0) == B to be true`. The patch edits source *inside* the
container; a running engine has already imported the old bytecode.

```bash
ssh $JUMP "ssh $DECODE_NODE 'docker cp $W/scripts/apply_p1v3.py bench_run:/tmp/ && \
  docker exec bench_run python3 /tmp/apply_p1v3.py'"
# -> patched OK - GLM52_P1V3 occurrences: 3

# The .pyc is STILL the image's. Delete it and relaunch:
ssh $JUMP "ssh $DECODE_NODE 'docker exec bench_run rm -f \
  /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/__pycache__/dsa_indexer*.pyc'"

ssh $JUMP "ssh $DECODE_NODE 'cd $W && ROLE=decode MY_IP=$DECODE_IP ETCD_IP=$PREFILL_IP \
  MTP=1 TAG=q4 bash scripts/start_leg.sh'"
```

Verify the **compiled bytecode**, not the source (see `patches/README.md` §4 for
the one-liner). Expect `1` and a `.pyc` mtime *after* the patch.

## 6. Router — Rust backend, on the PREFILL node

```bash
ssh $JUMP "ssh $PREFILL_NODE 'cd $W && BACKEND=rust bash scripts/start_router.sh'"
# -> router healthy (backend=rust pw=20.0 dw=2.0)
```

## 7. Verify every feature BEFORE spending 67 minutes

A green run that proves nothing is the default outcome here.

```bash
ssh $JUMP "curl -sf -m10 http://$PREFILL_IP:8100/health"
# {"active_workers":2,"status":"ok"}       <- 2, not 1
```

| feature | command | expected |
|---|---|---|
| PD | `curl .../v1/workers` | prefill `dp_size: null` + decode `dp_size: 8`, both `active` |
| prefill DPA **off** | `ps -eo cmd \| grep '[l]aunch_server'` on chi2835 | **no** `--dp-size`, **no** `--enable-dp-attention`, **has** `--ep-size 8`, `--chunked-prefill-size 16384` |
| decode DPA **on** | `docker exec bench_run ps -eo args \| grep -oE 'scheduler_DP[0-9]+' \| sort -u` | `scheduler_DP0` … `DP7` (8) |
| RDMA | `strings <leg>.log \| grep -cE 'MC_FORCE_TCP\|GID is NULL'` | **0** on both legs |
| MTP | `strings q4_decode.log \| grep -oE 'accept len: [0-9.]+'` | 1.5–3.9 band. **4.00 is BAD** (repetition loop) |
| kvd | `docker exec bench_run python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock` | all-zero before the run |

**Never probe a leg's own port directly — it hangs.** Always the router.
**Server logs contain binary bytes** — `strings <log> | grep`, never bare `grep`.

Smoke test through the router (cross-node PD proof):

```bash
ssh $JUMP "curl -sf -m150 http://$PREFILL_IP:8100/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{\"model\":\"glm5.2-mxfp4\",
  \"messages\":[{\"role\":\"user\",\"content\":\"What is 17 multiplied by 23? Answer with the number only.\"}],
  \"max_tokens\":200,\"temperature\":1.0,\"top_p\":0.95}'"
# -> 391, coherent, finish_reason: stop
```

## 8. Capture the pre-run baseline

```bash
for N in $DECODE_NODE $PREFILL_NODE; do
  ssh $JUMP "ssh $N 'docker exec bench_run python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock'" \
    > /tmp/par8.kvd_before_$N.json
done
```

## 9. Offline preview (no GPU, no server)

```bash
ssh $JUMP "cd $W/agbench && $W/venv/bin/python -m agent.agent_throughput \
  --workload-config $W/par8.yaml --mode preview --model glm5.2-mxfp4 \
  --tokenizer /mnt/vast/xiaobo/models/GLM-5.2-MXFP4"
```

**Expect three warnings, all real and all accepted deliberately** (they follow
from holding `new_session_rate` at 0.10, which was solved for N=32, while
lowering `initial_sessions` to 8 — the instruction was "其余不变"):

```
max_inflight=24 throttles once E2E exceeds ~7s
max_sessions=32 is reached once E2E exceeds ~15s
initial_sessions=8 is far from the steady-state range 22-56
```

Measured outcome: sessions **did** reach 32; in-flight peaked at **22** and
never pinned at 24. So the workload, not backpressure, set the load.

## 10. Run — 67 minutes

```bash
ssh $JUMP "cd $W && TAG=full RAMP=400 SUSTAIN=3600 setsid bash scripts/par8_run.sh < /dev/null"
```

No load knobs on the CLI — **the YAML is the single source of truth**. Passing
`--initial-sessions` here would silently shadow it.

Monitor; abort if `In-flight` pins at 24 (then backpressure sets the load and
the window is wasted):

```bash
ssh $JUMP "tail -2 $W/logs/par8_full.log | tr '\r' '\n' | tail -1"
```

## 11. Capture the after-state — BEFORE restarting anything

`/tmp/router.log` lives **inside the container** and is overwritten by the next
`start_router.sh`. Capture it first.

```bash
for N in $DECODE_NODE $PREFILL_NODE; do
  ssh $JUMP "ssh $N 'docker exec bench_run python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock'" \
    > /tmp/par8.kvd_after_$N.json
done

# per-rank routing distribution, SCOPED BY TIME (the log is appended all day and
# contains picks from previous legs — an unscoped read mixes deployments)
ssh $JUMP "ssh $PREFILL_NODE 'docker cp $W/scripts/cv_scoped.py bench_run:/tmp/ && \
  docker exec bench_run python3 /tmp/cv_scoped.py /tmp/router.log 2026-08-03T09:28'"

ssh $JUMP "ssh $PREFILL_NODE 'docker exec bench_run gzip -c /tmp/router.log' " > router.log.gz
```

Results land in
`$W/results/par8_full/par8_full/<timestamp>/{metadata.json,metrics.jsonl}`.

## Success criteria vs. measured

The spec set no numeric target for this run — it asked for *one full cycle at
reduced load with prefill DPA off*. Against the operational bars this stack uses:

| bar | source | measured | verdict |
|---|---|---|---|
| success rate ≥ 0.97 | `par8.yaml` `sla.success_rate` | **0.988** | **PASS** |
| TTFT p90 < 30,000 ms | `par8.yaml` `sla.ttft_p90_ms` | **4,948 ms** | **PASS** (6.1× margin) |
| E2E p50 < 4,500 ms | `par8.yaml` `sla.e2e_p50_ms` | **7,400 ms** | **FAIL — 1.64×** |
| full cycle completes | the instruction | 4,005 s, ramp+sustain+drain | **PASS** |
| in-flight not pinned at cap | method | max 22 of 24 | **PASS** |

> The E2E p50 miss is **expected and not a regression**: `e2e_p50_ms: 4500` is a
> *latency-floor* spec, not a capacity spec — the solo kits established that it
> is met only at concurrency 1. See `analysis/sli_percentiles.md`.
