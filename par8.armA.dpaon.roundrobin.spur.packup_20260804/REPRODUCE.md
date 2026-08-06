# REPRODUCE — par8 arm A (DPA on + round-robin) on spur

Top to bottom. Every script referenced lives in `scripts/`. Budget **~25 min
build** (parallel across both nodes) + **~12 min bring-up** + **67 min
measurement**. Cold start is 8–12 min per leg (weights → tilelang JIT → CUDA graph
capture) — that is **not** a hang.

```bash
# Constants used throughout. Substitute your own node IPs.
W=/shared_nfs/yihou_final_pr           # workspace on shared storage
PJOB=35682 ; PIP=10.245.156.167        # prefill: crsuse2-m2m-010
DJOB=35683 ; DIP=10.245.152.164        # decode:  crsuse2-m2m-081
```

> **Node substitution is allowed but not free.** Any two spur MI355X nodes work.
> `ETCD_IP` must be the **prefill** node, the router runs on the prefill node, and
> the bench's `ROUTER` follows it. Check `/tmp` health first — step 0.

---

## 0. Prerequisites

- **Two 8-GPU spur allocations, wall clock ≥ 24 h.** This is not padding: the
  first attempt at this arm died because a 12 h clock could not absorb a restart.

  ```bash
  sbatch -J par8 -p amd-spur -q amd-burst-qos -N1 -G8 -t 24:00:00 hold.sh
  spur show job <id> | grep -E 'JobState|NodeList'
  ```

  Size the clock for **build + bring-up + run + one full retry** (~3.5 h), not for
  a single clean pass. An allocation inherited from an earlier session may have
  minutes left — check `EndTime` before starting, and extend **before** any
  restart, never after.

- **Check `/tmp` on every node before building.** A node with `/tmp` as
  `drwxr-xr-x` (missing sticky + world-write) fails the build at
  `mkdir /tmp/dockercfg: permission denied`, which reads like a script bug:

  ```bash
  for j in $PJOB $DJOB; do spur exec $j bash -c 'ls -ld /tmp'; done   # want drwxrwxrwt
  ```

  Release and re-request any node that fails this.

- `export DOCKER_CONFIG=/tmp/dockercfg` before **every** docker call.

- **Never background a long docker client inside `spur exec`** — the exec
  namespace teardown kills it even under `nohup`/`setsid`.

- Model weights at `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (408 GB).
  Bench driver + venv per `environment.md`.

## 1. Stage the source and build the image on BOTH nodes

Branch `yihou.dev.glm52.agentic.final.pr` @ `97c2ff5`. Tar the worktree
(excluding `.git`) to `$W/build/src.tar`, then:

```bash
mkdir -p $W/build
for j in $PJOB $DJOB; do
  nohup bash scripts/build_image.sh $j > $W/build/drv_$j.log 2>&1 &
done
wait
cat $W/build/build_*.status          # want: ok sha256:...
```

The two nodes build independently and produce **different image ids**. Expected;
equivalence comes from the bytecode gate in step 2, never from the digest.

**This build is the first real gate.** The DSA layer applies patch 01's *anchors*
against real upstream sglang v0.5.15.post1 and verifies the result in freshly
compiled **bytecode**. Expect in the build log:

```
dsa_indexer.py :: _p1v2_trim               -> pyc=1
dsa_indexer.py :: _p1v2_rows               -> pyc=1     <-- GLM52_P1V3
dsa_backend.py :: _glm52_match_page_table_rows -> pyc=1
eagle_worker_v2.py :: requires_dp_attention_eager_forward -> pyc=2
```

## 2. Start the containers and pass the bytecode gate

```bash
bash scripts/start_ctr.sh $PJOB prefill
bash scripts/start_ctr.sh $DJOB decode
```

Expect on each:

```
GPUGATE True 8
RDMA PORT_ACTIVE in container: 1      <- 1 is CORRECT (the container sees only mlx5)
BYTECODE_GATE OK
```

> These nodes also carry 8 ACTIVE ionic rails on the **host** (visible in
> `env/*.txt`). They are not used and the container does not see them. See
> `environment.md` § RDMA fabric.

## 3. NO in-container patching — confirm, don't perform

The predecessor kits' `REPRODUCE.md` had a **MANDATORY** step here: hand-patch
`GLM52_P1V3` into the running container and relaunch. **That step is gone** — on
this branch P1V3 is baked in by the Dockerfile, and `_p1v2_rows` in step 2's gate
is what proves it (`_p1v2_trim` alone is satisfied by the *older* one-directional
revision and cannot tell them apart).

If your gate shows `_p1v2_trim` but not `_p1v2_rows`, you are on the wrong image.
Do not hand-patch; rebuild.

## 4. Start etcd (prefill node) and kvd (both nodes)

```bash
bash scripts/start_services.sh $PJOB prefill $PIP    # etcd + kvd
bash scripts/start_services.sh $DJOB decode  $DIP    # kvd only
```

Expect `etcd up`, `kvd socket OK`, and an all-zero `statctl` dump on both.

## 5. Launch the prefill leg — DPA **ON**, **GMU 0.70**

```bash
JOB=$PJOB MY=$PIP PIP=$PIP DPA=1 MTP=0 GMU=0.70 \
  bash scripts/ab_boot.sh prefill armA2
```

### `GMU=0.70` is the load-bearing argument of this step

**`0.80` will boot and then abort ~60 s into the run.** Every DPA-on leg on this
stack previously used 0.80, including the acceptance run of this exact image — so
the temptation to leave it is real. Under **round-robin** it does not survive:

```
HSA_STATUS_ERROR_OUT_OF_RESOURCES … Available Free mem : 52 MB
Fatal Python error: Aborted
```

at `token usage: 0.05` — the KV pool empty, i.e. **activation** memory. Round-robin
puts 4–5 DP ranks into prefill concurrently (kv-aware: 1–2), each holding its own
chunk's activations. Full mechanism in `notes.md` §1. (That attempt's log was
overwritten by its own restart — `logs/README.md` records what was lost.)

**Save the log before restarting.** Use a distinct tag per attempt
(`ab_boot.sh prefill armA_try1`, then `armA_try2`), or the relaunch truncates the
failed run's log in place — which is exactly what happened here.

`CHUNK` is **not** passed here. The default derives `ISL*TP` = 65,536 global, which
DPA divides to 8,192/rank. That matches arm B's *global* budget. Passing
`CHUNK=8192` would match per-forward work instead — a different, also-defensible
choice, and not the one this pair made (`patches/README.md`).

> `WARNING: engines still present after 40s` on a clean boot is a **false alarm** —
> the wait loop counts processes after `docker exec -d` has already started the new
> leg. Verify by timestamp (`docker exec … ps -eo pid,lstart,args`) if unsure.

## 6. Launch the decode leg — DPA ON, MTP ON, kvd OFF

```bash
JOB=$DJOB MY=$DIP PIP=$PIP DPA=1 MTP=1 KVD=0 \
  bash scripts/ab_boot.sh decode armA2
```

`KVD=0` is deliberate: a PD decode leg sets `disable_radix_cache=True` itself and
sglang rejects hierarchical cache alongside it. Decode's `mem-fraction-static`
stays at its 0.85 default — it never OOMed, and moving both would make the GMU
change a two-variable one.

Wait for both — **never grep an appended log for readiness**, it matches the
previous run within seconds:

```bash
PJOB=$PJOB PIP=$PIP DJOB=$DJOB DIP=$DIP bash scripts/ab_wait.sh 2400
# -> BOTH READY after 608s
```

## 7. Router on the prefill node — policy **round-robin**

```bash
JOB=$PJOB MY_IP=$PIP POLICY=round-robin PORT=8190 bash scripts/ab_router.sh
```

Expect:

```
{"status":"ok","active_workers":2}  router healthy (policy=round-robin)
router-policy=round-robin
```

`active_workers` must be **2**. The **absence** of an `overlap_weight=…` line is
the discriminator against the kv-aware arm — that line only prints for kv-aware.

> **There is no per-role policy switch.** `--router-policy` is a single global
> value (`infera/server/args.py:88`; the Rust backend likewise accepts only
> `round-robin|kv-aware` at `rust/router/src/config.rs:72`). "prefill and decode
> both round-robin" *is* `--router-policy round-robin`. The two pools then rotate
> independently because `RoundRobin` keys its counter on the candidate set.

> `--router-tokenizer-path` is passed but inert: it is `required=True` at the
> parser level regardless of policy, and `_build_round_robin(**_)` discards it.
> Omitting it exits 2 with a usage dump.

> On a restart **always pass a fresh port**. A router whose circuit breaker is
> still open returns 503 in ~0.4 s, which looks exactly like a backend failure.
> Read the latency: ~0.4 s is the breaker; 12–23 s is a real backend fault.

## 8. Verify every feature BEFORE spending 67 minutes

| feature | command | expected |
|---|---|---|
| prefill DPA **on** | `docker exec … ps -eo args \| grep '[i]nfera.engine'` | `--dp-size 8 --enable-dp-attention --ep-size 8 --mem-fraction-static 0.70` |
| decode DPA **on** | same on the decode node | `--dp-size 8 --enable-dp-attention`, `--speculative-algorithm EAGLE` |
| RDMA | `strings <leg>.log \| grep -cE 'MC_FORCE_TCP\|GID is NULL'` | **0** on both |
| kvd wired | `strings prefill.log \| grep -c 'infera-kvd adapter connected'` | **8** (prefill); **0** decode, by design |
| kvd baseline | `docker exec … python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock` | all-zero, both nodes |

Smoke test **through the router** (never a leg's own port — it hangs):

```bash
spur exec $PJOB bash -c 'export DOCKER_CONFIG=/tmp/dockercfg
  docker exec agbench_mtp curl -sf -m180 http://'$PIP':8190/v1/chat/completions \
  -H "Content-Type: application/json" -d "{\"model\":\"glm5.2-mxfp4\",
  \"messages\":[{\"role\":\"user\",\"content\":\"What is 17 multiplied by 23? Answer with the number only.\"}],
  \"max_tokens\":200,\"temperature\":1.0,\"top_p\":0.95}"'
# -> 391, coherent, finish_reason: stop
```

**Sampling matters**: `temperature 1.0 / top_p 0.95` per GLM-5.2's
`generation_config.json`. `temperature: 0` with MTP produces output
indistinguishable from KV corruption.

## 9. Capture the pre-run kvd baseline

```bash
for spec in "$PJOB:armA2_prefill" "$DJOB:armA2_decode"; do
  j=${spec%%:*}; n=${spec##*:}
  spur exec $j bash -c 'export DOCKER_CONFIG=/tmp/dockercfg
    docker exec agbench_mtp python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock' \
    > $W/results/$n.kvd_before.json
done
```

## 10. Run — 67 minutes

```bash
ROUTER=http://$PIP:8190 setsid bash scripts/ab_bench.sh par8_armA_rr2 < /dev/null &
```

The script prints the workload md5 — expect **`968b1543155839135dc9eaf6dd142626`**.

**No load knobs on the CLI**: the YAML is the single source of truth. Passing
`--initial-sessions` would silently shadow it and break the pairing with arm B.

Monitor (note the `\r` handling — the progress bar overwrites in place):

```bash
tail -c 400 /tmp/benchA2.log | tr '\r' '\n' | tail -1
```

**Watch the first 90 seconds.** Attempt 1 aborted at t+60 s. If the prefill leg is
going to die on activation memory, it does so almost immediately:

```bash
strings $W/logs/armA2_prefill.log | grep -cE 'HSA_STATUS_ERROR|Fatal Python'   # want 0
```

Abort the run if in-flight *pins* at 24 — backpressure would then be setting the
load. (It did not: 24 was touched on 31 of 3,820 ticks, 0.8 %.)

## 11. Capture the after-state — IMMEDIATELY, before anything expires

**Do this the moment the run finishes.** The arm B kit lost three artifacts to a
wall-clock reclaim that happened hours later; nothing warns you first.

```bash
# 1. kvd after-state, both nodes  -- THE headline evidence on this arm
for spec in "$PJOB:armA2_prefill" "$DJOB:armA2_decode"; do
  j=${spec%%:*}; n=${spec##*:}
  spur exec $j bash -c 'export DOCKER_CONFIG=/tmp/dockercfg
    docker exec agbench_mtp python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock' \
    > $W/results/$n.kvd_after.json
done

# 2. router metrics + pick log -- BOTH live inside the container and die with it.
#    NOTE the base64: a raw `gzip -c` through spur exec corrupts the stream
#    ("not in gzip format" on zcat). This was hit here.
spur exec $PJOB bash -c 'export DOCKER_CONFIG=/tmp/dockercfg
  docker exec agbench_mtp curl -s http://'$PIP':8190/metrics' > $W/results/armA2_router_metrics.txt
spur exec $PJOB bash -c 'export DOCKER_CONFIG=/tmp/dockercfg
  docker exec agbench_mtp bash -c "gzip -c /tmp/router.log | base64 -w0"' \
  | tr -d '\n' | base64 -d > $W/results/armA2_router.log.gz

# 3. environment snapshot, both nodes.
#    NOTE the `bash -c "bash …"`: `spur exec $j bash -s < file` yields an EMPTY
#    file (stdin is not forwarded) and `spur exec $j "bash …"` fails with
#    `nsenter: failed to execute`. Both were tried here.
for j in $PJOB $DJOB; do
  spur exec $j bash -c "bash $W/scripts/collect_env.sh" > $W/env/env_$j.txt 2>&1
done
```

Verify the pick log survived — this is the arm's primary result:

```bash
zcat $W/results/armA2_router.log.gz | grep -oE 'picked=[^ ]+' | sort | uniq -c
# -> 16 targets, 290-291 each
```

Results land in
`$W/bench/par8_armA_rr2/par8_armA_rr2/<timestamp>/{summary.json,metrics.jsonl,metadata.json}`.

---

## Success criteria vs. measured

The instruction set no numeric target — it asked for the par8 workload run
verbatim with DPA on and **round-robin routing on both prefill and decode**.
Against the bars in `spec/par8.yaml`:

| bar | source | measured | verdict |
|---|---|---|---|
| success rate ≥ 0.97 | `sla.success_rate` | **0.9854** | **PASS** |
| TTFT p90 < 30,000 ms | `sla.ttft_p90_ms` | **7,079 ms** (sustain) | **PASS**, 4.2× margin |
| e2e p50 < 4,500 ms | `sla.e2e_p50_ms` | **not recorded** | this driver emits no e2e percentile object |
| full cycle completes | the instruction | 4,006.7 s, ramp+sustain+drain | **PASS** |
| in-flight not pinned at cap | method | touched 24 on 31/3,820 ticks (0.8 %) | **PASS** |
| routing is round-robin on **both** pools | the instruction | 16 targets, 290–291 picks each, pools independent | **PASS** |
| workload byte-identical to par8 | the instruction | md5 match, only the tokenizer path retargeted | **PASS** |
