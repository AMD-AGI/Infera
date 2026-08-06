# REPRODUCE — par8 arm B (prefill DPA off + kv-aware) on spur

Top to bottom. Every script referenced lives in `scripts/`. Budget **~25 min
build** (parallel across both nodes) + **~10 min bring-up** + **67 min
measurement**. Cold start is 5–12 min per leg (weights → tilelang JIT → CUDA
graph capture) — that is **not** a hang.

```bash
# Constants used throughout. Substitute your own node IPs.
W=/shared_nfs/yihou_final_pr           # workspace on shared storage
PJOB=33490 ; PIP=10.245.158.155        # prefill: crsuse2-m2m-250
DJOB=33491 ; DIP=10.245.151.18         # decode:  crsuse2-m2m-251
```

> **Node substitution is allowed but not free.** Any two spur MI355X nodes work.
> `ETCD_IP` must be the **prefill** node, the router runs on the prefill node, and
> the bench's `ROUTER` follows it. Check `/tmp` health first — see step 0.

---

## 0. Prerequisites

- **Two 8-GPU spur allocations.** Set the wall clock explicitly; the default is
  24 h and an allocation inherited from a previous session may have minutes left:

  ```bash
  sbatch -J par8 -p amd-spur -q amd-burst-qos -N1 -G8 -t 24:00:00 hold.sh
  spur show job <id> | grep -E 'JobState|NodeList'
  ```

  Budget the *whole* pipeline against the remaining wall clock before starting.
  A build + bring-up + 67 min run needs ~2 h; a restart mid-way needs another
  ~1.5 h. Extend **before** restarting, not after.

- **Check `/tmp` on every node before building.** A node with `/tmp` as
  `drwxr-xr-x` (missing sticky + world-write) fails the build with
  `mkdir /tmp/dockercfg: permission denied`, which reads like a script bug:

  ```bash
  for j in $PJOB $DJOB; do spur exec $j bash -c 'ls -ld /tmp'; done   # want drwxrwxrwt
  ```

  Release and re-request any node that fails this.

- `export DOCKER_CONFIG=/tmp/dockercfg` before **every** docker call. Every
  script here does it; do it too in ad-hoc commands.

- **Never background a long docker client inside `spur exec`** — the exec
  namespace teardown kills it even under `nohup`/`setsid`.

- Model weights at `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (408 GB).
  Bench driver + venv per `environment.md`.

## 1. Stage the source and build the image on BOTH nodes

The branch is `yihou.dev.glm52.agentic.final.pr` @ `97c2ff5`. Tar the worktree
(excluding `.git`) to `$W/build/src.tar`, then:

```bash
mkdir -p $W/build
for j in $PJOB $DJOB; do
  nohup bash scripts/build_image.sh $j > $W/build/drv_$j.log 2>&1 &
done
wait
cat $W/build/build_*.status          # want: ok sha256:...
```

The two nodes build independently and produce **different image ids**. That is
expected; equivalence is established by the bytecode gate in step 2, never by
digest.

**This build is the first real gate.** The DSA layer applies patch 01's *anchors*
against real upstream sglang v0.5.15.post1 and verifies the result in freshly
compiled **bytecode**. Anchor drift fails here with `rc != 0` rather than at
runtime. Expect, in the build log:

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
RDMA PORT_ACTIVE in container: 1      <- 1 is CORRECT on spur (one mlx5 rail)
BYTECODE_GATE OK
```

**`BYTECODE_GATE OK` is 9 checks**, including the behavioural one that
`ALLOC_MEMORY_FUNCS["cuda"] is alloc_with_pin_memory` under `is_hip()`. A marker
in the bytecode proves the file was edited; that check proves the edit does what
it claims.

## 3. NO in-container patching — confirm, don't perform

The predecessor kits' `REPRODUCE.md` had a **MANDATORY** step here: hand-patch
`GLM52_P1V3` into the running container and relaunch, because their image only
carried the one-directional `_p1v2_trim` revision.

**That step is gone.** On this branch P1V3 is baked in by the Dockerfile. The
`_p1v2_rows` marker in step 2's gate is what distinguishes the two — `_p1v2_trim`
alone is satisfied by the *older* revision and cannot tell them apart.

If your gate shows `_p1v2_trim` but not `_p1v2_rows`, you are on the wrong image.
Do not hand-patch; rebuild.

## 4. Start etcd (prefill node) and kvd (both nodes)

```bash
bash scripts/start_services.sh $PJOB prefill $PIP    # etcd + kvd
bash scripts/start_services.sh $DJOB decode  $DIP    # kvd only
```

Expect `etcd up`, `kvd socket OK`, and an all-zero `statctl` dump on both.

> Each node's engine talks to its **own local** kvd socket. These are two separate
> stores, not two views of one — read the counters per node.

## 5. Launch the prefill leg — **DPA OFF, CHUNK 65536, GMU 0.70**

All three arguments are load-bearing. See `notes.md` §1–§3.

```bash
JOB=$PJOB MY=$PIP PIP=$PIP DPA=0 CHUNK=65536 GMU=0.70 MTP=0 \
  bash scripts/ab_boot.sh prefill armB
```

- `DPA=0` — the point of the arm. `--ep-size 8` stays (EP_DECOUPLE).
- `CHUNK=65536` — **global** budget. Without it, DPA-off would silently get ⅛ the
  per-step work of a DPA-on arm and the comparison would be meaningless.
- `GMU=0.70` — **0.80 does not boot** without DPA. It aborts with
  `HSA_STATUS_ERROR_OUT_OF_RESOURCES` at `token usage: 0.04`.

> `WARNING: engines still present after 40s` on a clean boot is a **false alarm** —
> the wait loop counts processes after the new leg has already started. Verify by
> timestamp if unsure (`docker exec … ps -eo pid,lstart,args`).

## 6. Launch the decode leg — DPA ON, MTP ON, kvd OFF

```bash
JOB=$DJOB MY=$DIP PIP=$PIP DPA=1 MTP=1 KVD=0 \
  bash scripts/ab_boot.sh decode armB
```

`KVD=0` is deliberate: a PD decode leg sets `disable_radix_cache=True` itself and
sglang rejects hierarchical cache alongside it.

Wait for both — **never grep an appended log for readiness**, it matches the
previous run within seconds:

```bash
PJOB=$PJOB PIP=$PIP DJOB=$DJOB DIP=$DIP bash scripts/ab_wait.sh 2400
# -> BOTH READY after 403s
```

## 7. Router on the prefill node — policy **kv-aware**

```bash
JOB=$PJOB MY_IP=$PIP POLICY=kv-aware PORT=8190 bash scripts/ab_router.sh
```

Expect:

```
{"status":"ok","active_workers":2}  router healthy (policy=kv-aware)
router-policy=kv-aware overlap_weight=1 prefill=20.0 decode=2.0
```

`active_workers` must be **2**, not 1.

> On a restart **always pass a fresh port**. A router whose circuit breaker is
> still open returns 503 in ~0.4 s, which looks exactly like a backend failure.
> Read the latency: ~0.4 s is the breaker; 12–23 s is a real backend fault.

## 8. Verify every feature BEFORE spending 67 minutes

A green run that proves nothing is the default outcome here.

| feature | command | expected |
|---|---|---|
| prefill DPA **off** | `docker exec … ps -eo args \| grep '[i]nfera.engine'` | **no** `--dp-size`, **no** `--enable-dp-attention`, **has** `--ep-size 8`, `--chunked-prefill-size 65536`, `--mem-fraction-static 0.70` |
| decode DPA **on** | same on the decode node | `--dp-size 8 --enable-dp-attention` |
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
`generation_config.json`. `temperature: 0` with MTP produces output that is
indistinguishable from KV corruption.

## 9. Capture the pre-run kvd baseline

```bash
for spec in "$PJOB:armB_prefill" "$DJOB:armB_decode"; do
  j=${spec%%:*}; n=${spec##*:}
  spur exec $j bash -c 'export DOCKER_CONFIG=/tmp/dockercfg
    docker exec agbench_mtp python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock' \
    > $W/results/$n.kvd_before.json
done
```

## 10. Run — 67 minutes

```bash
ROUTER=http://$PIP:8190 setsid bash scripts/ab_bench.sh par8_armB_nodpa < /dev/null &
```

The script prints the workload md5 — expect **`968b1543155839135dc9eaf6dd142626`**
for `spec/par8.yaml` with this cluster's tokenizer path.

**No load knobs on the CLI**: the YAML is the single source of truth. Passing
`--initial-sessions` would silently shadow it.

Monitor (note the `\r` handling — the progress bar overwrites in place):

```bash
tail -c 400 /tmp/benchB.log | tr '\r' '\n' | tail -1
```

Abort if `In-flight` pins at 24: backpressure, not the config, would then be
setting the load and the window is wasted. (It did not — max 24 never pinned.)

## 11. Capture the after-state — **BEFORE anything restarts or expires**

**This is the step this kit failed to complete.** The allocation was reclaimed at
the wall clock ~11 h later, and three artifacts were lost: the kvd deltas, the
router pick distribution, and the `collect_env.sh` snapshot. Do these immediately
after the run finishes:

```bash
# 1. kvd after-state, both nodes
for spec in "$PJOB:armB_prefill" "$DJOB:armB_decode"; do
  j=${spec%%:*}; n=${spec##*:}
  spur exec $j bash -c 'export DOCKER_CONFIG=/tmp/dockercfg
    docker exec agbench_mtp python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock' \
    > $W/results/$n.kvd_after.json
done

# 2. router metrics + log — BOTH live inside the container and die with it
spur exec $PJOB bash -c 'export DOCKER_CONFIG=/tmp/dockercfg
  docker exec agbench_mtp curl -s http://'$PIP':8190/metrics' > $W/results/router_metrics.txt
spur exec $PJOB bash -c 'export DOCKER_CONFIG=/tmp/dockercfg
  docker exec agbench_mtp gzip -c /tmp/router.log' > $W/results/router.log.gz

# 3. environment snapshot, both nodes
for j in $PJOB $DJOB; do
  spur exec $j bash -s < scripts/collect_env.sh > $W/env/env_$j.txt
done
```

Results land in
`$W/bench/par8_armB_nodpa/par8_armB_nodpa/<timestamp>/{summary.json,metrics.jsonl,metadata.json}`.

---

## Success criteria vs. measured

The instruction set no numeric target — it asked for the par8 workload run
verbatim with prefill DPA off and kv-aware routing. Against the bars in
`spec/par8.yaml`:

| bar | source | measured | verdict |
|---|---|---|---|
| success rate ≥ 0.97 | `sla.success_rate` | **0.9842** | **PASS** |
| TTFT p90 < 30,000 ms | `sla.ttft_p90_ms` | **6,389 ms** (sustain) | **PASS**, 4.7× margin |
| e2e p50 < 4,500 ms | `sla.e2e_p50_ms` | **not recorded** | see `README.md` |
| full cycle completes | the instruction | 4,007.7 s, ramp+sustain+drain | **PASS** |
| in-flight not pinned at cap | method | never pinned at 24 | **PASS** |
| workload byte-identical to par8 | the instruction | md5 match, only the tokenizer path retargeted | **PASS** |
