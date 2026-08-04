# REPRODUCE

Top-to-bottom, from a machine with cluster access. Total wall clock: **~25 min** bring-up
+ **~17 min** bench. Cold start is 3–4 min per leg (weights → tilelang JIT → cudagraph
capture) — that is not a hang.

```bash
# Constants used throughout
JUMP=root@149.28.124.225
W=/mnt/vast/c_huggingface/glm52_example_verify        # kit staging, shared by both nodes
PREFILL_NODE=chi2835 ; PREFILL_IP=10.2.122.78
DECODE_NODE=chi2879  ; DECODE_IP=10.2.122.10
IMAGE=infera/engine-sglang:merged-e
```

> **Node substitution is allowed but not free.** Any two MI355X nodes with the image and
> `ib_peer_mem` loaded work. If you change nodes: the IPs change, `PREFILL_IP` is the
> etcd host, and you **must re-run step 2** — `MC_GID_INDEX` and the active-rail list are
> per node, never copied.

---

## 0. Prerequisites

- **Slurm holds on both nodes belong to `yeandy-debug`. Never `scancel`.** Kill only your
  own containers and processes.
- Both nodes need: the image, `ib_peer_mem` loaded, `/mnt/vast` mounted, GPUs free.
- Secrets: cluster SSH only. No registry login, no API key. See `environment.md`.
- Repo: `AMD-AGI/Infera` @ `e2d462a` on branch `yihou.dev.glm52.example`, **with**
  `patches/kit_fixes.diff` applied to `examples/sglang_1p1d_glm5.2/`.

Check both nodes are actually free before spending a bring-up:

```bash
ssh $JUMP "for n in $PREFILL_NODE $DECODE_NODE; do echo \"== \$n\"; \
  ssh \$n 'rocm-smi --csv --showmeminfo vram | tail -8 | awk -F, \"{s+=\\\$3} END {printf \\\"VRAM %.0f GB used\\n\\\", s/1073741824}\"'; done"
```

Under ~5 GB total is idle. **Do not** derive this from `rocm-smi --showmemuse` percent —
it has read 0 % on a node at 98 % VRAM.

## 1. Stage the kit onto shared storage

The kit must be at the **same path on both nodes**; `up.sh` ssh's in and runs
`engine/leg.sh` from there.

```bash
ssh $JUMP "mkdir -p $W"
cd <your Infera checkout>
tar czf - examples/sglang_1p1d_glm5.2 | ssh $JUMP "cd $W && rm -rf kit && tar xzf - \
  && mv examples/sglang_1p1d_glm5.2 kit && rmdir examples"

# the site wrapper — this kit's is in scripts/
scp scripts/cluster.vultr.sh $JUMP:$W/kit/cluster/
```

`scripts/cluster.vultr.sh` is the **only** file carrying site values. Everything it sets
is either copied from step 2's report or listed in `environment.md`.

## 2. Preflight — this also picks your wrapper

**The kit's `preflight_rdma.sh mode` does not run on this image**:
`infera.tools.preflight.mooncake_mode` is absent from `merged-e`. Stage the repo's module
and bind-mount it in. (Gap #1 in `notes.md` — an image gap, not a script bug.)

```bash
cd <your Infera checkout>
tar czf - infera/tools/preflight | ssh $JUMP "cd $W && rm -rf repo_preflight \
  && mkdir repo_preflight && cd repo_preflight && tar xzf -"

for N in $PREFILL_NODE $DECODE_NODE; do
  ssh $JUMP "ssh \$N 'HL=\$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1); \
    docker run --rm --network host --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
      --group-add video --group-add render --cap-add=IPC_LOCK --ulimit memlock=-1:-1 \
      --security-opt seccomp=unconfined --ipc=host \
      -v /mnt/vast:/mnt/vast -v \$HL:/host-libionic/libionic.so:ro \
      -v $W/repo_preflight/infera/tools/preflight/mooncake_mode.py:/opt/infera/infera/tools/preflight/mooncake_mode.py:ro \
      $IMAGE python3 -m infera.tools.preflight.mooncake_mode'"
done
```

Expect on both nodes (ours are in `results/preflight_chi28*.txt`):

```
peer-mem module: PRESENT  (module:ib_peer_mem)
[A] bare ibv_reg_mr + peer-mem (default, no-pin, every rail)  — viable ★ best
  env: ... MC_GID_INDEX=1 ...
```

`peermem: present` ⇒ **mode A** ⇒ `cluster.peermem.sh`. Copy `MC_GID_INDEX` and the rail
list into the wrapper. **Read the rail list on both nodes** — ours differ (`ionic_5` is
`[DOWN]` on chi2879), and the wrapper must name their intersection.

## 3. Tear down anything already holding the GPUs

`scripts/teardown_prev.sh` removes **only our own** containers and waits for VRAM to
drain. Read it before running: it names the containers it will remove.

```bash
scp scripts/teardown_prev.sh $JUMP:$W/
for N in $PREFILL_NODE $DECODE_NODE; do ssh $JUMP "ssh \$N 'bash $W/teardown_prev.sh'"; done
```

Expect `GPUs idle` and ~0.3 GB/card. **The wait is the point** — relaunching before the
driver releases OOMs on a box that looks idle.

## 4. Bring up — one command

```bash
ssh $JUMP "cd $W/kit && setsid nohup bash cluster/cluster.vultr.sh up > $W/up.log 2>&1 < /dev/null &"
ssh $JUMP "tail -f $W/up.log"      # Ctrl-C on the tail stops nothing
```

Runs: containers on both nodes → etcd + kvd on prefill → both legs concurrently → wait →
router. Ours took **210 s** (prefill) / **225 s** (decode); see `logs/up_bringup.log`.

**Check this line before anything else:**

```
  RDMA PORT_ACTIVE visible in container: 8      <- chi2835
  RDMA PORT_ACTIVE visible in container: 7      <- chi2879
```

A **0** here means the host provider library was not injected and KV will not move over
RDMA — the leg will still boot and serve. The fixed `common.sh` prints a warning block;
an unfixed one prints a bare `0`. See `patches/README.md` §1.

## 5. Apply GLM52_P1V3 to the decode leg — then RELAUNCH

**Without this the decode leg crashes minutes into an agentic workload** with
`Expected lengths.size(0) == B to be true`. `merged-e` does not carry it; measured:

```bash
ssh $JUMP "ssh $DECODE_NODE 'docker exec glm52_pd grep -c _p1v2_clip \
  /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py'"
# 0  -> the patch is absent (an unpatched image also has _p1v2_trim = 4, which is P1V2, not V3)
```

Patch the source, then relaunch — a running engine has already imported the old module:

```bash
scp patches/apply_p1v3.py scripts/relaunch_decode.sh $JUMP:$W/
ssh $JUMP "ssh $DECODE_NODE 'docker cp $W/apply_p1v3.py glm52_pd:/tmp/ && \
  docker exec glm52_pd python3 /tmp/apply_p1v3.py'"
# -> patched OK - GLM52_P1V3 occurrences: 3

ssh $JUMP "ssh $DECODE_NODE 'bash $W/relaunch_decode.sh'"     # reaps, deletes .pyc, relaunches
```

Wait for it, then **verify the compiled bytecode — not the source**:

```bash
ssh $JUMP "ssh $DECODE_NODE 'docker exec glm52_pd curl -sf -m5 http://$DECODE_IP:30001/health >/dev/null && echo SERVING'"
ssh $JUMP "ssh $DECODE_NODE 'docker exec glm52_pd bash -c \"P=/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/__pycache__/dsa_indexer.cpython-310.pyc; \
  ls -la --time-style=full-iso \\\$P; strings \\\$P | grep -c _p1v2_rows; strings \\\$P | grep -c _p1v2_clip\"'"
# expect: a .pyc mtime AFTER the patch, and 1 / 1
```

A patch script that prints `patched OK` against a running engine has changed the source
and nothing else. This exact trap has invalidated a full experiment on this stack.

The router re-pairs through etcd automatically — no restart needed.

## 6. Verify every feature BEFORE spending a bench window

```bash
ssh $JUMP "cd $W/kit && bash cluster/cluster.vultr.sh smoke"
```

Read the blocks, not the exit code. Ours: `results/smoke_final.txt`. Expected:

| block | healthy reading |
|---|---|
| `/health` + `/v1/workers` | `active_workers: 2`, one prefill + one decode, both `active` |
| chat completion | a correct, coherent answer |
| mooncake | `MC_FORCE_TCP=0  GID-is-NULL=0  RDMA-devices-in-container=8` (and `7`) |
| DP-attention | prefill `dp_size=1 enable_dp_attention=False`; decode `dp_size=8 … =True` |
| MTP | `MEDIAN=2.88` — **read the median**, not the last batch |
| kv-aware / kvd | `router_policy: "kv-aware"`, `tokenizer-loaded=1`, 8 kvd adapters |

Two independent cross-checks, not covered by `smoke`:

```bash
# a) every feature again, off the LIVE cmdline rather than the log
scp scripts/feature_evidence.sh $JUMP:$W/ && ssh $JUMP "bash $W/feature_evidence.sh"

# b) prefix caching actually reports: same prompt twice, cached_tokens must rise
scp scripts/cache_probe.py $JUMP:$W/
ssh $JUMP "ssh $PREFILL_NODE 'docker cp $W/cache_probe.py glm52_pd:/tmp/ && \
  docker exec glm52_pd python3 /tmp/cache_probe.py'"
# round 1: prompt=1301 cached=None      <- cold, EXPECTED
# round 2: prompt=1301 cached=1280      <- the check
```

`cached=None` on a single cold request is **not** a missing `--enable-cache-report`.
Round 2 is the test.

## 7. Run the customer bench — AgentX Case-A

Follow `agentx.caseA.customer.packup_20260803/REPRODUCE.md` §1–4 to stage the customer
kit, materialize the corpus and build `aiperf-agentx:v1.0`. Then:

```bash
ssh $JUMP "mkdir -p /root/glm52_example_agentx && cp -a <staged bench dir> /root/glm52_example_agentx/"
scp scripts/run_agentx.sh $JUMP:/root/glm52_example_agentx/
ssh $JUMP "cd /root/glm52_example_agentx && setsid nohup bash run_agentx.sh > run_caseA.log 2>&1 < /dev/null &"
```

`run_agentx.sh` sets environment only; the customer's `replay_caseA.sh` is untouched
(`md5sum` = `7cde1afc627c7e4868eac0fd13741baa`). **`OUT` is deliberately inside `$HERE`** —
the customer script mounts only `$HERE`, so an `OUT` outside it writes results into the
container namespace where they die on exit, and the sweep then prints `FAILED` for a run
that succeeded.

~17 min per concurrency point (66 s warmup + 900 s profiling + teardown). Watch:

```bash
ssh $JUMP "grep -aiE 'Phase warmup complete|Phase profiling sending complete' \
  /root/glm52_example_agentx/bench/results/c8/art/logs/aiperf.log"
```

## 8. Read the result

```bash
ssh $JUMP "cd /root/glm52_example_agentx && python3 analyze.py bench/results/c8/art"
```

Expected (ours):

```
profiling requests : 225  (warmup 8, cancelled 0, ctx-overflow-skip 0)
window             : 913.5 s   0.246 req/s   261.4 out tok/s
TTFT  (ms)         : p50 5,546 | p90 22,068 | p99 36,914
ITL   (ms)         : p50 14.46 | p90 23.25
server cache hit   : 51.90 %  (from usage_prompt_cache_read_tokens, n=175)
```

**That 51.9 % is a metric-definition artifact, not your deployment.** The per-request
median is 88.2 %:

```bash
scp scripts/cache_by_turn.py $JUMP:/root/glm52_example_agentx/
ssh $JUMP "cd /root/glm52_example_agentx && python3 cache_by_turn.py bench/results/c8/art/profile_export.jsonl"
```

See `notes.md` §5 for why the two differ and which one to quote.

## Expected output vs. success criteria

| criterion | expected | ours |
|---|---|---|
| `up.sh` completes | router up, `active_workers: 2` | ✅ 210 s / 225 s |
| all six features positively verified | see §6 table | ✅ all six |
| customer bench runs unmodified | md5 unchanged, 0 errors | ✅ 225 req / 0 errors |
| performance comparable to the reference | ITL ≈ 14 ms, cache ≈ 88 % | ✅ 14.46 ms / 88.2 % |

## If it doesn't reproduce

`notes.md` has the full trap list. The three that cost the most:

1. **In-container RDMA device count is 0** → the host provider library is not being
   injected. `HOST_RDMA_LIB` + `HOST_RDMA_MOUNT` + `ENTRYPOINT_KEEP=1`, all three.
2. **Decode leg dies minutes in** with `Expected lengths.size(0) == B` → P1V3 did not
   reach the **bytecode**. Re-check step 5's `.pyc` verification.
3. **Reading a leg's own port directly hangs.** Always go through the router.
