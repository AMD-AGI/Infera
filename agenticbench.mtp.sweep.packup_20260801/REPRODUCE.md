# REPRODUCE

From two bare spur nodes to the 8-point sweep. **~3 h**, of which ~15 min is the
two parallel image builds, ~50 min is engine cold starts, and ~70 min is the
sweep itself.

Order matters. Every step has a verification, and each verification exists
because skipping it has silently produced a wrong result on this stack before.

## 0. Hold two nodes

```bash
cat > hold.sh <<'EOF'
#!/bin/bash
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-burst-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --time=24:00:00
sleep 86400
EOF
sbatch --parsable hold.sh     # x2
squeue -u $USER
```

Do **not** pass `-A`; it defaults to your account. `-q amd-burst-qos` matters —
it has the large node budget and schedules reliably under contention.

**If a job flips to `PENDING Reason=JobHoldMaxRequeue` within seconds**, it landed
on a node that accepts placement and then fails at launch. Catch the node and
exclude it — this happened here on `crsuse2-m2m-232`:

```bash
squeue -j <job> -h -o "%N"
scancel <job>
sbatch --parsable --exclude=<bad-node> hold.sh
```

Health-gate each node before trusting it:

```bash
spur exec <job> bash -c 'hostname; ip -o -4 addr show ens3 | awk "{print \$4}"; ls /sys/class/infiniband/'
```

`mlx5_0` must be present. Note each node's `ens3` IP — everything below takes
them as arguments; nothing is hardcoded inside a leg script.

Export them once:

```bash
export PJOB=<prefill job>  PIP=<prefill ens3 ip>
export DJOB=<decode job>   DIP=<decode ens3 ip>
```

## 1. Get the source, including the two uncommitted patches

```bash
git clone git@github.com:AMD-AGI/Infera.git && cd Infera
git checkout yihou.dev.glm52.merged.experiment
git log --oneline -1        # -> b92a1e8

git apply <KIT>/patches/dockerfile_add_rocm_hicache_layer.diff
mkdir -p deploy/docker/patches/sglang_rocm
cp <KIT>/patches/patch_hicache_rocm_host_alloc.py deploy/docker/patches/sglang_rocm/
```

**The patches are not optional on this cluster.** Without them the prefill leg
dies with `Memory access fault by GPU node-N` the moment kvd writes back at
long-context scale. See `patches/README.md`.

Stage the tree onto shared storage so both nodes build the same bytes:

```bash
bash <KIT>/scripts/stage_source.sh      # -> /shared_nfs/yihou_agbench_mtp/build/src.tar
```

It stages the **worktree**, not `git archive` of HEAD — `git archive` would drop
exactly the two uncommitted changes this cluster needs. `manifest.txt` records
the SHA and the dirty paths so the delta is never invisible.

## 2. Build the image on BOTH nodes, in parallel (~15 min each)

```bash
bash <KIT>/scripts/build_image.sh $PJOB &
bash <KIT>/scripts/build_image.sh $DJOB &
wait
cat /shared_nfs/yihou_agbench_mtp/build/build_*.status     # both must read "ok sha256:…"
```

Build **on each node** rather than moving a 28 GB tar: `docker save` backgrounded
inside `spur exec` dies at namespace teardown, and the claim under test is that
the Dockerfile reproduces the deployment.

The script keeps the `docker build` in the **foreground** of the `spur exec` on
purpose — a build client *backgrounded* inside `spur exec` is killed at namespace
teardown even under `nohup`/`setsid`. Keep the login-side process alive.

> **The two image ids WILL DIFFER.** Each node built independently, so Rust
> objects and layer timestamps differ. **Do not check for equal digests** —
> check content equivalence, which is the next step.

## 3. Containers, and the bytecode gate

```bash
bash <KIT>/scripts/start_ctr.sh $PJOB prefill
bash <KIT>/scripts/start_ctr.sh $DJOB decode
```

Each must end with **`BYTECODE_GATE OK`** — 8 assertions read from **freshly
compiled bytecode**, not from source. A build log saying a patch printed success
is not the same as the interpreter executing patched code; a stale `__pycache__`
entry has invalidated a full experiment on this stack twice.

| assertion | want |
|---|---|
| `GLM52_ROCM_HOST_ALLOC` in `pool_host/common` pyc | 1 |
| `wait_event` in mooncake `conn` / `utils` / `prefill` pyc | 2 / 1 / 1 |
| **`ALLOC_MEMORY_FUNCS["cuda"] → alloc_with_pin_memory` under HIP** | OK |
| `_p1v2_trim` (DSA p1) | ≥1 |
| `_glm52_match_page_table_rows` (DSA p2b) | ≥1 |
| `requires_dp_attention_eager_forward` (DSA p3) | ≥1 |

The bolded row is worth more than the marker rows: a marker proves the file was
edited, the dispatch check proves the edit *does what it claims*.

Also expect `GPUGATE True 8` and `RDMA PORT_ACTIVE: 9`.

## 4. etcd + kvd daemons

```bash
bash <KIT>/scripts/start_services.sh $PJOB prefill $PIP    # etcd + kvd
bash <KIT>/scripts/start_services.sh $DJOB decode  $DIP    # kvd only
```

Want `etcd up`, `kvd socket OK`, and all-zero counters on both.

Each node's engine talks to its **own local** kvd socket — these are two separate
stores, not two views of one. Read the counters per node accordingly.

## 5. Boot both legs (prefill ~5 min, decode ~10 min)

```bash
bash <KIT>/scripts/boot.sh prefill 262144 1 0 g1     # ctx, kvd=1, mtp=0
bash <KIT>/scripts/boot.sh decode  262144 0 1 g1     # ctx, kvd=0, mtp=1
bash <KIT>/scripts/wait_ready.sh 1800
```

`ctx=262144`, not 131072: at 131072 the Case A input distribution clamps 16.1 %
of requests (measured — the previous run's probe showed p90 *and* p99 both pinned
at exactly 131,072); at 262144 only ~1.4 % clamps.

The decode leg is legitimately slower (606 s vs 315 s here) — it also loads the
EAGLE draft model and captures its draft graphs. **Poll, do not kill.**

`wait_ready.sh` polls the HTTP endpoint rather than grepping the log for
`ready to roll`: the logs are appended across runs, so a grep matches a
*previous* run's line within seconds.

Then the router:

```bash
bash <KIT>/scripts/router.sh 8190      # -> "router healthy", 2 workers registered
```

It is `python -m infera.server` — **not** `infera.router`, which is a package
with no `__main__` and fails with a message that reads like a missing dependency.
Use the script.

## 6. Gate — every row, before spending a measured window

```bash
bash <KIT>/scripts/gate.sh g1
```

| row | prefill | decode |
|---|---|---|
| `ready to roll` | 1 | 1 |
| `Memory access fault` | **0** | **0** |
| `Scheduler hit an exception` | 0 | 0 |
| `Traceback` | 0 | 0 |
| `Errno 98` **after** the ready line | 0 | 0 |
| `context_length=262144` | 1 | 1 |
| `dp_size=8` | 1 | 1 |
| `speculative_algorithm='EAGLE'` | **0** | **1** |
| `disable_custom_all_reduce` | False | **True** |
| `enable_hierarchical_cache` | **True** | False |
| live `sglang::scheduler_DP` procs | 8 | 8 |

Two traps encoded in that script:

* **Grep through `strings`.** The logs contain binary bytes; a plain `grep -c`
  returns 0, which reads exactly like "the bad thing never happened".
* **Check `Errno 98` *after* the ready line.** A `--kv-snapshot-port` collision
  lets a leg log `ready to roll` and *then* die during etcd registration — it
  looks healthy and simply never appears in `/v1/workers`.
* `speculative_algorithm` is echoed **quoted** (`='EAGLE'`); matching a bare
  `=EAGLE` reports MTP absent on a leg that is running it.

Confirm both workers registered (through the router — **never probe a leg's own
port, it hangs**):

```bash
curl -s http://$PIP:8190/v1/workers
```

## 7. Correctness

```bash
spur exec $PJOB bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker exec agbench_mtp python3 /shared_nfs/yihou_agbench_mtp/scripts/correctness.py http://$PIP:8190"
```

Short factual **4/4**. The needle test will read **4/5 or 5/5** and that is
expected — see `notes/needle_resolved.md`: the failures are stochastic sampling
excursions, not KV corruption (the identical prompt against an identical warm
cache passes 3/6), and the benchmark's own prompts cannot reach that failure
mode. **Do not chase it.**

The probe deliberately sends GLM-5.2's own sampling (**temperature 1.0 /
top_p 0.95** from `generation_config.json`) and `max_tokens 2048`. Do not set
`temperature: 0`: under MTP that manufactures the exact corruption signature and
has produced a wrong verdict on this stack before.

## 8. Feature evidence (optional but cheap)

```bash
bash <KIT>/scripts/feature_evidence.sh g1
```

Records the PD/DPA/MTP/kvaware/kvd evidence in one file. To reproduce the two
non-obvious findings:

```bash
# the kv-event wire view (prefill emits plain ints, NOT bigrams -- see notes/)
docker exec -d agbench_mtp python3 .../kv_event_wire_probe.py tcp://$PIP:17568 8 60
# ...then drive traffic through the router while it listens

# the kvd SERVING proof (gets/hits climb, sets FLAT)
bash <KIT>/scripts/restart_replay.sh
```

The kv-events port is **not** fixed — read it from `/v1/workers`
(`kv_events_endpoint`), it is allocated per boot.

## 9. The sweep (~70 min)

```bash
bash <KIT>/scripts/sweep.sh sweep1
```

8 points, **one server**, nothing retuned between them:

| point | ISL | OSL | conc |
|---|---:|---:|---|
| p50 | 74,000 | 320 | 1, 32, 64, 128 |
| p90 | 155,000 | 3,300 | 1, 32, 64, 128 |

Design decisions that matter if you change it:

* **Paired percentiles, P99 dropped** (operator decision). Pairing each ISL with
  its own percentile's OSL keeps every point a real Case A request shape; a full
  3×3 cross would spend most of its wall clock on shapes the workload never
  produces.
* **One server sized for the largest point.** Headroom was computed, not assumed:
  the KV pool is 3,260,992 tokens/rank, so at the p90 point (158.3K tok/req) a
  rank holds ~20 requests and the node ~164 — conc=128 fits.
* **A distinct seed per concurrency.** `bench_serving` generates prompts
  deterministically, so with one fixed seed a larger N's prompt set is a
  **superset** of a smaller N's — run 128 then 64 and 64's "cold" run is silently
  warm.
* `--num-prompts = 2 × conc` (two full waves); conc=1 gets 4 so its percentiles
  are not a single sample.

## 10. Extract the tables

```bash
docker exec agbench_mtp python3 .../extract_sweep.py \
  /shared_nfs/yihou_agbench_mtp/bench/sweep1 <out>/sweep1_table.md

strings /shared_nfs/yihou_agbench_mtp/logs/g1_decode.log > /tmp/dec.txt
python3 <KIT>/scripts/accept_by_window.py /tmp/dec.txt <sweep.log> <out>/sweep1_accept.md
```

**The acceptance number needs the second script, and this is not optional.**
`bench_serving`'s own `accept_length` is `null` in every JSON (it reads
`<base_url>/server_info`, and `--base-url` is the router, which has none), and
the decode leg's `/server_info` value is a **cumulative, per-DP-rank** mean —
neither is a per-point number. Binning the decode log's 12,654 timestamped
`accept len:` samples into each point's window is. See `RESULTS.md`.

## 11. What "reproduced" means here

| check | expected |
|---|---|
| bytecode gate, both nodes | `BYTECODE_GATE OK` (8/8) |
| leg gate | every row |
| correctness, short factual | 4/4 |
| **kvd restart-replay** | **gets↑ hits↑ `sets` FLAT, misses 0**, gets = tokens/64×2 |
| **router `cache_hits` × 64 == engine `cached_tokens`** | exact identity |
| `accept len`, p50 point | **2.0–2.4** (not 4.00) |
| sweep | 8/8 JSONs, `real conc` ≈ requested |
| `Traceback` / `Memory access fault` | **0** on both legs throughout |

The two bolded rows are the ones that would go red if a fix were absent or wrong.
Everything else can pass for uninteresting reasons.

## If it doesn't reproduce

`notes/` in the order the traps bite. The three that cost the most time here:

1. **`pkill -f infera.engine.sglang` kills its own shell.** The pattern matches
   the `bash -c '...'` command string that *contains* it, so the step hangs
   forever with the engine already dead. Use a bracketed pattern
   (`"[i]nfera.engine.sglang"`). Cost a stalled round.
2. **A plain engine reboot does not empty the host tier**, so a "restart and
   replay" that only restarts the engine process proves nothing about L3. The
   first kvd proof came back a false negative this way — see
   `notes/kvd_serving_proof.md`.
3. **`temperature: 0` + MTP is indistinguishable from KV corruption**
   (`notes/needle_resolved.md`).
