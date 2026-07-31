# 10 — does the kv-aware router actually route? (2 decode workers)

**Ran:** 2026-07-30 13:00–13:40 · **Cost:** ~25 min (3 cold starts + 2 arms), 16 GPUs / 2 nodes
**Verdict:** ✅ **The scorer decides.** kv-aware → decodeA 17 / decodeB 0. round-robin → +6 / +8.

## What this experiment answers

**Does the kv-aware policy change where requests go, or is it merely
instantiated?**

Every earlier run could only confirm the role weights were *loaded* — the router
logged them, and that was the end of it. With **one** prefill worker and **one**
decode worker the scorer has no alternative to choose between: whatever cost it
computes, the request goes to the only worker in that role. It cannot express a
preference, so it cannot be tested.

This adds a **second decode worker** so a routing decision can exist, then runs
the same workload under two policies and counts where the decode work landed.

**Topology.** GLM-5.2-MXFP4, real mooncake RDMA, kvaware + kvd on:

| | host | TP | GPUs | port | `--kv-events-bind` | `--kv-snapshot-port` | gmu |
|---|---|---|---|---|---|---|---|
| prefill | chi2879 · 10.2.122.10 | 8 | 0–7 | 30000 | 5557 | 8801 | 0.88 |
| decodeA | chi2867 · 10.2.122.44 | 4 | 0–3 | 30000 | 5557 | 8801 | 0.70 |
| decodeB | chi2867 · 10.2.122.44 | 4 | 4–7 | **32000** | **5657** | **8802** | 0.70 |

**All three ports must differ between co-located workers** — unrelated code
paths, and one of them fails in a way that looks healthy (below). **gmu 0.70,
not 0.85**: TP4 doubles GLM-5.2's per-GPU weights to ~102 GB and 0.85 OOMs on a
390 MiB allocation.

Workload: the same prefix-reuse run in both arms — 4 sessions × 4 turns ×
2 phases, ~6200-token shared prefix, 32 requests. Only `--router-policy`
differs.

## Result

| policy | decodeA | decodeB |
|---|---|---|
| kv-aware (prefill 20.0 / decode 2.0) | **17** | **0** |
| round-robin | **+6** | **+8** |

Round-robin figures are deltas; the raw counters ran 17→23 and 0→8.
Re-derivable from the committed logs (see `results/routing_split_rederived.txt`):

```bash
grep -a 'Decode batch' pd_decodeA.log | grep -cE '2026-07-30 12:1[78]'        # 17
grep -a 'Decode batch' pd_decodeB.log | grep -cE '2026-07-30 12:1[78]'        #  0
grep -a 'Decode batch' pd_decodeA.log | grep -cE '2026-07-30 12:(19|20|21)'   #  6
grep -a 'Decode batch' pd_decodeB.log | grep -cE '2026-07-30 12:(19|20|21)'   #  8
```

Correctness held in both arms:

```
kv-aware    : 32/32, median latency 0.71s (cold) -> 0.27s (warm)
round-robin : 32/32, median latency 0.77s (cold) -> 0.26s (warm)
```

## The findings that matter

### 1. Same workload, same workers, flip the policy — the distribution inverts

This is the differential the earlier runs could not produce. Under kv-aware,
**decodeB ran nothing at all.** Not less — zero. And it was up and idle at the
time: it reached `ready to roll` at 12:15:03, about two and a half minutes
before the first arm's traffic. Under round-robin, with everything else
identical, it took 8 of 14.

So the kv-aware scorer is not merely instantiated. It is **deciding**.

### 2. All-to-one is the CORRECT behaviour here, not a load-balancing bug

Every request in this workload shares one prefix. decodeA holds it. The cost
function is

```
cost = w * (request_blocks - hits) + active_blocks
```

and with `w = 2.0` on the decode side the cache-locality term dominates the load
term at this concurrency. Sticking to the worker that already has the prefix is
exactly the point of the policy — sending half the traffic to a cold worker
would be the bug.

Under a workload with several distinct prefixes, or at concurrency high enough
for `active_blocks` to matter, the split would move.

### 3. A third default-port collision — and the worst-behaved of the three

**decodeB's first attempt died and looked completely healthy doing it.** It
loaded 408 GB, initialised its DP ranks, and printed the line everyone greps
for — then, *after* that, during etcd registration:

```
INFO:__main__:using etcd registration: endpoint=10.2.122.10:2379 ...
OSError: [Errno 98] error while attempting to bind on address ('0.0.0.0', 8801)
sys.exit(STARTUP_FAILURE)
```

`--kv-snapshot-port` defaults to 8801 on every worker. The engine was up; the
wrapper died around it. The worker simply never appeared in `/v1/workers`, and
the natural next move — investigating etcd or the router — is the wrong one.

That makes **three** kvaware-path ports that collide when two workers share a
host, all unrelated code paths:

| # | Port | Default | Fails |
|---|---|---|---|
| 1 | sglang `--kv-events-config` block | from `free_tcp_port_block` | **at** startup — `ZMQError: Address already in use`. A real bug (patch 0001): deterministic, not a race |
| 2 | `--kv-events-bind` | `tcp://0.0.0.0:5557` | **at** startup — second leg fails to bind |
| 3 | `--kv-snapshot-port` | `8801` | **after** `ready to roll` — looks healthy, worker absent |

Only #1 is a code bug; #2 and #3 are defaults that cannot be shared. #3 is the
one worth internalising. Diagnostic order and full write-up in
`results/snapshot_port_collision.txt`.

> **The check that catches it:** count the workers the *router* sees, not the
> `ready to roll` lines. `curl .../v1/workers | grep -o '"url"' | wc -l` must be
> 3. `scripts/run.sh` asserts this and greps for the Errno-98 signature on a
> shortfall.

This is also the run where **patch 0001 actually mattered** — two workers, one
host, kvaware on. It held:

```
decodeA  "endpoint": "tcp://*:25186"
decodeB  "endpoint": "tcp://*:13792"
```

Pre-fix both would have been `32764`, and with `dp_size=4` all four per-rank
publishers would have overlapped.

## How to reproduce

```bash
bash scripts/run.sh
```

~25 minutes. Writes `results/routing_kvaware.observed.txt` and
`results/routing_roundrobin.observed.txt`, each with decode-batch counters
before and after, plus a verdict table. Compare against
`results/step4_role_weights_routing.txt`.

```bash
KEEP=1 bash scripts/run.sh
DECODE_TP=4 DECODE_GMU=0.70 bash scripts/run.sh   # the defaults, spelled out
```

The script sets all three per-worker ports, **refuses to start** if
`--mem-fraction-static` is above 0.75 at TP≤4 (the TP4 OOM), and **fails loudly
if the router does not see 3 workers** — because with fewer than two decode
workers there is no routing decision and the experiment cannot produce a result.
It runs the kv-aware arm, restarts *only* the router onto round-robin, and runs
the identical workload again.

## Gotchas specific to this experiment

- **`ready to roll` is not a readiness check when two workers share a host.**
  See finding 3. Count workers at the router.
- **All three ports, every time.** `PORT`, `KV_PUB_PORT`, `KV_SNAP_PORT`. Fixing
  one does nothing for the others, and each fails differently.
- **`--mem-fraction-static` scales with TP.** 0.85 is a TP8 number. At TP4,
  GLM-5.2's per-GPU weights double to ~102 GB and DP3 dies on a 390 MiB
  allocation with 120 MiB free. Use 0.70. "Just split the node into two workers"
  is exactly the move that invites carrying the wrong number over.
- **Restart only the router between arms.** Restarting a leg changes its cache
  state and destroys the comparison. The whole design is "same workers, same
  warm caches, one variable".
- **Count `Decode batch` lines, not requests.** The router does not log
  placement, so the workers' own logs are the evidence. A long generation logs
  several lines, so this is a proxy — fine for a differential where the same
  workload ran twice, not a throughput measurement.
- **TP4 workers log 4 kvd adapters, not 8.** One per DP rank. Eight would be
  wrong here.
- **The arm labels come from the run transcript, not the logs.** The workers do
  not record the router's policy, and the router log is gone. The *split* is
  re-derivable; the *names* rest on the in-session record.

## What this does NOT prove

1. **The prefill weight (20.0) is unmeasured.** There is only one prefill
   worker, so the prefill half of the scorer had nothing to choose between —
   exactly the limitation this experiment fixed on the decode side. Measuring it
   needs two prefill nodes.
2. **This is affinity, not *correct* affinity.** One prefix, one holder. It
   shows the scorer sticks to a worker that has the data; it does not show that
   with N competing prefixes each lands on its own worker. That is the stronger
   test and it was not run.
3. **The load term never pushed back.** Requests were sequential, so
   `active_blocks` stayed near zero and locality won by default. Where the
   balance point is — the concurrency at which load starts to overcome
   locality — is completely unprobed, and it is the number an operator would
   actually want.
4. **Nothing about failover or worker health.** No worker was killed mid-run; no
   behaviour under a degraded or overloaded worker was observed.
5. **The latency figures are not a comparison between policies.** 0.71 vs 0.77
   cold and 0.27 vs 0.26 warm, single runs, no repeats. The dominant effect is
   cache warmth, not routing. Do not read a policy ranking out of them.
6. **kvd's counters were not read in this experiment.** Whether the routing
   decision changed kvd traffic is unknown.
7. **Two decode workers is the minimum, not a realistic fleet.** A 2-worker
   all-to-one result and a 20-worker all-to-one result are different claims.
8. **Single observation per arm.** No repeats. With counts this small (17/0,
   6/8) a repeat would be cheap and worthwhile; it was not done.

## Environment (verbatim in every packup so this folder stands alone)

**Cluster access.** Jump host `root@149.28.124.225`, then `ssh <node>`. Key-based,
no password appears in any script here.

```bash
J(){ ssh -o StrictHostKeyChecking=no root@149.28.124.225 \
       "ssh -o StrictHostKeyChecking=no $1 '$2'"; }
```

**Nodes** (8× AMD Instinct MI355X / gfx950 each, 128 threads, 3023 GB RAM):

| Host | Data-plane IP | amdgpu | Kernel |
|---|---|---|---|
| chi2879 | 10.2.122.10 | 6.16.13 | 6.8.0-124-generic |
| chi2867 | 10.2.122.44 | 6.16.13 | 6.8.0-107-generic |

**Fabric:** ionic RoCE v2, 8 rails/node (`ionic_0`…`ionic_7`), all PORT_ACTIVE.
Module `26.03.3.001`, NIC firmware `1.117.5-a-77`, routable GID at **index 1**
(hence `MC_GID_INDEX=1`). chi2879→chi2867 RTT 0.069 ms.

**Image:** `infera/engine-sglang:pd-unified`
sha256 `f8ec2d627392435b7cf4c97e47b93a3b36588bec43864a1758b7c0dc9405bd18`
(sglang 0.5.15.post1, torch 2.9.1+rocm7.2.0, ROCm 7.2.0). A **local build**, not
on a registry — the Infera PR #19 rebuild that makes mooncake cross-node RDMA
work. Distribute with `docker save ... | ssh <dst> docker load`.

**infera repo:** branch `yihou.dev.glm5.2.mxfp4.experiment`, commit `362192e7`.

**Models (absolute paths on the shared VAST NFS mount `/mnt/vast`):**
- GLM-5.2-MXFP4 — `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` (408 GB, 282 shards,
  `GlmMoeDsaForCausalLM`, 78 layers, 256 experts)
- Qwen3-1.7B — `/mnt/vast/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`

**Kit staging dir:** `/mnt/vast/c_huggingface/glm52_kvexp` — must be on the
shared FS so both nodes' containers see the same copy.

**Host `libionic` injection is mandatory.** Without it RDMA silently degrades to
TCP. Verify **inside** the container: `ibv_devinfo | grep -c PORT_ACTIVE` → `8`.

**Secrets:** cluster SSH only (key-based). No registry login needed (local
image). etcd runs unauthenticated on the data-plane IP.

## Traps that bite every experiment here

**1. `docker exec -d $CTR bash -lc '...'` does not persist.** The detached
login-shell form exits and takes the child with it. Symptom: no process, no log
file, no error. Bit us twice (router, kvd daemon). **Always** stage a script
file and run `docker exec -d $CTR bash /the_script.sh`, or use
`docker exec -d $CTR env VAR=... bash /script`.

**2. Nested ssh quoting silently mangles variables.** In
`ssh jump "ssh node '...$f...'"` the OUTER shell expands `$f`. Stage a script
file instead of fighting the quoting.

**3. Cold start is 6-12 min and looks like a hang.** GLM-5.2 loads 408 GB.
Watch the log growing (`wc -l`); don't kill it.

**4. Three kvaware ports collide when two workers share a host** — unrelated
code paths, fixing one does nothing for the others:

| Port | Default | Failure |
|---|---|---|
| sglang `--kv-events-config` block | from `free_tcp_port_block` | deterministic same base → `ZMQError: Address already in use` (this is **patch 0001**) |
| `--kv-events-bind` | `tcp://0.0.0.0:5557` | identical on every leg; 2nd fails to bind |
| `--kv-snapshot-port` | `8801` | **the nastiest** — leg prints `ready to roll`, *then* dies during etcd registration. Looks healthy; worker never appears in `/v1/workers`. |

**5. `--mem-fraction-static` is TP-dependent.** `0.85` suits TP8 (51 GB/GPU of
weights). At **TP4** weights double to 102 GB/GPU and 0.85 OOMs — use `0.70`.

**6. `--hicache-ratio` sizes the host pool off the KV pool.** Default 2.0 on a
small model tried to allocate **355 GB per DP rank**. Use `--hicache-size <GB>`
(absolute) instead.

**7. Never probe a PD leg directly.** `curl` to a leg's own port just hangs — a
PD leg only serves through the pair. Use the router, and use a differential run
(flip one thing, hold the rest) to isolate.

**8. Shared cluster hygiene.** Don't prune images, don't mount other people's
drives, don't `docker rm` a container you can't prove is yours
(`docker inspect` → Binds/Env/Created).
