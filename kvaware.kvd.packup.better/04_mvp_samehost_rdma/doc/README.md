# 04 — same-host PD over mooncake RDMA does not work (MVP round r3)

**Ran:** 2026-07-30 · **Cost:** ~8 min, one node (chi2879), 8 GPUs
**Verdict:** ◐ PARTIAL — startup fully fixed, KV transfer fails. A **same-host
limitation of this fabric**, orthogonal to kvaware and kvd.

## What this experiment answers

The three earlier fixes are in. So: **with both legs actually up, does a
co-located PD pair serve a request?**

No — and the reason is one layer below everything the previous rounds were
about. This is also the first round that could ask the question at all, since
it is the first with two live legs and therefore the first to start the router.

Configuration: Qwen3-1.7B, one node (chi2879), 1P1D — prefill TP4 GPU0-3
:30000, decode TP4 GPU4-7 :31000, infera router :8100, `infera.kvd` daemon on
`/tmp/kvd/kvd.sock`, `KVD=1 KVAWARE=1 POLICY=kv-aware`, mooncake transfer
backend on **real RDMA** (no `MC_FORCE_TCP`).

Carried in from earlier rounds: `--hicache-size 8` (bounds the host pool,
which defaulted to 354.94 GB/rank), legs' `--port` 1000 apart, and the patched
`free_tcp_port_block`.

## Result

**What now works — and this is the load-bearing part of the finding:**

```
prefill http://10.2.122.10:30000 kv_events_endpoint=tcp://10.2.122.10:17213
decode  http://10.2.122.10:31000 kv_events_endpoint=tcp://10.2.122.10:31215
```

Both legs `ready to roll`, both registered, **distinct** kv-event endpoints —
17213 and 31215 are two independent draws from the randomised port scan.
Pre-fix both would have been 32764 and the decode leg would have died.

**What fails:** every completion returns HTTP 500.

```
Failed to get kvcache from prefill instance
```

with, underneath it in the leg log:

```
worker_pool.cpp:408 ... local_nic: ionic_0, peer_nic: ...@ionic_4:
                        transport retry counter exceeded
rdma_endpoint.cpp:472  Invalid argument: received packet mismatch
```

## The finding that matters

**Two legs on one host means the mooncake KV transfer has to loop back across
RDMA rails** — out of `ionic_0` and back in at `ionic_4` on the same machine —
**and this ionic RoCE v2 fabric will not do it.** `transport retry counter
exceeded` is the QP giving up; `received packet mismatch` is the endpoint
seeing traffic it cannot reconcile.

This is a property of the **deployment shape**, not of kvaware, not of kvd, and
not of the model. On two nodes the transfer goes out one host's NIC and into
the other's — the ordinary path — and it works.

**Why this is not a kvaware/kvd result**, stated plainly because it is easy to
mis-cite:

- The failure is in the mooncake RDMA data path.
- Both switches were on, but neither is in that path — kvaware publishes events
  over ZMQ, kvd stores blocks over a unix socket.
- Every layer *above* the transfer is demonstrably healthy: legs start, DP ranks
  initialise, kvd connects, kv-events publish on separate sockets, both workers
  appear in `/v1/workers`. The only thing that fails is moving the bytes.

The workaround is `MC_FORCE_TCP=1`, this repo's known correct-but-slower path.

> **It is a workaround for the transport and buys nothing else.** In
> particular it does not make the output *right*. Getting a completion and
> getting a correct completion are different claims, and the next round
> (garbled text under `MC_FORCE_TCP`, on both the feature-on run and its
> switches-off baseline) is what settles that one. **For any correctness
> claim: two nodes, real RDMA.**

### The adjacent fix, and the third collision

This round also gave each leg its own `--kv-events-bind` (5557 / 5657) —
infera's *own* publisher socket, whose default `tcp://0.0.0.0:5557` is
identical on every leg.

That is the **third** distinct port collision in this deployment shape, all
three unrelated code paths, one found per debugging round. The full set,
with the diagnostic order for telling them apart, is in
`results/three_port_collisions.txt`. The short version:

| # | Port | Fails when | Signature |
|---|---|---|---|
| 1 | sglang `--kv-events-config` block, from `free_tcp_port_block` | at startup | `ZMQError ... 'tcp://*:<port>'` — **a real bug**, patch 0001 |
| 2 | `--kv-events-bind`, default 5557 | at startup | second leg fails to bind |
| 3 | `--kv-snapshot-port`, default 8801 | **after** `ready to roll` | leg looks healthy, never appears in `/v1/workers` |

#3 did not bite here — it surfaced later, in a GLM-5.2 run with two decode
workers on one host. It is recorded because it is the nastiest: "ready to roll"
is the line everyone greps for, and this failure comes after it.

## How to reproduce

```bash
TRANSPORT=rdma bash scripts/run.sh   # reproduce the failure (EXPECTED to fail)
TRANSPORT=tcp  bash scripts/run.sh   # MC_FORCE_TCP=1, completions come back
```

`TRANSPORT=rdma` leaves mooncake on RDMA; `TRANSPORT=tcp` sets
`MC_FORCE_TCP=1`. Both write `results/r3_samehost_rdma.observed.txt`; the
committed reference is `results/r3_samehost_rdma.txt`.

`scripts/probe.py` distinguishes three outcomes rather than two — **OK**,
**WRONG** (a completion came back with bad content) and **ERROR** (no
completion at all). That distinction is the whole point here: this round's
failure is ERROR, and the next round's is WRONG, and conflating them loses the
attribution.

The script uses a uniquely-named throwaway container and removes it on exit.

## Gotchas specific to this experiment

- **`active_ports=8` is a precondition, not a formality.** If host `libionic`
  injection fails, RDMA silently degrades to TCP — and then `TRANSPORT=rdma`
  "passes" for entirely the wrong reason. `scripts/run.sh` prints the count and
  says to stop if it is not 8. Also check `MC_FORCE_TCP` hits = 0 in the
  evidence block.
- **A "successful" rdma run is a failed run.** The pass condition is: 8 rails,
  `MC_FORCE_TCP` count 0, both legs ready, two workers registered with distinct
  endpoints, *and* 4/4 HTTP 500s with `Failed to get kvcache`. Points 1-4 are
  what make it a transport result; without them the failure is ambiguous.
- **`TRANSPORT=tcp` succeeding is a transport result only.** No 500s does not
  mean correct. The probe may report 0/4 correct with garbled content — that is
  a different failure needing a differential run to attribute.
- **Never probe a PD leg directly to isolate this.** `curl` to a leg's own port
  just hangs; a PD leg only serves through the pair. Use the router.
- **Rail pinning was never tried.** Whether forcing both legs onto the *same*
  ionic device would work around the loopback is **unknown** — `MC_FORCE_TCP`
  was the faster path to an answer and the real fix was to use two nodes.
- **Raw logs for this round no longer exist** (container removed before the
  GLM-5.2 runs). Everything in `results/` was captured in-session, and the full
  `worker_pool.cpp` / `rdma_endpoint.cpp` stanzas are among the things lost —
  only the quoted lines survive. See `logs/README.md`.

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
