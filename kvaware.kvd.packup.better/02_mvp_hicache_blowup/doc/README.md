# 02 — the `--hicache-ratio` host-memory blow-up (MVP round r1)

**Ran:** 2026-07-30 · **Cost:** ~5 min, one node (chi2879), 8 GPUs, no request served
**Verdict:** ❌ FAIL by design — two independent startup failures, neither a code bug

## What this experiment answers

The first attempt at bringing kvaware + kvd up on a PD pair. It answers, badly
and therefore usefully: **what breaks when you turn kvd on for the first time?**

Configuration: Qwen3-1.7B, one node, 1P1D — prefill TP4 on GPU0-3 port 30000,
decode TP4 on GPU4-7 port 31000, infera router :8100, `infera.kvd` daemon on
`/tmp/kvd/kvd.sock`, `KVD=1 KVAWARE=1 POLICY=kv-aware`.

Qwen3-1.7B was chosen deliberately: it is 4 GB and cold-starts in ~2 minutes
versus GLM-5.2-MXFP4's ~30, which is what made a five-round debugging loop
affordable in one afternoon. That choice is also, ironically, what caused
failure (b).

## Result

Neither leg reached `ready to roll`. Nothing was ever served.

**(a) prefill died on a derived ZMQ port:**

```
zmq.error.ZMQError: Address already in use (addr='tcp://127.0.0.1:30235')
RuntimeError: sglang subprocess exited with code -9 before reporting ready
```

**(b) decode asked for 1.4 TB of host RAM:**

```
[DP0 TP0 EP0] max_total_num_tokens=1547424, ...
[DP0 TP0 EP0] Allocating 354.94 GB host memory for hierarchical KV cache.
[DP1..DP3] (same, x4 ranks = 1.4 TB on a 3 TB box)
```

The two are unrelated. Fixing either does nothing for the other.

## The finding that matters

**354.94 GB is not a bug. It is `--hicache-ratio` doing exactly what it says**,
and the arithmetic closes to the printed digit:

```
bytes/token = 2 (K,V) x 28 layers x 8 kv_heads x 128 head_dim x 2 (bf16)
            = 114,688 B

device KV pool = 1,547,424 tok x 114,688 B = 177.47 GB
host pool      = 177.47 GB x hicache_ratio 2.0 = 354.94 GB   <- the log line
x 4 DP ranks   = 1.42 TB
```

(Full working, with every input traced to `config.json` or a log line, in
`results/hicache_sizing_arithmetic.txt`.)

The trap is upstream of the ratio: **`max_total_num_tokens` was 1,547,424 in the
first place.** Qwen3-1.7B's weights are ~0.9 GB per GPU at TP4, an MI355X has
far more HBM than that, and `--mem-fraction-static 0.60` handed the remainder to
the KV pool. So:

> **`hicache_ratio` couples the host pool to the device pool, and the device
> pool grows as the model shrinks. The smaller and cheaper the model you pick
> to debug wiring with, the larger the host allocation it demands.**

That is backwards from the intuition that a small model is the safe thing to
test with, which is exactly why it caught us.

The fix is `--hicache-size <GB>` — an absolute per-rank cap that overrides the
ratio and decouples the two pools entirely. Round r2 then printed
`Allocating 8.00 GB host memory for hierarchical KV cache.` The later GLM-5.2
runs used 16.

**Do not fix this by lowering the ratio.** Below roughly 1.5, sglang's
`prefetch_capacity_limit` computes to ~0 and the L3 tier is written but never
read — a silent no-op, much harder to spot than a 1.4 TB allocation failure.
Two failure modes on one dial, one loud, one silent. Set an absolute size.

On failure (a): port 30235 is not a port anyone configured. sglang derives a
block of internal ZMQ/nccl endpoints from `--port`, and the legs' ports were
close enough that the blocks overlapped. Spacing them 1000 apart (30000 /
31000) settled it. Note this is a *different* port bug from the one that kills
round r2 — that one binds `tcp://*:<port>` (wildcard) and originates in
infera's `free_tcp_port_block`; this one is loopback-scoped and originates in
sglang's own `--port` arithmetic. Same symptom string, different code, different
fix.

## How to reproduce

```bash
MODE=broken bash scripts/run.sh    # reproduce r1 as it happened (expect FAIL)
MODE=fixed  bash scripts/run.sh    # --hicache-size 16 + 1000-port gap
```

`MODE=broken` omits `--hicache-size` (so `hicache_ratio=2.0` applies) and sets
the port gap to 100. `MODE=fixed` sets `--hicache-size 16` and a 1000 gap.
Both write `results/r1_hicache_blowup.observed.txt`; compare against the
committed `results/r1_hicache_blowup.txt`.

The script uses a uniquely-named throwaway container and removes it on exit,
including on failure.

Under `MODE=fixed`, expect **prefill ready, decode possibly still dead** — that
is the next bug (both legs' `--kv-events-config` landing on the same base port),
a different failure with a different fix. This experiment is not about it.

## Gotchas specific to this experiment

- **`MODE=broken` really does attempt a ~1.4 TB host allocation.** On the
  3023 GB chi287x boxes it is refused rather than wedging the node, which is
  what happened on 2026-07-30. Do not run it on a smaller-RAM machine and do
  not run it next to somebody else's job.
- **A "successful" broken run is a failed run.** `prefill_ready=0` plus the
  354.94 GB line is the pass condition. If both legs come up, your box has a
  different `ip_local_port_range` (so the derived blocks happen not to overlap)
  or enough free RAM that the request succeeded — an environment difference to
  record, not a broken script.
- **The exact numbers are model- and flag-specific.** 354.94 GB follows from
  Qwen3-1.7B's 28/8/128/bf16 geometry, `--mem-fraction-static 0.60`, TP4 and
  DP4. Change any of those and you get a different number; the *mechanism* is
  what transfers.
- **Raw logs for this round no longer exist** — the container was removed before
  the GLM-5.2 runs. Everything in `results/` was captured in-session. The
  354.94 GB figure is independently re-derivable from `config.json`, so the
  headline does not rest on the lost log. See `logs/README.md`.

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
