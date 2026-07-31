# 07 — GLM-5.2 two-node PD with kvaware ON + kvd ON

**Ran:** 2026-07-30 11:20–11:35 · **Cost:** ~6 min cold start + ~1 min probe, 16 GPUs / 2 nodes
**Verdict:** ✅ PASS on correctness and wiring — ⚠️ **and kvd served zero traffic**

## What this experiment answers

**Does switching kvaware and kvd on break anything, and does the wiring
actually reach the engine?**

Both halves matter. The first is a correctness question and the answer is no —
4/4, identical to the switches-off run on the same topology. The second is a
"did it even attach" question, and it needed answering because two earlier
rounds had died before the features could be exercised at all (a deterministic
port collision, then a host-pool blow-up), and a third had produced garbled
output on a substrate that turned out to be the culprit.

**What it does not answer is whether kvd is worth anything.** It is not, on this
workload, and the section below says so in detail. Read that section before
quoting the 4/4.

**Topology.** GLM-5.2-MXFP4, two nodes, real mooncake RDMA:

| | prefill | decode |
|---|---|---|
| Host | chi2879 · 10.2.122.10 | chi2867 · 10.2.122.44 |
| Parallelism | TP8, DP-attention dp8/ep8 | TP8, DP-attention dp8/ep8 |
| Port | :30000 | :30000 |
| `--mem-fraction-static` | 0.88 | 0.85 |
| kvaware / kvd | ON / ON | ON / ON |

infera router on the prefill node :8100, policy **kv-aware**. `infera.kvd`
daemon on **both** nodes (each engine talks to its own local daemon over
`/tmp/kvd/kvd.sock`). etcd on the prefill node :2379. Host pool bounded with
`--hicache-size 16`. Correctness bar: `scripts/probe.py`, 4 temp=0 factual
prompts, ≥3/4.

## Result

| Check | Observed | Required |
|---|---|---|
| Correctness (`probe.py`) | **4/4** | ≥3/4 |
| `MC_FORCE_TCP` hits | 0 | 0 |
| `HIP dmabuf disabled` | 8 | 8 |
| `infera-kvd adapter connected` — prefill | **8** | 8 (one per DP rank) |
| `infera-kvd adapter connected` — decode | **8** | 8 |
| `KV plane up:` | present on **both** legs | present |
| Host pool allocation | 8 × **16.00 GB** | bounded, not the 2.0 ratio |
| `disaggregation-decode-enable-radix-cache` in decode argv | **1** (auto-appended) | 1 |
| kv-events endpoints | **25075** / **1649** (distinct) | distinct |
| **kvd traffic** | **gets=0 sets=0 hits=0 entries=0** | — |

All four completions were coherent and correct, matching the switches-off run
prompt-for-prompt. Verbatim from `results/step1_kvaware_kvd_4of4.txt`:

```
[OK] 'The capital of France is'  -> '... The capital of France is Paris. ...'
[OK] 'The capital of China is'   -> '... The capital of China is Beijing. ...'
[OK] '2+2='                      -> '... 2 + 2 = 4. ...'
[OK] 'The largest planet ...'    -> '... Mercury, Venus, Earth, Mars, Jupiter ...'

4/4 correct
```

## The findings that matter

### 1. The features are genuinely wired, on every rank, on both legs

Sixteen `infera-kvd adapter connected` lines — eight per leg, one per DP rank.
Not one leg, not rank 0 only. Alongside them the tree cache changes shape:

```
impl=HiRadixCache  hierarchical=True        (this run)
impl=RadixCache    hierarchical=False       (the switches-off run)
```

### 2. `KV plane up:` — this refuted a standing hypothesis

An earlier round suspected infera's own KV-event probe plane never attaches,
because `_find_radix_cache()` looks for the RadixCache in the **wrapper**
process while sglang runs in a **subprocess**, and no `KV plane up:` line had
ever been seen. It is there, on both legs:

```
INFO:__main__:KV plane up: events_bind=tcp://0.0.0.0:5557
  events_advertise=tcp://10.2.122.10:5557 snapshot=http://10.2.122.10:8801
  engine_block_size=64 index_block_size=64
```

The earlier rounds simply never got far enough to print it. Absence of evidence
in a truncated log had been read as evidence of absence — worth recording as a
method error, not just a resolved ticket.

### 3. The decode leg is legal only because infera auto-appended a flag

A PD decode leg sets `disable_radix_cache=True` on its own ("KV cache is forced
as chunk cache for decode server"), and sglang rejects
`enable-hierarchical-cache` alongside it. What legalises kvd on the decode side
is `--disaggregation-decode-enable-radix-cache`, which infera appends **only
when kv-events are enabled**. Confirmed present in this run's decode argv, with
`disable_radix_cache=False` as the consequence.

> **So turning kvaware OFF silently disables kvd on the decode leg.** The two
> switches are not independent, whatever the flag names suggest.

### 4. Patch 0001 held — the legs got different kv-events port bases

```
prefill  "endpoint": "tcp://*:25075"
decode   "endpoint": "tcp://*:1649"
```

Two independent draws from the randomised port scan. Pre-fix both callers got
`32764` deterministically. (On this two-node topology they would not have
actually collided — different hosts — but the *values* are the evidence that the
patched code path ran, and they must be checked, because the collision is fatal
the moment two workers share a host.)

### 5. **kvd served zero traffic — and this is the honest headline**

```
StatsResponse(entries=0, host_bytes=0, spillover_bytes=0, long_bytes=0,
              gets_total=0, sets_total=0, hits_total=0, misses_total=0,
              evictions_total=0)
```

Every counter zero. Four prompts of a few dozen tokens, sharing no prefix,
nowhere near enough to pressure the GPU KV pool: nothing gets evicted so nothing
is offloaded (`sets=0`), and no request repeats another's prefix so no lookup
would hit even if the store were full (`gets=0`). L3 prefetch is prefix-length
gated on top of that.

**The correct reading is "wired and harmless", not "useful".** That is a real
result — it is the precondition for measuring anything, and two bug fixes were
needed to reach it — but it is a weaker claim than the 4/4 invites. Full
statement of the boundary in `results/kvd_served_zero_traffic.txt`.

## How to reproduce

```bash
bash scripts/run.sh
```

Writes `results/step1_kvaware_kvd_4of4.observed.txt`, which includes the kvd
counters **before and after** the probe — the comparison that makes the zero
meaningful rather than merely unmentioned. Compare against the committed
`results/step1_kvaware_kvd_4of4.txt`.

```bash
KEEP=1 bash scripts/run.sh     # leave the deployment (and the kvd daemons) up
POLICY=round-robin bash scripts/run.sh   # features on, scorer off
WAIT_MIN=30 bash scripts/run.sh
```

The script preflights both nodes, injects host `libionic`, applies patch 0001,
starts a kvd daemon per node **from a staged script file** (the
`docker exec -d ... bash -lc` form does not persist — it cost two debugging
rounds), brings up both legs with `KVAWARE=1 KVD=1 --hicache-size 16`, starts
the kv-aware router, polls for readiness while printing the growing log
line-count, snapshots kvd, probes, snapshots kvd again, and prints a verdict
that explicitly refuses to let a PASS be read as "kvd is useful".

## Gotchas specific to this experiment

- **`--hicache-size`, never the default ratio.** `--hicache-ratio` defaults to
  2.0 and sizes the host pool off `max_total_num_tokens`. On a small model that
  computed to **354.94 GB per DP rank**. Use the absolute flag. But do not push
  the *ratio* below 1.5 either: sglang's `prefetch_capacity_limit` then computes
  to ~0 and L3 gets written and never read.
- **The `prefetch_threshold` warning at startup is cosmetic.** infera logs
  `SGLang version has no recognized prefetch_threshold field`. On 0.5.15.post1
  that field is not in `ServerArgs` at all; it is read from the backend
  extra-config (`hiradix_cache.py:675`), which is exactly where infera puts it.
  The value takes effect. Only infera's `ServerArgs`-probing fallback is looking
  in the wrong place. Do not chase it.
- **The two legs here share `KV_PUB_PORT=5557` and `KV_SNAP_PORT=8801` and that
  is fine — because they are on different hosts.** Put two workers on one host
  and both collide, along with a third (the sglang kv-events block). The 8801
  one is the nastiest: the leg logs `ready to roll` and *then* dies during etcd
  registration, so it looks healthy and simply never appears in `/v1/workers`.
- **kvd counters are not in the engine logs.** They come from the daemon over
  its unix socket (`scripts/kvdstats.sh`). Grepping the `.log` files for
  `gets_total` finds nothing — that is expected.
- **Both nodes need their own kvd daemon.** Each engine talks to its *local*
  socket; there is no shared store across the two hosts in this configuration.
  A consequence worth remembering when reading counters: prefill-side and
  decode-side numbers are two separate stores, not two views of one.

## What this does NOT prove

1. **That kvd stores or serves anything.** Counters all zero. The most that can
   be said is that it connected and was ready to. This is the single most
   important limitation and the easiest to lose when the 4/4 gets quoted.
2. **That kv-aware routing does anything.** The policy was `kv-aware` and the
   scorer was instantiated, but with **one** prefill worker and **one** decode
   worker it has no alternative to choose between. It cannot express a
   preference, so its being loaded is all that was shown. Both role weights were
   left at the default 1.0 here in any case.
3. **That the features are free.** No throughput, latency, or memory-overhead
   comparison against the switches-off run was made. 128 GB of host RAM per node
   was reserved for a pool that stayed empty; whether that costs anything under
   load is unmeasured.
4. **Anything at concurrency.** Every request was sequential. No batching, no
   queueing, no contention on the kvd socket.
5. **That the L3 tier works.** `long_bytes=0`. The L3 path was configured
   (`--long-path`) and never exercised. Its substrate in this run is the
   container overlay, which kvd itself flags.
6. **That correctness would hold under cache pressure.** 4/4 was achieved with
   an empty cache. A hit that returns *wrong* text is exactly the failure mode a
   KV cache can introduce, and no hit occurred here, so that risk is untested.
7. **Single observation.** One run. No repeat, no variance.

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
