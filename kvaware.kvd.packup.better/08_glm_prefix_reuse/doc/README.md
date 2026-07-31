# 08 — prefix-reuse workload: does kvd actually serve, or only connect?

**Ran:** 2026-07-30 11:50–12:10 · **Cost:** ~4 min per arm on a live deployment (no restart)
**Verdict:** ✅ kvd **serves** — `gets 0→170`, `hits=170`, `misses=0`, 573 MB resident, 32/32 correct
**⚠️ And the 2.7× speedup in this run is NOT kvd's.** See below before quoting it.

## What this experiment answers

A previous run had kvaware and kvd wired on every DP rank of both legs, scored
4/4, and kvd's counters were **all zero**. Four short prefix-disjoint prompts
give a KV-offload path nothing to store and nothing to fetch. So the standing
question was: *is kvd merely connected, or does it do work?*

This supplies a workload built for the answer: **4 sessions × 4 turns, every
turn sharing a ~6200-token system prefix, run twice** (a cold phase that
populates, then a reuse phase) = 32 requests. Then it reads the daemon's own
counters before and after.

The prefix length is not arbitrary. sglang's L3 prefetch is gated on it —
`hiradix_cache.py:1603` returns early when `prefetch_length <
prefetch_threshold` (256 by sglang's default; infera passes 64). A few dozen
tokens clears neither. ~6200 clears both.

**Topology** (unchanged from the run this follows — nothing was restarted except
the router):

| | prefill | decode |
|---|---|---|
| Host | chi2879 · 10.2.122.10 | chi2867 · 10.2.122.44 |
| Parallelism | TP8, DP-attention dp8/ep8 | TP8, DP-attention dp8/ep8 |
| `--mem-fraction-static` | 0.88 | 0.85 |
| kvaware / kvd | ON / ON | ON / ON |

Real mooncake RDMA, infera router :8100, `infera.kvd` daemon on **both** nodes
(each engine talks to its local socket — two separate stores, not two views of
one).

## Result

**Arm 1 — default overlap weights (1.0 / 1.0)**

```
prefix chars=24778 (~6194 tokens est.) sessions=4 turns/session=4

PHASE 1 (cold — populates cache)   16/16 correct | median latency 0.71s
PHASE 2 (reuse — same prefixes)    16/16 correct | median latency 0.83s
TOTAL 32/32 correct
```

kvd counters, prefill node chi2879:

| | entries | host_bytes | gets | sets | hits | misses |
|---|---|---|---|---|---|---|
| BEFORE | 0 | 0 | 0 | 0 | 0 | 0 |
| AFTER | **340** | **600837120** (573 MB) | **170** | 340 | **170** | **0** |

Decode node chi2867 went `entries 8 → 380`, `gets 0 → 170`, `hits 0 → 170`,
`misses 0`. (The 8 pre-existing entries are from the preceding run — so kvd was
not *completely* idle then; the prefill side was.)

**Arm 2 — role weights 20.0 / 2.0**, router restarted with
`--kv-overlap-weight 1.0 --kv-prefill-overlap-weight 20.0
--kv-decode-overlap-weight 2.0`, confirmed live:

```
INFO:__main__:router-policy=kv-aware overlap_weight=1 prefill=20.0 decode=2.0

PHASE 1   16/16 correct | median latency 0.26s
PHASE 2   16/16 correct | median latency 0.27s
TOTAL 32/32 correct
```

kvd counters after arm 2, prefill node: `entries=340 host_bytes=600837120
gets=170 sets=340 hits=170 misses=0` — **byte-for-byte unchanged**.

## The findings that matter

### 1. kvd is proven to serve, not merely to connect

`gets 0 → 170` with `hits=170 / misses=0` and 573 MB resident. The previous
experiment's open question is closed: given something to cache, kvd caches it
and reads it back.

### 2. **The 2.7× speedup is the GPU radix cache, not kvd**

This is the single most important sentence in this packup and the easiest number
to misquote. Between arm 1 (0.71 s) and arm 2 (0.26 s) the median latency fell
2.7×. Over that same interval:

```
kvd, prefill node, after arm 1:  entries=340 gets=170 sets=340 hits=170 misses=0
kvd, prefill node, after arm 2:  entries=340 gets=170 sets=340 hits=170 misses=0
                                 ^^^ not one counter moved
```

If kvd had served a single block during the fast run, `gets_total` would have
increased. It did not. The GPU-side radix cache was already warm from arm 1 and
satisfied every prefix lookup before anything reached the host tier.

**It is not the role weights either.** They were verified loaded and active —
but with one prefill worker and one decode worker the scorer has no alternative
to route to, so it cannot have changed where anything went, and therefore cannot
have changed the latency. What differed between the arms was cache warmth. The
router restart is a coincidence of timing.

Full statement in `results/speedup_is_not_kvd.txt`.

### 3. `hits=170 / misses=0` is same-process reuse only

A 100% hit rate is suspiciously clean, and the reason is that every lookup
repeated something *this same live process* had just stored, with a warm GPU
cache in front of it the whole time. kvd's actual selling point — a cache that
outlives the engine — is untested here.

### 4. A workload-design bug worth more than the fix

The **first attempt scored 1/32**. The cause was the prompt, not the engine: the
shared prefix said "Answer strictly from the reference material below", and
GLM-5.2 obeyed it exactly, correctly observing that ~6000 tokens of padding
about "system components" said nothing about Paris — and declining to answer.
Every response was **fully coherent**.

Two fixes, both needed: reword the prefix to "background context only — ignore
it unless the question is about it", and raise `max_tokens` 32 → 128 (GLM-5.2
emits a reasoning preamble; 32 truncated before the answer appeared). Then
32/32.

> **Lesson: when a correctness harness reports near-zero on a model that is
> visibly producing sane text, suspect the harness first.** The discriminating
> question costs one second — *is the output coherent?* Garbled points at the
> engine, cache or transport; coherent-but-"wrong" points at the prompt, the
> scoring rule, or the token budget. Only the first justifies opening the engine
> source. Full write-up in `results/workload_design_bug.txt`.

## How to reproduce

```bash
bash scripts/run.sh                  # both arms
ARMS=default bash scripts/run.sh     # just the arm where kvd's counters move
KEEP=1 bash scripts/run.sh           # leave the deployment up
```

Writes `results/step2_prefix_reuse_default.observed.txt` and
`..._weighted.observed.txt`, each with kvd counters **bracketing** the workload —
that bracketing is the design, since a single reading proves nothing and only
the delta is evidence. Compare against `results/step2_prefix_reuse.txt`.

The script preflights both nodes *and the workload*: it computes the prefix
length locally and fails before spending a 6-minute cold start if it would not
clear the prefetch threshold. It then injects host `libionic`, applies patch
0001, starts a kvd daemon per node from a staged script file (the
`docker exec -d ... bash -lc` form does not persist), brings up both legs with
`--hicache-size 16`, runs each arm, and prints a verdict that explicitly refuses
to attribute the latency drop to kvd.

## Gotchas specific to this experiment

- **Read kvd's counters, not the clock.** This is the whole methodological
  point. The engine log does not contain them; they come from the daemon over
  its unix socket (`scripts/kvdstats.sh`).
- **Each node has its own kvd store.** Prefill-side and decode-side counters are
  two separate daemons. Do not add them up or expect them to agree.
- **The prefix must clear the prefetch threshold.** Below it, the L3 path is
  skipped outright and you get a run that looks like the zero-traffic one, for a
  completely different reason. The preflight checks this.
- **`max_tokens` must leave room for GLM-5.2's reasoning preamble.** 128 works;
  32 scores zero on correct answers. A substring-match harness cannot tell
  truncation from a wrong answer, so this shows up as a correctness failure.
- **Do not restart the legs between arms.** Only the router. Restarting the
  engine changes the GPU cache state and destroys the comparison — which is, in
  fact, a *different* and worthwhile experiment, but not this one.
- **Sequential requests only.** The workload sends one request at a time.

## What this does NOT prove

1. **That kvd made anything faster.** No latency benefit is demonstrated
   anywhere in this experiment. The one speedup observed is attributed, with
   counter evidence, to the GPU radix cache instead.
2. **That kvd survives a restart** — its actual selling point. Every hit here
   repeated something the same live process had just stored. Showing survival
   requires killing the engine, keeping the daemon, verifying VRAM back at idle,
   and replaying; the signature to look for is gets/hits climbing while
   sets/entries/bytes stay flat.
3. **That the role weights do anything.** Verified *loaded*, not *effective*.
   One worker per role means the scorer has nothing to choose between. Measuring
   them needs ≥2 workers in a role.
4. **That the L3 tier was meaningfully exercised.** `long_bytes` did grow to
   573 MB so the write path works, but L3 here is the container overlay, which
   kvd itself flags (`ssd region long on overlay`). No block device, no
   O_DIRECT, no real storage measurement.
5. **Anything about eviction or capacity pressure.** 573 MB against a 16 GB host
   pool per rank. Nothing was ever evicted, so the eviction path, the L2→L3
   spill path under pressure, and the behaviour at the capacity limit are all
   untested.
6. **Anything at concurrency.** Sequential requests throughout. No contention on
   the kvd socket, no queueing, no batching interaction.
7. **That a cache hit returns *correct* text under adverse conditions.** It did
   here (32/32), on one prefix, at temp=0, with no eviction. That is a narrow
   slice of the ways a KV cache can be wrong.
8. **Single observation per arm.** No repeats, no variance estimate. The latency
   medians are over 16 requests each and should be read as indicative only.

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
