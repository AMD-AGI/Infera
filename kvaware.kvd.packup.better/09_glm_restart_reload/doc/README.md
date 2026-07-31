# 09 — cross-restart KV reuse: kill the engine, keep the daemon

**Ran:** 2026-07-30 12:15–12:45 · **Cost:** ~20 min (two cold starts + two workload passes), 16 GPUs / 2 nodes
**Verdict:** ✅ **CROSS-RESTART REUSE CONFIRMED** — `gets +170`, `hits +170`, `sets` flat, 31/32 correct

## What this experiment answers

**Is kvd the thing doing the work?**

A previous prefix-reuse run showed kvd serving 170 hits with a 100% hit rate.
That result had two confounders and neither is removable by measuring harder:

- every hit repeated something the **same live process** had just stored;
- the **GPU radix cache** was warm the whole time, and it is faster than kvd, so
  it could have been serving everything while kvd's counters merely happened to
  move.

This experiment removes both **by construction** rather than by argument.

**The design.** Kill *only* the sglang engine legs. Leave the kvd daemon
running.

- the engine dies → the GPU KV pool and HiRadixCache die with it; the
  replacement is a fresh process with an empty in-VRAM cache;
- kvd survives → it still holds the 340 entries / 573 MB the **old** engine
  wrote.

Then relaunch and replay the identical workload (4 sessions × 4 turns ×
2 phases, ~6200-token shared prefix, 32 requests).

**The signature that constitutes proof**, on the prefill node:

| counter | must do | why |
|---|---|---|
| `gets_total` | climb | the new engine read |
| `hits_total` | climb equally | and every read hit |
| `sets_total` | **stay flat** | it wrote nothing back |
| `entries`, `host_bytes` | **stay flat** | the store did not grow |

A rising `gets_total` alone would not be enough — `sets_total` staying flat is
what rules out "the new engine rebuilt everything and then hit its own fresh
writes".

**Topology.** GLM-5.2-MXFP4, prefill chi2879 (10.2.122.10) TP8 gmu 0.88 :30000,
decode chi2867 (10.2.122.44) TP8 gmu 0.85 :30000, DP-attention dp8/ep8 on both,
real mooncake RDMA, infera router :8100, etcd :2379, `infera.kvd` daemon on each
node, `--hicache-size 16`.

## Result

### Preconditions verified before relaunching — this *is* the experiment

```
chi2879 GPU[0] VRAM used ... 297840640 B   (~297 MB — idle baseline)
chi2867 GPU[0] VRAM used ... 297754624 B
kvd alive on both nodes .... pgrep -fc 'infera.kvd' = 1
kvd state preserved ........ entries=340, long_bytes=600837120
```

An engine holding a 408 GB model with a populated KV pool does not sit at
297 MB. The process is gone and its VRAM went with it.

### Counters — prefill node chi2879

| | BEFORE restart | AFTER restart + replay | delta |
|---|---|---|---|
| entries | 340 | 340 | **0** |
| host_bytes | 600837120 | 600837120 | **0** |
| long_bytes | 600837120 | 600837120 | **0** |
| sets_total | 340 | 340 | **0** |
| gets_total | 170 | 340 | **+170** |
| hits_total | 170 | 340 | **+170** |
| misses_total | 0 | 0 | 0 |

### Counters — decode node chi2867

| | BEFORE | AFTER | delta |
|---|---|---|---|
| entries | 402 | 424 | +22 |
| sets_total | 402 | 438 | +36 |
| gets_total | 170 | 340 | +170 |
| hits_total | 170 | 340 | +170 |
| misses_total | 0 | 0 | 0 |

### Correctness

```
PHASE 1   15/16 correct | median latency 0.76s
PHASE 2   16/16 correct | median latency 0.72s
TOTAL 31/32
```

## The findings that matter

### 1. A brand-new process read 170 blocks it never wrote

That is the whole result, and every alternative explanation is dead separately:

| Alternative | Killed by |
|---|---|
| "it was the GPU radix cache" | VRAM at 297 MB before relaunch — the cache did not exist |
| "same-process reuse" | writer process `pkill -9`'d, `engine procs left: 0`; reader is a different PID with a different randomised kv-events port (`27591` vs the writer's `25075`) |
| "it rebuilt and hit its own fresh writes" | `sets_total` and `entries` moved by **exactly zero** |
| "the daemon restarted and reloaded from disk" | `pgrep -fc 'infera.kvd'` = 1 throughout; the daemon never restarted |

Full argument in `results/confounders_removed.txt`.

Incidentally, the mechanism that makes this possible at all is
`PYTHONHASHSEED=0`, exported by the leg script: it makes block hashing stable
across processes. Without it a new engine computes different keys for identical
content and every get misses.

### 2. The single miss is a truncation, not a wrong answer

Session 2's "largest planet" question: GLM-5.2 spent its 128-token budget on a
reasoning preamble (`1. **Analyze the Request:** ...`) and never reached
"Jupiter". The same prompt passed in the other three sessions and in every prior
run. This is a harness-budget artifact — a substring-match check cannot
distinguish "truncated before the answer" from "answered wrongly" — and not a
cache or engine defect.

### 3. **Latency did NOT improve, and that is the honest headline**

`0.76 s` on the reload versus `0.71 s` on a cold run. Reading ~6200 tokens of KV
back over a unix socket from the host tier is not obviously cheaper than
recomputing prefill for a 6200-token prefix on 8× MI355X.

> **kvd's demonstrated win here is capacity and survival — the cache outlives
> the engine — explicitly not speed.**

A latency win would need a much longer prefix, a slower prefill, or GPU-direct
reads. None were tested.

### 4. The decode node is not clean, and that is recorded rather than smoothed

Its `sets_total` rose +36 and `entries` +22 while prefill's stayed flat. **Not
investigated.** The most likely explanation is the decode leg's own HiRadixCache
repopulating decode-side blocks that were never in kvd to begin with — but that
is a hypothesis, not a finding.

It does not weaken the conclusion, which rests entirely on the prefill node's
clean read-only delta. One node exhibiting all three required properties
simultaneously is sufficient to establish that the mechanism works.

## How to reproduce

```bash
bash scripts/run.sh
```

~20 minutes: cold start → populate → kill legs (not the daemon) → verify
preconditions → second cold start → replay → diff the counters. Outputs
`results/round1_populate.observed.txt`, `results/preconditions.observed.txt`,
`results/round2_reload.observed.txt`, plus a computed delta table and a verdict.

```bash
KEEP=1 bash scripts/run.sh                  # leave it up to pull logs
IDLE_VRAM_MAX=2147483648 bash scripts/run.sh  # loosen the VRAM gate (see below)
```

**The script gates on the VRAM precondition.** If VRAM is still high after the
kill it says so loudly and tells you not to cite the result, because a surviving
GPU cache would explain any reuse you then observe. That gate is not a
convenience — it is the difference between this experiment and the confounded
one it replaces.

## Gotchas specific to this experiment

- **Kill the engine, NEVER the daemon.** `scripts/restart_legs.sh` targets
  `infera.engine.sglang`, `sglang.launch_server` and `sglang::` and prints
  `kvd alive: <n>` afterwards so you can see it survived. A `pkill -f sglang`
  that also caught the daemon would destroy the very thing under test — and
  would still produce hits on the replay, from a rebuilt store, looking exactly
  like a pass.
- **Check VRAM, not `pgrep`.** A wedged engine process can be gone from `pgrep`
  while its VRAM allocation lingers, and a lingering GPU cache confounds
  everything. `rocm-smi --showmeminfo vram` is the authoritative check; idle on
  these boxes is ~297 MB.
- **The workload must be byte-identical between rounds.** Different text means
  different block hashes means guaranteed misses, and you would wrongly conclude
  kvd does not survive restarts. `scripts/prefix_reuse.py` is deterministic by
  construction (fixed prefix, fixed turns, temp=0).
- **`PYTHONHASHSEED=0` is load-bearing** and lives in the leg script's
  environment, so it does not appear in any log. Drop it and cross-restart
  lookup silently stops working.
- **The prefill log spans this experiment *and* a later one.** Requests at
  11:56–11:59 are this replay (32); 12:17–12:20 (64) belong to a subsequent
  routing run that reused the same prefill process. Window your greps — see
  `logs/README.md`.
- **Read the daemon's counters, not the clock.** Latency did not move here; the
  counters are the entire result.

## What this does NOT prove

1. **That kvd is faster.** It is not, at this prefix length, on this hardware.
   No latency benefit was observed and none is claimed.
2. **That the decode side works the same way.** The decode node's counters moved
   in an unexplained way. The claim rests on the prefill node only.
3. **That kvd survives a *daemon* restart.** The daemon ran continuously here.
   kvd does log `long region recovered from /tmp/kvd-long` at its own startup,
   so an L3 recovery path exists — it was not exercised or verified in this run.
4. **That the store is shared across nodes.** It is not. Each node runs its own
   daemon over a local unix socket; prefill and decode counters are two separate
   stores. Nothing here shows one engine reading another *host's* blocks.
5. **That reuse survives a config change.** Identical engine configuration in
   both rounds. Change the page size, the KV dtype, TP, or `PYTHONHASHSEED` and
   the keys almost certainly stop matching. Untested.
6. **Anything about capacity limits or eviction.** 573 MB in a 16 GB/rank pool.
   Nothing was ever evicted, so behaviour at the capacity limit — where a
   restart would find the store partially gone — is unknown.
7. **Anything at concurrency.** Sequential requests throughout.
8. **Single observation.** One kill, one relaunch, one replay. No repeat, no
   variance estimate.
9. **That 31/32 is a quality measurement.** It is a coherence gate on four
   factual questions at temp=0. The one miss was a token-budget truncation, and
   a substring check cannot tell that from a wrong answer without reading the
   text — which is why the text was read.

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
