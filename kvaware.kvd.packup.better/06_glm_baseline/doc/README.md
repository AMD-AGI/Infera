# 06 — GLM-5.2 two-node PD baseline: the CONTROL (kvaware OFF, kvd OFF)

**Ran:** 2026-07-30 · **Cost:** ~6 min cold start + ~1 min probe, 16 GPUs / 2 nodes
**Verdict:** ✅ PASS — 4/4 correct, fully coherent, on real mooncake RDMA

## What this experiment answers

**Is the two-node PD substrate clean before any KV feature is switched on?**

That is the whole job. Nothing here is about kvaware or kvd — both are off. The
run exists so that a later run *with* them on has something to be compared
against.

This matters more than it sounds. Every earlier attempt at this stack on a
single node produced garbled output, and the suspect list included the features.
Only a matched control on a known-good substrate can separate "the feature
works" from "the substrate was fine all along and the feature is inert". Without
this run, a 4/4 with the features on proves nothing: you cannot tell a working
feature from a no-op.

**Topology.** GLM-5.2-MXFP4, two nodes, real mooncake RDMA (not `MC_FORCE_TCP`):

| | prefill | decode |
|---|---|---|
| Host | chi2879 · 10.2.122.10 | chi2867 · 10.2.122.44 |
| Parallelism | TP8, DP-attention dp8/ep8 | TP8, DP-attention dp8/ep8 |
| Port | :30000 | :30000 |
| `--mem-fraction-static` | 0.88 | 0.85 |

infera router on the prefill node :8100, policy **round-robin**. etcd on the
prefill node :2379. Correctness bar: `scripts/probe.py`, 4 temp=0 factual
prompts, ≥3/4 required to pass.

## Result

| Check | Observed | Required |
|---|---|---|
| Correctness (`probe.py`) | **4/4** | ≥3/4 |
| Output coherence | full sentences, correct facts | eyeball |
| `MC_FORCE_TCP` hits in prefill log | **0** | 0 |
| `HIP dmabuf disabled` lines | **8** | 8 (one per ionic rail) |
| Both legs `ready to roll` | 1 / 1 | 1 / 1 |
| `kv_events_endpoint` at the router | **null** on both workers | null |
| `infera-kvd adapter connected` | 0 | 0 |
| `enable_hierarchical_cache` | False | False |

All four completions came back coherent — this is not a "it returned 200"
result. Verbatim, from `results/baseline_probe_4of4.txt`:

```
[OK] 'The capital of France is' -> '1.  **Identify the core question:** The user
     is asking for the capital of France.\n2.  **Retrieve knowledge:** ... France
     is a country in Europe. Its capital city is Paris. ...'
[OK] 'The capital of China is' -> '... The capital of the People's Republic of
     China is Beijing. ...'
[OK] '2+2=' -> '... 2 + 2 = 4. ...'
[OK] 'The largest planet in our solar system is' -> '... Mercury, Venus, Earth,
     Mars, Jupiter, Saturn, Uranus, Neptune ...'

4/4 correct
```

## The finding that matters

**The substrate is clean, and it was genuinely RDMA.**

Those are two separate claims and the second is the one that is easy to lose.
`MC_FORCE_TCP=1` is this repo's known correct-but-slow fallback; a run that
quietly fell back to it would still score 4/4 and would tell you nothing about
the RDMA path. Two counters rule it out:

```
grep -ac "MC_FORCE_TCP"        pd_prefill_base.log   -> 0
grep -ac "HIP dmabuf disabled" pd_prefill_base.log   -> 8
```

Zero `MC_FORCE_TCP` hits means the fallback was never taken. Eight
`HIP dmabuf disabled` lines is **one per active ionic NIC** — the mooncake RDMA
path initialising on all 8 rails. A degraded run does not produce them.

**And the switches confirmed themselves from the wire.** The router's
`/v1/workers` showed both legs registered with:

```
prefill http://10.2.122.10:30000 "kv_events_endpoint":null
decode  http://10.2.122.44:30000 "kv_events_endpoint":null
```

`null`, not absent — the field exists and is empty, which is the KV-events
switch reporting its own state rather than being taken on trust from the launch
command. Inside the engine the same story: zero `kv-events-config` occurrences,
zero kvd adapters, `enable_hierarchical_cache=False`, and the tree cache built
as plain `impl=RadixCache hierarchical=False`.

One incidental observation worth carrying forward, visible in the decode log:

```
[decode leg] disable_radix_cache=True
[prefill leg] disable_radix_cache=False
```

A PD decode leg turns its own radix cache off ("KV cache is forced as chunk
cache for decode server"). Harmless here because nothing asks for a hierarchical
cache. It becomes the central constraint the moment anything does.

## How to reproduce

```bash
bash scripts/run.sh
```

Writes `results/baseline_probe_4of4.observed.txt`; compare against the committed
`results/baseline_probe_4of4.txt`. Useful knobs:

```bash
KEEP=1  bash scripts/run.sh    # leave the deployment up (to pull logs, or to
                               # run a feature arm against the same containers)
WAIT_MIN=30 bash scripts/run.sh   # more patience on a cold NFS cache
```

The script does the preflight (image present on both nodes, model path visible,
`ip_local_port_range` sane), injects host `libionic`, applies the `net.py` port
fix, launches both legs with `KVAWARE=0 KVD=0`, starts the round-robin router,
polls for readiness while printing the growing log line-count, then collects the
evidence and prints a verdict.

It also handles the traps without being asked: it never uses the
`docker exec -d ... bash -lc '...'` form (which silently does not persist), it
stages every script as a file rather than fighting nested-ssh quoting, and its
readiness poll runs for 20 minutes by default so a 6-minute cold start is not
mistaken for a hang.

## Gotchas specific to this experiment

- **A 200 from `/health` means nothing here.** Both legs report ready and
  register with the router even when the KV transfer is completely broken; the
  failure only shows when a request actually needs KV moved from prefill to
  decode. That is why the gate is `probe.py`'s content check and not an HTTP
  status.
- **Do not `curl` a leg's own port to check on it.** A PD leg only serves
  through the pair; a direct probe just hangs and costs you a debugging round.
  Everything goes through the router at :8100.
- **`patch 0001` (the randomised `free_tcp_port_block`) is applied even though
  this run does not need it.** That function is unreachable with kvaware off. It
  is applied anyway so that this control and a feature run differ in *exactly*
  the features and nothing else — including the container's `net.py`.
- **The two legs use different `--mem-fraction-static` (0.88 / 0.85) on
  purpose.** These are the TP8 values. Do not carry them to a TP4 worker: at TP4
  the per-GPU weights double to ~102 GB and 0.85 OOMs.
- **`KIT` must be on the shared FS.** The script defaults to
  `/mnt/vast/c_huggingface/glm52_base06`. Logs written to a container-local path
  disappear with the container — which is exactly how an earlier round's logs
  were lost.

## What this does NOT prove

Stated plainly, because a clean control is easy to over-read:

1. **It says nothing about kvaware or kvd.** Both are off. This run cannot
   support any claim about either feature, positive or negative. Its only value
   is as the denominator of a later comparison.
2. **It does not prove the RDMA path is fast, only that it was taken.**
   `MC_FORCE_TCP=0` plus 8 dmabuf lines establishes *which* transport ran. No
   bandwidth or latency was measured, and the 4 probe prompts are far too small
   to stress it.
3. **4/4 on four short factual prompts is a coherence gate, not a quality
   evaluation.** It catches garbling, wrong-model, and broken-KV-transfer. It
   does not measure accuracy, long-context behaviour, or anything at
   concurrency — every request here was sequential.
4. **Nothing about a warm cache, prefix reuse, or capacity.** The four prompts
   share no prefix and are a few dozen tokens each. There is no cache behaviour
   to observe in this run at all, by design.
5. **Single observation.** One run, one moment. No repeat, no variance estimate.
   A flaky substrate that happened to be healthy for six minutes would look
   identical.
6. **The router log is gone** (container-local, removed at teardown). The
   `kv_events_endpoint=null` lines are an in-session capture. The leg logs
   independently show kvaware was never wired, so the conclusion stands, but the
   router's own view is not re-derivable from the files in `logs/`.

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
