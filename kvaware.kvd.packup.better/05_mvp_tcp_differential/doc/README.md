# 05 — the differential: is the garbled output kvaware/kvd's fault?

**Ran:** 2026-07-30 · **Cost:** ~18 min, one node (chi2879), two cold starts
**Verdict:** ✅ the differential is valid and answers its question —
**no-regression. NOT a correctness pass.** Both arms scored 0/4.

## What this experiment answers

The transport is working now (`MC_FORCE_TCP=1` sidesteps the same-host RDMA
loopback that returned HTTP 500 on every completion). Requests complete. And
the answers are nonsense.

So: **is the garbling caused by kvaware/kvd, or by the substrate they are
running on?**

The natural move — probe a PD leg directly to see which half emits nonsense —
**does not work**: `curl` to a PD leg's own port returns nothing, because a PD
leg only serves through the pair. A round was burned on that before switching
to the discriminating experiment, which is a **differential**: flip one thing,
hold everything else, compare.

Two arms, on one node, `MC_FORCE_TCP` both times:

| | arm A (round r4) | arm B (round r5) |
|---|---|---|
| kvaware | **ON** | **OFF** |
| kvd | **ON** | **OFF** |
| router policy | kv-aware | round-robin |
| everything else | identical | identical |

## Result

Both arms: both legs `ready to roll`, both workers registered, KV handoff
completes, **no HTTP 500s**. So both arms are genuinely answering — the question
really is about content, not transport.

**Arm A — kvaware ON, kvd ON:**

```
Q "capital of France" -> 'v4 freddy\n\nAists. Log In andapace.a\n\nWho wouldin %%%%3...'
Q "17*23"             -> 'S情辣梯neig治\n\n杖\n\n及格...'
```

**Arm B — both OFF:**

```
Q "capital of France" -> 'v4ই\n\n脐猫\n\nument=""&gt;&lt; t-tcan you-...'
```

**Equally garbled — and both arms open with the same `v4` token.**

## The finding that matters

Two runs with different features, a different router policy and different code
paths through the KV plane produce the same *kind* of wrongness and the same
opening token. Whatever generates this sits upstream of anything the switches
control.

> **The garbling belongs to the same-host PD + `MC_FORCE_TCP` substrate, not to
> kvaware and not to kvd.**

That closed the question for the cost of one extra ~2-minute round, instead of a
source-level hunt through the KV plane for a corruption bug that was never
there.

### The distinction this experiment exists to protect

> ### This is a NO-REGRESSION observation. It is **not** a correctness pass.

Spelled out, because this is the sentence that gets over-quoted:

- **Nothing here was correct.** Arm A: 0/4. Arm B: 0/4.
- Supported: *"the features did not cause the garbling."*
- **Not** supported, and not even addressed: *"kvaware+kvd produce correct
  output."* A comparison in which both arms fail can show the arms are alike;
  it cannot certify either one.
- Also not supported: *"the features were exercised."* Four short,
  prefix-disjoint prompts give a KV-offload path nothing to do. kvd's counters
  were never read in this round, and "connected" is not "serving".

The correct one-line citation:

```
On same-host PD over MC_FORCE_TCP, output was garbled with kvaware+kvd ON and
equally garbled with them OFF, so the garbling is not attributable to the
features. Neither arm was correct; this is not a correctness result.
```

`results/what_this_does_not_prove.md` sets out the boundary in full, with a
table of what each open question would actually take to close.
`scripts/compare_arms.py` encodes the same discipline — it **refuses to print
"PASS"** for the both-arms-fail case and prints the no-regression wording
instead.

### Why the investigation moved to two nodes

Two suspects remained, and they are **entangled**: `MC_FORCE_TCP` is in use
*only because* same-host mooncake RDMA fails on this fabric. No single-node
experiment can separate them.

Moving to two nodes removes both at once — the pair is no longer co-located,
real RDMA works, and `MC_FORCE_TCP` is no longer needed. It is also the
deployment shape anyone actually cares about. The single-node MVP existed to
shake out wiring cheaply, and it did: three bugs and a fabric limitation in five
rounds of ~2 minutes each. Once the wiring was clean, it had done its job.

**Which of the two suspects actually causes the garbling remains unknown.** They
were eliminated together, so neither was ever convicted individually.

## How to reproduce

```bash
bash scripts/run.sh              # ARMS=both (default) — runs A, then B, then compares
ARMS=A bash scripts/run.sh       # one arm only, for re-running a half that failed to launch
```

**Both arms must run in the same invocation.** An arm measured on a different
day, after a reboot, or beside a different neighbouring job is not a control.
A single arm proves nothing.

`scripts/up.sh` is one launcher for both arms, taking exactly three variables
(`KVD`, `KVAWARE`, `POLICY`). That is deliberate: the cheapest way to guarantee
two arms differ in one dimension is to make it structurally impossible for them
to differ in any other. The kvd daemon starts in *both* arms (unused in B) so
the process table matches too.

`scripts/probe.py` dumps each completion with `repr()` — control characters and
CJK must survive being diffed — and writes JSON for
`scripts/compare_arms.py`, which prints the per-prompt comparison, flags any
shared opening prefix automatically, and applies the verdict table.

**The pass condition is a valid comparison, not correct output.** Check, in
order: both arms reached `p=1 d=1`; both show `MC_FORCE_TCP > 0`; arm A's
workers show a non-null `kv_events_endpoint` and >0 kvd adapter connections
while arm B's show null and 0; and neither arm errored. Only then does the
comparison mean anything.

## Gotchas specific to this experiment

- **Never vary `MC_FORCE_TCP` between arms.** It is a prime suspect for the
  garbling; varying it would confound the exact attribution the differential
  exists to make. `scripts/leg.sh` says so at the knob.
- **Verify the switches from the wire, not the launch command.** A worker's
  `kv_events_endpoint` is `null` when kvaware is off — that is the switch
  confirming itself. `run.sh` dumps `/v1/workers` for both arms.
- **`KVAWARE=0` alone would have disabled kvd on the decode leg anyway.** A PD
  decode leg sets `disable_radix_cache=True` by itself and sglang forbids
  `--enable-hierarchical-cache` alongside it; what legalises kvd there is
  `--disaggregation-decode-enable-radix-cache`, which infera auto-appends *only
  when kv-events are enabled*. So turning both off together is the honest OFF
  state, not an over-reach.
- **Do not read "no HTTP 500" as "correct".** They are different claims, and
  this round is precisely the case where the first holds and the second does
  not. `probe.py` reports **OK / WRONG / ERROR** as three distinct outcomes for
  that reason.
- **Never probe a PD leg directly to isolate content problems.** It hangs. A
  round was lost to this.
- **The surviving evidence is three output fragments**, and only **one** matched
  arm-A/arm-B pair (the France prompt). The original arms may not have used
  byte-identical prompt sets — arm A's record shows `17*23` where the standard
  probe uses `2+2=`. That is a real weakness in the original comparison, which
  is why `probe.py` now pins the case list in code. The negative finding
  survives it; a positive one would not have. See `logs/README.md`.
- **Raw logs no longer exist** — the container was removed before the GLM-5.2
  runs. Everything in `results/` was captured in-session.

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
