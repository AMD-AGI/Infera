# Notes — corrections, dead ends, unknowns, operational findings

Written in the order a reader benefits from, not the order things happened. §1
and §5 are the two entries most likely to save someone else a day.

---

## 1. CORRECTION: `server_args` records what you ASKED for, not what the scheduler resolved

**What was believed.** During round 0 I reported that the KDA-pool clamp — the
documented trap where the mamba state cache silently caps concurrency — had
**not** fired. I cited the resolved server args:

```
'max_running_requests': None
```

**Why it was wrong.** `server_args` is the *request*, echoed back. The scheduler
computes its own effective value afterwards and only says so in the log. In
round 1, where I grepped the log instead, every rank printed:

```
[TP0..TP3] max_running_requests is capped to 200 by the mamba state cache
  (max_mamba_cache_size=1000, 5 state slots per request). To raise it: increase
  --mamba-full-memory-ratio or --max-mamba-cache-size, or halve the state size
  with --mamba-ssm-dtype bfloat16.
```

with `Mamba Cache is allocated. max_mamba_cache_size: 1000, conv_state size:
1.17GB, ssm_state size: 33.24GB` per rank.

**What it cost.** One incorrect claim shipped in a status report and had to be
retracted. Nothing downstream depended on it, so the cost was reputational
rather than technical — but the same misreading on a run where concurrency
mattered would have produced a throughput number silently capped at 200 and an
analysis that never looked for the cap.

**How to avoid it.** For anything the scheduler can override — `max_running_requests`,
`max_total_tokens`, chunked-prefill sizing, graph bs lists — **read the resolved
value out of the worker log, never out of `server_args`.** Treat `server_args`
as an input record only.

**Unknown, and it is unrecoverable:** whether the clamp also fired in **round 0**.
That container was replaced before I ran the grep, and its log is gone. The
check that would settle it — `grep 'capped to' <round-0 worker log>` — has no
surviving file to run against. Assume it did (nothing differed in the config
that would change the arithmetic), but it is not measured.

---

## 2. Predicting the fusion outcome from the index, before launching

Not a mistake — recorded because the method transfers.

The question was whether FP8-Flash needs `--disable-shared-experts-fusion`, which
is mandatory for MXFP4-Flash. Rather than copy the flag across or try it and see,
the precondition was checked directly in the checkpoint index:

```python
import json, collections
d = json.load(open("/apps/data/models/GLM-5.3-Flash/model.safetensors.index.json"))["weight_map"]
c = collections.Counter(k.rsplit(".",1)[-1] for k in d if "shared_experts" in k)
# -> {'weight': 129, 'weight_scale_inv': 129}
```

A strict 1:1 pairing means the shared experts are **uniformly block-FP8**, the
same precision as the routed experts. The MXFP4 failure requires shared experts
at a *higher* precision than routed ones; that precondition is absent, so fusion
is legitimate. Launched without the flag, and it loaded.

**The general form: the flag is a fix for a precision mismatch, not a property of
the model family.** Check the index, not the model name.

---

## 3. Three environment traps, all first-hand, all cheap to hit again

### (a) etcd: the three peer flags move together or not at all

Overriding only `--listen-peer-urls` to a non-default port leaves
`--initial-cluster` at its default `default=http://localhost:2380` while
`--initial-advertise-peer-urls` is derived from the detected host. etcd exits 1:

```
--initial-cluster has default=http://localhost:2380 but missing from
--initial-advertise-peer-urls=http://10.235.192.137:23796
```

Fix: pass `--listen-peer-urls`, `--initial-advertise-peer-urls` **and**
`--initial-cluster` consistently (`scripts/mix_up.sh` step 4). A script that
hardcodes 2380 never sees this, which is exactly why it bites the first time you
land somewhere 2380 is taken. **This was the second independent hit on this bug
on this project** — the big-model track hit it on a different node and port — so
it is a property of etcd, not of any one script.

### (b) Never edit a script on NFS while bash is running it

The round-0 bring-up died at the router step with:

```
./mix_up.sh: error reading input file: Stale file handle
```

Cause: I edited `mix_up.sh` on NFS *while bash was still executing it*. Bash
reads a script **incrementally**, so an in-place edit corrupts everything it has
not yet read. The failure surfaced ~40 lines below the edit, which invites
debugging the wrong step. The worker was unaffected; the router was started by
hand. Since then the scripts are run from a `/tmp` copy
(`cp mix_up.sh /tmp/ && bash /tmp/mix_up.sh`), and `SELF` is pinned to the
workspace path rather than derived from `BASH_SOURCE` so the copy still finds
`mix_worker.sh`.

### (c) `docker rm -f` returns long before KFD frees the VRAM

Measured: **100.1 GiB per card still held ~10 s after teardown returned.** The
next launch measured that, believed the node was busy, and sized itself to 0.50
instead of 0.60 — i.e. it sized against **its own corpse**. `scripts/mix_up.sh`
now waits for our own usage to drain before measuring, and treats a *plateau* as
"that is somebody else, proceed" rather than waiting forever on a neighbour.

---

## 4. The throughput numbers are the LEAD's, not this operator's — and the "crash" was a wrong-port health check

`results/sweep_f8_by_lead.txt` (and `logs/f8_c*_by_lead.jsonl.gz`) contain a
two-point sweep against this deployment: conc 1 and 8, isl 7400 / osl 320.

**The team lead measured them**, on this deployment, between roughly 11:27 and
11:35 UTC — visible in the daemon event log as execs interleaved with this
operator's own 7-minute health probes. The lead also ran the HIP IPC probe
against the same container, `docker cp`'d the artifacts out, and removed both
containers at **11:35:42 UTC**.

They are reported here because they are useful and are the first direct
FP8-vs-MXFP4 comparison at fixed topology. They are attributed to the lead and
**not** claimed by this packup's operator, at the lead's instruction and for a
reason that survives knowing who took them: **contention was not sampled during
either arm.** Given §5, that is the measurement that decides whether the
absolutes mean anything. Full provenance and limits: `results/README.md`.

### The `flash_fp8_crash` directory name — RESOLVED, and the resolution is the lesson

An earlier version of this note recorded as **unknown** why the artifacts were
copied into a directory named `flash_fp8_crash`. It is now closed, by the lead:

> *"I named the directory `flash_fp8_crash` because I wrongly believed the engine
> had crashed. I had checked `/health` on `localhost:30000`; the engine binds
> `192.168.3.26:31400`. I got `000`, concluded it was dead, and created a
> directory named for a crash that never happened."*

**There was no crash.** The directory has been renamed `flash_fp8_sweep`.

Worth keeping is that **the artifacts were right and the label was wrong**, and
the artifacts said so at the time: the copied worker log stops at 11:21 and shows
no fault; both arms report full success counts (10/10, 80/80); and this
operator's own probe returned `ENGINE_OK` **8 seconds before** the kill. The
investigation's conclusion at the time — *"whoever named that directory knows
something the artifacts do not show"* — was the one wrong inference in it. They
did not. They had made an error the artifacts correctly refused to support.

**The rule: a health check proves nothing unless you verify the address you
probed.** A connection failure to the wrong port is indistinguishable from a
dead engine, and it reads as the more alarming of the two. Read the bound
address out of the log (`server_args ... host=..., port=...`) before concluding
anything from a non-response.

This is the mirror image of the failure recorded in the project's own CLAUDE.md —
*a 200 is not evidence a leg is alive* — and both happened on this project within
the same hour, in opposite directions: one operator called a live engine dead off
the wrong port, another called a dead engine healthy off a router-served 200.
**The rule that survives is the one that does not depend on who is checking:
prove liveness by making the engine emit a token, at an address you read from its
own log.**

---

## 5. OPERATIONAL FINDING: a node that oscillates on a quarter-hour cycle cannot host a quotable benchmark

This is the second entry of the same species as §1 — a reading that looked
sufficient and was not — and it is the more reusable of the two.

**Method.** After the correctness work finished, the deployment was held while
the node was polled on a fixed **7-minute cadence, 22+ cycles**, sampling on each
poll: presence of any foreign `infera-engine-*` / `infera-etcd-*` container,
per-GPU `rocm-smi --showmeminfo vram` on all 8 cards, and an engine `/health`
probe.

**What it measured.** The colleague's job cycles continuously:

- cards 0-3 move between **0 GiB and ~30 GiB** (typically 23-24, peaking 28-30);
- our cards 4-7 move correspondingly between **161 and 190 GiB** as their share
  comes and goes on top of our 161;
- the period is **7-15 minutes**;
- **three fully-idle windows** were observed in the first ~22 cycles, the
  longest **~14 minutes**.

Separately and earlier, a foreign `infera-engine-20260902-045954-*` container
(image `rocm/infera:sglang-dev`) appeared and took **161.8 GiB on card5 and
161.6 GiB on card6** for about five minutes, then vanished.

**The conclusion, and why it is a finding rather than caution.** A sweep long
enough to be worth quoting straddles at least one transition, and **the failure
is silent** — a transition does not announce itself in the numbers, it just
inflates a tail that then gets attributed to the model. So the correct response
was **to decline to benchmark, not to benchmark and caveat**. A caveat on a
contaminated number does not survive being quoted; the number does.

**The counter-observation, recorded because it broke my own ceiling.** Later the
same session an idle window ran **~28 minutes** — double the previous maximum —
and then, after the containers were gone, all 8 cards sat at 0.3 GiB for over
three hours. So the 7-15 minute characterisation describes *that afternoon's*
workload, not a law about the node. It was correct on the evidence available
when the decision was made, and it stopped being correct later. Both halves
belong in the record.

**A process lesson from the same episode, worth more than the incident:** the
lead, seeing one idle polling cycle, approved a bounded 10-minute two-point
sweep; I had six cycles of context and had declined. The approval was later
withdrawn with the reasoning *"when an operator with more samples and an operator
with more authority disagree about a node, the samples win."* Recorded here
because the general form outlives the run.

---

## 6. Numbers that are evidence, not results

- **Decode CUDA graph capture at TP4 costs 15.4-17.4 GB per rank** —
  `Capture target decode CUDA graph end. elapsed=33.47 s, mem usage=17.43 GB`.
  The TP8 packup measured **~1.4 GB**. The ~11× difference is the sharding: each
  TP4 rank owns twice the model. **Do not carry a TP8 graph-memory figure into a
  TP4 plan.** Capture *time* was ~33 s in both, so time transfers and memory does
  not.
- **The graph bs list did not truncate.** All 11 requested sizes
  `[1,2,4,8,16,24,32,48,64,96,128]` were captured, because the resolved
  `max_running_requests` was 200 and every size sits under it. Push the list past
  200 on this model and expect the truncation MXFP4 showed.
- **~12-15 tok/s graphs off → ~110 tok/s graphs on**, single stream, 64-token
  prompt. Cited **only** as evidence the graph path engaged; the 7.5× step
  matches the TP8 packup's 15.3 → 106.85. Not a performance result, and not
  comparable to anything measured at a different ISL/OSL.

---

## 7. Things deliberately not done

- **No `patches/`.** This run needed no code fix. The image is the repo's
  `Dockerfile.sglang.glm53` unmodified, and no engine source was touched.
  The directory is omitted rather than shipped empty.
- **`reset_gpus.sh` was never run on this node**, by instruction and by
  judgement: there was nothing of ours to reclaim and it must not go near a
  colleague's processes. `scripts/mix_up.sh` does not call it.
- **Nothing was killed, stopped or removed that we did not create.** Teardown in
  `scripts/mix_up.sh` hard-refuses any container name not matching `yihou_f8_*`
  and exits 1 rather than proceeding.
