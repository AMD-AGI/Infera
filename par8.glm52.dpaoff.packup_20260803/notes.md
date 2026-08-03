# Notes — traps, wrong turns, and what this run could not answer

Ordered by how much they would cost someone repeating this. The first three
each **produce no error** — a leg that boots, serves, and returns a clean run of
the wrong deployment.

---

## Trap 1 — `CHUNK=` is silently dropped (NEW, caught before it cost a window)

**What.** `start_leg.sh` hardcodes `ISL=8192 TP=8` in its `docker exec ... env`
block and never forwards `CHUNK`. `glm52_leg.sh:73` derives it instead.

**Why it matters.** sglang divides `chunked_prefill_size` by `dp_size` **only**
under DP-attention (`server_args.py:4902`):

| | passed | engine divides? | per-forward |
|---|---|---|---|
| DPA=1 | 65536 | yes, ÷8 | 8192 |
| DPA=0 | 8192 | **no** | 8192 |

Both land on 8192 — which is *why the earlier DPA-off solo run was a fair
comparison*, and also why the problem is invisible. A DPA-off leg **cannot be
given a larger chunk from outside**: the outer `CHUNK=` vanishes and the leg
boots at 8192 while the operator believes 16384.

**How it was caught.** By reading `glm52_leg.sh:73` *before* launching. This is
the one trap in this kit that cost nothing.

**Context.** Naively reusing Case A's `65536` under DPA-off would have made
per-forward prefill work **8× larger**, not equal — a silent 8× change to the
thing under test. The number must be re-derived whenever DPA changes, never
copied. Fix: `patches/README.md` §1.

## Trap 2 — the `.pyc` trap: source patched, bytecode not (hit live)

**What.** `apply_p1v3.py` printed `patched OK - GLM52_P1V3 occurrences: 3`
against the decode container. The engine was **already running**, so it had
imported the old module:

```
$ ls -la .../dsa/__pycache__/dsa_indexer.cpython-310.pyc
-rw-r--r-- 44050 Aug  1 05:59      <- image build time, NOT the patch time
```

**Why it matters.** GLM52_P1V3 is the fix without which the decode leg *crashes
mid-run* under MTP+DPA on an IDLE rank. The run would have died 2–13 minutes in,
after the bring-up cost was already paid — and the operator would have believed
the fix was applied, because the patch script said so.

**How it was caught.** By listing the `.pyc` mtime instead of trusting the
script's success line. Fix: delete the `.pyc`, relaunch the leg, then verify the
**compiled bytecode** contains `_p1v2_rows` (one-liner in `patches/README.md` §4).

**Context.** CLAUDE.md principle 5 — *verify bytecode, not source* — and this
has invalidated a full experiment in this tree before. Same failure class as
Trap 1 and as the DPA_PASSTHROUGH trap: **the thing you edited and the thing
that ran are different objects.**

## Trap 3 — router.log is appended all day and mixes deployments

**What.** `/tmp/router.log` lives **inside the container** (not on the host —
`ls /tmp/router.log` on the node finds nothing) and is appended across every leg
restart. An unscoped parse of this run's log returns picks addressed to
`10.2.122.3` — the **previous decode leg on chi2878**, a different node.

**Why it matters.** Those stale picks silently pollute a per-rank distribution,
which is precisely the analysis the log exists for.

**How it was caught.** The distinct-target count came back as **9** for an
8-rank leg. Nine ranks on an eight-rank deployment is the tell.

**Context.** Two habits, both used here: scope by timestamp
(`cv_scoped.py <log> 2026-08-03T09:28`), and **read the targets before counting
them**. Also: the log is ANSI-colourised and the escape codes split fields
across lines, so a bare `grep -c "picked="` returns **0** on a log containing
5,768 picks. Strip ANSI first — `cache_view.py` and `cv_scoped.py` both do.

Same family as the fault-grep false positives on this stack: `server_args=`
contains `abort_on_priority_when_disabled`, so `grep -ic error` over
`q4_decode.log` returns **1,492** hits, all from one argument dump. Excluding
`server_args=|aiter|fused_moe` leaves **0** real faults. Always
`strings <log> | grep` — these logs contain binary bytes.

---

## The node eviction — and why the decode leg moved

**What.** The decode leg on chi2878 died at **22:54:23** the night before:

```
[22:54:23] deregistered worker 10.2.122.3:30000 (lease revoked)
[22:54:23] SIGTERM received. Draining requests and shutting down...
[22:54:23] Subprocess scheduler_0 crashed with exit code -15
```

**`exit code -15` is SIGTERM**, preceded by a clean deregistration — killed from
outside. Not OOM, not a crash, not the DSA/MTP bug.

**Why.** `sinfo` showed chi2878 had gone to **`resv`** — reserved for another
job. The node was then at load average 70, RAM 2,495/3,023 GB, with dozens of
another user's python processes on GPU 1 being OOM-killed.

**How it was resolved.** Relocated decode to **chi2879**, which was verifiably
idle (0 VRAM used, no KFD processes, RAM 144/3,023). Prefill on chi2835 was
**not touched** — it had been up 18 h and restarting it would have been a
gratuitous variable.

**Context.** These nodes are shared and slurm holds belong to `yeandy-debug`.
The operating rule: *kill only our own processes, never `scancel`,* and check
both **slurm state** and **real GPU load** — they disagree routinely. Deriving
"is this node busy" from `rocm-smi --showmemuse` percent is unreliable (it has
read 0 % on a node at 98 % VRAM); use `--showmeminfo vram` absolute bytes.

## A documented asymmetry: chi2879 runs 7 rails, not 8

`ionic_5` on chi2879 is **PORT_DOWN**. `glm52_leg.sh:75-81` enumerates only
`PORT_ACTIVE` ionic devices, so the decode leg came up with 7 rails
automatically. Prefill (chi2835) had all 8.

**This is not controlled for, and its effect on KV transfer time is
unmeasured.** It is recorded because a reproducer on two healthy nodes will have
a slightly different transport and should not be surprised by a small delta.

---

## The two structural findings

### 1. kv-aware routing steered nothing — on either leg

Full treatment in `analysis/routing_and_kvaware.md`. In one line each:

- **Prefill**: DPA-off ⇒ `dp_size=1` ⇒ one routing target. Nothing to choose.
- **Decode**: `cache_hits` **0 on all 2,884 picks** — the leg runs `ChunkCache`,
  which never emits `BlockStored`, so the router's KV view is permanently empty
  and the cost function's overlap term cancels.

**Verified on the wire, not inferred**: subscribing to each engine's kv-event
socket during the run gave prefill **30 messages / 15 s** and decode **0**.

> **The prefix cache worked (88.9 %); kv-aware routing did not run.** These are
> different mechanisms reported through the same `cache_hits` field, which is
> how they get conflated. Prefill's hits are sglang's own radix tree *inside*
> the single engine.

### 2. MTP × decode-radix is mutually exclusive upstream — with no stated reason

`arg_groups/pd_disaggregation_hook.py:29-56` raises on
`--disaggregation-decode-enable-radix-cache` + `--speculative-algorithm`. The
mission requires MTP, so decode radix is unreachable. **Our launcher never
touches a radix flag — this is upstream's default path.**

**Upstream research found no rationale.** The guard was added silently in
[PR #19746](https://github.com/sgl-project/sglang/pull/19746) (merged
2026-05-01), whose body never mentions speculative decoding; none of its 30
review comments question it; there is no test for it; the docs assert it without
reason; [PR #28238](https://github.com/sgl-project/sglang/pull/28238) preserved
it without explaining it. The only reasoning anywhere is
[PR #32170](https://github.com/sgl-project/sglang/pull/32170) — **unmerged, CI
red, non-maintainer** — arguing the gate is *unjustified*: *"the radix tree
itself is already EAGLE-aware … a conservative gate rather than a structural
limitation."*

Reading the code, three genuine conflicts exist, any one sufficient:

| | conflict | citation |
|---|---|---|
| A | decode-radix admission/eviction has **no spec over-allocation term**; `_pre_alloc` hard-sets `committed == allocated` | `decode.py:1296-1306, 1334-1336` vs `pool_configurator.py:426-434` |
| B | spec writes real KV slots for **unverified drafts** into `req_to_token[committed:allocated)` | `eagle_utils.py:785-798, 843-849`; `common.py:659-680` |
| C | EAGLE rewrites radix keys to **bigram** (`len = raw−1`), changing slice semantics | `kv_cache_builder.py:216`; `radix_cache.py:96-99, 142-153` |

**Which one motivated the guard is NOT DETERMINED.** Removing it and running is
the only way to find out; the assertion most likely to fire first is
`decode.py:1355-1360` (`"KV cache is full! Bug in memory estimation."`).

**And one plausible-sounding explanation is refuted by the code**: rejected
draft tokens *cannot* pollute the prefix tree — every insert path slices at
`kv_committed_len` (`radix_cache.py:443-451`; `decode.py:1452-1465, 1957-1963`).

**Consequence for us**: `decode_prefix_len` ≡ 0, so every turn re-transfers the
**entire** prompt KV (`prefill.py:273-281`; default 0 at
`base/conn.py:134-135`). A prefill-side cache hit does not shrink it — 
[PR #29316](https://github.com/sgl-project/sglang/pull/29316) only sends the
cached prefix *earlier*, overlapping transfer with the suffix forward. **Whether
this costs us anything measurable is unmeasured**; the discriminating
measurement is `transfer_total_bytes` vs TTFT.

---

## The error class: 15 × context overflow, zero timeouts

All 15 errors are HTTP 502 wrapping an engine **400**:

```
"Requested token count exceeds the model's maximum context length of 262144
 tokens. You requested a total of 265545 tokens"
```

**Rejected before generation.** Zero requests hit the driver's 240 s client
timeout (E2E max 239.2 s, 6 requests over 200 s).

**Root cause is workload arithmetic, inherited from Case A.**
`max_input_tokens: 260000` clamps the *input* only; `max_tokens` is sampled
independently with an unclamped upper bound of ~451K. Nothing checks
`input + output ≤ 262144`. Needs the joint tail, hence 0.52 %.

**Left unfixed deliberately** — Case A carries the identical gap (0.60 %), and
fixing it would break comparability. A future workload should clamp jointly.

## Output-length control: exact, and the 31-token spike is the sampler

`max_tokens = generation_length`, `ignore_eos: true`, **no `stop` field**
(`agent_throughput.py:2229-2236`) — so `max_tokens` is the sole terminator.
Neither early stop nor overrun is possible.

303 of 2,768 completions (11 %) are exactly **31 tokens**. This is
`PercentileSampler`'s floor `vmin = p50²/p90 = 320²/3300 = 31.03`
(`sampling.py:106`), which offline catches **9.9 %** of draws. **Workload
property, not engine behaviour.**

*(Measured min is 30, one below the floor. The driver re-encodes accumulated
stream text rather than reading `usage.completion_tokens`
(`agent_throughput.py:2305`), so token-boundary drift is the likely cause —
**not verified**. Settling it needs both numbers logged side by side.)*

## Open: the driver hardcodes `temperature: 0.0`

Every request payload carries `"temperature": 0.0` (`agent_throughput.py:2233`
and three sibling call sites); there is no CLI override. This conflicts with a
standing rule on this stack:

> **`temperature: 0` + MTP is indistinguishable from KV corruption.** Use
> GLM-5.2's own `generation_config.json` (temp 1.0 / top_p 0.95).

The leg runs with `sampling_defaults='model'`, which *may* mean the model's own
generation config overrides the request — **this was not verified.** It is
relevant to the 61 `accept len: 4.00` batches (3.8 % of 1,594) noted in
`analysis/sli_percentiles.md`.

**Neither question is answered by this run.** Two cheap measurements would
settle them, and neither needs a measured window:

1. Send the same prompt twice through the router with `temperature` 0.0 and 1.0;
   identical output ⇒ the request value is being ignored.
2. Dump output token IDs for the requests in the `4.00` batches and check for
   repetition.

The smoke test (`17×23 → 391`, coherent, `finish_reason: stop`) was run with an
explicit `temperature: 1.0`, so it does **not** exercise the driver's path.
