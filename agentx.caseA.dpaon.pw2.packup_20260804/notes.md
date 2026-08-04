# Notes — traps, wrong turns, and what this run could not answer

Ordered by how much they would cost someone repeating this. The first four each
**produce no error at the layer you are watching** — a leg that will not start
for a reason unrelated to your config, a policy that reports success while
running the old one, a run that succeeds and is reported as FAILED, and a
routing policy that scores every candidate identically while logging normally.

---

## Trap 1 — prefill DPA cannot start on a node whose ephemeral range starts at 1024

**What.** `DPA=1` on chi2835 died in 16 seconds with:

```
File "/opt/infera/infera/common/net.py", line 107, in free_tcp_port_block
  randomised = [random.randint(1024, highest) for _ in range(_PORT_BLOCK_TRIES)]
ValueError: empty range for randrange() (1024, 1017, -7)
```

**Why.** `free_tcp_port_block` prefers a port block **below** the ephemeral
range, computing the window as `[1024, low - count]`. chi2835 had
`ip_local_port_range = 1024 65535`, so `low - count = 1016 < 1024` and the
window is empty. Only `dp_size > 1` reaches this code (`worker.py:77` routes
`count <= 1` to `free_tcp_port()`), which is why **the identical image with DPA
off had been running for days**.

**How it was caught.** Reading the traceback. The trap is that it looks like a
config error in *your* launch and is not.

**Context.** `p8_prefill.log` from 2026-08-02 contains the same traceback — an
earlier DPA-on attempt failed here and was never diagnosed. Fix and its
regression tests: `patches/`.

## Trap 2 — the port-block fix is not enough; the block must be *outside* the ephemeral range

**What.** With the fallback active (scanning inside the ephemeral range) the leg
got further and then died at rank 7:

```
base 37059 -> DP0..DP6 bound 37059..37065
DP7: zmq.error.ZMQError: Address already in use (addr='tcp://*:37066')
```

**Why.** The probe **must** release the block before returning — a retained
`127.0.0.1` reservation would shut the engine's own child out. That leaves a
window between probe and bind, and inside the ephemeral range the kernel can
hand the port to any `bind(("",0))` caller in that window. Port pressure was
trivial at the time (47 distinct local ports), so this is not congestion; it is
sglang's own rank-init grabbing ports in a burst.

**Resolution.** chi2835 was the **only** node of four reading `1024 65535`;
chi2879/chi2867/chi2872 all read the kernel default `32768 60999`. Resetting it
restores `free_tcp_port_block`'s original path — scanning `[1024, 32760]`, which
the kernel never allocates — so the race is **eliminated by construction**, not
narrowed. The next leg took base 11271 and all 8 ranks bound cleanly.

**The lesson.** The in-range fallback keeps a DPA leg *startable* on such a node,
but it is strictly weaker. Prefer fixing the node.

## Trap 3 — the router reports the policy you asked for while running the old one

**What.** `start_router_pol.sh` printed

```
router healthy (backend=rust policy=round-robin pw=20.0 dw=2.0)
```

while `ps` showed a process from **two days earlier** still serving with
`--router-policy kv-aware`.

**Why.** The script's `pkill -9 -f infera-router` did not match inside the
`docker exec` session. The new router started, `exec`'d correctly, hit
`Error: Address already in use (os error 98)` on 8100 and exited — but the
health check that follows passes anyway, because the *old* router is still
answering `/health`.

**How it was caught.** Only because the script also reads the policy back out of
`ps`. That line was added while writing this kit; without it the whole
round-robin leg would have silently measured kv-aware.

**How to avoid.** Never trust the requested value. `start_router_pol.sh` now
prints `resolved --router-policy:` from `ps`; if it disagrees with what you
asked, find the surviving PID and `kill -9` it **by PID** (not by pattern), then
confirm the engine count is unchanged before and after.

## Trap 4 — restarting the prefill leg silently breaks the decode leg

**What.** After relaunching prefill, `/health` reported `active_workers: 2`,
`/v1/workers` showed both legs `active`, the router logged normal picks — and
every chat request timed out with `HTTP=000`.

**Why.** The decode leg still held the bootstrap connection to the *previous*
prefill instance:

```
prefill: KVTransferError(bootstrap_room=...): Aborted by AbortReq
decode : Lost connection with prefill instance (bootstrap_addr: 10.2.122.78:8998)
```

Its reconnect attempts all cluster at the moment of the prefill restart and then
stop. etcd registration and `/health` know nothing about this.

**Resolution.** Relaunch the decode leg too. **Any prefill restart requires a
decode restart** in this PD setup — REPRODUCE step 4.

**Cost.** ~10 minutes and one confusing round of "the router is fine, the legs
are fine, nothing works".

## Trap 5 — 6,396 stack traces from one harmless source (inherited)

`grep -i error` on `aiperf.log` returns thousands of
`ValueError: Buckets are required for histogram time series` from
`aiperf/server_metrics/storage.py:525`. All are the optional server-metrics
scraper failing to parse a Prometheus histogram. They do not touch dispatch or
any reported metric.

Also inherited: the sweep prints **`FAILED`** for a run that succeeded, because
`OUT` sits outside `$HERE` and the container mounts only `$HERE`. Both legs here
completed with 0 errors and both were reported `FAILED`.

**Always exclude `server_metrics|Buckets are required` before counting faults,
and read the rescued artifacts rather than the sweep's verdict.**

---

## The mid-run crash: prefill DP6, HSA out-of-resources

**What.** At 05:04 UTC, during the round-robin c16 leg:

```
:0:rocdevice.cpp:3582 Aborting with error : HSA_STATUS_ERROR_OUT_OF_RESOURCES
Fatal Python error: Aborted
```

in `deepseek_v2.py forward` → `eager_runner._execute_extend` →
`prefill.py:525 event_loop_overlap_disagg_prefill`. Rank 6's scheduler process
disappeared; the other seven kept serving.

**Why it is not KV exhaustion.** The batch line at the time reads
`token usage: 0.07` — the KV pool was 93 % empty. `#new-token: 16384`,
`#pending-token: 105419`. This is DP-attention **activation** memory: each rank
holds its own chunk's activations, and the transient peak exceeded what
`1 - mem_fraction_static` left behind.

**Same family as par8** (`start_leg.sh:54-70`), where the cure was *lowering*
gmu 0.88 → 0.80. **Here gmu was already 0.70 and it still crashed.**

**Mitigation used.** Halve the per-forward prefill work: CHUNK 131072 → 65536,
i.e. 16384 → 8192 after sglang's `//dp_size`. The kv-aware leg then ran 915 s
with all 8 ranks alive.

**What this costs the comparison.** The 20260803 run did 16384 tokens per
forward. This kit does 8192. The DPA change is therefore **not** a single
variable — see the four-way delta in `README.md`.

**Detection matters.** A dead rank does not stop the run. The router keeps
routing to 8 targets (`expand_targets` reads `dp_size` from etcd, not liveness),
requests to the dead rank fail or hang, and the aggregate numbers quietly
degrade. Check the scheduler count **after** a run, not only before:

```bash
docker exec bench_run ps -eo args | grep -oE 'scheduler_DP[0-9]+' | sort -u | wc -l
```

---

## What this run could not answer

### 1. Why the router's cache view is empty — NOT DETERMINED

**This is the headline gap.** On 37 prefill picks carrying real prompts
(mean 1,436 blocks) the router logged `cache_hits=0` and `active_blocks=0`
every single time, so the kv-aware cost function was identically zero and the
policy degenerated to "first target".

Eliminated, each with evidence (`analysis/policy_ab.md` has the table): stale
subscription port, unbound per-rank publishers, dead ZMQ link, engine with no
cache to report (60/1129 prefill batches had `#cached-token > 0`), dead kvd
(gets 576 → 1,864, hits == gets, misses 0).

The two probes that would settle it, in order: (a) confirm which socket carries
`BlockStored` for a **prefill-role** leg — the leg logs
`KvEventPublisher started: endpoint=tcp://0.0.0.0:5557` (infera's own publisher)
while the router subscribes to the sglang-side block at 26803+rank; (b) if
events do flow, compare the router's `BlockHasher` output against the ids the
engine publishes — a tokenizer mismatch produces exactly this signature.

**Until then `pw=2.0` is a derivation with no measurement behind it.**

### 2. Whether DPA-on prefill helps or hurts this workload — NOT SEPARABLE

Four things changed at once versus 20260803: DPA on, gmu 0.80 → 0.70,
per-forward prefill 16384 → 8192, and the prefill delayer returning (it is
scoped to the DPA branch in `glm52_leg.sh:151-156`). TTFT p50 went 5,146 →
25,188 ms and throughput 0.256 → 0.148 req/s. **No single-knob attribution is
possible** and none is offered.

### 3. Whether 16384/forward is usable at all under DPA at gmu 0.70

One crash, one data point, no repeat attempt. It may be load-dependent — DP6
died during the c16 leg, not c8.

### 4. Why ~60 requests per leg are turn ≥ 1 yet report no cache field

In the 20260803 run every record missing `usage_prompt_cache_read_tokens` was a
first turn (54/54 `turn_index == 0`). Here only ~35 of ~70–98 missing records
are first turns. A different condition, unexplained. It makes the per-request
p50 cache figures **not** strictly comparable across kits; the token-weighted
figure is unaffected.

### 5. Whether a warm decode leg confounds the comparison

The decode leg had been up 18 hours before this session and was restarted
mid-session (05:42). Its radix/kvd state was never cold. `statctl` snapshots
are in `env/kvd_{before,after}_*.json`; no attempt was made to normalise.

---

## Operational notes

- **`pkill -f rescue_artifacts.sh` kills your own ssh connection** — `-f`
  matches the remote command line containing that string. Use
  `pkill -f "rescue_[a]rtifacts"`. Same family as the standing rule against bare
  `pkill -f infera.kvd`, where `.` is a wildcard that also matches the engine's
  `--infera-kvd-socket`.
- **Stop the rescue loop only after the aiperf container is gone.** I stopped it
  while the c16 container was still draining and lost that leg's complete
  artifacts — 64 of ~68 records. That leg is archived as-is and marked
  unusable.
- **Nested `docker exec` + `ssh` quoting silently produces wrong output.** One
  verification here printed `engines after: 0` from a quoting error, not from a
  dead engine. Re-check with a simpler command before believing a scary number.
- **Server logs contain binary bytes** — `strings <log> | grep`, never bare
  `grep`. And scope by time window: these logs are appended all day and contain
  previous runs.
- **Never probe a PD leg's own port directly** — it hangs. Always via the router.
- The deployment's **slurm holds belong to `yeandy-debug`**; never `scancel`.
  Only our own engine/router processes were killed, and the engine count was
  checked before and after each kill.
