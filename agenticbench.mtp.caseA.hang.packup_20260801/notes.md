# Notes — the wrong turns, and what is still open

Three of the wrong turns below are mine, and each cost a round. They are recorded
in the order they happened because the *sequence* is instructive: each one made
the next diagnosis harder.

---

## 1. `pkill -f infera.engine.sglang` kills its own shell

**What.** The kvd restart-replay step hung for ~12 minutes with no output, engine
already dead.

**Why.** The pattern matches the `bash -c '...'` command string that *contains*
that text — the very shell running `pkill`.

**Fix.** Bracket it: `pkill -9 -f "[i]nfera.engine.sglang"`.

**Context.** `router.sh` already documented this trap for `infera.server`; I
reintroduced it elsewhere. The same latent bug sat in `boot.sh` masked by
`|| true`, which meant its wait-for-teardown loop — the entire point of the step
— never ran.

## 2. I called a hang "saturation" and rewrote the config around it

**What.** When the rate-0.15 run pinned at the in-flight cap, I concluded the
offered load was too high, and rewrote `caseA_full.yaml` with a long note
blaming my own reasoning about MTP and prefill-bound workloads. Then I re-ran at
a lower rate.

**Why it was wrong.** The metrics say the opposite. For the first 556 s the run
was healthy — 401 requests completed, in-flight oscillating **8–29 against a cap
of 48**, **zero** ticks at the cap, cache hit 0.8897. Then completions froze at
exactly 401 while in-flight ratcheted to the cap.

**The distinguishing rule, which I should have applied first:** a saturated
server keeps completing requests, just slowly. **Completions going to exactly
zero while in-flight climbs is a hang.** And the server-side numbers were
decisive and available the whole time: prefill at **3 % KV usage** with a queue
depth of 0.7.

**Context.** The operator's question is what forced the re-read: *"the previous
MTP-off experiment passed with the same config, and in-flight never hit max
either — so is the data before the hang correct?"* It is, and checking it
properly showed the run had a **2× margin to the cap** when it died. The lower-rate
rerun that followed was therefore diagnosing a phenomenon that did not exist —
and it ran against an already-dead decode leg, so it was void twice over.

## 3. Restarting one leg killed the other

**What.** After rebooting only the decode leg, attempt 3 produced **zero**
completions from t=0. Decode was healthy; **prefill** was now `DEAD`.

**Why.** The two legs share torch-distributed / TCPStore state. Restarting one
orphans the survivor, which then dies in
`ProcessGroupNCCL::HeartbeatMonitor::runLoop` with `TCPStore recvValue failed`.

**Fix.** Always restart **both** legs together.

## 4. The causal order was the reverse of my first reading

**What.** I initially reported the decode leg as the failure and prefill as
collateral.

**Why it was wrong.** Timestamps, once actually compared:

    12:22:03-05   prefill still emitting normal Prefill batch lines
    12:21:57      decode DP5: "#token: 0, #running-req: 1"  (drained, still holding)
    12:22:06      prefill rank5+6: TCPStore recvValue FAILED   <-- FIRST failure
    12:22:08      decode's 8 ranks stop logging
    ~12:22        driver completions freeze at 401

And the TCPStore peers are `crsuse2-m2m-253:43517` plus its slurm alias — the
**prefill leg's own ranks**, an intra-leg store, not the cross-node one. So the
first failure is inside prefill and decode's `all_gather` stall is downstream.

**Context.** I had the decode py-spy dumps first and reasoned from the richest
evidence I happened to hold, rather than from the earliest timestamp. Getting
this backwards would have pointed any fix at the wrong leg.

## 5. A latent two-variable trap in my own A/B, found by a question

`CUSTOM_AR` defaulted to *follow* MTP:

```bash
CUSTOM_AR="${CUSTOM_AR:-$([ "$MTP" = "1" ] && echo 0 || echo 1)}"    # WRONG
```

so the MTP-**off** control arm would have silently re-enabled the aiter custom
all-reduce kernel that is known to deadlock on gfx950 — making "MTP on vs off" a
two-variable comparison and rendering the control worthless.

The converged kit's leg script carries a comment saying it had removed this exact
trap for this exact reason. I reintroduced it. Now:

```bash
CUSTOM_AR="${CUSTOM_AR:-0}"        # independent of MTP, by design
```

**Context.** Surfaced only because the operator asked whether
`--disable-custom-all-reduce` is still needed post-patch. The answer to *that*
question is "yes, keep it" (8/8 legs across four kits ran it on; the path
`remains unexercised` without it) — but checking it exposed a real bug in this
kit's scripts.

---

## Traps carried over that still bite

* **Grep engine logs through `strings`** — they contain binary bytes and a plain
  `grep -c` returns 0, which reads exactly like "the bad thing never happened".
* **Never probe a PD leg's own port from outside** — it hangs. Go via the router.
  (During the hang I probed the router with a 300 s timeout and it timed out too;
  that was the router's event loop being busy with 48 long requests, **not** a
  router fault — its CPU was 7.9 % and it was still returning `200 OK`.)
* **VRAM release is asynchronous** — poll to 0 % before rebooting or the next boot
  OOMs.
* **`--dashboard-mode` or nothing structured is persisted** — without it this
  freeze would have left no time series and no evidence.

---

## What is established, and what is not

**Established (first-hand):**

* The freeze is real, reproducible across three attempts, and leaves **no**
  traceback, GPU fault, or scheduler exception anywhere.
* At the moment of freezing the deployment was **not** saturated: in-flight 8–29
  of 48, prefill at 3 % KV usage, queue 0.7.
* The stalled shape: 7 decode ranks in `prepare_mlp_sync_batch → all_gather`,
  DP1 in `mooncake/conn.py:1923 send_metadata → zmq send`.
* The first observable failure is prefill's intra-leg `TCPStore recvValue failed`
  at 12:22:06, two seconds before decode goes quiet.
* Both an AgenticBench client and an independent `bench_serving` client stall
  identically against the same router, so the client is not the cause.

**NOT established — deliberately left open:**

* **Whether MTP causes it.** The MTP-off Case A on this cluster completed 67
  minutes / 2,919 requests, and MTP is the salient deployment difference — but
  that run also used a different image. **The single-variable control arm
  (`REPRODUCE.md` §6) has not been run.** Until it is, MTP is a leading suspect,
  not a cause.
* **Why DP1 blocks in `send_metadata`.** Whether the ZMQ peer stopped receiving,
  a high-water mark was hit, or the prefill-side bootstrap went away first — not
  investigated. The live hang was released before this was probed, on the
  operator's instruction to prioritise getting data.
* **Whether the prefill TCPStore failure is cause or symptom.** It is the
  earliest *logged* event, which is not the same as being the root cause; a
  silent stall could precede it.
* **The trigger conditions.** The 70-minute `bench_serving` sweep on the same
  deployment — including conc=128 × 155K-token prompts — did **not** trigger it,
  and neither did a 900 s Case A probe. Something about sustained multi-turn
  Case A traffic specifically does. Duration, session churn, and mixed request
  shapes are all confounded.
* **Whether IndexShare-off avoids it.** `exp2_indexshare_off` shows that route
  working for this bug class at small scale; untested here.

## Why the existing "MTP is fixed" kits do not settle it either way

Four kits validate the patch set and all PASS — at **~96 requests, 512-token
outputs, conc ≤ 64, ctx 32768**, with their own notes recording
`conc=128 was not run`. This workload is 401+ requests, 74K–235K-token inputs,
multi-turn, and survived 556 s. Their green results do not cover this regime, so
they neither contradict nor explain the hang.
