# Notes — traps, wrong turns, and the one correction

The single most useful thing in this kit: **three of the four ways this
experiment could have failed produce no error at all.** They yield a leg that
boots, serves, and returns a clean 105-sample run — of the wrong deployment, or
with a wrong explanation attached. Each is recorded below as
**what / why / how it was caught / context**.

---

## Trap 1 — `--ep-size` was scoped to the DPA branch

**What.** In `glm52_leg.sh`, `--ep-size "$TP"` lived inside
`if [ "$DPA" = "1" ]`. Running with `DPA=0` dropped the flag, and sglang
resolved `ep_size` **8 → 1**.

**Why it matters.** MoE expert-parallelism would have collapsed at the same
moment as attention DP. Two variables change together, and the headline "2×
faster" could not be attributed to DP-attention at all. It would also be a
materially different deployment — `ep_size=1` runs MoE as pure TP, with
different memory and performance characteristics.

**How it was caught.** By resolving the arguments offline before launching:

```python
prepare_server_args(<base args, no --ep-size>)
#  -> ep_size = 1   dp = 1   dpa = False
```

**Context.** The fix (`patches/0006`) hoists the flag out of the branch. Because
that edits a line the DPA-**on** path also executes, the DPA-on argument list was
dry-run before and after and compared — same flags, same values, order only.
Without that check, "the fix only affects DPA=0" would have been an assumption
sitting under the whole comparison.

---

## Trap 2 — `DPA=1` was hardcoded in the launcher (the expensive one)

**What.** `start_leg.sh` wrote a literal `DPA=1` into its `docker exec ... env`
block. The outer `DPA=0` was shadowed and discarded.

**Why it matters.** The launcher still printed its success line:

```
[p7] prefill launched on chi2879 mtp=0 ctx=262144 gmu=0.80 -> .../p7_prefill.log
```

and the leg came up **with DP-attention enabled**. Had this gone unnoticed, the
run would have produced 105 clean, plausible, entirely meaningless samples — a
second copy of the baseline, reported as the DPA-off result. Nothing downstream
would have flagged it: no error, no warning, sensible-looking numbers.

**How it was caught.** By reading back the **live process command line** after
launch rather than trusting the launcher's echo:

```bash
ssh <node> 'ps -eo pid,lstart,cmd | grep "[s]glang.launch_server" | head -1'
```

It still contained `--dp-size 8 --enable-dp-attention`.

**Context — the generalisable rule.** *A launcher's success message is not
evidence that the launcher did what you asked.* Verify the running process, not
the wrapper. This is the same failure class as "verify the **loaded** module,
not the file on disk" (stale `__pycache__`), which has invalidated an experiment
in this tree before. Both are cases where the thing you edited and the thing
that ran are different objects.

Corroborating signals, once fixed — worth knowing because they are cheap to
check and hard to fake:

```
DSA with TP mode is active, dp_size=1, tp_size=8
[TP0 EP0] max_total_num_tokens=3263680 ... max_running_requests=2048
```

Rank prefixes change from `DP0 TP0 EP0` to `TP0 EP0`, and
`max_running_requests` goes from 256 (= 2048/8, per-rank schedulers) to 2048
(one scheduler). Either alone confirms the mode.

---

## Trap 3 — a fault grep that matches a boot line

**What.** Scanning the prefill leg for faults over the run window returned
**1 hit**, which reads as a real fault.

**Why it matters.** It is the disaggregation warmup line, which contains
`'num_retractions': 0` — matching a `retract` pattern. Reporting it would have
cast doubt on a clean run.

**How it was caught.** By printing the matching line instead of trusting the
count.

**Context.** This is the second false positive of its kind on this stack; the
baseline kit documents `server_args=` containing the substring
`abort_on_priority_when_disabled`, which matches `abort`. **Engine logs are
appended all day and contain giant argument dumps — a substring grep over them
will find almost any word you look for.** Two habits follow, both used here:
scope by timestamp (`^\[2026-08-02 0[56]:`), and read the hits before counting
them. Also always `strings <log> | grep` — these logs contain binary bytes and
bare `grep` goes blind.

Real fault count for this run: **0 on both legs.**

---

## The correction — why kvd started getting read

**This kit's first draft got the mechanism wrong, and the wrong version is more
plausible than the right one.**

**The wrong explanation.** The DPA-off leg boots with

```
HiCache host KV pool (356,160 tokens) is smaller than the device pool
(3,263,680 tokens); L2 cache effectiveness is reduced.
```

Since DPA-off frees attention-weight memory and grew the per-rank KV pool
2,829,952 → 3,263,680 (+15.3 %), the natural story is: *the device pool grew,
`--hicache-size 16` was not re-tuned, so the host tier is now relatively too
small — a fixable configuration gap.*

**Why it is wrong.** The host pool is **356,160 tokens in both runs**, and the
**same warning appears in the baseline's log** against a 2,829,952-token device
pool. Nothing about the host tier changed. A +15 % device pool cannot explain an
**8.5× rise in evictions**.

**The right explanation.** Look at aggregate, not per-rank, capacity:

```
DPA-on   [DP0 TP0 EP0] max_total_num_tokens=2829952  max_running_requests=256
DPA-off  [TP0 EP0]     max_total_num_tokens=3263680  max_running_requests=2048
```

Under DPA there are **8 schedulers, each owning a distinct KV shard**
(the request budget 2048 is split 8 ways → 256). Under pure TP there is **one
scheduler and one KV pool, replicated across the TP ranks.**

| | per rank | distinct shards | aggregate |
|---|---|---|---|
| DPA-on | 2,829,952 | 8 | **22,639,616 tok** |
| DPA-off | 3,263,680 | 1 | **3,263,680 tok** |

**−85.6 % aggregate capacity.** Evictions +8.5×, sets +9.9× — both consistent
with that one number, neither consistent with a +15 % change.

**Context — why this correction is the point of the experiment.** DP-attention
is a **latency-for-capacity trade**: it spends per-request latency to buy
aggregate KV, and therefore throughput headroom. This run happens to measure
both currencies simultaneously — TTFT halves, aggregate KV falls ~86 %, and the
kvd spill tier goes from *never read* (+0 gets) to *doing real work* (+432).
The kvd counter is not a curiosity; it is the capacity side of the trade
becoming visible.

**Lesson.** A metric that is *per-rank* and a metric that is *aggregate* can move
in opposite directions. Reading the boot log's `max_total_num_tokens` as "the KV
pool" without asking *"how many ranks hold a distinct copy?"* produces a
confident, coherent, wrong story — and the wrong story even comes with a
supporting warning message printed by the engine itself.

---

## Secondary observations

**The two TTFT outliers are cache stalls, not compute.** 11,397 ms at 77K input
in a run that served 215K in 6,098 ms is not size-scaling. With the working set
no longer resident, a request that must re-fetch through the host tier pays
seconds. Consistent with 77 requests under 1 s and two that stall.

**The prefill delayer disappears with DPA, and it does not matter.**
`--enable-prefill-delayer` is scoped to the DPA branch. It is a real config
difference, but the baseline's log shows it triggering **0 times** in its window
— at concurrency 1 no queue ever forms. Checked rather than argued.

**kv-aware routing on prefill becomes a no-op.** With `dp_size=1`,
`is_rank_multiplexed()` is false, so `expand_targets()` yields **1** prefill
target instead of 8 (`/v1/workers` shows `"dp_size": null`). This is inherent to
DPA-off, not an independent variable, and near-harmless at concurrency 1 — but
this configuration must not be used for a kv-aware routing study.

**The +7 % TPOT regression is unexplained and left open.** Decode was never
restarted: same PID 2420132, same bytecode, engine-side `accept len` mean 2.834
over 1,768 batches (healthy — `4.00` would signal a repetition loop). The
per-token floor barely moved (7.78 → 7.43 ms), which argues the compute path is
unchanged. Plausible mechanism: a 2× faster prefill hands KV to decode sooner,
so decode batches pack more densely. **One run against one run — recorded as an
observation, not a finding.**
