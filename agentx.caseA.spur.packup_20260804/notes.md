# Notes — traps, wrong turns, and what this run could not answer

Ordered by what they would cost someone repeating this.

---

## Trap 1 — `-G8` does not give you the GPUs (cost: two dead legs)

Both legs died at startup with
`HIP out of memory ... GPU has a total capacity of 287.98 GiB of which 0 bytes
is free`, while PyTorch itself had allocated only 60–66 GiB. The missing
~220 GB belonged to **another tenant** on the same node — `rocm-smi --showpids`
showed live PIDs holding ~270 GB/card with active SDMA.

**spur's `-G8` reserves a scheduling slot, not exclusive GPU memory.** Screening
nodes by "does it have the image" is not enough; you must also check free VRAM.
`scripts/screen.sh` does both and is the thing to run before committing a node:

```
crsuse2-m2m-276 img=1 maxUsedGB=278     <- has image, GPUs FULL
crsuse2-m2m-268 img=0 maxUsedGB=0       <- GPUs free, no image
```

## Trap 2 — the prebuilt image and free GPUs are anti-correlated

`infera/engine-sglang:merged-mtp` is **node-local** (built in place; no registry,
no exported tar). Of ~25 nodes screened, only 2 carried it — and both had their
GPUs fully occupied, repeatedly, across three allocation rounds. The nodes with
the image are the popular nodes precisely because they have been used.

The operator's instruction was "avoid rebuilding the image". It could not be
honoured: transferring 108 GB between nodes is slower than rebuilding, and
`--exclude`-driven re-screening never surfaced an image-carrying node with free
GPUs. **The rebuild took ~4 min per node** from `src.tar` + layer cache, run in
parallel on both.

The rebuild turned out to be strictly better anyway: it produces `final-pr`,
which has **GLM52_P1V3 baked in**, so no in-container patching was needed. On
`merged-mtp` the gate correctly reported `_p1v2_rows pyc_hits=0  MISSING`.

## Trap 3 — the packup's hicache verification command is stale

`agenticbench.mtp.caseA.packup_20260801/environment.md` gives:

```python
from sglang.srt.mem_cache.memory_pool_host import ALLOC_MEMORY_FUNCS, alloc_with_pin_memory
```

This now raises **ImportError**. The patch is present and live — the module was
refactored to `sglang.srt.mem_cache.pool_host.common`. Reading the ImportError
as "the fix is missing" would send you rebuilding an image that is already
correct. The behavioural check that works:

```python
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS, alloc_with_pin_memory
assert ALLOC_MEMORY_FUNCS["cuda"] is alloc_with_pin_memory
```

`start_ctr.sh`'s bytecode gate already uses the correct path.

## Trap 4 — the artifact dir must be inside the mounted path

The customer's `replay_caseA.sh` mounts only `$HERE` into the aiperf container.
Setting `OUT` outside it makes `--output-artifact-dir` resolve **inside the
container namespace**; the CSV dies with the container and the host-side
`[ -f "$csv" ]` test fails. `summary.csv` then records `FAILED` **for a run that
completed and produced full metrics**.

Fix is env-only, no edit to the customer script: `OUT=$HERE/results`.
Both points wrote clean rows this time.

## Trap 5 — `PORT_ACTIVE: 1` is correct here, `9` was the wrong signal

`start_ctr.sh` counts `PORT_ACTIVE` across all IB devices. On the first node
pair it read **9** (8 × ionic + mlx5_0); on the final pair, **1**. The 1 is the
right answer: spur's KV fabric is **mlx5_0**, and the 8 ionic devices are the
vultr fabric that must not be bound here. A leg script that auto-discovers ionic
will silently take the wrong path on spur.

Confirmed on the wire rather than assumed: `MC_FORCE_TCP` / `GID is NULL`
occurrences = **0** on both legs, i.e. no silent TCP fallback.


## Trap 6 — the Case-A kit's known-good GMU crashes under this corpus

**What.** Arm 2's first attempt used `GMU=0.80` under prefill DPA=8 — the value
`agenticbench.mtp.caseA.packup_20260801` records as the fix for exactly this
failure. It died 3 minutes in:

```
:0:rocdevice.cpp:3582 ... HSA_STATUS_ERROR_OUT_OF_RESOURCES ... Available Free mem : 362 MB
Fatal Python error: Aborted
[2026-08-04 05:17:53 DP2 TP2 EP2] Prefill batch ... token usage: 0.05
```

**Why it matters.** `token usage: 0.05` is the tell — the KV pool was 5 % full,
so this is **not** KV exhaustion. It is DP-attention activation memory: under
dp8 every rank holds its own chunk activations, and this corpus's ISL p99 of
245 K is well past what the Case-A kit exercised. A known-good GMU is known-good
*for a workload*, not for a deployment.

**How it was fixed.** **Lower** `mem-fraction-static` to 0.70 — more room left
outside the static pool. This is the counter-intuitive direction and the
opposite of the decode-side retract fix. Diagnose by phase: prefill
HSA-OOM/Aborted ⇒ lower; decode retract/NotImplemented ⇒ raise.

**Context.** Arm 1 (DPA off) already ran at 0.70 for the same reason, so both
arms end up there by different routes — which means GMU is *not* a controlled
variable between them. It happens not to differ, but that is luck, not design.

Artifacts: `logs/prefill_dpa8.gmu080_crashed.log.gz`,
`results/dpa8_c8/rank_samples.attempt1_crashed.jsonl` (35 samples).

## Trap 7 — restarting the router destroys the evidence you need

**What.** `/tmp/router.log` lives **inside the container** and is truncated by
the next `router*.sh`. It carries the per-decision lines
(`picked= cache_hits= request_blocks= active_blocks= w_overlap=`) — the only
direct record of *why* the policy chose a given rank.

**Why it matters.** Arm 2 found the prefill ranks badly imbalanced
(max/min 5.49×). The natural next question is whether the kv-aware cost function
caused it or whether it is a dispatch/tie-break artefact. **That question cannot
be answered from this run** — the log was overwritten when the router was
restarted to change the policy, before anyone thought to save it.

**How to avoid.** Capture it *before* any router restart:
`docker exec <ctr> gzip -c /tmp/router.log > router_<arm>.log.gz`.
Added as `REPRODUCE.md` step 13a.

**Context.** The vultr par8 kit records the same file as a trap for a different
reason (it is appended across leg restarts and mixes deployments, so an unscoped
parse counts stale picks). Both failure modes come from the same fact: it is
container-local mutable state that no one owns.

## Trap 8 — a cumulative counter reads "balanced" when nothing is happening

**What.** `sample_ranks.sh` records **cumulative** per-rank batch counts.
`rank_balance.py` differences consecutive ticks. During aiperf's long-output
decode phases no new prefill batches arrive, so every delta is 0 — and a naive
reading of "all ranks equal" looks like perfect balance.

**Why it matters.** In arm 2's crashed first attempt this masked a **dead leg**
for 8 minutes: the analyser printed all-zero shares and `no active ticks yet`,
which reads as "idle", not "the engine aborted". The crash was only found by
checking whether the log file was still growing.

**How to avoid.** The analyser now skips ticks with `tot_d <= 0` and reports the
active-tick count separately. When it says `(no active ticks yet)` for more than
a couple of minutes, **check the leg is alive** (`ls -la` the log twice) before
concluding anything about balance.

---

## Trap 9 — killing the wrapper leaves the engine holding all the VRAM

**What.** `pkill -f infera.engine.sglang` kills only the **wrapper** process.
The actual engine runs in a `sglang.launch_server` subprocess with its own
`sglang::scheduler_DP*` children, and those are re-parented to init, keep
running, and **keep every byte of VRAM**.

**How it manifests.** `rocm-smi` reads unchanged (measured: **273 GB/card still
used, flat for 100 s** after the "kill"). It looks like a slow teardown, so the
natural response is to wait — and waiting never helps. Relaunching in that state
gives an immediate HIP OOM that reads like a sizing problem.

**How to avoid.** Kill the whole family, and confirm by VRAM, not by process
name:

```bash
docker exec $CTR bash -c "pkill -9 -f sglang.launch_server; pkill -9 -f 'sglang::'; \
                          pkill -9 -f infera.engine.sglang; true"
# then poll until this reads ~0 on BOTH nodes:
rocm-smi --showmeminfo vram | grep -oE 'Used Memory \(B\): [0-9]+' | ...
```

Zombies (`Z`/`<defunct>`) in `ps` after this are fine — the memory is already
released; they are only waiting to be reaped.

> REPRODUCE.md step 12's teardown always had both patterns. This was hit by
> issuing the kill by hand and copying only the `infera.engine` half.

---

## Trap 10 — the bench driver must run on the HOST, not in the engine container

**What.** The customer's `replay_caseA.sh` calls `docker run` to launch aiperf.
The engine container has **no docker client** (`which docker` → nothing), so
running the driver via `docker exec $CTR ...` fails instantly.

**How it manifests.** `summary.csv` records **`FAILED`** and the per-point log
is **0 bytes** — the script's `run_aiperf ... > "$rd.log" 2>&1` captured a
command that never produced output. Total elapsed: 0 seconds. Reads exactly like
Trap 4 (artifact dir outside the mount), but the cause is the opposite end of
the pipeline.

**Tell them apart by the log size:** Trap 4 leaves a *full* log and a missing
CSV; Trap 10 leaves an *empty* log.

**How to avoid.** Run the driver in the `spur exec` namespace, which is the host
and does have a docker client:

```bash
nohup spur exec $PREFILL_JOB bash -c \
  'export DOCKER_CONFIG=/tmp/dockercfg; OUT=... CONCS=8 bash $W/run_caseA.sh' \
  > $W/logs/run_x.log 2>&1 &
```

**Second-order gotcha met here:** the nohup target must be writable *by you*. A
log path previously created by a `docker exec -d` is owned by **root**, and the
redirect dies with `permission denied` before the driver ever starts — the
directory being writable is not enough.

---

## Trap 11 — the bench is CLOSED-LOOP, so TTFT alone is unreadable

**What.** The customer's kit has a deterministic *corpus* but **not**
deterministic *send timing*. `aiperf/timing/strategies/agentic_replay.py`
dispatches turn N+1 **from turn N's return callback**:

```
handle_credit_return (:1214)
  "if not the final turn, dispatch the next turn honoring trace delay_ms.
   If the final turn just completed, recycle the trace_id..."
lane ledger (:138)
  "a lane's session slot is held until its whole TREE drains"
```

So `--concurrency 8` is **8 serial trajectory chains in parallel**, and trace
`delay_ms` is think-time layered *on top of* the response. **Arrival rate is an
output of server speed, not an input.**

**How it manifests.** A change that makes decode *slower* lowers the arrival
rate, drains the prefill queue, and **improves TTFT**. Measured in arm 3:
ITL +68 % → arrival 0.257 → 0.196 req/s (**−24 %**) → prefill `#queue-req`
2.00 → 0.50 → **TTFT p50 −37 %**, with the prefill leg byte-identically
configured. Read alone, that looks like a prefill improvement. It is not.

**How to avoid.** Always pull arrival rate from `request_start_ns` in
`profile_export.jsonl` and report it beside TTFT
(`scripts/pd_bottleneck.py --jsonl`). If arrival moved, TTFT is measuring load,
not speed. Corollary: **TTFT and throughput moving in opposite directions here
is one cause, not a trade-off.**

> This one cost four wrong explanations in a row. The deeper error: this kit was
> recorded as "open-loop" in the project's own CLAUDE.md (on the strength of the
> frozen corpus), and that **second-hand note was then used to reject a correct
> first-hand observation**. Deterministic corpus ≠ open loop. When a document
> and a measurement disagree, go read the source.

---

## Trap 12 — decode counters are PER RANK; mixing them invents a number

**What.** Decode runs dp8, so **each of the 8 schedulers prints its own**
`#running-req` / `#prealloc-req` / `#transfer-req`, tagged `[DPn TPn EPn]`.
Prefill here has DPA **off** — one global scheduler, no `DPn` tag, and its
counters are a single real queue.

**How it manifests.** Averaging decode lines across ranks yields a value that is
not any queue's depth. Per-rank `#running-req` ≈ 1.0 reads as "decode is idle",
when the cross-rank per-second sum is **p50 6, max 10** — the C=8 load is fully
present, just spread over 8 ranks. The per-rank cap is also the DP-*adjusted*
`max_running_requests` (**256**, the second value printed in `server_args`), not
the 2048 passed on the CLI.

**How to avoid.** Split on the `DPn` tag, report per rank, and use the
cross-rank per-second **sum** as the only aggregate.
`scripts/pd_bottleneck.py` does both and was validated against hand-derived
values on both arms.

**Related, and easy to misread as a fault:** prefill `#inflight-req` runs higher
than decode `#transfer-req` **by construction**. They end at the same event
(`KVPoll.Success`) but start at different ones — prefill's starts when *compute
finishes* (`prefill.py`, req enters `disagg_prefill_inflight_queue`), decode's
starts when *the KV landing address is sent* (`decode.py:1090-1111`,
`send_metadata`). Prefill's window contains decode's. State 2 > state 4 is
normal, not backpressure.

---

## What this run did NOT establish

**1. The 66 % → 88 % gap is measured, not explained.** Turn-0 cache-bust fully
accounts for the 66 % → 51 % part (turn 0 is 0.0 % across 53 and 73 requests).
The residual — the corpus's constructed ~88 % against ~66 % realized on turns
1+ — is *consistent with* block-vs-page granularity, but that is inference, not
measurement. Settling it needs per-request `cached_tokens` correlated against
the trace's own `hash_ids` overlap for the same request. Not captured here.

**2. `submission_valid=true` despite `--unsafe-override`.** The customer script
hardcodes `--unsafe-override` plus `--trajectory-start-min-ratio 0.90 /
--max-ratio 0.98`, while the scenario's defaults are 0.0/1.0. The expectation
going in was `submission_valid=false`. All three runs came back **true**. The
likely reading is that `default_trajectory_start_*_ratio` are *defaults*, not
locks, so overriding them violates nothing and `--unsafe-override` becomes a
no-op — but **this was not verified against the validator's source**, and the
distinction matters if the customer treats the flag as disqualifying.

**3. No DPA-on arm.** Prefill DPA was off by instruction. The TTFT numbers
therefore cannot be attributed to DPA-off versus the chunk/GMU pairing that
accompanies it — those moved together. A controlled comparison needs the same
YAML re-run with prefill DPA on and chunk back to a DPA-matched value.

**4. kvd `gets = 0`, again.** 168,866 sets against **zero** reads, matching what
both reference kits recorded. Whether prefill's HiCache ever consults kvd under
this workload is still unestablished; the discriminating measurement is
per-request HiCache tier accounting, which this run did not capture.

**5. MTP's 9.1 % degenerate batches.** 509 of 5,564 decode batches report
`accept len` ≥ 3.99 (the ceiling). The mean of 3.06 is healthy, but a ceiling
pile-up is the known tell for a repetition loop. Cheap check, not run: dump
output token IDs for those batches and look for repetition.

## Node ledger

Jobs `33488/33490/33491/33505` and later `35682/35683` (`par8ab`) belong to a
**different, parallel experiment**. They were never touched. All screening
allocations made here were released after use; the two nodes actually used were
`crsuse2-m2m-268` (prefill, job 35748) and `crsuse2-m2m-288` (decode, job 35749).

An earlier attempt on `crsuse2-m2m-276` + `crsuse2-m2m-253` was abandoned to
Trap 1; those jobs were cancelled, not left holding GPUs.

**6. Why the prefill ranks skew monotonically (arm 2).** DP0 took 21.1 % of
prefill batches and DP7 took 3.8 % — max/min **5.49×**, CV **0.494**, worsening
through the run (2.91× at 5 min → 5.49× at 17 min). A monotone DP0>DP1>…>DP7
gradient is not what minimising `W*(miss) + active` produces from a symmetric
workload; it points at dispatch order or a tie-break favouring the lowest rank.
**Undetermined.** Three discriminating experiments, none run: (a) re-run under
`--router-policy round-robin` — if the gradient survives, the cost function is
exonerated; (b) parse the router's per-decision log (**unavailable for this
run**, see Trap 7); (c) sweep W ∈ {1,5,20} and watch CV — flat CV means the
weight is not the lever.

**7. Whether the 5.0/1.0 weights are better than 20.0/2.0.** They were *derived*
from arm 1's measured miss distribution (see `analysis/dpa8_arm.md`), not
*validated* against it. Arm 2 changed DPA and the weights together, so its
numbers cannot attribute anything to the weights. The clean test is one
concurrency point, DPA fixed, W swept.

**8. Token-level rank skew.** The sampler counts **prefill batches** per rank,
not tokens. Ranks drawing longer prompts do more work per batch, so the
token-level imbalance may be larger or smaller than 5.49×. Not captured.

**9. What the decode radix cache contributes on its own (arm 3).** It cannot be
enabled without disabling MTP — SGLang rejects
`--disaggregation-decode-enable-radix-cache` under `--speculative-algorithm`, so
`infera/engine/sglang/args.py:261-278` declines to append it whenever EAGLE is
on. Arm 3 therefore measures **the pair**, and the server-measured cache rate
came back **flat** (50.8 % → 49.5 %, per-turn nearly identical). Isolating it
would need an upstream change or an infera flag to suppress the radix cache
independently; neither exists. See `analysis/nomtp_radix_arm.md`.

**10. ~~Why arm 3's TTFT improved 37 %.~~ RESOLVED — see Trap 11.** The driver
is closed-loop; slower decode lowered the arrival rate by 24 % (0.257 → 0.196
req/s, measured from `request_start_ns`), which drained the prefill queue
(2.00 → 0.50) and improved TTFT. Not a server-side gain. Bottleneck verdict from
the five-state analysis: **prefill-bound in both arms, transfer exonerated,
decode at 0.1–0.3 % of its running cap** (`analysis/pd_bottleneck_arm1_vs_arm3.txt`).

**11. What the decode radix cache contributes.** Unobserved, not just
unexplained. It cannot be enabled without disabling MTP (see §9), and the one
cache metric available — `usage.prompt_cache_read_tokens` — reports the
*prefill*-side prefix match, so it would read flat regardless. No decode-side
hit/miss accounting was captured.

**12. An iso-load comparison of the two arms.** Because the driver is
closed-loop, arm 3 offered 24 % fewer requests in the same 900 s. The
prefill-bound verdict holds in each arm independently, but queue *magnitudes*
across arms compare two different load levels. Fixing this needs an open-loop
driver (`--request-rate`), which would no longer be the customer's benchmark —
so it is a genuine limit of running their kit unmodified, not an oversight.
