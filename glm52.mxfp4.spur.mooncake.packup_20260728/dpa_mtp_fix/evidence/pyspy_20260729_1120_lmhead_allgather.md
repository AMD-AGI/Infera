# py-spy evidence, 2026-07-29 ~11:20 — the hang is at the LM-head all-gather

Captured from the live wedged decode leg (job 9006, container `dbg2`, server
started 10:59:06, hung inside PD-disaggregation warmup). All 8 DP ranks dumped.

**This supersedes the hang location recorded in earlier notes.** Prior documents
said the ranks were stuck in `dp_gather_replicate` ← `prepare_mlp` ←
`deepseek_v2.py:2242`. That was true *before* Fix A/A2. With Fix A/A2 applied,
`dsa_backend` no longer appears in any stack, and the deadlock has **moved** to a
different collective.

## What all 8 ranks have in common

Every rank is blocked in the *same* collective — the LM-head vocabulary
all-gather over the **full 8-way TP group**:

```
all_gather_into_tensor (torch/distributed/distributed_c10d.py:4049)
...
tensor_model_parallel_all_gather (distributed/communication_op.py:47)
__call__ (triton_symm_mem_ag.py:515)
_get_logits (logits_processor.py:930)
forward (logits_processor.py:442)
forward (deepseek_nextn.py:409)      <-- the MTP/nextn draft model
```

## What differs — the ranks split 3 / 5 across two MTP pipeline stages

| rank | runner entry | MTP stage | call site |
|------|--------------|-----------|-----------|
| DP0 | `_execute_decode` (eager_runner.py:251) | `draft()` → `draft_forward` | eagle_worker_v2.py:697 ← 550 ← **1246** |
| DP1 | `_execute_decode` (eager_runner.py:251) | `draft()` → `draft_forward` | eagle_worker_v2.py:697 ← 550 ← **1246** |
| DP2 | `_execute_decode` (eager_runner.py:251) | `draft()` → `draft_forward` | eagle_worker_v2.py:697 ← 550 ← **1246** |
| DP3 | `_execute_idle` (eager_runner.py:409) | `_draft_extend_for_decode` | eagle_worker_v2.py:965 ← **1259** |
| DP4 | `_execute_idle` (eager_runner.py:409) | `_draft_extend_for_decode` | eagle_worker_v2.py:965 ← **1259** |
| DP5 | `_execute_idle` (eager_runner.py:409) | `_draft_extend_for_decode` | eagle_worker_v2.py:965 ← **1259** |
| DP6 | `_execute_idle` (eager_runner.py:409) | `_draft_extend_for_decode` | eagle_worker_v2.py:965 ← **1259** |
| DP7 | `_execute_idle` (eager_runner.py:409) | `_draft_extend_for_decode` | eagle_worker_v2.py:965 ← **1259** |

Common frames below the split, identical on all 8:

```
run_batch (scheduler.py:3259)
event_loop_overlap_disagg_decode (decode.py:1848)
dispatch_event_loop (scheduler.py:4220)
```

## Why this is a collective-*count* divergence, not a speed difference

In `forward_batch_generation` the decode branch runs three stages in a fixed
order (eagle_worker_v2.py):

```
1246   draft()                      <-- DP0,1,2 are here
1248   verify()
1259   _draft_extend_for_decode()   <-- DP3..7 are here
```

RCCL/NCCL matches collectives by **issue order**, not by identity. Ranks 3–7 are
*two stages further along* than ranks 0–2, so when ranks 3–7 issue the LM-head
all-gather belonging to `draft_extend`, ranks 0–2 are still issuing the one
belonging to `draft`. Those two calls get matched to each other, and the next
one has no partner — deadlock.

So the ranks did not merely drift apart in time; they issued a **different number
of collectives**. Something on the idle ranks skipped, or on the busy ranks
added, at least one collective before this point.

Note the runner split lines up exactly with the stage split: the 5 ranks that ran
`_execute_idle` are the ones that are ahead. The idle path is doing *less* work
per stage but is *further along*, which is consistent with the idle path
issuing fewer collectives somewhere earlier in the iteration.

## The measurement this motivates

Every previous fix (hoist_sync, Bug 3 broadcast, Bug 4 uniform event) addressed
*how to re-synchronize after divergence*. None measured *why the counts diverge*.

`probe_stage_count.py` instruments exactly that:

* a per-rank monotonic counter of LM-head all-gathers (`AG_ENTER`/`AG_EXIT`),
  tagged with the current MTP stage — the counts must agree across ranks, so
  the first iteration where they disagree localizes the defect;
* `STAGE_ENTER`/`STAGE_EXIT` around draft / verify / draft_extend, plus an
  `ITER` line carrying `forward_mode`, `bs`, `speculative_num_steps` and the
  `spec_info` type — tells us which branch produced the extra/missing call;
* `DRAFT_LOOP` with the trip count and the idle flag — if
  `speculative_num_steps` ever differs per rank, the draft loop itself issues a
  different number of forwards.

Logs are written per rank to `/tmp/stage_probe_dp{0..7}.log` so they can be
diffed rank-against-rank.

## Reproduction detail worth keeping

The wedged server released **all** VRAM on `kill -9` of the 9 sglang PIDs
(launcher + 8 schedulers), going 86% → 0% on all 8 GPUs. A container recreate
was therefore *not* required, which preserved all five in-container patches.
Earlier notes assumed a recreate was necessary; it is not, as long as the
processes are killed by explicit PID rather than a broad `pkill -f` pattern.
