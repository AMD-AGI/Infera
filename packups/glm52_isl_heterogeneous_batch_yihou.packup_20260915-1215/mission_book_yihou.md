# MISSION — heterogeneous-ISL batches in the internal SGLang decode harness

**Started** 2026-09-15 11:14 UTC. **Workspace**
`temp_workspace/isl_hetero_batch_yihou_20260915-1114/`.
This file is authoritative and is re-injected every 10 minutes. If anything below conflicts with
memory, this file wins.

## The task in one sentence

Make `sglang_decode_internal_bench_and_profiling` able to run a batch whose requests have
**different input lengths**, via **Plan 1** (every DP rank receives the *same* ISL multiset), and
expose four ways to specify that multiset.

## Scope — decided by the user 2026-09-15

**Plan 1 only.** Every attention-DP rank gets an identical ISL multiset, so all ranks stay in
lockstep and the cross-rank determinism check survives intact. Plan 2 (a global distribution split
unequally across ranks) is **out** — it would delete the harness's strongest invariant.

### The four ISL modes

| mode | spec | semantics |
|---|---|---|
| `uniform` | one integer | today's behaviour; `--input-len 70000` must keep working unchanged |
| `bimodal` | two values + a ratio | **the minority side rounds UP (`ceil`)** |
| `normal` | mean + stddev | rounded to int, clamped; realized list recorded |
| `list` | explicit sequence, or a file | most general; the other three reduce to it |

`bimodal` ratio semantics, fixed by the user: with `local_batch_size` requests and a ratio, the
side with the **smaller** share gets `ceil(local_batch_size * share)` requests. Example: 32
requests, 10 % short -> `ceil(3.2) = 4` short, 28 long.

## What is already established — do not re-derive

These are first-hand reads done on 2026-09-15 before the team was formed. Treat as settled.

**F1. The attention backend is already ragged-safe; it needs ZERO changes.**
`dsa_backend.py:1143-1153` (read inside container `yihou-glm52-tp8ep1-pr50-51` on n10-29):
```python
cache_seqlens_int32 = (forward_batch.seq_lens + draft_token_num).to(torch.int32)
max_seqlen_k = int(forward_batch.seq_lens.max().item()) + draft_token_num
```
`seq_lens` is a per-request tensor and the max is derived, not assumed. This is the normal serving
path. CUDA graphs are captured per batch size, not per sequence length, so they are unaffected too.

**F2. Acceptance is a single scalar per iteration, broadcast to the whole batch — and that is
upstream behaviour, not ours.** `spec_utils.py:411-419` calls `sample_simulated_acc_len` once and
then `sim_accept_index[:, :simulate_acc_len]`. Therefore `len(set(accept_lens)) == 1` in
`topology.py:69` is TRUE and must be KEPT. Only the `len(set(previous_lens)) == 1` half is the
thing that has to go. **Do not weaken the acceptance half of that assertion.**

**F3. KV allocation is already decoupled from the logical ISL.** In `allocate_batch`,
`batch.prepare_for_extend()` (line 228) runs BEFORE the per-request truncation loop (lines
231-235). Allocation uses one uniform max row width; the logical prefix is truncated afterwards,
per request, in a loop that is already elementwise. So short requests simply over-reserve.

**F3a. Consequence, and a real limitation to state in the final report:** a mixed batch costs the
same KV as an all-max batch. A `{8k, 70k}` mix does NOT fit anywhere an all-70k batch does not.
Fixing that means changing the allocation path and is explicitly OUT of scope.

**F4. `validate_worker_progress` (`batch_state.py:8-13`) is already elementwise — zero changes.**

## The one real design point

`topology.py:66-79` compresses the whole batch into 7 ints, all-gathers them, and demands exact
equality across ranks. Under Plan 1 every rank holds the same ISL multiset, so equality is still
achievable — but `previous_lens[0]` is only meaningful if the per-rank ORDER is also identical.
Either guarantee identical ordering, or replace the positional field with an order-insensitive
digest. Decide it explicitly and write down which one, with the reason.

## Core principles for this task

**Suspend, don't conclude.** Report what was measured or read, and stop. In particular: whether a
heterogeneous batch's TPOT is interpretable at all is an OPEN question (DSA is sparse with
`index_topk = 2048`; the per-step cost-vs-length relationship has NOT been measured). Do not
predict it, do not assert it, do not design around a guess about it.

**A backend that was requested is not a backend that ran.** Unchanged from the previous task. Any
run must assert from the log which kernel actually executed.

**Determinism is the product.** `realized_accept_length` is the correctness gate and is unaffected
by ISL. A change that makes a uniform-ISL run produce a different `realized_accept_length` than the
published baseline is a bug, no matter how good it looks.

## Backward compatibility is a hard requirement

`--input-len 70000` must keep producing byte-identical behaviour. The published baseline
(**TPOT 25.6740 ms**, `realized_accept_length 3.6134393063583814`, `verify_iterations 2768`) is the
regression reference. Existing sweep scripts, `verify_point.py`, and `config_yihou.json` consumers
must not break.

## Rules (from the user, in force at all times)

1. `mission.md` is re-injected every 10 min so it never falls out of context.
2. Work as an agent team. Leader polls every 20 min; a problem seen once is only recorded, and only
   escalated if the next poll shows it unresolved.
3. Follow the user-level `CLAUDE.md`; ask interactively before deviating.
4. research -> gather -> analyse -> plan -> sub-workspace -> CLAUDE.md -> then work.
5. Prefer running in a docker container over touching the host.
6. Work through LSP and serena.
7. **Never delete any file whose path does not contain `yihou`.**
8. English at work; Chinese only when reporting to the user.

## Hard rules inherited from the user-level CLAUDE.md

Work inside a `yihou/` directory. Outside one you may read and create, but **never delete** — no
`rm`, no `--delete`, no truncate-and-rewrite of a file you did not create, no `docker rm` of a
container you did not start. **Never write a recursive delete whose target is a variable.**
DCO sign-off (`git commit -s`, as yourself) on every commit. English at work.

## STATUS 2026-09-15 12:05 UTC — DONE (code + GPU verified)

> The 11:35 entry below said the GPU smoke was BLOCKED by tenancy. **Superseded.**
> The user authorised preempting non-Slurm workloads on a node Slurm shows as ours;
> the condition was verified, two containers were `docker stop`ped (not removed), and
> all three GPU runs completed. Preemption record: results/implementation_status_yihou.md.

**Regression gate met bit-for-bit:** `realized_accept_length = 3.6134393063583814`,
`verify_iterations = 2768`, `complete = True` at C=256 / ISL 70000 / OSL 10000.
**Ragged batch runs on 8 GPUs** with CUDA graph ON; every acceptance quantity is
identical to the uniform control. TileLang confirmed as the kernel that actually ran.
TPOT differences in these runs are single-shot and are NOT conclusions.

### Original 11:35 entry (kept for the record)


Full detail in `results/implementation_status_yihou.md`. Summary:

- `60 passed, 40 subtests passed`; baseline before any change was `26 passed`.
- All four ISL modes verified end-to-end through the real CLI at tp8/ep1/dpa/C=256. Both ceil
  directions correct (`ratio=0.1` -> 4/28, `ratio=0.9` -> 28/4).
- **Measured, not argued:** `verify_iterations`, `realized_accept_length` and
  `useful_output_tokens` are byte-identical across uniform / bimodal / a 61808-wide spread. The
  existing acceptance gate therefore applies to ragged runs unchanged.
- **Measured:** `verify_point.py` still passes all 11 published pre-change result files.
- Both teammates delivered and are idle. Nothing outstanding in their files.

**BLOCKER: `smci355-ccs-aus-n10-29` is fully occupied by three other workloads** as of 11:30 UTC
(evidence: `results/tenancy_20260915-1130_yihou.txt`). One of them is a **4-node distributed**
MLPerf Flux training job (`torchrun --nnodes=4 --node_rank=3`); another is someone else's
`profile_decode.py` against `/perf_apps/xiaobo/models/`. ~293 GB VRAM per GPU is spoken for.
**Touch none of them.** Our Slurm hold (job 30723) does not make the GPUs free — containers on this
host ignore the scheduler. Re-check tenancy before attempting the smoke run; do not start it while
any of those three are live.

## Machine

`smci355-ccs-aus-n10-29`, shared. Four of our containers still exist
(`yihou-glm52-tp8ep1-{round2-ptpc1,pr50-51-ptpc1,pr50-51,oldpin}`). Other people's containers run
on the same host — touch none of them. **Never run on the local host.**

Most of this task is CPU-only Python and needs no GPU at all. Only the final smoke run does.
