# Plan — finish validating and land the three upstream sglang GLM-5.2 PRs

Read `context.md` first for the environment facts and the traps; read
`working_process.md` for what already happened and why. This file is only the
remaining work, in the order it should be done.

## Definition of done

For each of #33968 / #33970 / #33973: reviewed, validated on gfx950 hardware,
flipped from draft to ready with the pr-gate green, and the local records
(`deploy/docker/patches/*/**.upstream.status.yaml`, `pr.done.md`) updated to match.

## The gating principle

**A patch does not get flipped to ready on the strength of a clean rebase or a
passing validator script.** `validate_{A,B,C}.py` check that the upstream diff is
equivalent in effect and correct in scope against the proven local patch — they do
**not** require the defect to reproduce, and they cannot tell you the fix works.
Only the hardware runs below can do that.

Corollary that already bit once: for any A/B, **the positive control comes first**.
Stock must reproduce the defect before a clean result on the patched tree means
anything.

## Step 1 — #33970 (mooncake KV race). Nearest to done.

The single-node TP4+TP4 approach is sound; r01 proved its four premises, including
a mooncake loopback transfer with the real engine. r02 failed to launch for two
reasons that are both single-host artifacts, neither a defect in the patch.

**1a. Fix the bring-up.** In `rounds/r02-stock-positive-control/scripts/up_singlenode.sh`:

- Separate the two legs' `--port` by more than `ZMQ_TCP_PORT_DELTA (233) +
  NUM_DERIVED_PORTS (6)`. Use 30000 and 31000. The current 30000/30001 makes the
  two legs' derived ZMQ ranges overlap on shared `127.0.0.1` (both containers are
  `--network=host`), and prefill dies with
  `Address already in use (addr='tcp://127.0.0.1:30235')` -> `exited with code -9`.
- Move decode's `--kv-events-bind` and `--kv-snapshot-port` off 5557/8801. Decode
  binds them **even with `KVAWARE=0`** — infera defaults `--kv-events-bind` to
  `tcp://0.0.0.0:5557` (`/opt/infera/infera/engine/sglang/args.py:180`).
- Do **not** loosen the arm guard. It greps `wait_event` in `mooncake/conn.py`
  (stock=0, patched=9) and aborts on a mismatch; that is what stops a mislabelled
  A/B from being recorded as a result.

**1b. Fix the needle probe's chunk arithmetic.** `rounds/.../scripts/needle.py`
computes which chunk each needle lands in. It must use the **resolved**
`chunked_prefill_size` read off the leg's own `server_args` line, not the value
passed on the command line: with DP-attention on, sglang divides the global budget
by `dp_size`, so `--chunked-prefill-size 131072` at `dp_size=4` resolves to 32768.

This matters because **only non-final chunks can be corrupted**. The final chunk
goes through the sampling path, which already has a real `copy_done.synchronize()`,
so a final-chunk needle is retrieved correctly even on a broken build — that would
read as a false PASS. The probe already reports non-final and final scores
separately; keep that, it is the built-in control separating "the race" from "the
model cannot do this".

**1c. Run the A/B.** Stock arm first and require a degraded non-final-chunk score.
If stock comes back clean, **stop** — the configuration is not reproducing the
defect and the patched arm proves nothing. Then the patched arm.

**1d. Measure the `synchronize()` cost.** Prefill throughput, patched vs stock,
via `engine/bench.sh` or `sglang.bench_serving`. Currently unmeasured. The patch's
own note flags it and it is the first thing a reviewer will ask: the new
`synchronize()` blocks the transfer worker, trading transfer overlap for
correctness.

**1e. Write the honest scope statement into the PR.** Single-node loopback CAN
establish the correctness claim and the cost. It CANNOT establish behaviour under
real cross-node RDMA latency — loopback is *faster*, so the race window is
*narrower* and reproduction *harder*. A positive reproduction here implies the
cross-node case is at least as bad; a failure to reproduce here would not clear it.

## Step 2 — #33973 (DSA decode DP-divergent D2H syncs). Not started.

Needs 1x gfx950 with PD + DP-attention + MTP. Success criterion: the DP group does
not deadlock on the first routed request, and `py-spy dump` on every rank shows no
rank parked inside `dsa_backend`.

The `up_singlenode.sh` from step 1 is the natural harness — set `MTP=1` on the
decode leg (it is forced to 0 in the current script) and keep `DPA=1`. Same
positive-control discipline: stock must deadlock first.

One semantic check is already done and need not be repeated: `dsa_backend.py:1055`
constructs `DSAMetadata(seq_lens_sum=...)` on a path DRAFT_EXTEND_V2 reaches, and
the concern was that dropping the `.cpu()` mirrors strands that field.
`metadata.seq_lens_sum` has **zero** readers repo-wide (checked
`dsa_indexer_metadata.py`, `dsa_topk_backend.py`, `nsa_backend.py`; no `asdict` /
`astuple` / `replace`).

## Step 3 — #33968 (HiCache ROCm host-pool allocator). Blocked on hardware.

n06-33 measures `same=True` for all four allocation strategies at every size from
8 MiB to 7.33 GB — it does **not** reproduce the fault, so it is a negative control
like gfx942. Size is ruled out; no mechanism is claimed. See `context.md`.

On a candidate machine, run `scripts/probe_host_devptr_sizes.py` first. **If it
does not print `same=False`, that machine cannot validate this PR** — do not
proceed to the HiCache write-back repro, and do not flip the PR. If it does, the
repro is: stock must fault at the host VA, patched must not.

Worth recording on whatever machine reproduces it: `amdgpu` version. This box runs
6.14.14, which the patch record attributes to the MI300X negative control, and the
original gfx950 fault report does not record its driver version. That is the only
known uncontrolled variable left — it is a lead, not a conclusion.

## Step 4 — review passes

Not yet done for any of the three. Two passes per PR, per `pr.verify.md`:
the code-review skill on the upstream PR, then a deeper read using LSP + serena
(the open-source-pr skill). Do this before flipping, not after.

## Step 5 — flip and record

Per PR, once its hardware run has passed:

1. `gh pr ready <n>`.
2. Confirm the pr-gate goes green. The current red on all three is the
   `Block draft PR` repo policy, not a real failure — it should clear on its own
   once the PR is no longer a draft.
3. Update `deploy/docker/patches/*/**.upstream.status.yaml` and re-run
   `scripts/validate-patch-status.py`.
4. Update `pr.done.md` to replace the historical-evidence caveats with what was
   actually measured, and on which machine.

## Step 6 — the stale record

`pr.done.md` states that upstream **#30350** is the better fix for
`patch_hicache_rocm_staged_write_back` and that the action is a re-review nudge.
**#30350 was closed unmerged on 2026-08-17 by its author** (Emmanuel0612), so that
record is wrong. Per the user's decision this is a **TODO only** — they will look at
it themselves. Do not open a replacement PR, and do not widen scope to the four
patches that correctly get no PR (`work.todo.md` explains which and why).

## Constraints that apply throughout

- **DCO**: every commit needs `-s`, signed off as the actual author. Never a bot or
  assistant identity, never a copied colleague line. Upstream also rejects
  assistant `Co-Authored-By` trailers.
- Work in English; report to the user in Chinese.
- Keep temporary artifacts in a scratch workspace, not in the repo tree.
- Suspend, don't conclude: state what was measured. Where a mechanism is unknown,
  say so and name the measurement that would settle it.
