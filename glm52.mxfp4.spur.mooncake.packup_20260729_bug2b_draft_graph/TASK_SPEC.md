# Task as issued

The task was given conversationally, not as a spec file. Recorded verbatim so the kit is
self-contained and the goalposts cannot drift retroactively.

## The instruction (2026-07-29, user, verbatim)

> 现在开始尝试自行精细定位draft cuda graph打开后的同步卡死bug到底在哪里。并尝试修复
> （可以和关闭draft cuda graph对拍）（可以尝试修改warmup代码）（可以使用二分定位，打点，
> 从最简单的能跑通的mvp开始，逐渐缩小问题域等方式）自行解决问题，期间用户不在，不要询问用户。
> 开始之前。把除了实验需要的额外资源释放了。先收集资料，再plan, 再设置goal，并更新CLAUDE.md，
> 设置当前任务为主要任务。然后开始工作。

Plus a debug-methodology rule set, summarized:

* **Loop**: understand context → form method from materials → run → if fail, analyze → adjust
  → repeat.
* **Methodology, at least 3 of**: (1) traces/stacks/logs, (2) source analysis both
  top-down and bottom-up from the failure, (3) **对拍 — mandatory when a reference arm
  exists**, (4) official docs / web research, (5) rigorous consistent recording.
* **Strategies**: direct iteration; progressive (shell → MVP → single-node → full);
  decomposition by module; binary search once the problem domain is closed. Pick by cost.
* **Rules**: at least 3 methods, 对拍 required when available; **every round gets its own
  directory** recording command / hypothesis / log / result, indexed from one global file
  (`working_process.md`) so no round's evidence is lost.
* **Standing constraints** (carried from earlier in the session): never delete an
  experiment workspace, local or remote, without explicit instruction; files over 4 MB
  need approval before being added to git; ask whether logs should be included; **the
  user submits PRs — do not push** (later amended: this round was explicitly asked to
  push to the PR worktree).

## Prior state this task inherited

From `glm52.mxfp4.spur.mooncake.packup_20260728/dpa_mtp_fix/`:

* Bug 1, 5, 6 and the nextn bf16 patch: fixed, PD+DPA+MTP reaching 640/640 at conc=128 —
  **but only with the draft CUDA graph disabled** (Variant B).
* Bug 2b — the draft-graph deadlock — localized to `eagle_worker_v2.py::draft()` and
  known to be rank-divergent, but **not fixed**. One attempt (removing the
  `not is_idle()` term) had failed and moved the failure earlier, into warmup.
* A retracted claim on record: "the graph path issues 0 all-gathers" — a probe blind spot,
  since a replayed CUDA graph runs no Python.

## Goal set at the start of this round (before any work)

Written into `PLAN.md` and `CLAUDE.md` up front, binary, five parts — all five or it is
not a fix:

1. PD warmup completes;
2. 4 × 24-token → 4/4 with `spec_accept_length > 1`;
3. 1 × 512-token → 512/512;
4. conc=128 × 512 → 0 hangs, 0 `KVTransferError`;
5. **the graph path is provably taken** (marker count > 0).

Criterion 5 is the one that matters: Variant B already satisfied 1–4 by disabling the
graph, so only 5 distinguishes a fix from the existing workaround.

**Outcome: all five met**, plus a mix regression and an 8-round durability run —
2540/2540 requests, against 0/4 for the same-node control with only the fix reverted.
