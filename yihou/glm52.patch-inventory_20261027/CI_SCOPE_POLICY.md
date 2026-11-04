# CI watch scope

Owner instruction, November 3, 2026: when our PR does not change shared code,
focus only on **PR Test (Base)**.

- Assess the current PR diff and affected behavior. Record the exact head and
  scope reason; reassess after a head change. A shared filename alone is not
  evidence of shared behavior: a backend-specific guarded path can remain
  isolated. Shared interfaces, scheduling, dispatch, imports, or fallback
  behavior used by other paths can justify broader coverage.
- With no shared-code change, monitor, diagnose, retry, and repair Base issues
  only. Do not turn unrelated AMD, Xeon, ARM64, Extra, MLU, or other workflow
  failures into CI-watch blockers or repeatedly request their investigation.
- With a demonstrated shared-code change, additionally track only the affected
  workflows and explain their connection to the diff. An uncertain scope is a
  diff-review question, not authorization to pursue every failing workflow.
- Retain historical non-Base evidence. Remove an old out-of-scope CI concern
  from active notifications with a documented `scope_excluded` action; this
  means the monitoring scope changed, not that the test passed or was fixed.
  Split mixed findings so remaining Base, review, and correctness work is kept.
- Existing reviewer requests, source provenance, numerical qualifications, and
  actual upstream merge requirements remain separate ledger facts. Base success
  does not certify them. Do not invent additional CI gates for this watch.
- Retry authorization and limits are unchanged. The supported helper may rerun
  several workflows as a side effect of a justified Base retry; disclose that
  scope. Never request a retry solely for an out-of-scope failure.

The worker prompt implements this policy. The nine-PR watch list and 90-minute
polling schedule are unchanged.

## Execution priority

Owner instruction, November 4, 2026: **SGLang #33059, #46820 and #46830 are
the first maintenance group because they block other projects.** This group
takes precedence over broader deployment P0 labels. Check them first and
prioritize actionable diagnosis, small validated repairs and eligible retries.
When one is queued or running, work on another actionable priority PR. Advance
each actionable priority PR or record its concrete current blocker before
lower-priority repairs or retries. Report all nine monitored PRs. Within the
eligible priority group, retry the oldest-created Base run first, subject to
the existing helper and author cooldowns. Do not cancel or duplicate runs to
manipulate queue position; upstream runner scheduling and approval remain
outside our control.

### Current owned diagnostic action: #46820

The November 4 audit of [Base job 125750869472](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869472)
at head `ac205f829173029ce17c1898fd9c8d9ab2f861fd` identified a 10-second
timeout while submitting work to the LLaVA CPU process pool, before awaiting
the worker result. Pool replacement, warmup HTTP 500 and the outer 1800-second
timeout follow. This is the first observed failure, not a proven root cause or
proof that it is transient or patch-related.

`_create_cpu_executor` defaults to `os.cpu_count()` workers and fork; the first
submit starts workers. The existing real-worker-exit regression uses one
worker. Upstream fixes #44143, #36983 and #35700 are already in this head;
main's import cleanup #47731 is not an established repair for this failure.
Our next bounded diagnostic is CPU-only: measure first-submit time and thread
stacks, record worker count and actual CPU quota, and compare one worker with a
safely bounded representative pool. Do not launch an unbounded default pool on
a shared host. Distinguish process-startup cost from a lock stall before
proposing a fix or a retry; preserve the socket-contract regressions and avoid
speculative KV publisher changes. No new GPU allocation is needed for this
initial investigation.

**Follow-up after the 08:30 UTC cycle:** the bounded probe completed 24
submissions with one/eight workers under Python 3.13; first submits took
3–28 ms. It executed the tested head's extracted factory and initializer,
but used an `abs(-7)` payload and did not call the real LLaVA image-processing
method or execute the existing pool-recovery tests. This did not reproduce
the CI failure and does not qualify the full processing path.

The next diagnostic remains worker-owned and CPU-only: provision an isolated
Python 3.12 environment under the verified task NFS runtime, match the CI
version and relevant dependencies, and exercise the real LLaVA method and
existing pool-recovery regressions with bounded worker counts and first-submit
stack capture. Preserve fixture, source and environment identities and record
any missing dependencies. Do not require a GPU allocation or maintainer action
before attempting this local step. If it still cannot reproduce the failure,
the original CI worker count, resource quota, parent memory and blocked-thread
stack are the specific evidence to obtain from the runner owner. Do not turn a
successful simplified probe into a transient-failure classification or a fix.
