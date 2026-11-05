# CI watch scope

Owner instruction, November 5 at 10:12 UTC: **the owner lifted the #33059 and
#46820 read-only holds, requested both rebased onto current main, and requires
fresh Base CI to pass completely before merge, as required by maintainers.**
Their prior FDFO incident report remains attribution evidence; it is not a merge
waiver or a substitute for a passing Base run on the published heads. The old
#46820 exact-attempt deferral is withdrawn. Prioritize these PRs together with
#46830, recover eligible failed/cancelled runs, and validate small fixes before
pushing. Preserve active runs; do not duplicate requests or reset healthy CI.
Record exact heads, run IDs, attempts and executed test results. An unexecuted
or skipped required shard is not a pass. Keep review requirements separate.

Re-read `read_only_prs` before remote mutations. Future owner holds apply across
heads and attempts until explicitly lifted; the list is currently empty.

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

The worker prompt implements this policy. The nine-PR watch list is unchanged;
the clarified recovery policy below uses a 30-minute check cadence.

## Owner clarification: recovery first, hourly retry opportunities

On November 4 the owner explicitly prioritized ongoing CI recovery over passive
diagnosis and record keeping. Inspect terminal failed/cancelled Base attempts
and use the `owner_requested` helper reason when no known code correction needs
to be applied first. This is a request to recover CI, not a claim that an unknown
failure has been proved transient. Do not require full root-cause proof before
using an available recovery opportunity.

Check every 30 minutes, allow one automated request per hour across this author's
watched PRs, and remove the former two-retries-per-head lifetime cap. Preserve
ownership, run-ci, exact head/attempt, active-run exclusion and durable duplicate
prevention. Retain successful jobs through the upstream failed-only mechanism;
an already-passing gate does not create a new two-hour wait. If a gate actually
fails its upstream cooldown, respect that observed deadline through `stale_gate`.

Every cycle must verify the request, handler, new attempt and actual test steps.
An eligible priority PR with an available slot must receive recovery action,
not only another table entry. A no-action result must name the current blocker
and next eligible time where applicable. Existing deterministic code failures
still need a small validated fix; do not spend all opportunities on one known
unchanged regression while another priority PR can progress.

## Execution priority

The November 5 10:12 UTC requirement above controls #33059/#46820: continue
until Base passes on their published heads. Preserve the [FDFO incident
report](CI_INCIDENT_46820_FDFO_20261105.md) as historical attribution, including
the failed old run and unexecuted files; do not use it to defer current coverage.
For other PRs, an evidenced unchanged-path failure with a matching upstream
baseline failure and merged correction can still be reported for maintainer
assessment. Include immutable logs/source and remaining unexecuted jobs; never
infer an exception for queued jobs, unknown failures or similar error names.

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

Before expanding that diagnostic, verify the actual CI launch configuration:
the modified metrics send returns early when KV events are disabled, and the
VLM test source does not enable them. Record whether the failing run could
execute the changed path. Recovering this Base shard remains in scope; repairing
the whole VLM subsystem is not automatically a prerequisite for this PR.
An isolated Python 3.12 real-LLaVA/pool-recovery probe is an available CPU-only
diagnostic if needed to choose a justified recovery action, not a new mandatory
validation gate. Keep worker counts bounded and capture first-submit stacks.
If local evidence is insufficient, the CI worker count, quota, parent memory
and blocked-thread stack are specific evidence to obtain from the runner owner.
Do not turn a successful simplified probe into a transient-failure classification.

## Recover existing CI without expanding the patch

The November 4 owner clarification is to finish the three priority PRs, not
accumulate unrelated development or self-imposed approval gates.

- Follow the existing current-head Base run when it is queued or active. Do
  not rebase merely to refresh it: a new head starts new CI and cannot retain
  the previous head's passing results. Rebase only for a concrete integration
  need, such as a verified upstream repair or an actual conflict.
- A same-head failed-job retry retains passing results and may execute failed,
  cancelled or dependent jobs. The supported PR-wide helper can also select
  other unsuccessful workflows. Disclose those side effects without expanding
  our patch-repair scope or overriding retry eligibility limits.
- Report separately: remaining Base tasks, known upstream-required checks,
  failures attributable to the changed path, and historical review/evidence
  notes. A red workflow alone does not prove patch causality or that every red
  check is a mandatory merge gate. Inspect actual requirements before claiming one.
- #46830's tokenizer TTFT test runs Qwen3-30B-A3B without the changed DSA/spec
  path. Retain its red CI result, but do not turn it into a tokenizer optimization
  project or new author validation gate without changed-path evidence.
- #33059's native MI300 coverage note remains historical evidence context.
  An author's own request for additional coverage is not itself a new maintainer
  blocking review. Preserve actual review requests and source qualifications;
  do not infer new hardware prerequisites from a missing matrix alone.
