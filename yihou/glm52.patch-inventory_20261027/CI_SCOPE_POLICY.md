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
