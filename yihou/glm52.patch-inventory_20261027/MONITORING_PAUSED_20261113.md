# GLM-5.2 PR monitoring handoff

## Monitoring paused by owner — 2026-11-13

The owner explicitly requested that recurring monitoring stop until they ask to resume. The three-hour CI watch and daily PR ledger cron commands are commented out; the watch runtime config has `enabled=false`. Historical evidence and unresolved items are retained. No automatic retries, reviews or ledger refreshes are scheduled. No active watcher or ledger worker was found at pause time. Existing upstream GitHub runs are not cancelled.

| PR | Recorded state | Next action after explicit resume |
|---|---|---|
| SGLang #33059 / #46820 / #46830 | Merged; Base passed; #46820 required NVIDIA Extra passed | No retry needed |
| [SGLang #49732](https://github.com/sgl-project/sglang/pull/49732) | Open, Ready; Base passed (attempt 3) | Review/approval and merge follow-up |
| [SGLang #45524](https://github.com/sgl-project/sglang/pull/45524) | Open, Ready; Base passed (attempt 1) | External-author review/approval and merge follow-up |
| [SGLang #47278](https://github.com/sgl-project/sglang/pull/47278) | Open, Ready; Base passed (attempt 2) | Prioritize MLA kernel review; retain source/numerical qualifications |
| [SGLang #47897](https://github.com/sgl-project/sglang/pull/47897) | Open, Ready; Base passed (attempt 2) | Kernel review; retain non-Base and numerical qualifications |
| [SGLang #47276](https://github.com/sgl-project/sglang/pull/47276) | Optional, outside llr37; Base attempt 29 failed | Inspect latest failed job before deciding recovery; prior repeats showed FlashInfer MLA graph mismatch; causality unproven |
| [SGLang #46918](https://github.com/sgl-project/sglang/pull/46918) | External original; Base cancelled | Reference only; owned continuation #49732 passed |
| SGLang #47872 | Owner-paused pending redesign; not queried | Remain paused unless explicitly resumed |
| [Mooncake #5005](https://github.com/kvcache-ai/Mooncake/pull/5005) | Merged, confirmed by live query on November 13 | No merge follow-up needed |

Base success is not a claim that every platform workflow passed. Four open priority PRs had no APPROVED review at the last explicit review audit (November 12); that review snapshot was not re-queried at pause time. Upstream state may change after this snapshot.

Runtime pause evidence: `runtime/ci-watch-20261103/paused-20261113/` (ignored local artifacts), including the previous crontab and last cycle/notification snapshots. The last completed watcher cycle succeeded at 2026-11-13 03:12:53 UTC. Restart requires an explicit owner request and fresh head/CI/review checks; do not treat this intentional pause as an outage.
