# GLM-5.2 Internal Decode Implementation Plan

**Goal:** Run 16 scheduler-free requests from 70000 to 80000 committed tokens with real MTP compute and expected accept length3.61.

**Architecture:** Real pinned TpModelWorker/EAGLEWorkerV2, fixed ScheduleBatch, physical synthetic-prefix KV, one-token untimed bootstrap, real draft/verify/extension and explicit state relay. Leader owns environment/run scripts; teammate owns bench/tests.

**Tech stack:** Python, PyTorch ROCm, SGLang402df1e1, AITER2c71811b, Docker, existing Spur allocation126175.

## Constraints
All writes under this workspace or its owned Docker container. No node234/036, allocation submission/cancellation, host changes, other-user cleanup, source repo edits, commits or pushes. No Scheduler or PD. Preserve real routing. See design.md for timing/accounting contract.

## Steps
- [x] Research pinned APIs and reference limits; obtain plan approval.
- [x] Create workspace and task instructions; snapshot evolving packup inputs.
- [x] Confirm own-container view exposes8 MI355X; verify containerd storage has24TiB available. First read-only probe lacked a temp directory; second probe with workspace TMPDIR succeeded.
- [ ] Verify archive hash/zstd and load pinned image; inspect exact import/source identities.
- [ ] Teammate: test CPU acceptance/output-cap/page-reserve helpers, then implement bench/profile_decode.py and supporting state code. Main entry spawns TP ranks itself; no torchrun.
- [ ] Leader: create guard-railed run script using existing allocation and owned container; syntax-test and exercise negative input/node guards.
- [ ] Run --help/import probe; then eager few-iteration small-context smoke with real weights.
- [ ] Diagnose each failure from log, pinned source, reference and official documentation; localize one change per iteration.
- [ ] Graph smoke; long-context smoke; assert graph/backend engagement and capacity.
- [ ] Refresh packup evidence and execute full workload, plus bounded repeat.
- [ ] Summarize measurements/provenance/limitations with exact commands and retained logs.

## Tests
CPU helper tests run without GPU dependencies using `python3 -m unittest discover -s tests -v`. GPU launch uses `bash scripts/run_decode.sh <unique-iteration> --batch-size 16 --input-len 70000 --output-len 10000 --accept-length 3.61 --tp-size 8`. `--max-steps` is smoke-only and cannot satisfy completion. Full result must report160000 useful emitted tokens and all16 completed requests.
