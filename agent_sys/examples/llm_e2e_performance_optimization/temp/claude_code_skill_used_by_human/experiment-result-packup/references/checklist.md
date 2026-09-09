# Completeness checklist

Walk this before Phase 4. Each item is something a cold reproducer needs. If an
item genuinely doesn't apply (e.g. no patches for a clean bench), mark it N/A —
don't silently skip. If an item can't be captured (env not snapshotted at run
time), record the gap explicitly in the file rather than guessing.

## 1. Time
- [ ] Experiment date (absolute, not "yesterday") in README.
- [ ] Per-run timestamps preserved (the `_YYYYMMDD_HHMMSS` dir names are fine).
- [ ] If multi-day, the date range and what changed between days.

## 2. Purpose + spec
- [ ] Goal stated in README in 2-3 sentences.
- [ ] Originating spec markdown copied into packup, OR linked by exact path if
      too large / still live.
- [ ] Success criteria copied verbatim from the spec (the concrete bar).
- [ ] Actual result stated against each criterion (pass/fail + margin).

## 3. Environment — hardware
- [ ] Exact machine hostnames + roles (e.g. prefill vs decode node).
- [ ] GPU model + count per node.
- [ ] CPU model, RAM.
- [ ] GPU driver / ROCm (or CUDA) version.
- [ ] RDMA fabric: type (IB / RoCE / ionic), driver version, active rails,
      data-plane IPs. (This repo's runs live or die on the fabric — be precise.)

## 4. Environment — software
- [ ] Docker image tag used.
- [ ] Base image `sha256` digest (not the floating tag).
- [ ] Dockerfile used (copied or path'd).
- [ ] Git branch AND commit SHA of the repo the run used.
- [ ] Any key library/kernel versions the result is sensitive to (e.g. sglang /
      aiter / vllm build, kernel date).

## 5. Reproduction commands / scripts
- [ ] Every script that ran copied VERBATIM into `scripts/`.
- [ ] `REPRODUCE.md` gives the ordered, copy-pasteable command sequence.
- [ ] Every script referenced in REPRODUCE.md exists in `scripts/`.
- [ ] Env vars that affect the result are captured (in the scripts or listed).
- [ ] Non-obvious flags explained (why `--attention-backend dsv4`, etc.) — brief.

## 6. Debug / fix outcomes (if any)
- [ ] Each patch/fix in `patches/` as a .patch/.diff.
- [ ] Each patch has a note: **what** changed, **why** it was needed, **how**
      applied, **context** (the symptom it cured).
- [ ] If the fix is inline in a script rather than a patch, it's called out.

## 7. Dependencies — absolute paths / uncommitted files
- [ ] Model weights path (absolute, + which shared FS).
- [ ] Datasets / input files (path + how to obtain).
- [ ] Uncommitted local config or files the run reads.
- [ ] Host-injected libs (e.g. libionic) — where they come from.

## 8. Required secrets (names + source only, NEVER values)
- [ ] Registry / docker login — which account, where creds live.
- [ ] Cluster / SSH access — how it's arranged.
- [ ] Any API keys / tokens / S3 / etcd creds — named, sourced, not pasted.
- [ ] Double-check no real secret VALUE leaked into any copied script/log.

## 9. Explanatory notes
- [ ] `notes.md` captures gotchas & wrong turns as what/why/how/context.
- [ ] Known failure modes + how to avoid (e.g. GPU mem must return to baseline
      between RDMA rounds; CUDA-graph capture is slow ~30min — wait, don't kill).

## 10. Results / evidence
- [ ] Machine-readable numbers (CSV/JSON), not just screenshots.
- [ ] Plots/summary tables if they existed.
- [ ] Results tie back to the success criteria.

## 11. Hygiene rules (from SKILL.md)
- [ ] No original file deleted (local or remote). Everything was COPIED.
- [ ] Files > 4 MB: user asked before including.
- [ ] Logs: user asked whether to include (and trim/gzip if so).
- [ ] Self-contained: cold-read pass done — no dangling pointers into scratch.
