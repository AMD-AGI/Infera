# Synthetic GLM-5.2 eight-expert PD — completed

## Result
Operational smoke and one fixed-length CONC32 benchmark passed on2026-09-23. This is an intentionally non-faithful model for KV-capacity experiments, not original GLM-5.2 performance or quality.

- Prefill138 GPU0–3; decode136 GPU0–3. Each TP4/DP4/DPA on/EP4.
-75 target MoE layers:8 routed experts, all IDs0–7 selected,2 local routed experts per EP rank. Shared expert retained separately; shared fusion disabled.
- MTP OFF, simulated acceptance OFF, HiCache OFF; triton DSA; FP8 KV; memory fraction0.85.

## Measured memory
| Per rank | Prefill | Decode |
|---|---:|---:|
| Post-weight-load usage (runtime GB label) |31.43|31.44|
| KV pool (runtime GB label) |205.23|210.74|
| Allocated KV tokens |3,990,400|4,097,408|

Actual loader retained5,019 tensors /44,848,135,008 bytes and skipped112,391 tensors before materialization. Independent metadata accounting matches. Same-EP4 theoretical expert+gate saving is87.06GiB/rank; this is not a measured256EP4 A/B. Historical119.24GB/rank baseline differs in EP/MTP/fusion settings. KV allocated capacity is not a guarantee of sustainable fully occupied long-context service; this task tested4096/1024 only.

## Fixed-length benchmark
32 warmup +256 measured requests, exact4096 input/1024 output tokens from server usage. All288 requests successful; peak client inflight32. Measured window184.09s, zero failures.

| Metric | Result |
|---|---:|
| Output throughput |1,424.02 tokens/s|
| Input+output throughput |7,120.09 tokens/s|
| Mean TTFT |1.126s|
| p50 / p99 TTFT |0.810 /4.202s|
| Mean TPOT |20.87ms|
| Mean /p99 request latency |22.476 /25.544s|

Single-request128/16 smoke and16-request256/32 burst at concurrency8 passed. Final health: prefill/decode/router HTTP200. No traceback, native fault, NCCL error or NaN marker in successful-run engine logs. Transfer counters advanced on all4 DP ranks; prefill KV transfer count delta sums288 (warmup+measured). Do not equate health-probe-inflated stage counters with benchmark request count.

TTFT uses first nonempty text stream chunk; TPOT=(last-first)/(output_tokens-1). Chunk intervals are not guaranteed per-token ITL. No semantic-quality validation or full-capacity soak performed.

## Reproduce
All authored artifacts are inside this directory. Original checkpoint remains read-only. Both nodes require image infera-sglang:v0519-yihou-0917-nextnfix-hicache, SHA recorded in spec/mission.md. The model overlay references /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4.

```bash
W=/home/yihou/dev/git/infera.glm52.pd/bench/glm5p2_pd/results/yihou-expert8-pd
# Existing containers must first be deliberately stopped before a new launch.
# Choose a fresh round path; existing artifacts are never overwritten.
R="$W/rounds/yihou-reproduction-$(date -u +%Y%m%dT%H%M%S)"
python3 "$W/scripts/rdma_probe.yihou.py" --out "$R/rdma"
bash "$W/scripts/launch.yihou.sh" "$R"
bash "$W/scripts/smoke.yihou.sh" "$R"
bash "$W/scripts/bench_fixed.yihou.sh" "$R/benchmark"
```

Model preparation: scripts/prepare_model.yihou.py (see component handoff); exact source file mounts: patches/mounts.yihou.tsv, patch: patches/yihou-expert8.patch. Source files are frozen while file-bound into running containers; editing them requires fresh containers to avoid stale NFS handles.

## Evidence
- spec/mission.md and spec/plan.md: scope and approved plan.
- rounds/000-research/: GPU RDMA8/8 byte verification,40–42GB/s; HIP/NFS tmp diagnosis.
- rounds/001-component/expert8-handoff.md: source changes and focused tests.
- rounds/001-component/expert8/memory-accounting.json: checkpoint-only byte accounting.
- rounds/004-json-override/: successful launch, smoke, before/after snapshots.
- rounds/003-fixed-c32/benchmark/{summary.json,requests.jsonl}: benchmark evidence.
- rounds/003-fixed-c32/verification.json and after-benchmark/: effective settings, counters, logs.

## Bring-up fixes and limits
- NFS TMPDIR caused HIP 'Failed to unbundle code object' then SIGSEGV, even pure torch fill. Workspace-scoped container tmpfs resolves the demonstrated case; exact COMGR/NFS cause uninvestigated.
- Set JSON_MODEL_OVERRIDE_ARGS AFTER sourcing inherited config; its brace expansion corrupts a pre-set JSON string. Initial failure preserved in round002.
- No node137 interaction, no original checkpoint/shared tracked code modification, no commit. Services are left running for follow-up experiments; this completed task does not launch extra sweeps.
