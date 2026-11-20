# Notes — gotchas, wrong turns, open questions (what / why / how / context)

## Gotchas that each cost a run (all fixed in the kit)
1. **Docker cannot mount HOME NFS on openai nodes** (root_squash). Symptom: `mkdir /home/yihou/dev: permission denied`.
   Fix: `stage.sh` copies kit + inferencex-e2e to /perf_apps; run_point/selftest mount only staged copies.
2. **vLLM frontend 600 s engine-registration timeout** when 4 servers load 476 G from NFS at once. Symptom: `engine input
   registration timed out after 600s`. Fix: serve.sh sets `VLLM_ENGINE_READY_TIMEOUT_S=3600` (setup only).
3. **aiperf offline tokenizer lookup by HF id needs a COMPLETE cached snapshot** (incl. 48 weight shards). Symptom:
   `IncompleteSnapshotError`, client exits before any request. Fix: run_point passes `TOKENIZER=<local snapshot dir>`
   (InferenceX replay.py accepts a path). Servers that never got a prompt token can be re-benched (checked via metrics).
4. **vllm-router soft RLIMIT_NOFILE=1024 in a default container** → at c384 the warmup "cache-pressure" burst (~550 in flight)
   exhausts fds → `error sending request` → circuit breaker → 503 → aiperf aborts on a failed root warmup request.
   Repro: 700 parallel tiny requests → 507x200/2x500/191x503. Fix: `--ulimit nofile=1048576:1048576`.
5. **Shared 0777 NFS + world-writable run dirs** = injection risk (entry.sh executed as container root). Fix: executed/sourced
   files in owner-only `runs/<tag>/kit/` (0755 dir, 0644 files, mounted :ro); log dirs sticky 1777; config owner/mode checked.
6. **ssh launch can hang silently** → node never got the command. Fix: `launch_remote.sh` uses `ssh -n -o BatchMode=yes`,
   requires 'launched' and the run dir to appear.
7. **rocm-smi GPU index != HIP index** on these nodes (e.g. HIP 4,5 → smi 4,7). Check per-container VRAM before alarming.
8. **Workflow subagents stalled twice on trivial shell calls** (suspected hidden permission prompt). Leader finished the PD kit inline.

## Findings with evidence level
- [FH, code + run] Any non-UMBP KV connector disables the AITER sparse-MLA fast path (`only_restores_canonical_kv`,
  vllm models/deepseek_v41/amd/rocm.py:80-109) and then `--kv-cache-dtype fp8_e4m3` is rejected at load
  (deepseek_v4/amd/rocm.py:167-172) → PD must run fp8_ds_mla + Triton sparse. Measured colocated cost c128: -13.5% (E4b vs R1).
- [FH, code read] Mooncake on main transfers compressed KV, indexer K, all SWA windows (clipped) and the compressor ring; D
  recomputes Engram history + last token (bounded replay N-1). No missing state found (research/g3_pd_state_transfer.md).
- [INF] Safe override of the fast-path gate for Mooncake needs a P/D config fingerprint at handshake (research/g1_fastpath_override.md).
- [FH] Reproduction gap vs CI ≈ -5% (R1, R2) with 3-4 servers per node; replicate spread 4.3% → not yet attributable.
- [FH raw] R1 knee is above c256 (c128 → c192 +26%, c256 +3% more); R2 (UMBP) c384 = 407k.

## Wrong turns (kept so nobody repeats them)
- E4 first tried "SPARSE_MLA=False with fp8_e4m3" — impossible combo (load error #1 in Findings); split into E4b/E4c.
- First report on c192/c256 called the windows partial (read before flush); they were complete (3628/3630 s).
- R4 (TP2 c32, CI 107,989) could not be reproduced: its image tag is gone from Docker Hub (404).

## Open
- Is the -5% vs CI from concurrent servers on one node? (needs an isolated R2 run)
- UMBP external hit differs 2-4x between the two DP ranks (7-14% vs 33-58%) — cause unknown.
- E7 PD correctness + throughput (running on crusoe v3-186 since 2026-11-20 05:22Z).
- E12/E13 prefill and decode per-GPU rates (decide whether +50% is reachable with PD; research/dynamo_vllm.md budget).
