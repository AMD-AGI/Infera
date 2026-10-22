# GLM-5.2-MXFP4 v0.5.21 same-node P4D4 — OffloaderV2 G4 at C56: KV-transfer sharding and chunk size (task "g4tune", 2026-10-19 05:2x → 2026-10-20 01:3x UTC)

Follow-up to `../glm52.v0521-offloaderv2-burst-c56c64.packup_20261019` (G2/G3/G4 at C56/C64). This kit packs the
g4tune rounds g4xb1–g4xb5, g4xm1–g4xm4, g4xc1, g4xc2 and the machine microbenchmarks.
**Status: IN PROGRESS** — g4xc2 (chunk 65536 on m2m-250) was started 2026-10-20 01:27:27Z and had no result when this kit was built (01:3x Z).

## Question (user, 2026-10-19; full brief in `spec/task_brief.CLAUDE.md`)

Base = best point so far, ofg4b A2 23,771 tok/s/chip (C48, OffloaderV2 G4K1 step 1, pf90 recipe, sim 3.61). Get more throughput at **C56** (then C52):
1. Explain the prefill "inflight" stage (last chunk sent → KV transfer Success: ~5 s at C48, 10–12 s at C56, 15–19 s at C64; ~7.5–8 GB per request). Suspects: the machine; Mooncake.
2. Larger chunked-prefill size (user hypothesis: small chunks cannot hide OffloaderV2's group-MoE weight load).
3. Debug at C56; C52 when there is time. Decode always `--tokenizer-worker-num 4 --detokenizer-worker-num 4`.

Correctness is waived by construction (`DECODE_SIMULATE_ACC_LEN=3.61`); accept length is not a correctness result.

## Result — C56, one variable at a time vs R1 (score = `request_metrics.throughput.per_gpu.total_tput_tps`, n = 1 each)

| round | variable | node | A1 | **A2 (scored)** | A2 TTFT mean/p95 (s) | A2 TPOT (ms) | prefill transfer_kv p50/p90/avg (s) | kv speed p50 (GB/s) | prefill_forward p50/avg (s) |
|---|---|---|---|---|---|---|---|---|---|
| g4xb5 | R1 baseline, `YIHOU_XFER_SHARD=dest` (= upstream) | OAI m2m-250 | 20,606 | **21,262** | 20.59 / 61.90 | 13.3 | 7.31 / 21.76 / 9.98 | 0.86 | 2.58 / 4.18 |
| g4xm4 | R2 `SHARD=mix` (same node as R1) | OAI m2m-250 | no JSON (EDQUOT) | **22,854** | 14.83 / 49.37 | 15.0 | 1.89 / 4.01 / 2.16 | 3.09 | 3.08 / 4.62 |
| g4xm3 | R2 `SHARD=mix` | burst v3-017 | 21,769 | **22,495** | 15.88 / 58.43 | 14.4 | 1.96 / 4.12 / 2.22 | 2.94 | 3.20 / 4.82 |
| g4xc1 | R3 `PREFILL_CHUNK_SIZE=65536` (per-DP 16384) | OAI m2m-097 | stopped in warm-up | — | — | — | — | — | ~3m15s per 16384-token/DP step |
| g4xc2 | R3b chunk 65536 on another machine | OAI m2m-250 | **in progress** | — | | | | | |

Six more rounds (g4xb1–4, g4xm1–2) were lost to burst preemption; see `results/results.csv` and `results/rounds/`.

**What the measurements show (no mechanism claimed):**
- Same node (250), mix vs dest: A2 +7.5 % (22,854 vs 21,262); prefill transfer_kv avg 2.16 vs 9.98 s; TTFT mean 14.8 vs 20.6 s; TPOT 15.0 vs 13.3 ms; prefill_forward avg 4.62 vs 4.18 s.
- **User decision 2026-10-20: `SHARD=mix` is NOT adopted** — it makes prefill and decode interfere and hampers the analysis. R1 (`dest`) stays the baseline. (From the user via the task brief; not yet written in the ledger when this kit was built.)
- With `dest`, every prefill DP used ONE sender shard; per-DP transfer rate in the profiling phase of g4xb5 was mostly 1–4 GiB/s (range 0.2–4.2) per 5-min window while that single worker was busy ~50–100 % (`results/xfer_sum.g4xb5_g4xm4.txt`, re-computed for this kit).
- Machine (v3-021): Mooncake HIP transport idle ~47–51 GB/s per pair; with a bf16 matmul burn on the sender GPU (GPU1–3) and ≥ 2 pairs: 0.36–0.4 GB/s (GPU0 unaffected); receiver load no effect; `MC_HIP_STREAM_PRIORITY=high` / `HSA_ENABLE_SDMA=1` no change; `HSA_ENABLE_SDMA=0` hung. torch peer copy idle 59.7 GB/s. Mechanism not established. Details in `notes.md` §4.
- Chunk 65536 on m2m-097: first two 16384-token steps 4 s apart, then ~3m15s per step with GPU0-3 100 % busy; py-spy showed the scheduler in `torch.synchronize` ← `process_batch_result_disagg_prefill` (prefill.py:962). Cause not measured. User: chunk changes worked on spur before → rerun on another machine (g4xc2).

## Navigate

`REPRODUCE.md` · `environment.md` · `notes.md` (narrative, gotchas, open questions) · `patches/` (engine overlays + diffs + what/why/how) ·
`results/results.csv` · `results/bench/<round>-c56-<A>/` (agentx JSON + server_metrics CSV) · `results/hist.g4tune.jsonl` (sglang histograms) ·
`results/xfer_sum.g4xb5_g4xm4.txt` · `results/rounds/<round>/` (status, chain/pipeline logs, census, audit, bench console) · `results/dryruns/` ·
`results/launch/` · `results/machine/` (microbench logs) · `scripts/` · `analysis/ledger/poll.md` (snapshot 2026-10-20 01:19Z) · `analysis/research_wf.20261019.json`.

Rebuild the copied part: `bash scripts/tools/packup_copy_g4tune.yihou.sh` (copy-only). Files > 4 MB are listed, not copied, in `results/not_copied.tsv`.
