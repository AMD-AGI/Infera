# GLM-5.2 PD 4+4 C80 (135 -> 136): trace bottleneck report, ATOM/Dynamo-ranked experiments, replicate (2026-10-24..25)

Follows `yihou/glm52.llrepro-kf-ar-topk.packup_20261024` (best then: llr7 = 41,679 = llying 4+4 kit + KF decode kernels +
decode custom AR + decode `SGLANG_OPT_USE_TOPK_V2=false`). All runs: prefill crsuse2-m2m-135 GPU4-7 (PP4 x TP1), decode
crsuse2-m2m-136 GPU4-7 (TP4/DP4 DPA, EAGLE 5/1/6), AgentX C80, 3600 s, one A1 run each, **sim acceptance 3.61 - no
correctness claim**; score = total tok/s per GPU over 8 GPUs. One variable per run vs the stated reference.

## Results

| run | change (vs reference) | score | qps | ISL | TTFT p50/p90 s | TPOT ms | KV xfer p50/p90 s | verdict |
|---|---|---|---|---|---|---|---|---|
| llr7 | reference (previous pack-up) | 41,679 | 2.744 | 120.6k | 4.45 / 12.32 | 15.4 | 1.99 / 4.72 | |
| llr7r | llr7 replicate + torch profile / probes | 38,935 | 2.618 | 118.0k | 4.46 / 15.04 | 15.8 | 2.05 / 4.78 | perturbed by profiling (trace save stalled each PP stage ~45 s) |
| llr9c | llr7 + decode graph capture 11 sizes (`--cuda-graph-max-bs-decode 96 --cuda-graph-bs-decode 1 2 4 8 12 16 24 32 40 48 64 80 96`) | 37,990 | 2.587 | 116.5k | 4.14 / 12.29 | **19.1** | 1.78 / 4.29 | not kept: decode step p50 50.5 -> 60.1 ms (padding to the next captured size); per-rank capture capped at 64 = max_running 256 / dp 4; frees only ~0.9 GB |
| llr10 | llr7 + decode mem 0.93 | 41,453 | 2.735 | 120.3k | 4.24 / 11.29 | 15.6 | 1.96 / 4.65 | kept as base (score neutral, TTFT p90 -8 %); later NOT robust (llr17) |
| **llr11** | llr10 + prefill `--pp-async-batch-depth 1` | **42,506** | 2.797 | 120.6k | 3.65 / 9.52 | 16.1 | 1.54 / 3.85 | kept (+2.5 %; prefill PP0 1.74 -> 1.99 batch/s) |
| llr12 | llr11 + decode `--disaggregation-decode-enable-radix-cache` + router `--session-affinity both` | 40,814 | 2.712 | 119.4k | 3.78 / 13.69 | 16.4 | 1.37 / 3.72 | not kept: transfer MB p50 864 -> 45 (reuse works) but sticky sessions unbalance DP ranks (running 12.5/13.0/8.3/10.4) |
| llr13 | llr11 + decode `--stream-interval 30` | 42,892 | 2.821 | 120.7k | 3.64 / 9.90 | 15.7 | 1.58 / 4.04 | kept (+0.9 %, inside noise) |
| **llr14** | llr13 + decode `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8` | **43,043** | 2.829 | 120.8k | 3.47 / 9.04 | 15.8 | 1.51 / 3.65 | best single run (+0.35 %, inside noise) |
| llr14r | llr14 replicate | 42,366 | 2.791 | 120.5k | 3.71 / 10.06 | 16.0 | 1.53 / 4.03 | run-to-run spread >= 1.6 %; llr14 config mean of 2 = 42,705 |
| llr15 | llr14 with `--pp-async-batch-depth 2` | 42,411 | 2.797 | 120.4k | 4.05 / 10.64 | 15.7 | 1.48 / 3.84 | not kept (prefill_forward p50 1.62 -> 1.97 s) |
| llr16 | llr14 + prefill HiCache io `direct` / `page_first_direct` | - | - | - | - | - | - | stopped early: prefill PP0 0.21 batch/s, queue 38 |
| llr17 | llr14 + prefill chunk 8192 -> 4096 | - | - | - | - | - | - | VOID: decode HSA OUT_OF_RESOURCES 46 min in (mem 0.93, DP3 running 33, usage 0.98); guard stopped AgentX |
| llr17b | chunk 4096 with decode mem 0.92 | running at pack-up | | | | | | |

**Best:** llr14 config, 43,043 (single run) / 42,705 (mean with its replicate). Defensible gain over llr7 is about
+2-3 %, mostly `--pp-async-batch-depth 1`. llr13/llr14 increments are inside the measured >= 1.6 % spread.

## Analyses (all verified by an independent agent; `analysis/`)
- `analysis/llr7r/REPORT.md` - trace bottleneck report at the llr7 config: decode step 51 ms of which the verify graph
  <= 42 ms (kernels inside it unrecorded); decode KV usage >= 0.95 for 25-28 % of the time; prefill PP stages 71-76 % busy
  with 88-101 ms CPU-handshake gaps; RDMA mean 8 % of line rate, 0 retx; decode bs per rank uneven in the trace.
- `results/llr/llr7-c80.qseries.txt` and each run's `qseries.txt` - decode/prefill queue counters per 5-10 min bucket:
  from mid-run decode usage ~0.9, transfer-req rises, running falls, prefill batch rate drops.
- `analysis/kv_transfer_latency_data.md` - the ~1.5 s `kv_transfer_latency` does not depend on size (llr12 < 10 MB
  p50 1.04 s); ~0.2-0.6 s at low load, ~4 PP0 iterations (~430 ms) at C80; INFERENCE: release waits for the PP
  consensus round (prefill.py:1029 -> 1199). llr17/llr17b test this with chunk 4096.
- `analysis/audit_ours_vs_llr7.md` - R2 (`--decode-input-demand`) is llying's (Infera #188), ON in all llr runs; no
  "measured gain on our stack + absent from llr7 + applicable" item cleanly qualifies; tokenizer 4 / detok 1 in llr7.
- `analysis/coderead_r2_ec_kf.md` - R2 status; the other session's decode empty_cache patch ported (`analysis/ec_port/`,
  not run); KF indexer does not engage at verify rows <= 48 because the built-in gfx950 fused indexer serves them.
- `analysis/coderead_decode_radix_pp_overlap.md` - decode radix needs `--session-affinity both`; overlap schedule is
  forced off for pp > 1, `--pp-async-batch-depth` is the PP lever.
- `PLAN.priority.md` - ranking of the ATOM / Dynamo+SGLang ideas (Infera e186e8e5 REPORT.md P1-P28) against our data.

## Layout
- `REPRODUCE.md`, `notes.md`, `environment.md`.
- `scripts/` - site files llr7r, llr9*, llr10-llr17b; `run_llr.yihou.sh`; `guard_llr.yihou.sh` (v2; stops our AgentX
  if the decode stops batching or logs a fatal, armed only in the profiling phase); `prof_llr7r.yihou.sh` +
  `node_probe1s.yihou.sh` (profile/probe capture); `qseries.yihou.py`, `bneck.yihou.py`, `llsum.yihou.py`, `argdiff.yihou.py`.
- `results/llr/<run>/` - agentx json, runtime.env, engine-config.env, workers.json, server metrics (gz) + `hist.json`,
  decode/prefill key lines, router startup config, bneck, qseries, run/guard logs; `results/llr/summary.txt`.
- `analysis/probe_llr7r/` - 1 s RDMA/CPU/GPU probes and py-spy dumps (gz). Torch traces are NOT included (8.5 MB x 4
  decode, ~67 MB x 4 prefill): `/shared_nfs/yihou_glm52/llrepro_20261023/trace_llr7r.yihou/{decode_early,prefill_early}/`.
