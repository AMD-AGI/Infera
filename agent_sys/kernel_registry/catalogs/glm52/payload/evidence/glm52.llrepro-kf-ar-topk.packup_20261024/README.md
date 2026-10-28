> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# contributor 4+4 C80 base + KF decode kernels, decode custom AR, KF top-k (REDACTED_NODE -> REDACTED_NODE, 2026-10-24)

Follow-up to `registry/glm52.llrepro-pp4-dpa4.packup_20261024` (llr1..llr4: contributor's PP4 prefill + DPA4 decode reproduced
on our nodes, 37,392 vs his 37,310; index_share kept -> llr3 37,743). User order: (1) reproduce contributor, (2) our e2e options
one at a time, (3) the KF decode kernels. This pack-up is step (3) plus two performance-first options
(user 10-24: "custom-all-reduce开开，先不要在乎乱码和正确性的问题，先把性能压上去。再调正确性").

**Results (AgentX C80, 3600 s, one A1 run each, sim acceptance 3.61 - NO correctness claim; tok/s per GPU over 8 GPUs):**

| run | change vs previous row | score | qps | ISL mean | TTFT p50/p90 s | TPOT ms | KV xfer p50/p90 s | decode step ms p50 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| llr3 | reference (contributor kit + index_share) | 37,743 | 2.574 | 116.4k | 3.68 / 9.76 | 19.8 | 1.70 / 4.14 | 70.1 | base |
| **llr5** | + KF decode kernels (sparse MLA, indexer logits, MoE) | **40,777** | 2.707 | 119.6k | 4.22 / 10.84 | 16.5 | 1.97 / 4.68 | 55.3 | kept (+8.0 %, qps +5.2 %, ISL +2.8 %) |
| **llr6** | + decode custom all-reduce ON | **40,937** | 2.721 | 119.4k | 4.25 / 12.25 | 15.7 | 1.97 / 4.67 | 52.9 | kept (+0.4 %, TPOT -4.8 %) |
| **llr7** | + decode `SGLANG_OPT_USE_TOPK_V2=false` (KF indexer top-k engages) | **41,679** | 2.744 | 120.6k | 4.45 / 12.32 | 15.4 | 1.99 / 4.72 | 50.5 | kept (+1.8 %, qps +0.8 %, ISL +1.0 %) - best |
| llr8 | + decode `--mem-fraction-static 0.95` | VOID | | | | | | | decode aborted 04:55:27 (~7 min into profiling): `HSA_STATUS_ERROR_OUT_OF_RESOURCES ... Available Free mem : 0 MB`; after graph capture 2.22 GB free (llr7 10.46 GB); KV 3.04 M tokens/rank (llr7 2.86 M) |

llr3 -> llr7: +10.4 % score, +6.6 % qps, TPOT 19.8 -> 15.4 ms, with a 3.6 % longer mean ISL.
GPU / CPU prefix hit stayed 0.49 / 0.15 for llr5-7 (llr3 0.41 / 0.12). 0 Traceback / Memory access fault in all three.

## Where the time goes at llr7 (measured; `results/llr/bneck.*.txt`, `results/probe/`)
- Decode: GPU 99-100 % busy (rocm-smi, llr5), step p50 50.5 ms at ~10.5 running requests per rank; KV usage mean 0.79,
  max 1.00 (~15 % of log lines >= 0.95); 3.0 M KV tokens per rank, ~180k tokens per running request.
- As the decode got faster (llr3 -> llr7) running requests per rank fell 13.0 -> 10.5 while the upstream queues grew:
  prefill queue-req 1.2 -> 2.2, prefill pending-token mean 148k -> 289k, decode transfer-req per rank 3.0 -> 4.5,
  prealloc frac>0 0.17 -> 0.24.
- RDMA (llr6, 120 s average): rails 4-7 only, REDACTED_NODE tx 5.3 / 2.9 / 4.8 / 4.6 GB/s, 0 retransmits, 0 ACK timeouts. A
  120 s average cannot show burst saturation.
- CPU (llr6): REDACTED_NODE no core > 44 %; REDACTED_NODE one core 100 %, two > 80 % (one sglang thread 99.9 %, four python3 threads ~98 %).
  Whether that thread delays GPU launches is not measured.

## Addendum (2026-10-24 07:5xZ)
- llr8 VOID (above). Logs: `results/llr/llr8-c80/decode.abort.txt`, `decode.keylines.txt`, `decode.last_batches.txt`.
  The AgentX client kept running against the dead decode for 55 min (our patrol grepped only the last 5000 lines and
  missed it); decode mem stays 0.92 with custom AR + KF. Not yet tried: 0.95 with the other session's decode
  empty_cache patch, or 0.93.
- KF coverage at llr7 (`results/llr/llr7-c80.rowbuckets.txt`): in the steady window 33.3 % of decode log lines have
  TARGET_VERIFY rows <= 48 (running <= 8 per rank), where only `-> orig` was logged for indexer logits / top-k; 65.2 % are
  in 49-128 (KF), 1.6 % > 128. Cause for rows <= 48 not found yet (notes.md section 2).

## Layout
- `REPRODUCE.md` - commands. `notes.md` - KF engagement per shape, caveats, open questions. `environment.md`.
- `kit/sglang_2p1d_glm5.2/` - the kit exactly as run (previous pack-up's kit + decode hook `DECODE_CUSTOM_AR`).
- `kf_overlay/` - the KF kernel overlay mounted read-only at `REDACTED_LOCAL_PATH` on the decode (from the kernel session
  `historical-integration-branch`; source path and md5 in `kf_overlay/MD5SUMS.source`).
- `scripts/` - `llr5..llr8.site.env`, `run_llr.registry.sh`, hooks `kit_kfhook.registry.sh` / `kit_arhook.registry.sh`,
  `llsum.registry.py` (summary row), `bneck.registry.py` (steady-state queue / step stats), `node_probe.registry.sh`, run logs.
- `results/llr/<run>/` - agentx json, runtime.env, engine-config.env, workers.json, server metrics (gz), decode / prefill
  key lines (first 400, incl. the `[infera-kernels]` install / CHECK / per-shape lines); `results/llr/summary.txt`; `results/probe/llr6.*` (RDMA / CPU / GPU / top -H).
