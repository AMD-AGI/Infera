# GLM-5.2 PD 4+4 (135 -> 136): PP comm overlap (new best), chunk 4096, failed diagnostics, C88 (2026-10-25)

Follows `yihou/glm52.llrepro-atomdyn-tune.packup_20261025` (best then: llr14 config, 43,043 single / 42,705 mean).
Same nodes, image, kit (+ new `PREFILL_ENV_EXTRA` hook), KF overlay; AgentX 3600 s, sim acceptance 3.61 (**no
correctness claim**); one variable per run.

## Results

| run | change | score | qps | ISL | TTFT p50/p90 s | TPOT ms | KV xfer p50/p90 s | verdict |
|---|---|---|---|---|---|---|---|---|
| llr14 config | reference: 3 runs llr14 / llr14r / llr19 | 43,043 / 42,366 / 42,659 (mean 42,689) | 2.79-2.83 | 120.5-120.8k | 3.47-3.80 / 9.04-10.06 | 15.8-16.0 | 1.51-1.64 / 3.65-4.13 | base |
| llr17b | llr14 + prefill chunk 4096, decode mem 0.92 | 42,225 | 2.791 | 120.1k | 3.19 / 10.64 | 15.5 | 1.17 / 3.47 | not kept (prefill queue p90 3.6 s); latency -23 %, iteration period -37 % |
| llr18 | llr14 + `--disable-decode-cuda-graph` (diagnostic) | - | | | | | | FAILED at start-up: eager draft_extend `topk.hip:184 lengths must have one entry per score row`, then Memory access fault |
| llr19 | llr14 replicate + rocprofv3 `--attach` to decode DP0/DP2 | 42,659 | 2.806 | 120.6k | 3.80 / 9.89 | 15.9 | 1.64 / 4.13 | replicate; rocprofv3 reported success but wrote no output |
| llr20 / llr20b | llr14 + prefill `SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM=1` | - | | | | | | llr20: NFS quota (AgentX dataset copy); llr20b: prefill PP3 `AssertionError prefill.py:896` (incompatible with prefill MTP) |
| **llr21** | llr14 + prefill `SGLANG_PP_COMM_OVERLAP=1` | 42,983 (from jsonl) | - | 121.5k | 3.17 / 6.81 | - | - | agentx json lost to NFS quota; score recomputed with `jsonl_score.yihou.py` (reproduces llr14/llr19 exactly) |
| **llr21r** | llr21 replicate (WORK_DIR on 135 local disk) | **43,716** | 2.876 | 120.6k | 2.99 / 6.62 | 16.2 | 1.24 / 2.56 | best single run |
| llr22 | llr21 + `YIHOU_KF_FUSED_INDEXER=1` | 43,497 | 2.859 | 120.7k | 3.09 / 7.16 | 16.4 | 1.23 / 2.77 | flag is a NO-OP on this image (patch target module absent) -> 3rd llr21 run |
| llr23 | llr21 config at **C88** | 44,233 | 2.932 | 119.8k | 4.09 / 11.96 | 16.9 | 1.75 / 4.18 | +1.9 % vs C80 mean, TTFT p90 +75 %; decode KV mean 0.85, prealloc waits 0.33-0.36 of the time |

**Best at C80: llr21 config (llr14 + prefill `SGLANG_PP_COMM_OVERLAP=1`), 3 runs 42,983 / 43,716 / 43,497, mean
43,399** (+1.7 % vs llr14 config mean, +4.1 % vs llr7 41,679); TTFT p90 6.6-7.2 s vs 9.0-10.1 s (no overlap between the
two sets); prefill PP0 2.18 batch/s (2.09), decode transfer-req/rank 2.6 (3.3), prealloc frac 0.10-0.13 (0.15).

## Layout
- `results/llr/<run>/` - agentx json (where written), runtime.env, engine-config.env, workers.json, server metrics csv
  (gz) + hist.json, decode/prefill key lines, router startup, bneck, qseries, run/guard logs; `summary.txt`;
  `llr21-c80.jsonl_score.txt`.
- `scripts/` - site files llr17b-llr23; `run_llr.yihou.sh` (WORK_DIR may be node-local; compact results rsynced to NFS),
  `guard_llr.yihou.sh`, `prof_llr18/19.yihou.sh`, `jsonl_score.yihou.py`, analysis scripts, kit `worker.sh`/`common.sh`
  with the `PREFILL_ENV_EXTRA` hook.
- `notes.md`, `environment.md`. Reproduction: as in the previous pack-up's REPRODUCE.md, with these site files.
