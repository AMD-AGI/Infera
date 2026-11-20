# InferenceX: DeepSeek-V4.1-Flash (dsv41flash) AgentX entries, recipes, harness contract

Collected 2026-11-19 (research only, no GPU runs). Repo: `SemiAnalysisAI/InferenceX`
(the InferenceMAX repo moved there; `gh search repos InferenceX`). Shallow clone at
`source/InferenceX` = commit `db3fc79642b97a5321ff1c567ebfbd580642d4c8` (2026-11-18 22:28 -0800).
Published numbers come from the public API `https://inferencex.semianalysis.com/api/v1/...`
(OpenAPI saved at `notes/ix/openapi.json`). Raw pulls kept in `notes/ix/`:

- `bench_dsv41_agentic_20261119.json` (current published rows, 227 rows) + `table_all.txt` (flattened)
- `bench_dsv41_agentic_asof_{2026-10-10,10-20,11-01,11-15}.json` + `table_asof_*.txt` (historical snapshots)
- `rankings_{fastest,cheapest}_20261119.csv`, `reproduce_<rowid>.json` (exact launch commands), `pr_<n>.json` (PR bodies)
- `changelog_dsv41.yaml` (179 perf-changelog entries whose config-keys contain dsv41)
- Flattener script: `scripts/ix_table.py`

Evidence tags: [FH] = first-hand (file:line / API row id / PR# read by me), [2H] = second-hand
(a PR author's local, non-CI claim, quoted), [INF] = my inference.

Note on naming: in InferenceX, `dsv4*` keys = DeepSeek-V4-Pro; `dsv41flash*` keys = DeepSeek-V4.1-Flash.
Only `dsv41flash*` is covered below unless stated.

---

## 0. Decision-relevant summary

1. Metric: `tput_per_gpu` = (all input tokens incl. prefix-cache hits + output tokens) / benchmark
   duration / (prefill GPUs + decode GPUs). [FH `infx/results/agentic/request_metrics.py:250`,
   `infx/results/agentic/__init__.py:132,237-241`]. Output is <1% of it (MI355X vLLM c384: 3,378 of
   427,451 tok/s/GPU) [FH API row 448242]. The dashboard frontier is p90 interactivity
   (1/p90 ITL, tok/s/user) vs tok/s/GPU [FH `docs/agentx-standalone.md`].
2. Highest published DSv4.1-Flash AgentX tok/s/GPU on MI355X (any engine), 2026-11-19:
   **427,451** (vLLM, 2 GPUs, TP1xDP2 attention + TP MoE, UMBP host-DRAM KV offload, c384,
   p90 interactivity 15.8) [FH API row 448242, run 43861172833, 2026-11-18, PR #5036].
   It is colocated (single server), not PD.
3. No DSv4.1-Flash **vLLM PD** entry exists on MI355X: published, merged recipe, or open PR. The only
   MI355X PD entry is TileRT (vLLM TP4 prefill + TileRT TP8 decode over Mooncake), c1 only,
   2,829 tok/s/GPU [FH API row 447684, PR #4756]. Searched: InferenceX PRs (open+closed), API rows,
   `vllm-project/vllm` and `vllm-project/recipes` PR search (section 8).
4. All high-throughput PD entries are NVIDIA: GB300 Dynamo-SGLang 1,071,944 tok/s/GPU (34 GPUs,
   P9xDEP2 / D1xDEP16, c7480); GB300 Dynamo-vLLM 1,011,514 (40 GPUs, P12xDEP2 / D1xDEP16, c9888);
   B300 Dynamo-vLLM 828,209 (18 GPUs, P5xDEP2 / D1xDEP8, c3584) [FH API, section 2].
5. Leaderboard ranking at interactivity tier 125 (API `views/rankings`, kind fastest-gpu):
   GB300 1,048,780 (Dynamo SGLang, disagg) > B300 726,160 (Dynamo vLLM, disagg) > GB200 372,362 (vLLM)
   > B200 353,905 (vLLM) > **MI355X 150,079 (SGLang, colocated)** [FH `notes/ix/rankings_fastest_20261119.csv`].
   The interpolation the API uses for "tier 125" was not verified.
6. The dashboard replaces a whole curve on each normal run [FH openapi `/api/v1/benchmarks`
   description]. The current MI355X vLLM curve therefore shows only the 3 UMBP/DP2 points
   (c128/256/384); the TP2/TP4/TP8 c1-c64 points (dated 2026-11-09) are visible only in historical
   snapshots [FH API as-of 2026-11-15 vs current].

---

## 1. Current leaderboard top per vendor (DSv4.1-Flash, AgentX, `sequence=agentic-traces`)

Source: `GET /api/v1/benchmarks?model=DeepSeek-V4.1-Flash&sequence=agentic-traces`, fetched 2026-11-19.
All FP4 checkpoint (`deepseek-ai/DeepSeek-V4.1-Flash`), DSpark 5 draft tokens, synthetic AL 3.51 unless noted.
"Max tok/s/GPU" is the max over the curve, regardless of interactivity.

| Vendor | Best curve | Max tok/s/GPU | Point | p90 intvty | GPUs | Date | Run |
|---|---|---:|---|---:|---:|---|---|
| NVIDIA (any) | vr200 dynamo-vllm PD | 2,993,171 | P8x(TP2 DEP2) + D1x(DEP16), c15800, DRAM offload | 180.8 | 32 | 2026-11-13 | NVIDIA/nv-perf-sa-review run 43515830826 |
| NVIDIA (GB300) | dynamo-sglang PD | 1,071,944 | P9x(DEP2) + D1x(DEP16), c7480 | 105.6 | 34 | 2026-11-13 | IX run 43029773811 |
| NVIDIA (GB300) | dynamo-vllm PD | 1,011,514 | P12x(DEP2) + D1x(DEP16), c9888 | 75.5 | 40 | 2026-10-20 | IX run 39207089292 |
| NVIDIA (GB300) | vllm colocated | 863,068 | TP4 DPA, c1024 | 26.1 | 4 | 2026-11-10 | IX run 42241164555 |
| NVIDIA (B300) | dynamo-vllm PD | 828,209 | P5x(DEP2) + D1x(DEP8), c3584 | 83.8 | 18 | 2026-11-17 | IX run 43769338888 |
| NVIDIA (B300) | vllm colocated | 625,966 | TP8 DPA (DEP8), c1280 | 35.9 | 8 | 2026-11-10 | IX run 42108214901 |
| AMD (MI355X) | vllm colocated | 427,451 | TP1xDP2 attn, TP MoE, UMBP, c384 | 15.8 | 2 | 2026-11-18 | IX run 43861172833 |
| AMD (MI355X) | atom colocated | 389,350 | TP2 DEP2 DPA, c256 | 37.8 | 2 | 2026-11-13 | IX run 43180092301 |
| AMD (MI355X) | sglang colocated | 181,052 | TP2 DP2-attn, c64 | 73.3 | 2 | 2026-11-17 | IX run 43344632854 |
| AMD (MI355X) | tilert PD | 2,829 | P1x TP4 (vLLM) + D1x TP8 (TileRT), c1 | 1333.3 | 12 | 2026-11-07 | IX run 42340993534 |
| AMD (MI325X) | vllm | 50,733 | TP2, c32 | 23.9 | 2 | 2026-11-11 | IX run 40744174186 |
| AMD (MI300X) | vllm | 28,308 | TP4, c32 | 25.8 | 4 | 2026-11-06 | IX run 41950365710 |

[FH all rows: `notes/ix/table_all.txt`; run URLs `https://github.com/SemiAnalysisAI/InferenceX/actions/runs/<id>`.]
The vr200 row's run URL points to `NVIDIA/nv-perf-sa-review`, not the InferenceX repo [FH API].
Whether it is an "official" leaderboard entry was not checked; it is not in the tier-125 ranking CSV.

Ranking at interactivity tier 125 (fastest-gpu): GB300 1,048,780 Dynamo-SGLang disagg; B300 726,160
Dynamo-vLLM disagg; GB200 372,362 vLLM; B200 353,905 vLLM; MI355X 150,079 SGLang. Cheapest-gpu
($/M total tokens): GB300 0.000612; B300 0.000865; B200 0.00136; GB200 0.00139; MI355X 0.00278
[FH `notes/ix/rankings_*_20261119.csv`].

---

## 2. Every DSv4.1-Flash PD (disagg) number published

Source: current API rows with `disagg=true` [FH `notes/ix/table_all.txt`]. All DSpark-5 / AL 3.51,
FP4, KV offload to DRAM on all NVIDIA Dynamo PD points. Score = tok/s/GPU over P+D GPUs.
TTFT columns in seconds.

| HW | Engine (image) | P workers x shape | D workers x shape | GPUs | Conc | tok/s/GPU | p90 intvty | TTFT p50 / p90 | Date |
|---|---|---|---|---:|---:|---:|---:|---|---|
| vr200 | dynamo-vllm (`vllm-cu134-vr1111f-b1`) | 8 x TP2 DEP2 | 1 x DEP16 | 32 | 15800 | 2,993,171 | 180.8 | 1.8 / 3.8 | 11-13 |
| vr200 | dynamo-vllm | 6 x DEP2 | 1 x DEP16 | 28 | 11500 | 2,632,993 | 259.7 | 1.8 / 4.3 | 11-13 |
| vr200 | dynamo-vllm | 6 x DEP2 | 1 x DEP32 | 44 | 11000 | 1,695,282 | 411.5 | 1.7 / 4.1 | 11-13 |
| vr200 | dynamo-vllm | 3 x DEP4 | 1 x DEP32 | 44 | 6600 | 1,097,953 | 561.8 | 0.6 / 1.4 | 11-13 |
| vr200 | dynamo-vllm | 1 x DEP2 | 3 x TP4 | 14 | 310 | 167,331 | 840.3 | 0.8 / 3.4 | 11-13 |
| vr200 | dynamo-vllm | 1 x DEP2 | 4 x TP4 | 18 | 260 | 112,598 | 1030.9 | 0.6 / 2.4 | 11-13 |
| gb300 | dynamo-sglang (`lmsysorg/sglang:nightly-dev-cu13-20261111-314bce7c`) | 9 x DEP2 | 1 x DEP16 | 34 | 7480 | 1,071,944 | 105.6 | 4.2 / 12.3 | 11-13 |
| gb300 | dynamo-sglang | 7 x DEP2 | 1 x DEP16 | 30 | 5540 | 992,956 | 161.8 | 3.8 / 11.1 | 11-13 |
| gb300 | dynamo-sglang | 6 x DEP2 | 1 x DEP32 | 44 | 3018 | 474,138 | 321.5 | 0.7 / 2.3 | 11-13 |
| gb300 | dynamo-vllm (`vllm/vllm-openai:cu134-nightly-aarch64-dev-20261019`) | 12 x DEP2 | 1 x DEP16 | 40 | 9888 | 1,011,514 | 75.5 | 6.9 / 12.3 | 10-20 |
| gb300 | dynamo-vllm | 9 x DEP2 | 1 x DEP16 | 34 | 7480 | 993,521 | 145.6 | 9.0 / 17.8 | 10-20 |
| gb300 | dynamo-vllm | 16 x DEP2 | 1 x DEP32 | 64 | 10888 | 954,146 | 196.5 | 3.3 / 8.0 | 10-20 |
| gb300 | dynamo-vllm | 6 x DEP2 | 1 x DEP32 | 44 | 3018 | 461,635 | 331.1 | 1.5 / 3.4 | 10-20 |
| gb200 | dynamo-vllm (`...aarch64-01fba62-1111d`) | 5 x DEP4 | 1 x DEP16 | 36 | 5600 | 817,333 | 125.6 | 3.0 / 6.9 | 11-13 |
| gb200 | dynamo-vllm | 4 x DEP4 | 1 x DEP16 | 32 | 4096 | 761,086 | 191.9 | 2.5 / 5.3 | 11-13 |
| gb200 | dynamo-vllm | 3 x DEP4 | 1 x DEP16 | 28 | 3018 | 662,528 | 229.9 | 2.7 / 5.3 | 11-13 |
| gb200 | dynamo-vllm | 3 x DEP4 | 1 x DEP32 | 44 | 3018 | 444,467 | 327.9 | 2.5 / 4.9 | 11-13 |
| b300 | dynamo-vllm (`vllm/vllm-openai:cu134-nightly-96710245...`) | 5 x DEP2 | 1 x DEP8 | 18 | 3584 | 828,209 | 83.8 | 8.2 / 13.2 | 11-17 |
| b300 | dynamo-vllm | 4 x DEP2 | 1 x DEP8 | 16 | 2048 | 738,119 | 120.5 | 0.9 / 2.7 | 11-17 |
| b300 | dynamo-vllm | 3 x DEP2 | 1 x DEP8 | 14 | 1280 | 576,607 | 175.4 | 0.8 / 2.0 | 11-17 |
| b300 | dynamo-vllm | 2 x DEP2 | 1 x DEP8 | 12 | 768 | 420,185 | 214.1 | 0.6 / 1.6 | 11-17 |
| b300 | dynamo-vllm | 2 x DEP2 | 1 x DEP8 | 12 | 256 | 154,022 | 274.7 | 0.2 / 0.6 | 11-17 |
| gb300 | tilert (vLLM prefill + TileRT decode) | 1 x TP4 | 1 x TP8 | 12 | 1 | 2,840 | 1515.2 | 0.4 / 0.8 | 11-11 |
| gb300 | tilert | 1 x TP4 | 1 x TP24 | 28 | 1 | 1,237 | 1724.1 | 0.4 / 0.8 | 11-11 |
| mi355x | tilert (`ghcr.io/tile-ai/tilert-rocm-decode:0.1.6`, prefill `tilert-rocm-prefill:0.1.7`) | 1 x TP4 | 1 x TP8 | 12 | 1 | 2,829 | 1333.3 | 0.3 / 0.5 | 11-07 |

Pattern visible in all NVIDIA PD rows [FH]: many small DEP2 (or DEP4) prefill workers, one wide DEP
decode (DEP8/16/32), concurrency in the thousands, prefill host-DRAM offload. On B300 the PD curve's
best point (828k, 18 GPUs) is 1.32x the colocated vLLM best (626k, 8 GPUs) [FH]. No claim about why.

B300 vLLM PD recipe (as committed, `benchmarks/multi_node/srt-slurm-recipes/dsv41flash/vllm/b300-fp4/agentx/disagg-variants.yaml`) [FH]:
- Dynamo 1.6.0.dev20261010 frontend, `router-mode kv`, session-affinity TTL 14400 s; fastokens tokenizer.
- Prefill (per worker, 2 GPUs): TP1 DP2 + EP, `MultiConnector[NixlConnector(push, LIBFABRIC) + SimpleCPUOffloadConnector(75 GB, lazy)]`,
  `kv-cache-dtype fp8`, `attention-config {"backend":"FLASHMLA_MEGA_ATTN_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}`,
  `engram-config {"cpu_offload":true,"embedding_across_dp":true}`, `moe_backend deep_gemm_mega_moe`, `prefill-schedule-interval 2`,
  `max-num-batched-tokens 16384`, `long-prefill-token-threshold 8192`, `max-num-seqs 64`, `prefix-cache-retention-interval 32768`.
- Decode (8 GPUs): TP1 DP8 + EP, NixlConnector `kv_consumer`, EPLB (16 redundant experts), `no-enable-prefix-caching`,
  `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP=4`; per-point `max-num-seqs` / FULL_DECODE_ONLY graph sizes in overrides.
- The DSpark spec-config is identical on both roles.

---

## 3. MI355X DSv4.1-Flash: all published numbers (colocated unless stated)

### 3a. vLLM, current published curve (2026-11-18, PR #5036, image `vllm/vllm-openai-rocm:nightly-60bd6dd2edd175a1c55fd3488f871e4c3d2ef27d`) [FH API rows 448240-448242]

| Topology (2 GPUs) | KV offload | Conc | tok/s/GPU | output tok/s/GPU | p90 intvty | TTFT p50 / p90 (s) |
|---|---|---:|---:|---:|---:|---|
| TP1 x DP2 attn, TP MoE, fp8_e4m3 KV | UMBP DRAM 449 GB | 384 | 427,451 | 3,378 | 15.8 | 1.6 / 3.7 |
| same | UMBP | 256 | 401,896 | 2,956 | 23.7 | 1.0 / 2.7 |
| same | none | 128 | 271,329 | 2,233 | 45.1 | 0.6 / 1.5 |

c384 row details [FH row 448242 / server-log API]: 27,066 requests in 3630 s; mean input 113,745 tok,
mean output 906 tok; GPU prefix hit 88.3%, CPU/UMBP hit 5.7%, theoretical 96.4%; GPU KV usage 85.8%.

### 3b. vLLM historical curves (as-of snapshots; superseded on the dashboard) [FH `notes/ix/table_asof_*.txt`]

| Date (image) | TP2 c64 | TP2 c32 | TP2 c16 | TP2 c8 | TP2 c1 | TP4 c64 | TP4 c1 | TP8 c1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-10-02 (`nightly-rocm100-ac9126e5`) | 128,237 (p90i 31.5) | 95,826 | 54,107 | 29,243 | 11,255 | 83,732 | 5,976 | - |
| 2026-10-20 (`nightly-e68efc68`) | 184,835 (64.4) | 113,560 | 57,398 | 31,500 | 14,023 | 109,120 (104.3) | 7,649 | - |
| 2026-10-27 (`nightly-e68efc68`, PR #4480 caps) | 186,280 (66.8) | 110,462 | 57,244 | 31,601 | 13,465 | - | 7,399 | - |
| 2026-11-09 (`nightly-4fab5a07`, PR #4876) | 186,109 (62.7) | 112,238 (186.9) | 58,279 (302.1) | 32,274 (452.5) | 13,136 (452.5) | - | 7,917 (675.7) | 4,016 (704.2) |

(p90 interactivity in parentheses where shown.) Row ids for the 11-09 curve: 447797-447806.

### 3c. ATOM (2026-11-13, `rocm/atom-dev:nightly_202611101510`, run 43180092301) [FH API]

| Topology | Conc | tok/s/GPU | p90 intvty | TTFT p50 / p90 |
|---|---:|---:|---:|---|
| TP2 DEP2 DPA (2 GPUs) | 256 | 389,350 | 37.8 | 5.8 / 19.2 |
| TP2 DEP2 DPA | 128 | 264,138 | 52.1 | 0.9 / 4.6 |
| TP2 DPA EP1 | 64 | 195,738 | 90.8 | 0.7 / 2.8 |
| TP2 | 32 | 114,016 | 155.3 | 0.6 / 2.4 |
| TP4 | 32 | 62,395 | 209.2 | 0.2 / 0.6 |
| TP2 | 16 / 2 / 1 | 57,815 / 15,486 / 15,105 | 246.3 / 471.7 / 561.8 | - |
| TP4 | 8 / 4 / 2 / 1 | 17,175 / 10,341 / 8,150 / 7,994 | 505-694 | - |

ATOM c256 row: mean input 129,455, GPU prefix hit 94.3%, no KV offload [FH row 448099].

### 3d. SGLang (2026-11-17, `lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261111`, run 43344632854) [FH API]

| Topology | Conc | tok/s/GPU | p90 intvty | TTFT p50 / p90 |
|---|---:|---:|---:|---|
| TP2 DP2-attn | 64 | 181,052 | 73.3 | 0.8 / 2.4 |
| TP2 | 32 | 110,904 | 197.6 | 0.5 / 10.2 |
| TP2 DP2-attn | 32 | 110,744 | 131.9 | 0.6 / 1.7 |
| TP2 | 16 / 8 / 4 / 2 / 1 | 57,345 / 32,240 / 19,130 / 15,060 / 15,083 | 336.7 / 387.6 / 606.1 / 485.4 / 606.1 | - |
| TP4 | 16 / 8 / 4 / 2 / 1 | 29,846 / 17,200 / 10,343 / 8,493 / 8,015 | 434.8-775.2 | - |

### 3e. TileRT PD (2026-11-07): see section 2 (c1, 2,829 tok/s/GPU, 12 GPUs).

---

## 4. Exact MI355X recipes (command lines as launched by CI)

Source: `GET /api/v1/reproduce?id=<row>` (saved `notes/ix/reproduce_<id>.json`) and recipe files.
CI cluster: `cluster:mi355x-amds`, nodes `mia1-p01-g50` etc., CPU AMD EPYC 9575F 64-core, srtctl
(NVIDIA/srt-slurm) single-node launcher [FH reproduce JSON]. Model weights resolved as HF
`deepseek-ai/DeepSeek-V4.1-Flash` (TileRT snapshot hash `314113f8a7dadb388215396ba5fbee2c610a04d6`).

### 4a. vLLM DP2 + UMBP, c384 (current MI355X best, row 448242) [FH]

Image `vllm/vllm-openai-rocm:nightly-60bd6dd2edd175a1c55fd3488f871e4c3d2ef27d@sha256:67a3a0fb79893fc5a0fcff5f9b739ba8fec7e920875a12e0574c2e9f50cd814c`.

```bash
# service (starts before workers; /dev/shm mounted at /umbp-host in every step)
UMBP_STANDALONE_ADDRESS=unix:///umbp-host/umbp/standalone.grpc.sock UMBP_DRAM_CAPACITY=449000000000 \
UMBP_DRAM_USE_HUGEPAGES=0 UMBP_SSD_ENABLED=0 MORI_UMBP_LOG_LEVEL=INFO \
bash -c 'P=$(python3 -c "import mori, os; print(os.path.dirname(mori.__file__))") && LD_LIBRARY_PATH=$P:$LD_LIBRARY_PATH exec $P/umbp_standalone_server unix:///umbp-host/umbp/standalone.grpc.sock'

# engine
AITER_TRITON_LOG_LEVEL=ERROR OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE \
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=True \
VLLM_USE_BREAKABLE_CUDAGRAPH=1 VLLM_USE_RUST_FRONTEND=1 \
vllm serve deepseek-ai/DeepSeek-V4.1-Flash --host 0.0.0.0 --port 6100 --device-ids 0,1 \
  --attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' --block-size 128 \
  --data-parallel-size 2 --tensor-parallel-size 1 --disable-uvicorn-access-log --enable-auto-tool-choice \
  --engram-config '{"cpu_offload":true}' --gpu-memory-utilization 0.9 --kv-cache-dtype fp8_e4m3 \
  --kv-transfer-config '{"kv_connector":"UMBPStoreConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"mode":"standalone","endpoint":"unix:///umbp-host/umbp/standalone.grpc.sock"}}' \
  --language-model-only --max-cudagraph-capture-size 1024 --max-model-len 1048576 --max-num-batched-tokens 16384 \
  --moe-backend aiter --prefill-schedule-interval 11 --reasoning-parser deepseek_v41 \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 5, "draft_sample_method": "probabilistic", "rejection_sample_method": "synthetic", "enable_adaptive_verification": false, "synthetic_acceptance_length": 3.51}' \
  --tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41

# frontend
vllm-router --worker-urls http://<host>:6100 --host 0.0.0.0 --port 8000 --intra-node-data-parallel-size 2 \
  --worker-startup-timeout-secs 3600 --policy consistent_hash --prometheus-host 127.0.0.1 --prometheus-port 18000 \
  --request-timeout-secs 14400 --disable-retries      # vllm-router 0.1.14

# client
agentx profile --url http://<host>:8000 --model deepseek-ai/DeepSeek-V4.1-Flash --tokenizer deepseek-ai/DeepSeek-V4.1-Flash \
  --concurrency 384 --benchmark-duration 3600 --tokenizer-trust-remote-code \
  --server-metrics http://<host>:6100/metrics http://<host>:9405/metrics \
  --output-artifact-dir /logs/agentic/aiperf_artifacts --public-dataset semianalysis_cc_traces_weka_062126
```
c256 = same with `--concurrency 256`; c128 = same without UMBP service and without `--kv-transfer-config`
[FH `.../dsv41flash/vllm/mi355x-fp4-mtp/agentic.yaml` overrides `override_dp2_*`].

### 4b. vLLM TP2 c64 (pre-UMBP curve, row 447802, 2026-11-09) [FH]

Image `vllm/vllm-openai-rocm:nightly-4fab5a07475822c6eaa971dceb2e2220d0446808@sha256:6f126f27...`. Same env as 4a.
```bash
vllm serve deepseek-ai/DeepSeek-V4.1-Flash --host 0.0.0.0 --port 8000 --device-ids 0,1 \
  --attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' --block-size 128 \
  --disable-uvicorn-access-log --enable-auto-tool-choice --engram-config '{"cpu_offload":true}' \
  --gpu-memory-utilization 0.9 --language-model-only --max-cudagraph-capture-size 1024 --max-model-len 1048576 \
  --max-num-batched-tokens 16384 --moe-backend aiter --monokernel --reasoning-parser deepseek_v41 \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 5, "draft_sample_method": "probabilistic", "rejection_sample_method": "synthetic", "enable_adaptive_verification": false, "synthetic_acceptance_length": 3.51}' \
  --tensor-parallel-size 2 --tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41
```
Per-point overrides in `agentic.yaml` [FH]: TP2 c1-c4 `max-num-seqs 1, max-cudagraph-capture-size 8,
max-num-batched-tokens 32768`; TP2 c8 `seqs 2, capture 16, mbt 8192`; TP2 c16 `seqs 4, mbt 8192`;
TP2 c32 `seqs 8, mbt 8192`; TP4/TP8 c1 `engram cpu_offload false`. `--monokernel` covers decode steps
up to 48 rows (8 seqs x 6 tokens) [FH changelog PR #3959/#4319].

### 4c. ATOM TP2 DEP2 DPA c256 (row 448099) [FH]
```bash
AITER_LOG_LEVEL=WARNING ATOM_DISABLE_MMAP=true ATOM_DP_LB_REQ_EQUIV=512 ATOM_DP_SESSION_AFFINITY=1 \
ATOM_ENABLE_PREFILL_DELAYER=1 ATOM_NUMA_BIND=0 ATOM_PREFILL_DECODE_INTERVAL=10 GPU_MAX_HW_QUEUES=5 OMP_NUM_THREADS=4 \
python3 -m atom.entrypoints.openai_server --model deepseek-ai/DeepSeek-V4.1-Flash --host 0.0.0.0 --server-port 6100 -tp 2 \
  --attn-prefill-chunk-size 16384 --block-size 16 --cudagraph-capture-sizes '[1,...,32,48,64,128,256]' --cudagraph-mode FULL \
  --dp-load-balance least_tokens --enable-dp-attention --enable-expert-parallel --enable_prefix_caching \
  --gpu-memory-utilization 0.9 --index-cache-dtype fp4 --kv_cache_dtype fp8 --level 3 --max-num-batched-tokens 16384 \
  --method dspark --num-speculative-tokens 5 --spec-decode-acceptance-length 3.51 \
  --state-checkpoint-interval-tokens 8192 --timeout-keep-alive 900 --tool-call-parser dsml_v41 --trust-remote-code
atomesh launch --worker-urls http://<host>:6100 --host 0.0.0.0 --port 8000 --backend atom
```

### 4d. SGLang TP2 DP2-attn c64 (row 448180) [FH]
Env: `GPU_MAX_HW_QUEUES=5 ROCM_QUICK_REDUCE_QUANTIZATION=INT8 SGLANG_AITER_CAPTURE_COPY_IN=1 SGLANG_DEFAULT_THINKING=1
SGLANG_DP_USE_GATHERV=1 SGLANG_DP_USE_REDUCE_SCATTER=1 SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank SGLANG_DSV41_REASONING_EFFORT=high
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 SGLANG_HIP_DSV41_INDEX_K_PAGE8=1 SGLANG_HIP_FP4_INDEXER_ROWGROUP=1 SGLANG_OPT_HIP_FLYDSL_MXFP8=1
SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=true SGLANG_SHARED_EXPERT_TP1=1 SGLANG_SIMULATE_ACC_LEN=3.51 SGLANG_SIMULATE_ACC_METHOD=match-expected
SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token SGLANG_USE_AITER=1 TORCH_BLAS_PREFER_HIPBLASLT=1 TRITON_HIP_USE_ASYNC_COPY=0`
```bash
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V4.1-Flash --chunked-prefill-size 32768 \
  --cuda-graph-backend-prefill disabled --cuda-graph-max-bs-decode 64 --data-parallel-size 2 --enable-decoder-swa-bounded-replay \
  --enable-dp-attention --enable-dp-attention-local-control-broadcast --enable-dp-lm-head --enable-metrics --enable-prefill-delayer \
  --enforce-shared-experts-fusion --fp8-gemm-backend aiter --load-balance-method total_requests --max-running-requests 64 \
  --mem-fraction-static 0.85 --prefill-decode-interval 16 --prefill-delayer-token-usage-low-watermark 0.7 --reasoning-parser auto \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 --swa-prefix-tails 4096 --tensor-parallel-size 2 \
  --tokenizer-worker-num 2 --tool-call-parser auto --trust-remote-code --watchdog-timeout 3600
python -m sglang_router.launch_router --worker-urls http://<host>:6100 --port 8000 --policy cache_aware --dp-aware \
  --request-id-headers x-correlation-id --request-timeout-secs 14400 --disable-health-check --disable-retries
```

### 4e. TileRT PD c1 on MI355X (row 447684) [FH]
Prefill: vLLM TP4 in `ghcr.io/tile-ai/tilert-rocm-prefill:0.1.7` (vLLM nightly e68efc68 + Mooncake), same vLLM flags as
4b plus `--compilation-config {"cudagraph_capture_sizes":[1..8192]}`, `--max-num-seqs 1`, `--engram-config {"cpu_offload":false}`,
`--kv-transfer-config {"kv_connector":"DsV41TileRTConnector","kv_connector_module_path":"tilert.pd_vllm.dsv41_prefill_connector","kv_role":"kv_producer",...,"tilert_transport":"mooncake"}`,
env `MOONCAKE_DISABLE_HIP_DMABUF=0 RDMAV_FORK_SAFE=1 NCCL_IB_HCA=rdma0..rdma7`. Decode: `python -m tilert.pd_vllm.decode_server --engine tilert --model dsv41 --transport mooncake --with-mtp --num-mtp 5` (TP8, `TILERT_SIMULATE_ACC_LEN=3.51`). Router: `tilert.pd_vllm.pd_router` with a custom chat template (`configs/dsv41flash-tilert-chat-template.jinja`).
Full recipe: `benchmarks/multi_node/srt-slurm-recipes/dsv41flash/tilert/mi355x-fp4/agentx/disagg-1p1d-tp4-tp8-mtp.yaml`.

---

## 5. AgentX harness contract (how to run it ourselves)

All [FH] unless marked.
- Client: `agentx-harness==1.0.14` (PyPI, Python 3.11-3.13), command `agentx profile` (an AIPerf preset)
  [`docs/agentx-standalone.md`, `infx/bench/agentic/requirements.txt`]. CI argv built in
  `infx/bench/agentic/replay.py:replay_argv` (example of a CI argv in 4a).
- Dataset: `--public-dataset semianalysis_cc_traces_weka_062126` (HF `semianalysisai/cc-traces-weka-062126`,
  full/uncapped). dsv41flash is not in `CAPPED_FAMILIES` (only dsr1, qwen3.5, qwen3.8next use the 256k corpus)
  [`infx/bench/agentic/traces.py`]. Observed workload on MI355X rows: mean input 114k-134k tokens/request,
  mean output ~0.9-1.0k, theoretical prefix hit 96-97% [API rows 448242/448180/448099].
- Duration 3600 s per point (`DEFAULT_AGENTIC_DURATION_SECONDS`, `infx/matrix/validation.py:21`); <900 s requires
  `--unsafe-override`; "fast" mode = 1200 s with 1 warmup request per lane (`replay.py:24-26`).
- Warmup: the agentx preset supplies 10 warmup requests per lane (standalone doc); older changelog entries
  say 5 per lane for ATOM [`perf-changelog` PR #3387/#3603]. Which applies to a given CI run: read its aiperf.log.
- Random seed 42 (preset default). Restart server per concurrency point, fresh output dir, do not reuse a live
  server across points.
- Optional session affinity for DP routers: `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true`
  (adds `X-Dynamo-Session-ID`), `AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true`.
- `--server-metrics <url>/metrics`; vLLM recipes require prefix `vllm:` (`AIPERF_REQUIRED_SERVER_METRIC_PREFIX`).
- Speculative decoding for throughput runs: synthetic acceptance from the golden table; evals use real verification.
  For vLLM the connector rewrites the spec-config to `rejection_sample_method: "synthetic", synthetic_acceptance_length: <AL>`
  (`infx/srt_slurm/synthetic_acceptance.py:289-322`); SGLang uses `SGLANG_SIMULATE_ACC_LEN/METHOD/TOKEN_MODE`;
  ATOM `--spec-decode-acceptance-length`; TileRT `TILERT_SIMULATE_ACC_LEN`.
- Golden AL table for DSv4.1-Flash DSpark (probabilistic draft, block rejection), measured on SPEED-Bench
  qualitative coding, vLLM `deepseekv41-flash-0909` TP4, temp 1.0, output 4096
  [`infx/golden_al_distribution/deepseek-v4.1-flash__dspark__native__probabilistic-block.yaml`, source run 34493175056]:

  | num_speculative_tokens | 1 | 2 | 3 | 4 | 5 |
  |---|---:|---:|---:|---:|---:|
  | thinking_on (used for AgentX) | 1.81 | 2.43 | 2.90 | 3.26 | **3.51** |
  | thinking_off | 1.88 | 2.62 | 3.27 | 3.73 | 4.07 |

  Draft lengths 6-8 were rejected by that image (no AL measured) [same file].
- Score/metrics: `total_tput_tps = (total_input + total_output) / duration` (`infx/results/agentic/request_metrics.py:250`), divided by
  P+D GPU count. Interactivity = 1 / p90 ITL (dashboard x-axis). Evals: GSM8K run at one concurrency per key
  (e.g. c384 for the UMBP key); thresholds in `infx/evals/thresholds.yaml` (not read in detail).
- Published-curve snapshot tool: `uv run --locked python -m infx.results.published_curve --model DeepSeek-V4.1-Flash ...`
  (binds each row to its producing commit/recipe; executes repo code) [`docs/agentx-standalone.md`].
- Reproduce/logs API: `/api/v1/reproduce?id=<row>`, `/api/v1/server-log?id=<row>&file=<path>`,
  `/api/v1/server-log-search?id=<row>&q=<text>` (used above).
- `dram-utilization` in master config sizes `TOTAL_CPU_DRAM_GB` (0.60 -> 449 GB for the 2-GPU share in the UMBP key) [`amd-master.yaml`, PR #5036].

---

## 6. MI355X vLLM: changes that moved the curve (from perf-changelog / PR bodies)

All [FH `notes/ix/changelog_dsv41.yaml`] unless marked. Chronological.
- PR #2962: first MI355X vLLM draft (TP4, image `deepseekv41-flash-0909`); `VLLM_USE_BREAKABLE_CUDAGRAPH=1` required
  (model not torch-compiled; FULL_AND_PIECEWISE aborted). Adaptive verification must be off on ROCm (indexer backend
  cannot trim on device).
- PR #3058: `--moe-backend aiter` (CK a8w4 experts) instead of `aiter_triton_mxfp4_bf16`: MoE GEMM 2.72->2.30 ms/step c1,
  9.06->7.18 at c16 (131k ctx); c1 ITL 4.01->3.91 ms [2H: numbers from the PR author's traces].
- PR #3326: Engram tables cost 47.2 GiB/rank at TP4 and 94.4 GiB/rank at TP2 when resident; host offload makes TP2
  feasible. `--no-swa-bounded-replay` was needed then (ROCm prefill kernels did not clamp at replay_start ->
  HSA memory fault). Indexer buffers scale ~4.4 MiB per batched token.
- PR #3571: `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=True` (Gluon sparse MLA), MXFP4 sparse-logits indexer + block 128
  (+14.7% p50 interactivity, throughput unchanged [2H]); c128 dropped (TP2 c64 dominated).
- PR #3690/#3959/#4319: mono decode (`--monokernel`, <=48 rows) on all 40 layers and the 3 DSpark draft layers;
  decoder SWA bounded replay on (vllm#59039). Prefix-cache SWA replay still forced off on ROCm.
- PR #4409/#4480: per-point running caps (`max-num-seqs`) to raise p90 interactivity; TP4 kept only at c1/c2.
- PR #4666: TP8 c1 point (vllm#64101 mono decode at TP8); QuickReduce INT4 turned off (`NONE`).
- PR #5036 (merged 2026-11-18): DP2 attention + TP MoE, `prefill-schedule-interval 11`, no monokernel, fp8_e4m3 KV
  (vllm#66036), UMBP (vllm#64715) at c256/c384. Author's local fast-run (1200 s, cold cache) table [2H]:
  c384 no-offload 143,888 -> UMBP 291,619 tok/s/GPU; c256 250,435 -> 275,836; c128 193,099 -> 183,984;
  c544 46,641 -> 287,025 (not in PR). CI then measured 427,451 / 401,896 / 271,329 [FH API]. UMBP GSM8K
  0.9212 baseline vs 0.9333/0.9295 [2H, copied from vllm#64715].

## 7. Open InferenceX PRs touching DSv4.1-Flash (as of 2026-11-19) [FH `gh search prs`]

| PR | Author | Updated | What | Perf claims |
|---|---|---|---|---|
| #5125 | fanxingran | 11-19 | MI355X vLLM TP4 c1-c48 with running caps, image `nightly-rocm100-8ca76632` | local 900 s, interactivity only: TP4 c16 576 vs official TP2 302 p90 [2H] |
| #5098 | Fangzhou-Ai | 11-17 | MI355X vLLM DEP2 (AITER MegaMoE over MoRI) c128-384 + UMBP | none measured; earlier local work found MegaMoE ~2x slower/layer at 2 ranks [2H] |
| #5056 | Fangzhou-Ai | 11-16 | MI355X vLLM DEP8 (TP1xDP8+EP, MegaMoE/MoRI) c256/512/1024, fp8_e4m3 KV | local 1 h c256: 174.3k-182.7k tok/s/GPU, p90i 120.6-131.8 [2H] |
| #4873 | Fangzhou-Ai | 11-13 | MI355X vLLM DP2 with cross-DP prefill delayer (needs vllm#65655, unmerged) | none in body [2H] |
| #5096, #5034, #5062 | AMD (xiaofei-zheng, limou102, lishuoshuo-amd) | 11-13..11-19 | MI355X SGLang: fastokens, DP2 up to c128/c256, monokernel at TP2 c32 | not read |
| #5012 | yhl-amd | 11-13 | MI355X ATOM TP4 c1 on 1111 nightly | not read |
| #5014, #5046, #5058 | Klaud-Cold / AMD | 11-13..11-19 | MI325X/MI300X vLLM image bumps, SWA bounded replay | not read |
| #5015, #5135 | Juntian777 | 11-19 | GB300 Dynamo-vLLM: vLLM main 01fba62, TP4 specialized decode, 5P/7P/12P disagg | not read |
| #4991 | jh-nv | 11-16 | GB300 Dynamo + TensorRT-LLM disagg | not read |
| #5065 | weireweire | 11-18 | GB300 Dynamo-SGLang megakernel agg c4/c8/c32 | not read |
| #5093/#5094 | TileRT | 11-18 | GB300 vLLM prefill + TileRT DEP32 decode | not read |
| #5105 | fanyang-real | 11-17 | TokenSpeed GB300 | not read |
| #4972 | hshrivastava-droid | 11-12 | B300 vLLM update | not read |
| #4993 | functionstackx | 11-11 | A100 vLLM AgentX | not read |
| #4774 / #4770 | functionstackx | 11-04 | SMG router experiments (GB300 / MI355X DP-attention) | not read |

Related MI355X vLLM PD precedent on another model (not DSv4.1) [FH PR bodies]:
- #5090 (open, whx-sjtu, 11-16): MiniMax-M3.1-Flash MI355X **vLLM PD over MooncakeConnector**, 1P TP4 + 2D TP4
  (12 GPUs), vllm-router 0.1.16, c40-c80; author's local 3600 s runs 25,974-50,441 tok/s/GPU, "+11% to +28% above
  the single-node curve at equal p90 interactivity" [2H]. Needs vllm-project/recipes#1262 (Mooncake PD on AMD) and
  vllm#67857. Uses `framework: vllm-disagg` as a separate curve category.
- #4973 (closed 11-16, superseded by #5090): same model over MoRI-IO RDMA WRITE (vllm#65822, srt-slurm#662).
- DeepSeek-V4-Pro (not Flash) SGLang PD on MI355X with MoRI + UMBP linker: #2823 (merged 09-15), #3664 (merged 10-04).
- Closed DSv4-Pro MI355X vLLM disagg smoke tests: #1707, #2081 (closed 09-25, no numbers read).

## 8. Gaps / not found

- DSv4.1-Flash vLLM PD on MI355X: not found in InferenceX (published rows, master config, recipes, open/closed PRs)
  nor via `gh search prs` in vllm-project/vllm ("DeepSeek-V4.1 disagg ROCm", "DSv4.1 MoRIIO",
  "DeepSeek-V4.1 MooncakeConnector", "moriio deepseek v4", "UMBP DeepSeek-V4.1", "DeepSeek-V4.1 PD"; only hit:
  vllm#62311 MoRIIO attention-backend bugfix, closed) or vllm-project/recipes (DSv4.1 entries are colocated or GB300 PD #1137).
- Whether NixlConnector / MoRIIOConnector / MooncakeConnector handle DSv4.1's KV layout (fp8_ds_mla or fp8_e4m3 +
  MXFP4 indexer cache + SWA/compressed layers + Engram) on ROCm: not verified (no code read in this pass).
- How the rankings API interpolates "tier 125" (MI355X 150,079 vs highest measured SGLang point with p90i >= 125 of
  110,744 at c32 DPA): not verified.
- vr200 row provenance (NVIDIA repo run) and whether it counts as official: not checked.
- SGLang/ATOM changelog entries for MI355X beyond the summaries above, and NVIDIA recipe details other than B300 vLLM
  PD: only skimmed. Full text in `notes/ix/changelog_dsv41.yaml`.
- `/api/v1/views/overview` rejects `models=DeepSeek-V4.1-Flash` (allowed: default/all); not pursued.
- Warmup count actually used in the 11-18 CI run: not read from its aiperf.log.
