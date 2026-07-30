# Environment

## When
Bring-up + verification: **2026-07-23**. Single session, ~15 min bring-up.

## Hardware / node
| role | node | GPUs used |
|------|------|-----------|
| single-node mix (prefill+decode) | **chi2866** | card4-7 |

- **GPU:** AMD Instinct **MI355X** (gfx950), 8 per node, ~283 MB/card idle baseline.
- chi2866 is also the slurm jump host; card0-3 held a foreign `titan` training
  container (+ `zirui` / `primus-*` service containers) throughout — untouched.
  That is why this run used card4-7 (`HIP_VISIBLE_DEVICES=4,5,6,7`).
- **No RDMA fabric used.** Single-node PD-mix is one server on one node; no
  ionic / MoRIIO / Mooncake / cross-node KV transfer is involved.

## Software
- **Docker image:** `lmsysorg/sglang:v0.5.14-rocm720-mi35x` (SGLang **0.5.14**,
  ROCm 7.2.0, MI35x build). It implements `GlmMoeDsaForCausalLM`
  (`python/sglang/srt/models/glm4_moe.py`, a `DeepseekV2ForCausalLM` subclass →
  DeepSeek MLA + DSA lightning-indexer path). No infera-built sglang image was on
  the node; this lmsys image is the newest/correct one for GLM on this HW.
- **Run as:** privileged container, `--entrypoint ""` (single-node needs no ionic
  injector), `--ipc host --shm-size 64g`, `-v /mnt/vast:/mnt/vast`.
- This packup is engine-image only — no infera repo build was involved (the
  server is launched directly via `sglang.launch_server`). For the infera-native
  launch (via `python -m infera.engine.sglang`) see the e2e test added alongside
  this kit (`tests/e2e/pd_mixed/sglang/matrix.py`).

## SGLang auto-config for GlmMoeDsa (do NOT override — this is why the launch is minimal)
The server auto-selected, and these should be left alone:
```
attention_backend   = dsa            (NOT dsv4; NOT the fp16 default)
page_size           = 64             (aiter preshuffle paged-MQA available)
kv_cache_dtype      = bfloat16
dsa_prefill_backend = tilelang
dsa_decode_backend  = tilelang
dsa_topk_backend    = sgl-kernel
decode CUDA-graph   = full (52 batch sizes captured)
prefill CUDA-graph  = disabled
max_total_num_tokens= 651712 ; context_len = 202752
```

## Per-run env vars
```
HIP_VISIBLE_DEVICES=4,5,6,7
ROCM_VISIBLE_DEVICES=4,5,6,7
SGLANG_USE_AITER=1
```

## External dependencies (absolute paths, not in repo)
- **GLM weights:** `/mnt/vast/xiaobo/models/GLM-5.1-FP8` (VAST shared mount, same
  path in-container).
- **Shared work dir:** `/mnt/vast/c_huggingface/` (scripts, `glm_sglang_mix.log`).
  `/mnt/vast` is the shared VAST filesystem; `/tmp` is NOT shared.

## Required secrets (names only — no values)
- **Cluster SSH:** ProxyJump preconfigured in `~/.ssh/config` (`ssh chi2866`).
- **Docker image:** present on the node; if re-pulling, standard DockerHub for
  `lmsysorg/sglang`.

## Not captured (honest gaps)
- Exact host kernel / ROCm driver point-versions were not snapshotted. The result
  is a correctness bring-up (not a perf number), so kernel drift should not change
  the verdict. Run `collect_env.sh` on the node if reproducing elsewhere.
- No performance numbers were taken (throughput/latency) — this was a correctness
  bring-up only.
