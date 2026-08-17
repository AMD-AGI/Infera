# Kimi-K3 on AMD Instinct (MI355X)

Serve **Kimi-K3** on AMD Instinct GPUs. Per the AMD "Kimi-K3 on AMD Instinct GPUs"
guide, low-bit (MXFP4 / A8W4) is done **at runtime by aiter** from the native
checkpoint — no separate quantized download.

## Base images
- **vLLM:** `vllm/vllm-openai-rocm:kimi-k3`
- **SGLang:** `lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727`

## Model
`moonshotai/Kimi-K3` — native checkpoint (~1.5 TB, 96 shards), cached at
`/mnt/vast/yaocheng/models/moonshotai/Kimi-K3`. The aiter env flags
(`AITER_SITUV2_A8W4=1`, `AITER_BF16_FP8_MOE_BOUND=0`) do the low-bit MoE at load.

## Run (8-GPU node, TP8)

```bash
bash vllm_serve.sh            # TP8 on the cached checkpoint, port 8000
MODEL=<path-or-repo> TP=8 PORT=8000 bash vllm_serve.sh   # overrides
docker logs -f kimi_k3_vllm
```

The vLLM image ENTRYPOINT is `/bin/bash`, so the script serves via `vllm serve
<model> <flags>` (entrypoint overridden). Flags/env match the AMD guide:
`VLLM_ROCM_USE_AITER=1`, `--moe-backend auto`, `--tensor-parallel-size 8`,
`--gpu-memory-utilization 0.95`, `--mm-encoder-tp-mode data`,
`--tool-call-parser kimi_k3 --reasoning-parser kimi_k3`.

## Smoke test

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"kimi-k3","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

SGLang launcher: TODO (base image above).
