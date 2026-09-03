---
language:
- en
- zh
library_name: transformers
license: mit
pipeline_tag: text-generation
base_model:
- zai-org/GLM-5.3-Flash
tags:
- glm5_next
- quark
- mxfp4
- rocm
- sglang
---

# GLM-5.3-Flash-MXFP4

## Model Overview

- **Model architecture:** GLM-5.3-Flash
  - **Input:** text and images
  - **Output:** text
- **Source checkpoint:** [zai-org/GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16)
- **Validated hardware:** 4× AMD Instinct MI350 GPUs (gfx950)
- **Validated software:**
  - ROCm 7.2.4
  - PyTorch 2.11.0+rocm7.2
  - Transformers 5.12.1
  - SGLang 0.5.18 development image with [PR #36607](https://github.com/sgl-project/sglang/pull/36607)
  - AMD Quark 0.12.post1 checkpoint format
- **Inference engine:** [SGLang](https://github.com/sgl-project/sglang)
- **KV cache used for validation:** BF16

This is the OneNexus V29 mixed-precision MXFP4 checkpoint of GLM-5.3-Flash. It was quantized from the BF16 checkpoint, not from the published FP8 checkpoint.

The checkpoint contains 227,496,639,296 bytes (211.87 GiB) of indexed model weights. The published GLM-5.3-Flash FP8 checkpoint contains 328,337,455,672 bytes (305.79 GiB) of model weights, so this checkpoint reduces weight storage by 30.71%.

## Model Quantization

AMD Quark applies OCP MXFP4 E2M1 quantization to the routed MoE expert weights. Weights use static 1×32 block scaling with E8M0 scales; expert activations are quantized dynamically with the same 1×32 block layout. The Hugging Face quantization metadata uses normalized `model.layers.*` module names, consistent with the convention used by [amd/GLM-5.2-MXFP4](https://huggingface.co/amd/GLM-5.2-MXFP4).

The following paths remain in BF16:

- attention and DSA projections;
- router gates, dense/shared MLP projections, and `lm_head`;
- routed experts in layers 3, 5, and 6;
- routed experts in MTP layer 45.

Layer 4 remains MXFP4. Fifteen layer-4 experts use a checkpoint-only, folded intermediate SmoothQuant transform selected from held-out BF16 activation traces. This transform does not require a custom runtime operation.

The baseline Quark recipe is:

```bash
cd Quark/examples/torch/language_modeling/llm_ptq/
python quantize_quark.py \
  --model_dir zai-org/GLM-5.3-Flash-BF16 \
  --output_dir GLM-5.3-Flash-MXFP4 \
  --quant_scheme mxfp4 \
  --exclude_layers "*self_attn*" "*mlp.gate" "*lm_head" \
    "*mlp.gate_proj" "*mlp.up_proj" "*mlp.down_proj" \
    "*layers.45.*" \
  --file2file_quantization
```

V29 adds the mixed-precision expert protections and folded SmoothQuant refinement described above. The exact machine-readable exclusions and quantization parameters are stored in `config.json`; the refinement summaries are stored in `mixed_precision_correction.json`, `quantization_correction.json`, and `mxfp4_smoothquant_optimization.json`.

## Deployment

### SGLang on AMD MI350/MI355

The checkpoint was validated with the GLM-5.3-Flash ROCm support introduced by SGLang [PR #36607](https://github.com/sgl-project/sglang/pull/36607), native AITER MXFP4 MoE kernels, TileLang DSA backends, and a BF16 KV cache. PR #36607 has been merged; when using the validated SGLang v0.5.18 ROCm image, mount a checkout containing that change as shown below. A newer image that already contains the merged change does not need the source overlay.

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
git fetch origin pull/36607/head:pr-36607
git checkout pr-36607

docker run --gpus all \
  --shm-size 32g \
  -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$PWD/python/sglang:/sgl-workspace/sglang/python/sglang:ro" \
  --env SGLANG_USE_AITER=1 \
  --env SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 \
  --ipc=host \
  lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260822 \
  python3 -m sglang.launch_server \
    --model-path OneNexus/GLM-5.3-Flash-MXFP4 \
    --tp-size 4 \
    --quantization quark \
    --trust-remote-code \
    --disable-cuda-graph \
    --context-length 65536 \
    --mem-fraction-static 0.80 \
    --max-running-requests 32 \
    --chunked-prefill-size 4096 \
    --max-prefill-tokens 16384 \
    --dsa-prefill-backend tilelang \
    --dsa-decode-backend tilelang \
    --kv-cache-dtype bfloat16 \
    --moe-runner-backend aiter \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --mm-feature-transport cpu \
    --host 0.0.0.0 \
    --port 30000
```

The validation server dispatched native AITER FP4 MoE kernels (`torch.float4_e2m1fn_x2`, 1×32 quantization) rather than dequantizing the checkpoint to BF16 GEMMs.

## Evaluation

The checkpoint and its BF16 oracle were evaluated with [sgl-eval](https://github.com/sgl-project/sgl-eval) using `temperature=0`, `seed=0`, and `reasoning_effort=max`. “8192” and “16384” below are maximum output-token limits, not input-context limits.

Definitions:

- **Completed:** requests with a recorded evaluator result.
- **Raw accuracy:** correct ÷ completed.
- **Truncated:** requests ending because the maximum output-token limit was reached.
- **Excluding truncation:** correct ÷ (completed − truncated).
- **Recovery:** MXFP4 accuracy ÷ BF16 accuracy.

### Accuracy

| Benchmark | Model | Completed | Correct | Raw accuracy | Truncated | Excluding truncation | Recovery (raw / excl. trunc.) |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K, 8192, flexible extract | BF16 oracle | 500/500 | 486 | 486/500 = 97.20% | 3 | 486/497 = 97.79% | — |
| GSM8K, 8192, flexible extract | MXFP4 V29 | 500/500 | 491 | 491/500 = 98.20% | 1 | 491/499 = 98.40% | 101.03% / 100.62% |
| MMLU, 8192 | BF16 oracle | 500/500 | 436 | 436/500 = 87.20% | 41 | 436/459 = 94.99% | — |
| MMLU, 8192 | MXFP4 V29 | 500/500 | 428 | 428/500 = 85.60% | 42 | 428/458 = 93.45% | 98.17% / 98.38% |
| GPQA, 16384 | BF16 oracle | 198/198 | 132 | 132/198 = 66.67% | 64 | 132/134 = 98.51% | — |
| GPQA, 16384 | MXFP4 V29 | 198/198 | 131 | 131/198 = 66.16% | 63 | 131/135 = 97.04% | 99.24% / 98.51% |

On the exactly matched GSM8K rows, MXFP4 and BF16 correctness agree on 487/500 questions (97.4%). On the exactly matched MMLU rows, correctness agrees on 474/500 questions (94.8%); among the 442 questions for which both models produce a parsed answer, the selected answer agrees on 440/442 (99.55%). On GPQA, the two models choose the same answer on all 114 questions for which both produce a parsed answer; most raw-score differences are caused by which requests reach the output-token cap.

### Reproduction

After starting the SGLang endpoint, install `sgl-eval` and run:

```bash
sgl-eval run gsm8k \
  --num-examples 500 \
  --num-threads 32 \
  --max-tokens 8192 \
  --temperature 0 \
  --seed 0 \
  --reasoning-effort max \
  --base-url http://localhost:30000/v1 \
  --model OneNexus/GLM-5.3-Flash-MXFP4

sgl-eval run mmlu \
  --num-examples 500 \
  --num-threads 32 \
  --max-tokens 8192 \
  --temperature 0 \
  --seed 0 \
  --reasoning-effort max \
  --base-url http://localhost:30000/v1 \
  --model OneNexus/GLM-5.3-Flash-MXFP4

sgl-eval run gpqa \
  --num-examples 198 \
  --num-threads 32 \
  --max-tokens 16384 \
  --temperature 0 \
  --seed 0 \
  --reasoning-effort max \
  --base-url http://localhost:30000/v1 \
  --model OneNexus/GLM-5.3-Flash-MXFP4
```

For benchmark and leaderboard reproduction, keep GLM-5.3-Flash at `reasoning_effort=max`. For chat use, follow the source model’s guidance on `clear_thinking`.

## Limitations

- This is a post-training mixed-precision quantization. It can differ numerically and behaviorally from BF16, especially on long reasoning traces near an output-token limit.
- Validation used the SGLang ROCm/AITER path described above. Other inference engines and kernel implementations require separate compatibility and accuracy checks.
- The accuracy results are deterministic single runs on the stated subsets and settings; they are not claims for every evaluation protocol.

## License

This checkpoint is distributed under the source model’s MIT License. See `LICENSE` and the [GLM-5.3-Flash-BF16 model card](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) for source-model details and citation information.
