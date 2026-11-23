#!/usr/bin/env python3
"""Build the FP4-dispatch overlay (two files) from the vLLM image's own v0.34.0 files:
  vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py
  vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py

The GLM-5.2 decode runs MoE with naive DP dispatch: an all-gather of the BF16 hidden states (12,288 B / token), after
which aiter's fused_moe quantizes the gathered rows to MXFP4 (per 1x32). The quantization is row-wise, so quantizing
each rank's own rows before the all-gather moves about a quarter of the data (3,072 B packed FP4 + 192 B e8m0 scales
per token). This is MoRI's FP4 dispatch (prepare_finalize/mori.py:98-105: `get_hip_quant(QuantType.per_1x32)(a1,
shuffle=False)`, scales left unshuffled) applied to the naive all-gather; aiter's fused_moe_2stages has the matching
branch ("Input is already quantized to fp4x2 (e.g., from FP4 dispatch), skip re-quantization, only sort the scale").
- The FP4 and scale tensors are gathered as BF16 views so the aiter custom all-gather keeps handling them.
- Only when 128 < tokens gathered <= 2048: aiter picks the MoE kernels by next-power-of-two token count, and the
  block_m 16 kernels it uses up to 128 tokens read raw BF16 and fault on pre-quantized input (aiter fused_moe.py
  comment at the prequant check); 256-2048 are tuned block_m >= 32 FP4-activation kernels.
- Only with VPD_FP4_DISPATCH=1, MXFP4 W4A4 experts that asked for unquantized inputs, no LoRA.
- rocm_aiter_moe.py: AiterExperts.workspace_shapes sizes the output as (M, K) with K = the dispatched width; with
  packed FP4 input that is half the hidden size (MoRI already handles this the same way).
  make_fp4dispatch_overlay.py NAIVE_ORIG NAIVE_OUT AITER_ORIG AITER_OUT
"""
import sys

ENV = "_VPD_FP4_DISPATCH = _vpd_os.environ.get(\"VPD_FP4_DISPATCH\") == \"1\"\n"

NAIVE = [
    ("import torch\n",
     "import os as _vpd_os\n\nimport torch\n\n" + ENV + (
         "\n"
         "\n"
         "def _vpd_fp4_tokens_ok(n_local: int) -> bool:\n"
         "    from vllm.forward_context import get_forward_context\n"
         "\n"
         "    try:\n"
         "        sizes = get_forward_context().dp_metadata.get_chunk_sizes_across_dp_rank()\n"
         "    except Exception:\n"
         "        return False\n"
         "    total = sum(sizes) if sizes else n_local\n"
         "    return 128 < total <= 2048\n"
     )),
    ("        a1q, scales, a1q_scale_orig = _quantize_and_setup_dispatch(\n"
     "            a1, quant_config, defer_input_quant\n"
     "        )\n"
     "\n"
     "        # When LoRA is active, dispatch the per-token LoRA id along with\n",
     "        fp4_dtypes = None\n"
     "        if (\n"
     "            _VPD_FP4_DISPATCH\n"
     "            and defer_input_quant\n"
     "            and quant_config.use_mxfp4_w4a4\n"
     "            and self._lora_context is None\n"
     "            and _vpd_fp4_tokens_ok(a1.shape[0])\n"
     "        ):\n"
     "            from aiter import QuantType, get_hip_quant\n"
     "\n"
     "            q, s = get_hip_quant(QuantType.per_1x32)(a1, shuffle=False)\n"
     "            s = s[: a1.shape[0]].contiguous()\n"
     "            fp4_dtypes = (q.dtype, s.dtype)\n"
     "            a1q, scales, a1q_scale_orig = q.view(torch.bfloat16), [s.view(torch.bfloat16)], None\n"
     "        else:\n"
     "            a1q, scales, a1q_scale_orig = _quantize_and_setup_dispatch(\n"
     "                a1, quant_config, defer_input_quant\n"
     "            )\n"
     "\n"
     "        # When LoRA is active, dispatch the per-token LoRA id along with\n"),
    ("        return a1q, a1q_scale, None, topk_ids, topk_weights\n",
     "        if fp4_dtypes is not None:\n"
     "            a1q = a1q.view(fp4_dtypes[0])\n"
     "            a1q_scale = a1q_scale.view(fp4_dtypes[1])\n"
     "        return a1q, a1q_scale, None, topk_ids, topk_weights\n"),
]

AITER = [
    ("import torch\n", "import os as _vpd_os\n\nimport torch\n\n" + ENV),
    ("        packed_fp4_input = (\n"
     "            self.moe_config.use_mori_epv1_kernels and self.quant_config.use_mxfp4_w4a4\n"
     "        )\n",
     "        packed_fp4_input = self.quant_config.use_mxfp4_w4a4 and (\n"
     "            self.moe_config.use_mori_epv1_kernels\n"
     "            or (_VPD_FP4_DISPATCH and K * 2 == self.moe_config.hidden_dim)\n"
     "        )\n"),
]


def patch(src, out, reps):
    s = open(src).read()
    for old, new in reps:
        n = s.count(old)
        if n != 1:
            sys.exit(f"expected exactly one {old.strip()[:60]!r} in {src}, found {n}")
        s = s.replace(old, new)
    open(out, "w").write(s)
    print(f"wrote {out}")


def main():
    patch(sys.argv[1], sys.argv[2], NAIVE)
    patch(sys.argv[3], sys.argv[4], AITER)


if __name__ == "__main__":
    main()
