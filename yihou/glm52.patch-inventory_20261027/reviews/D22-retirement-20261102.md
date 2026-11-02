Closing this proposal because the reason for the opt-in has been superseded by #47180 (merged as `b50d4394945d3dcacea05ee4afd334a9b225cbdc` on November 1).

At current main `acd6f5570`, `ModelRunner._prepare_replicated_q_proj()` accepts per-tensor FP8 `w_kc` / `w_vc`, applies the rank-local scale in BF16 before the gather, and prepares replicated Q without disabling the default GLM weight quantization. This removes the original need for `SGLANG_DSA_DCP_KEEP_KV_B_BF16` to activate replicated Q.

I checked the merged implementation and completed 13 CPU probes against its unchanged helper and loader quantizer: both FP8 formats, scalar and one-element scales, unsupported per-channel scales, BF16/FP16/FP32 controls, and two actual `input_to_float8` → absorb-conversion cases. The GLM quantizer returns a zero-dimensional scale; these two production-shape cases return BF16 and match the explicit dequantization reference.

Correction to the initial validation wording: the first probe assumed that a one-element FP32 vector scale also preserves BF16; PyTorch instead promotes that result to FP32. The final probe records this distinction and verifies the loader's actual scalar scale separately. This is CPU conversion coverage only, not distributed gather, GPU, or model-accuracy qualification. #47180 provides its own separately attributed hardware evidence.

Keeping original BF16 weights is numerically different from dequantizing the FP8 weights, so this is not a claim that #47180 implements this flag verbatim. There is no isolated current-main evidence justifying that additional memory/precision option after #47180. Historical T10 measurements cannot establish an incremental benefit over this new baseline. We will not add a redundant option on that basis. The original patch and T10 archive remain preserved for reference.

The trial rebase was stopped after discovering this change; the PR head is unchanged. No replacement PR is needed. The separate PD transport/enablement work remains #46824 and #46826.
