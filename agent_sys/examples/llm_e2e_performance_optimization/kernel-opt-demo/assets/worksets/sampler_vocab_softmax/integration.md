# Integration — where this operator plugs back into sglang

Stage 4 (kernel optimization) does **not** perform the integration; stage 5
does. This file exists so that stage 4 optimizes something that *can* be
integrated, and so stage 5 does not have to re-derive where it goes.

## The call site

`python/sglang/srt/layers/sampler.py:183`, sglang v0.5.14
(commit `49e384ce9d304648e9959666ecb8ce8cd98d0deb`), inside the sampler's
forward path:

```python
logits[:] = torch.softmax(logits, dim=-1)
```

Read it carefully — three facts are load-bearing and all three are in that one
line:

1. **`logits[:] = ...` is an in-place write into an existing buffer.** The
   replacement must write into a caller-provided `out`, not allocate. That is
   why the workset's signature is `sampler_softmax(logits, out)` and why the
   driver pre-allocates `out`.
2. **`dim=-1` over the vocabulary**, which is the contiguous dimension. A
   replacement may rely on that contiguity; it must then say so, because a
   non-contiguous `logits` would silently produce wrong results rather than an
   error.
3. **The result is a probability distribution consumed by `torch.multinomial`.**
   Not by an `argmax`, where a monotone-preserving approximation would be
   harmless. Rows must sum to 1. This is why the workset enforces a
   rows-sum-to-1 bar at `1e-4` on top of SNR and `allclose` — an SNR of 130 dB
   with rows summing to 0.999 is a sampler drawing from the wrong distribution,
   and no SNR threshold notices.

## What a replacement must preserve

| | |
|---|---|
| Signature | `sampler_softmax(logits: Tensor[B, V], out: Tensor[B, V]) -> Tensor` |
| Write | in place into `out`; return `out` |
| dtype | fp32 in, fp32 out |
| Layout | row-major, vocabulary contiguous |
| Device | one ROCm device; no cross-device movement |
| Build | none. Triton or plain torch. **No compile step may be added** — the driver imports the module directly |

## What stage 5 will have to do, and is not done here

- Wire the replacement in behind a flag, so the ATen path stays reachable.
- Decide the fallback: what happens at a vocabulary or dtype the replacement
  does not handle. Silently producing wrong numbers is the failure to design
  against.
- Re-run the end-to-end checks: short/long text, needle, llm-eval, and an
  end-to-end performance regression.
- Confirm the kernel-level speedup survives at the service level. It may not:
  the traced softmax is 14.5% of *decode GPU time*, so by Amdahl a 2.8× on it
  bounds the decode-GPU-time saving at ~9.3%, and the end-to-end effect is
  smaller again once CPU-side scheduling is counted. **Nobody has measured
  that**, and stage 4 must not claim it.

## Sibling operators at the same call site

The same sampler does three vocabulary-wide reductions per step, ~39% of decode
GPU time between them. Only the softmax is in this workset:

| kernel | µs/call | share | in this workset? |
|---|---:|---:|---|
| `reduce_kernel<…, ArgMaxOps<float>, …>` | 57.68 | 15.05% | no |
| `cunn_SoftMaxForwardGmem<4, float, …>` | 55.59 | 14.50% | **yes** |
| `reduce_kernel<…, func_wrapper_t<float>, …>` | 34.81 | 9.08% | no |

Worth knowing before optimizing in isolation: if all three were fused into one
pass over the vocabulary, the win would be larger than optimizing any one of
them. That is a design change to the sampler, not a kernel swap, and it is
out of scope for this workset — recorded so the option is not lost.
