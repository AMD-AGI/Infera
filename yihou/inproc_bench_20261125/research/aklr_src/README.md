# Model-level micro-benches

Each directory benchmarks one model on sglang: it times every operator on its
own, sums them over the layer stack, and checks the sum against a full forward.
How to run one, and how to read its report, is in that model's README.

| file | holds |
|---|---|
| `<model>/profile_decode.py`, `<model>/profile_prefill.py` | The entry points, one per phase. Copied per model on purpose: each reports its model's own sections. |
| `<model>/model_bench.py` | That model's parts: its layer tree, input recording, batch and request setup, memory accounting, server arguments. |
| `sglang_bench.py` | Code that is the same in every model: forward setup, batch preparation, CUDA-graph and eager timing, MoE parallel layouts, version and table output. |
| `main.py` | The e2e stage's entry point, the same for every model: `python main.py <model> <decode\|prefill> [the model's flags] --result-json PATH` prefills pinned real text (`TEXTS`, from the HF cache), runs teacher-forced decode steps, and writes `{"tokens_per_s", "latency_ms"}` to PATH. E2e checks performance only; operator accuracy remains checked by the bench stage. |
| `kernel_breakdown.py` | The per-kernel breakdown of an operator's replay, from Kineto. Used by the decode scripts that report kernels. |

The e2e entry point uses SGLang v0.5.22's request API: prefill chunks retain
the same requests and KV rows, advancing `prefix_len` and `extend_end` before
each forward. The final batch continues into timed decode without resetting KV.

Each profile script appends this directory to `sys.path` before importing
`sglang_bench` and `kernel_breakdown`. Code that differs between models stays
in that model's `model_bench.py`.
