# Environment and provenance

Raw probes: `env/host_smci355-ccs-aus-n06-25.txt` and `env/container_yihou-glm52-tp8ep8-0914.txt`,
both captured **from the same container instance that executed all five runs** (it was still
running at capture time).

## Hardware — `smci355-ccs-aus-n06-25` (single node, the only node used)

| Item | Value |
|---|---|
| OS / kernel | Ubuntu 22.04.5 LTS, `6.8.0-107-generic` (x86_64) |
| CPU | 2× AMD EPYC 9575F 64-Core, 256 logical CPUs |
| RAM | 3.0 TiB |
| GPU | **8× AMD Instinct MI355X**, `gfx950`, 288 GB each |
| GPUs used | all 8, `HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` |
| Allocation | none — bare metal, no Slurm/spur; host verified idle before each run |

RDMA fabric present but **irrelevant**: every run is single-node, so no cross-node traffic occurred
and no network claim is made.

## Software

| Item | Value | Source |
|---|---|---|
| Container image | `sha256:b5aa5bd3d828285bff8dd01bb92e6106f569f5b1eab5de24558790b1b10a5dc0` | pinned **by digest**, re-asserted before every run |
| SGLang | `402df1e1e453e1e85ec0f5ac4052d36598cc691a` | `git -C /sglang rev-parse HEAD` + `pinned_sglang` in every result |
| AITER | `2c71811b32c8ce2e1266aedaec199df7d90f597d` | `git -C /aiter rev-parse HEAD` |
| torch | `2.9.1+rocm7.2.0.lw.git7e1940d4` | `pip list` |
| triton | `3.7.0+amd.rocm7.2.0.git89002410` | `pip list` |
| ROCm | 7.2.0 | container probe |
| Python | 3.10.12 (venv at `/opt/venv`) | container probe |
| Model | `/perf_apps/data/models/GLM-5.2-MXFP4` (read-only NFS, 408 GB) | `config_yihou.json` |
| Wrapper repo | branch `dev.yihou.sglang.bench.fast.script`, profiling feature at **`23f472d2`** | `evidence/code_snapshot/git_head.txt` |

**Which HIP runtime is actually loaded** — checked, because torch sometimes ships its own:
`/proc/<pid>/maps` shows `/opt/rocm-7.2.0/lib/libamdhip64.so.7.2.70200` and
`/opt/rocm-7.2.0/lib/libroctracer64.so.4.1.70200`. `torch/lib/` contains neither. So the ROCm-side
env vars in `research/rocm_profiling_env.md` refer to the runtime that actually ran.

**Kineto uses roctracer on this build** (`libtorch_hip.so` links `libroctracer64.so.4`,
`libroctx64.so.4`, `librocprofiler-register.so.0`), not rocprofiler-sdk.

## Profiling tooling available in the container

`rocprofv3` (1.1.0), `rocprofv2`, `rocprof`, `rocprof-compute`, `rocprof-sys-*`.
**`rocpd` and `roctx` Python packages are already installed** at
`/opt/rocm/lib/python3.10/site-packages` — not on the venv `sys.path`, but reachable with
`PYTHONPATH=/opt/rocm/lib/python3.10/site-packages` alone (verified: both import, plus
`rocpd.{schema,summary,query}`). No build, no install, no `.pth` needed. `rpdTracerControl` /
`rocmProfileData` are **not** installed — the RPD path SGLang documents would need building.

## Runtime environment variables in effect

Set by `scripts/create_container_yihou.sh`: `HIP_VISIBLE_DEVICES=0..7`,
`AITER_JIT_DIR=/tmp/yihou-aiter-b5aa5bd3d828`, `AITER_USE_FLYDSL_MOE_SORTING=1`,
`SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1`, `SGLANG_OPT_USE_TOPK_V2=false`, `HF_HUB_OFFLINE=1`,
`HF_HOME=/tmp/yihou-hf-cache`, `PYTHONNOUSERSITE=1`, `PYTHONUNBUFFERED=1`, plus the host-specific
`HOME`/`USER`/`LOGNAME` and the explicit `PATH`.

Baked into the image: `SGLANG_USE_AITER=1`, `SGLANG_MOE_PADDING=1`,
`SGLANG_ROCM_FUSED_DECODE_MLA=1`, `SGLANG_SET_CPU_AFFINITY=1`, `SGLANG_USE_ROCM700A=1`,
`HIP_FORCE_DEV_KERNARG=1`, `HSA_NO_SCRATCH_RECLAIM=1`, `NCCL_MIN_NCHANNELS=112`,
`ROCM_QUICK_REDUCE_QUANTIZATION=INT8`, `TORCHINDUCTOR_MAX_AUTOTUNE=1`,
`PYTORCH_ROCM_ARCH=gfx942;gfx950`.

**Profiling-specific: none were set.** `SGLANG_PROFILE_V2`, `SGLANG_TORCH_PROFILER_DIR`,
`SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE`, and every `DEBUG_CLR_*` / `DEBUG_HIP_*` were left at
their defaults for all five runs, because the published throughput number used stock defaults.
`SGLANG_TORCH_PROFILER_DIR` is set per-run by the bench, into the iteration's own `profile/`.

## Resolved benchmark configuration

Identical for every run except `--disable-cuda-graph` and the `--profile-*` window:

```
--model-path /perf_apps/data/models/GLM-5.2-MXFP4 --tp-size 8 --ep-size 8 --dp-size 8
--moe-a2a-backend none --max-running-requests 256 --enable-aiter-allreduce-fusion
--enable-fused-qk-norm-rope --mem-fraction-static 0.85 --enable-dp-attention
--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-num-draft-tokens 6
--speculative-eagle-topk 1 --kv-cache-dtype fp8_e4m3 --dsa-decode-backend flydsl
--dsa-prefill-backend flydsl --dsa-topk-backend aiter --cuda-graph-bs-decode 32
--cuda-graph-max-bs-decode 32 --random-seed 1234 --disable-radix-cache --skip-tokenizer-init
--disable-overlap-schedule --trust-remote-code
```

Benchmark-side: `input_len 70000`, `output_len 10000`, `accept_length 3.61`
(`match-expected`, `real-draft-token`), `warmup_steps 10`, `seed 1234`, `max_steps 0`.
Attention backend resolves to `DeepseekSparseAttnBackend` for both target and draft.
Model: 78 hidden layers, 1 nextn-predict layer (verified in `config.json`).

## Gaps

- **No repeats.** Every run is a single run; run-to-run variance is unmeasured.
- **Only rank 0 was traced.** Per-rank kernel distribution unverified.
- **No `rocm-smi` time series** during the runs; only the harness's own
  `max_memory_allocated_bytes`.
- **The `max_streams_` readout failed.** The `LOG_DEBUG` line reporting it never emitted despite the
  format string being present in the loaded `libamdhip64.so`; `AMD_LOG_LEVEL=4` yields only level-1
  and level-3 lines and `AMD_LOG_MASK=0xFFFFFFFF` yields *fewer* lines than no mask. Unresolved —
  which is why `research/rocm_profiling_env.md` §2 is marked provisional.
