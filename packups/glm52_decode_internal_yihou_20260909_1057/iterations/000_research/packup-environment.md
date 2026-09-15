# Environment and external dependencies

Evidence priority: raw runtime artifact/source > historical process narrative. This file is not a freshly collected node snapshot. The original nodes are no longer held; `scripts/collect_env.sh` is included only for a future authorized rerun.

## Historical machine

| Field | Evidence / qualification |
|---|---|
| Cluster / role | crsuse2-m2m, amd-spur; single aggregated inference server on crsuse2-m2m-153, historical job120922 |
| GPUs | 8 AMD Instinct MI355X, gfx950; TP rank0–7 in server logs; GSM8K environment identifies gfx950:sramecc+:xnack- |
| CPU visibility | AMD EPYC9575F64-Core Processor;236 visible logical CPUs,2 sockets,59 cores/socket,2 threads/core,2 NUMA domains,KVM; taken from saved eval environment, not a bare-metal topology guarantee |
| RAM | Historical `free -g` report2751 means **GiB**, not decimal GB; raw full memory snapshot absent.3023 in benchmark metadata is configured `TOTAL_CPU_DRAM_GB`, not measured RAM |
| Kernel visible in eval container | Linux6.8.0-107-generic x86_64 |
| GPU kernel-driver version | **Not captured**. HIP/ROCm runtime versions below do not supply the host driver version |
| RDMA fabric / rails / driver / data-plane IP | **Not captured**. This experiment is single-node; no cross-node transport claim is made |
| Spare hold | crsuse2-m2m-168, historical job120923; no measurement there. Hardware/storage equivalence and usability not established |
| Docker | Historical builder report29.7.2, overlayfs, DockerRootDir `/var/lib/docker`; raw host daemon/storage snapshot missing. `df` from a spur namespace does not prove host backing-store capacity |

## Image and source identity

| Component | Fixed identity | Evidence |
|---|---|---|
| rocm-llm-bench | main @ `e6fea39083d05d184732ff8f7e2891067b30e640` | Read-only HEAD/status; original perf stack.txt |
| InferenceX | `6d6d296c6d323540e566b005e68c6522c816ed23` | Git submodule status, included tracked snapshot |
| aiperf | `754356e9a39acc6cc6afb242d123bb57c3fb6f75` | Nested gitlink/status, included tracked snapshot |
| Image used | `sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d` | Both result stack files, original successful build log |
| Historical image tag | `rocm-llm-bench:latest` | Mutable alias, not identity |
| Base | `lmsysorg/sglang:v0.5.18-rocm720-mi35x` | Original Dockerfile |
| Base registry digest | `sha256:6d68cd19206716cb3f1e31e2ad89cd0852d7ae614a792773c30a4277f8955c72` | docker_pull.log Digest line |
| SGLang | `402df1e1e453e1e85ec0f5ac4052d36598cc691a`, xiaobochen-amd/sglang | Dockerfile and build checkout message |
| AITER | `2c71811b32c8ce2e1266aedaec199df7d90f597d`, xiaobochen-amd/aiter | Dockerfile and build checkout message |
| lm-evaluation-harness | Intended source pin `b315ef3b05176acc9732bb7fdec116abe1ecc476` | benchmark_lib.sh; actual result reports version0.4.9.2 but git_hash null, so exact installed commit is not independently stamped |

SGLang/AITER source is fetched by the included original Dockerfile at these pins; AITER recursively checks out pinned submodules. Network/continued repository access is an external requirement for rebuilding. The top-level and benchmark submodule bytes are included; original git configuration/credential stores and parent history are not. `scripts/bootstrap_repo.py` restores authentic pinned shallow Git tips fully offline from included original commit/tree/blob identity. Generated audit checkout Git objects are retained under audit-work, not copied from credential-bearing Git configuration.

## Runtime versions captured in artifacts

GSM8K `results_2026-09-08T09-37-13.323592.json` contains `pretty_env_info`:

- Ubuntu22.04.5 LTS; Python3.10.12; GCC11.4.0; glibc2.35.
- PyTorch runtime `2.9.1+rocm7.2.0.git7e1940d4`; pip metadata `torch==2.9.1+rocm7.2.0.lw.git7e1940d4` (preserve this distinction).
- ROCm build7.2.26015-fc0010cf6a; HIP runtime7.2.26015; MIOpen3.5.1.
- Triton3.7.0+amd.rocm7.2.0.git89002410; triton_kernels1.0.0+amd.rocm7.2.0.git89002410.
- Eval transformers5.12.1; numpy2.2.6; lm_eval0.4.9.2.

Separate AIPerf virtualenv installation is recorded in original agentx_c1.log: uv0.12.10, CPython3.11.16, aiperf0.12.0 from the pinned checkout, datasets5.0.1, huggingface-hub1.30.0. [Observed constraints](dependencies/aiperf-observed-constraints.txt) contain the130 other resolved package versions. They are **not** a wheel-hash lock or a complete server environment. Original installers include floating constraints/downloads and tolerate some eval-install errors; exact cold binary equivalence remains unproven.

## External image archive (excluded, read-only)

- Absolute path: `/shared_nfs/yihou/playground/glm52_mix_repro_20260908/image/rocm-llm-bench.tar.zst`
- Compressed size: **23,912,216,852 bytes** (23.912GB decimal; approximately22.27GiB).
- SHA256: `a6d047de9e3791b3c819f9035b0c3a4f632663b5cf8f85c3147b4f5b8204f9a2`.
- The checksum was recomputed read-only during initial packaging and matched; the continuation audit rechecks it and records the result. Historical zstd test passed; docker load was not tested during the09-08 mix run. A separate09-09 load recovered the exact image and source pins; see [later validation](provenance/later-image-validation.md). No assertion that load requires92GB root free is supported.
- The digest is an archive checksum, not the image ID or base digest. Rebuild alternative in REPRODUCE.md may produce a different image ID even with pinned source; no bit-identical build claim.

## Model and datasets (not included)

- Weights/tokenizer/config: `/shared_nfs/models/GLM-5.2-MXFP4` on shared NFS; launcher mounts read-only. Baseline named `/perf_apps/xiaobo/models/GLM-5.2-MXFP4`. Directory naming does not prove identical weights. Full weight hashes / original model revision are **missing**; reproducer must arrange and verify model access separately.
- AgentX: `semianalysisai/cc-traces-weka-062126`, split train,393 entries; original cache revision `23f152f6f0f9399a85901b89a6458def0ef16729`.
- GSM8K: `openai/gsm8k`, config main, test1319/train fewshot; original cache revision `740312add88f781978c0658806c59bc2815b9866`.
- The source cache is `.../glm52_mix_repro_20260908/repo/.cache/hf`; only the two known revision markers were copied, not credentials, token files, cache internals or the ~1.85GB AgentX dataset blob. Cold acquisition uses pinned revisions. Public-dataset behavior was recorded; no HF token was required for this run.
- No separately injected host library is referenced by the original launcher. Docker device bindings `/dev/kfd`, `/dev/dri` and video/render group access are required. Do not infer that host GPU driver/fabric details are unnecessary merely because they are missing.

## Required access, never secret values

Arrange cluster account/QOS and access to an exclusive MI355X node via your cluster administrator; historical job IDs are not reusable. Docker daemon permission, shared model/NFS permission and outbound access to GitHub, Docker Hub, PyPI/uv, and Hugging Face are required for a cold setup. Registry authentication, if rate-limited, must come from the reproducer's approved credential provider. No credential stores were read or packed. The local inference endpoint used placeholder API key `EMPTY`, not a paid API secret. Avoid setting actual API keys when logging eval commands: the original eval helper uses shell tracing.
