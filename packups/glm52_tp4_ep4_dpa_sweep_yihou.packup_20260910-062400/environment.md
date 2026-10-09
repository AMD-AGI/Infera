# Environment and provenance

## Measured hardware
Nodecrsuse2-m2m-056, existingjob130737, four AMD Instinct MI355X selected as0,1,2,3. Driver6.14.14; gfx950:sramecc+:xnack-,294896MiB reported GPU memory. Runtime probe verifies four visible Torch devices; rocm-smi can still enumerate all eight host GPUs. Docker uses containerd snapshotter; actual NVMe backing probe showed27.9TiB total,24.7TiB available. Raw probes under evidence/iterations/000_environment/ are historical, not fresh packaging queries.

**Missing:** node056 CPU/RAM snapshot, RDMA fabric/rails/network-driver/data-plane IPs were not captured for this sweep. Do not copy node055's old CPU/RAM values into this run. Single-node TP/EP was used; no cross-node network claim. GPU exclusivity was inspected at setup and process exit, not continuously instrumented.

## Software
- Exact imageID sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d; archive tag rocm-llm-bench:latest is mutable and was not the selection criterion.
- Base tag lmsysorg/sglang:v0.5.18-rocm720-mi35x; inherited base digest sha256:6d68cd19206716cb3f1e31e2ad89cd0852d7ae614a792773c30a4277f8955c72 from prior optimization-kit evidence, not freshly checked registry metadata.
- SGLang xiaobochen-amd/sglang commit402df1e1e453e1e85ec0f5ac4052d36598cc691a; AITER xiaobochen-amd/aiter commit2c71811b32c8ce2e1266aedaec199df7d90f597d, directly verified in the measured container. Build recipe checks out commits (detached); original Dockerfile included under references/optimization_stack/.
- Torch2.9.1+rocm7.2.0.git7e1940d4 directly probed. Prior identical-image evidence gives Python3.10, Triton3.7.0+amd.rocm7.2.0.git89002410, FlyDSL0.3.2,transformers5.12.1; no fresh full pip inventory for056 is claimed.
- Wrapper repo branchdev.yihou.sglang.bench.fast.script baseline4d982fa (full SHA in provenance/git_head.txt). New changes are uncommitted; provenance/tracked_changes.diff omits untracked files by Git design, so complete verbatim scripts/original and per-run bench snapshots/hashes are authoritative. No remote credentials/Git configuration shipped.

## Inputs and flags
External archive /shared_nfs/yihou/playground/glm52_mix_repro_20260908/image/rocm-llm-bench.tar.zst,23912216852bytes,SHA256 a6d047de9e3791b3c819f9035b0c3a4f632663b5cf8f85c3147b4f5b8204f9a2. Sweep setup rehashed/loaded exact image; packaging only preserves that evidence, does not rehash/copy23.9GB again.

External model /shared_nfs/models/GLM-5.2-MXFP4 on NFS,read-only. Config hash8b46225f7afd7181735c2dd97b925bcb70dfebb72b1a105d4ac0be7583a5c726; safetensors-index hashfd42188894abe9196fb70a2113c4fd1d0569b29a307a89c0e18e200111456fc6. Full weight hashes/revision absent. No dataset or SIKL runtime required.

TP4EP4,DP1off/DP4on,FP8KV,FlyDSL requested,AITERtopk,MoEa2a=none(still communicates),fused allreduce/QKnorm-rope,memfraction0.85,EAGLE5/6/topk1,seed1234,warmup10. DPAon localC/4. Environment in original creation script includes HIP_VISIBLE_DEVICES0,1,2,3; AITER_USE_FLYDSL_MOE_SORTING1; SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK1; SGLANG_OPT_USE_TOPK_V2false; offline HF; image-keyed AITERJIT. See exact config_yihou.json per run.

Required access: authorized existing cluster allocation,Docker daemon,shared model/archive; optional registry/GitHub/package access for an explicitly different rebuild. No inference API/HF secret required. No credential values included. Full runtime binaries rely on external image; no fully locked source rebuild is promised.
