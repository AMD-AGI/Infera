# Findings, wrong turns and boundaries

## Actual expert reduction
What:8 real routed experts per target MoE layer,2 per EP rank, plus original shared expert. Why:free weight VRAM for KV experiments. How:filter original checkpoint names BEFORE get_tensor, CPU materialize retained tensors, slice gate/correction bias0:8; fixed all-eight sigmoid routing retains normalization/scaling. Context:masking routing while retaining256 expert allocation would not answer the question. Full load logs and metadata accounting agree on44,848,135,008 retained bytes.

## Real DSA inheritance
Initial research looked at glm4_moe.py; GlmMoeDsaForCausalLM is a thin DeepseekV2ForCausalLM subclass. The final hook is DeepseekV2MoE in deepseek_v2.py, not the unused GLM4 block. Final mount manifest contains only helper/deepseek_v2/loader/topk. Imports and integrated routing tested before full startup.

## Shared NFS temporary directory
What:pure torch.fill_ and Mooncake probe faulted on both nodes. Why observed:HIP logs 'Failed to unbundle code object'; native stack in libamdhip64. Exact COMGR/NFS mechanism remains OPEN. How:workspace-scoped container tmpfs for TMPDIR resolves minimal fill and all8 directed GPU RDMA transfers. Context:allocation/register succeeded, so those alone were insufficient GPU health tests. Torch-before-Mooncake ordering did NOT fix full probe; preserved as a rejected hypothesis. Do not diagnose nodes as broken based on this wrapper-induced failure.

## JSON source-order failure
Initial model launch failed before loading with extra trailing brace in JSON_MODEL_OVERRIDE_ARGS. Inherited shell default expansion corrupts a pre-set JSON value. Reference deployment assigns override AFTER source; final config does so. No source model change was needed. Both failed and successful launch logs are archived.

## JIT and monitoring
CUDA graph/JIT cold builds can take30minutes. Inspect build progress and live docker logs, not only a detached file follower. Runtime successful-run errors were scanned after benchmark. Health checks can add stage-counter samples; the288 benchmark requests are proven by per-request records and transfer counts, not every stage histogram.

## Measurement boundaries
-31.43/31.44 runtime-GB weight usage is measured;205.23/210.74 runtime-GB KV pools and3,990,400/4,097,408 token capacities are allocated per rank.
-87.06GiB/rank saved at equal EP4 is theoretical checkpoint expert+gate accounting; no full256EP4 A/B was performed. Historical119.24GB includes a different configuration.
-Only4096/1024 C32 ran. Large allocated KV capacity is not a full-occupancy throughput/long-context stability guarantee.
-TTFT is first nonempty text chunk. TPOT uses first/last chunk times divided by1023. Stream chunk intervals need not equal per-token ITL.
-Generated quality is intentionally not evaluated. No semantic Jupiter/tool-call assertions used.

## Packaging
Verbatim scripts/source live in payload. restore_workspace.yihou.py creates a fresh destination and changes only absolute workspace paths; it must not overwrite scratch or modify live file mounts. Experiment-time snapshots and later environment captures remain distinct. Gzip key logs and per-request data approved by user. No weights, image layers, caches, core dumps or credentials included. No original files/containers removed, no services restarted, no git commit.

## Explicit provenance gap
Final image is locally addressable only; export/load a verified image for exact reproduction. Ancestor Dockerfiles and patches explain lineage, but full image source/build identity is not completely captured and rebuilding is not asserted bit-for-bit. No persistence of registry credentials is attempted.
