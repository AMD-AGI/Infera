# Patch application and rationale

## yihou-expert8.patch
What:4file synthetic-only SGLang patch (new helper, deepseek_v2.py, model_loader/loader.py, layers/moe/topk.py).
Why:actual8 expert allocation with original256checkpoint; all-eight routed execution without shared-expert fusion.
How:recommended exact-image read-only mounts listed in mounts.yihou.tsv; restoration rebases local source paths, container destinations remain exact. Patched files and originals are preserved in payload/source. Do not apply patch on top of already mounted patched sources.
Context:GlmMoeDsaForCausalLM inherits DeepseekV2ForCausalLM; filtering before tensor materialization avoids loading discarded experts on GPU. Tests and full smoke/C32 passed. Original behavior retained when synthetic marker is absent.

## Harness changes already in verbatim scripts
What:workspace/model/source mounts, exact image-ID guard, workspace-scoped tmpfs TMPDIR and caches, task-only stop scope, NCCL_SOCKET_IFNAME.
Why:isolate this experiment and avoid HIP code-object extraction failure on NFS. Remove unrelated generic preflight-container stop from copied harness.
How:use restored scripts/config.yihou.expert8.sh plus scripts/bench-harness. JSON override is explicitly reassigned after inherited config to avoid extra closing brace.
Context:initial launch JSON failure and pure GPU fill failures are documented in notes and key logs. No shared tracked harness modifications were made.

## image-provenance/ — archival only
Historical NextN fusion and HiCache fixes are already baked into the pinned image. These are NOT additional expert8 steps. Dockerfiles, source diff, and build verification explain ancestry. Ancestor router affinity patches explain load-bearing PD routing. Historical Dockerfiles contain non-yihou deletion operations, so do not run them verbatim under this task's deletion rule. Obtain the verified final image as described in REPRODUCE instead.
