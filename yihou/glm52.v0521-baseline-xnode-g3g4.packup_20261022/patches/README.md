# Patches

| file | what | why | how applied in these runs |
|---|---|---|---|
| `0001-upstream-pr45483-idlefix.patch` (+ `pr45483.body.md`) | `EagerRunner._execute_idle` also inits the ForwardContext attention backend when it differs from `model_runner.attn_backend` | EAGLE v2 idle draft steps ran the DSA indexer on a previous batch's metadata -> top-k over-read -> decode "Memory access fault" with index_share=true | image-version copy `scripts/mounts-src/eager_runner.yihou-idlefix.py` bind-mounted over `/sgl-workspace/sglang/python/sglang/srt/model_executor/runner/eager_runner.py` (`mounts.yihou.v2-xferfix2-gtt-idlefix.tsv`); PR against upstream main 93088901 |
| `0002-upstream-pr45484-offloader-is-shuffled.patch` (+ `pr45484.body.md`) | `_CpuParamOffloader.create_device_tensor` keeps `is_shuffled` on the device copy | `.to("cuda")` drops the attribute, so the offloaded MoE layers lose the preshuffled-weight tag read by aiter fused_moe / mxfp4 paths | runs used the NFS copy `/shared_nfs/yihou_glm52/v0521cap2_20261013/offload/offloader.yihou-attrs.py` (prefill-only mount `mounts.yihou.offload-attrs.tsv`) |
| `scripts/mounts-src/hipxfer_trace.yihou.c` | LD_PRELOAD timing shim (log-only) | split Mooncake HIP batch time into host enqueue / sync wait / GPU copy span | g4h1 only, prefill container, via `config.yihou.xfertrace.sh` |

Earlier patches still in the image tree (xferfix2 conn.py, PR #42553 GTT host pool, kv_events non-blocking, forward_mla_rocm,
nonfinal event) are documented in `glm52.v0521-idxshare-maf-fix.packup_20261021/patches/` and earlier packups.
Upstream PRs: https://github.com/sgl-project/sglang/pull/45483 , https://github.com/sgl-project/sglang/pull/45484 (both open,
opened 2026-10-21 from fork dorado269).
