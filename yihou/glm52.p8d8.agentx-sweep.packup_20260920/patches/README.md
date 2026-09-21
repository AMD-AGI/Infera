# patches — what is baked into the image, vendored here for reference

Neither patch is applied at run time. Both are **already inside**
`infera-sglang:v0519-yihou-0917-nextnfix-hicache` (`sha256:fd7220a57b7d…`), which this
run transferred rather than rebuilt. They are vendored so the image's content can be
audited, or the image reconstructed, without the original build context.

| file | what | status |
|---|---|---|
| `pr37152.sources.yihou.diff` | sglang PR **#37152**, ROCm HiCache JIT copy rounds + K-only host pool. Three source hunks; the PR's two test files were dropped. | **OPEN upstream, never merged.** Upstream's own AMD ROCm CI on it is **RED**. Applied because the user asked for it. Acts on the **prefill** leg only here, since `engine.sh:76` hard-rejects decode HiCache together with MTP. |
| `nextn-fusion-fork-4350d37c5.patch` | GLM-5.2 NextN draft was missing its `fused_shared_experts_architecture` override, silently losing shared-experts fusion on the draft model. | Not in `sgl-project/sglang` main. Origin: `xiaobochen-amd/sglang`, commit `4350d37c5ba7aaa9dfaa91261f57e1aae1fd49da`. |

## Verifying they are present in the image

```bash
docker run --rm --entrypoint bash <image> -lc '
S=/sgl-workspace/sglang/python/sglang
grep -c pick_group_bytes   $S/kernels/jit/csrc/kvcacheio/hicache.cuh   # expect >= 1
grep -c _tiles_across_lanes $S/kernels/ops/kvcache/hicache.py          # expect >= 1
grep -c "can_use_jit = (_is_cuda or _is_hip)" $S/srt/mem_cache/pool_host/mha.py
grep -c "fused_shared_experts_architecture = .GlmMoeDsaForCausalLMNextN." $S/srt/models/glm4_moe.py
'
```

All four were verified non-zero on **both** nodes before the run — see
`analysis/image_transfer.yihou.md`.

**Do not apply `nextn-fusion-fork-4350d37c5.patch` with `git apply`** — the anchor line
moves between sglang trees. The source kit uses an idempotent applier that anchors on
the class statement.
