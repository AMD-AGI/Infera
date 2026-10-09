# Expert8 patch handoff

## Implemented
- Exact-image DSA hierarchy: GlmMoeDsaForCausalLM in glm4_moe.py inherits DeepseekV2ForCausalLM. Synthetic routing hook is therefore in DeepseekV2MoE in deepseek_v2.py; no glm4_moe source delta remains.
- New models/yihou_expert8.py validates opt-in config, filters checkpoint names before safetensors get_tensor, materializes only retained tensors on CPU, slices gate/correction bias, skips NextN layer78 for target-only load.
- DefaultModelLoader._get_weights_iterator dispatches to this iterator only for config.yihou_synthetic_expert8=true. This bypasses bulk/fast GPU shard materialization and avoids GPU allocation for experts8..255.
- DeepseekV2MoE selects route_all_eight and STANDARD top-k output for synthetic mode; forbids shared fusion, NextN and hash modes. Physical FusedMoE count follows config8 and EP4.
- TopK select_experts synthetic branch emits all0..7 and applies routed scaling only when the backend requests scaling in top-k. Existing padding/physical-ID postprocessing remains intact.
- Model overlay contains copied config/metadata and symlinks to original read-only shards. Original model unchanged.

## Tests passed
- tests/test_yihou_expert8.py: instrumented CPU checkpoint handles prove discarded tensors never call get_tensor; gate/bias match first8 slices; shared weights retained; NextN omitted.
- GPU route eager for0,1,32,128 tokens; finite weights, exact IDs0..7, normalized sigmoid reference.
- CUDA graph capture/replay with changed logits.
- Actual select_experts branch exercised with scaling2.5; postprocessing mocked for this focused test, so it is not EP dispatch validation.
- Fresh-container imports confirm actual DSA hierarchy contains synthetic hook and DefaultModelLoader contains filtering hook.
- Source AST parsing passes.

## Pending gate
Full four-GPU FusedMoE EP4 allocation/forward and quantized full-model load are not yet tested. Leader's next startup is this gate; do not present route tests as full EP4 validation.

## Integration
- patches/mounts.yihou.tsv contains4 exact file mounts (helper, deepseek_v2.py, loader.py, topk.py).
- patches/yihou-expert8.patch and source-hashes.yihou.json capture the changes.
- No extra environment knobs. Required --disable-shared-experts-fusion and target-only/no MTP already configured by leader.
- Important: editing a file after NFS file-bind mount can produce stale handle. Launch fresh containers after patch edits; final sources are now frozen.
- Component containers yihou-expert8-component and yihou-expert8-component-final stopped (retained evidence).138 has no KFD processes and284MiB/card baseline.
- No commits and no deletion performed.
