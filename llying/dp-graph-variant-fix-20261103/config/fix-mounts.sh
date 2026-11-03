# Sourced by the arm files: mount the fix (patches/fix, 6 files) into the decode container. FIX_SET=debug swaps in the
# decode graph runner with the variant audit (patches/debug, debug only) and turns the audit on. Needs PKG.
FX=$PKG/patches
SB=/sgl-workspace/sglang/python/sglang/srt
for f in layers/attention/graph_variants.py model_executor/runner/decode_cuda_graph_runner.py \
    managers/scheduler_components/dp_attn.py managers/schedule_batch.py model_executor/forward_batch_info.py \
    batch_overlap/two_batch_overlap.py; do
    src=$FX/fix/$f
    [[ "${FIX_SET:-fix}" == debug && -f $FX/debug/$f ]] && src=$FX/debug/$f
    DECODE_MOUNT_EXTRA="$DECODE_MOUNT_EXTRA $src:$SB/$f:ro"
done
export DECODE_MOUNT_EXTRA
[[ "${FIX_SET:-fix}" != debug ]] || export DECODE_ENV_EXTRA="$DECODE_ENV_EXTRA SGLANG_DEBUG_GRAPH_VARIANT_AUDIT=1"
unset FX SB f src
