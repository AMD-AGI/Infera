#!/usr/bin/env bash
# Apply, inside a container of the stock lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018
# image, the five SGLang fixes this recipe still needs on that base, and verify
# each one reached the source. Idempotent: a second run reports "already applied".
#
#   B1 patch_dsa_indexer_hip_dp_padded_rows.py  sglang#33059  aiter indexer rows under DP padding
#   B2 dsa_dp_sync.diff                         sglang#46830  DP-rank D2H sync deadlock
#   B6 patch_kv_metrics_send_noblock.py         sglang#46820  scheduler blocks on KvMetrics PUSH
#   B7 patch_deepseek_v2_pp_embed_dim.py        sglang#47184  PP middle stage has no embedding
#   B8 patch_mla_absorb_bmm_default_config.py   sglang#46829  absorb-BMM config aborts Triton
#
# Not applied, by reading this base's source: B3 (index_share_for_mtp_iteration:false
# keeps page-table and top-k rows equal), B4 (only the overlap path is patched;
# PP prefill runs --disable-overlap-schedule), B5 (the HiCache controller moves
# indices per pool on this base), and the Responses / decode-radix patches.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="${SGLANG_DIR:-/sgl-workspace/sglang}"
SRT="$SGLANG_DIR/python/sglang/srt"
git_sgl() { git -c safe.directory='*' -C "$SGLANG_DIR" "$@"; }

echo "[stock-patches] B1 dsa_indexer DP-padded rows"
python3 "$here/patch_dsa_indexer_hip_dp_padded_rows.py"

echo "[stock-patches] B2 dsa_dp_sync.diff"
if git_sgl apply --check "$here/dsa_dp_sync.diff" 2>/dev/null; then
    git_sgl apply "$here/dsa_dp_sync.diff"
    echo "  applied"
elif git_sgl apply --check -R "$here/dsa_dp_sync.diff" 2>/dev/null; then
    echo "  already applied"
else
    git_sgl apply --check "$here/dsa_dp_sync.diff"
    exit 1
fi

for p in patch_kv_metrics_send_noblock.py patch_deepseek_v2_pp_embed_dim.py \
    patch_mla_absorb_bmm_default_config.py; do
    echo "[stock-patches] $p"
    python3 "$here/$p"
done

# module path : text that must be in the patched source
markers=(
    "layers/attention/dsa/dsa_indexer.py:_p1v2_trim"
    "layers/attention/dsa_backend.py:must not be None for DRAFT_EXTEND_V2"
    "managers/scheduler_components/kv_events_publisher.py:_infera_kv_metrics_noblock"
    "models/deepseek_v2.py:_infera_pp_embed_dim"
    "models/deepseek_common/attention_forward_methods/forward_mla_rocm.py:_infera_absorb_bmm_default"
)
fail=0
for spec in "${markers[@]}"; do
    f="$SRT/${spec%%:*}" m="${spec#*:}"
    rm -f "$(dirname "$f")/__pycache__/$(basename "$f" .py)".*.pyc
    if grep -qF -- "$m" "$f"; then
        echo "  ok   ${spec%%:*}"
    else
        echo "  MISSING ${spec%%:*} :: $m" >&2
        fail=1
    fi
done
[[ "$fail" == 0 ]] || { echo "[stock-patches] verification failed" >&2; exit 1; }
echo "[stock-patches] all five applied"
