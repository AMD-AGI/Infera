# GPU tests (1 GPU, gfx950) of the Triton PREFILL top-k (scripts/vpd_triton_topk_prefill.py) and its overlay
# (scripts/make_tritontopk_prefill_overlay.py), before any S_TOPK_PREFILL=1 pair. Builds, from the image's files, the
# idxfp4 overlay (scripts/make_idxfp4_overlay.py) and on top of it the prefill top-k overlay (the FP8 ops file from the
# image's own, the MXFP4 ops file from idxfp4's), then with the idxfp4 files mounted (its ops file replaced by ours, as
# p4d4-pair14.sh mounts the prefill) and both ops files + the two kernel modules mounted over the image's:
#   1. hooks: switch on, both patched modules import, each hook is exact on a prefill-shaped call and declines the
#      shapes it does not cover;  off: switch unset, the modules import without the kernel and every hook declines;
#   2. exact: exactness against torch.topk, rows 1 .. 4096, lengths 2K .. 1M, 9 distributions incl. ties / -inf /
#      out-of-row garbage; per-path row counts (whole-row fallback must stay rare);
#   3. time: against aiter's one-block top-k (MXFP4 path: top_k_per_row_decode; FP8 path: top_k_per_row_prefill) on
#      full-budget prefill chunks (rows x width = 128M fp32), + a collect BLOCK / num_warps sweep;
#   4. regression: test_idxfp4.py gpu with VPD_TOPK_IMPL=triton (the decode hook of the same, now chained, MXFP4 ops
#      file) and test_tritontopk.py on the decode overlay built from v3comm's ops file (unchanged decode path).
# ~20 min. Results in $B/results/topk-prefill-1gpu (one file per step, summary.txt). ONCE=test-topk-prefill-1gpu
# (copy this file to $R/once/ first: run_once.sbatch runs $R/once/$ONCE.sh).
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}   # run_once.sbatch does not export R
cd "$R" || exit 0
B=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-pd-opt-20261123
O=$B/results/topk-prefill-1gpu
IMAGE=infera-vllm:v0340-0c5e5e89
SP=/usr/local/lib/python3.12/dist-packages
OPS=$SP/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
MX=vllm/v1/attention/ops/rocm_paged_mxfp4_indexer.py
mkdir -p "$O"
cp "$0" "$O/once.sh" 2>/dev/null
gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
image_id=$(docker image inspect -f '{{.Id}}' "$IMAGE")
jit=vpd-aiter-jit-${image_id:7:12}
ov=$R/runs/topk-prefill-1gpu-$JOB-ov
rm -rf "$ov"; mkdir -p "$ov/idxfp4" "$ov/topkp" "$ov/mods" "$ov/dec" && chmod -R 777 "$ov"
cp "$B/scripts/vpd_triton_topk_decode.py" "$B/scripts/vpd_triton_topk_prefill.py" "$ov/mods/"
cp "$R/overlay-v3comm/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py" "$ov/dec/v3comm_ops_rocm_aiter_mla_sparse.py"
docker run --rm --network none -v "$B:$B:ro" -v "$ov:$ov" --entrypoint bash "$IMAGE" -c \
    "md5sum $OPS $SP/$MX && python3 $B/scripts/make_idxfp4_overlay.py $SP $ov/idxfp4 \
     && python3 $B/scripts/make_tritontopk_prefill_overlay.py $OPS $ov/topkp/ops_rocm_aiter_mla_sparse.py \
            $ov/idxfp4/rocm_paged_mxfp4_indexer_ops.py $ov/topkp/rocm_paged_mxfp4_indexer_ops.py \
     && python3 $B/scripts/make_tritontopk_overlay.py $ov/dec/v3comm_ops_rocm_aiter_mla_sparse.py $ov/dec/ops_rocm_aiter_mla_sparse.py" \
    > "$O/build.txt" 2>&1 \
    || { echo "$(date -u +%m%d-%T) topk-prefill-1gpu: overlay build failed, see $O/build.txt" >> logs/summary.out; exit 0; }
# the prefill's mounts: idxfp4's files with its ops file replaced by ours (one mount per destination), our FP8 ops file,
# the two kernel modules next to vllm
pf_v=()
while IFS=: read -r f rel; do
    src=$ov/idxfp4/$f
    [[ "$rel" == "$MX" ]] && src=$ov/topkp/rocm_paged_mxfp4_indexer_ops.py
    pf_v+=(-v "$src:$SP/$rel:ro")
done < "$ov/idxfp4/mounts.txt"
pf_v+=(-v "$ov/topkp/ops_rocm_aiter_mla_sparse.py:$OPS:ro")
pf_v+=(-v "$ov/mods/vpd_triton_topk_decode.py:$SP/vpd_triton_topk_decode.py:ro" -v "$ov/mods/vpd_triton_topk_prefill.py:$SP/vpd_triton_topk_prefill.py:ro")
printf '%s\n' "${pf_v[@]}" | grep -v '^-v$' > "$O/mounts.txt"
dups=$(grep -o ':/[^:]*:ro$' "$O/mounts.txt" | sort | uniq -d)
[[ -z "$dups" ]] || { echo "$(date -u +%m%d-%T) topk-prefill-1gpu: destination mounted twice: $dups" >> logs/summary.out; exit 0; }
gpu() {   # NAME TIMEOUT ENV CMD...: run CMD (bash -c) on GPU 0 with the prefill mounts, output in $O/NAME.txt
    local name=$1 t=$2 envs=$3; shift 3
    local e=(); for x in $envs; do e+=(-e "$x"); done
    timeout "$t" docker run --rm --init --network none --device /dev/kfd --device /dev/dri "${gids[@]}" -e HIP_VISIBLE_DEVICES=0 \
        -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 "${e[@]}" -v "$jit:$SP/aiter/jit" -v "$B:$B:ro" -v "$ov:$ov:ro" \
        "${pf_v[@]}" --entrypoint bash "$IMAGE" -c "cd /tmp && $*" \
        2>&1 | grep -v -E "^\[aiter\]|queue-intercept" > "$O/$name.txt"
    local rc=${PIPESTATUS[0]}
    echo "$name rc=$rc $(grep -a -E "passed|FAILED|Error|error" "$O/$name.txt" | tail -1)" | tee -a "$O/summary.txt"
}
: > "$O/summary.txt"
T="python3 $B/scripts/test_tritontopk_prefill.py"
P2="vllm.v1.attention.ops.rocm_aiter_mla_sparse vllm.v1.attention.ops.rocm_paged_mxfp4_indexer"   # the mounted (patched) modules
gpu hooks 900 "VPD_TOPK_PREFILL=1" "$T hooks $P2"
gpu off 600 "" "$T off $P2"
gpu exact 2400 "VPD_TOPK_PREFILL=1" "$T exact"
gpu time 2400 "VPD_TOPK_PREFILL=1" "$T time"
gpu idxfp4_decode_hook 900 "VPD_TOPK_IMPL=triton" "python3 $B/scripts/test_idxfp4.py gpu"
gpu decode_overlay 900 "VPD_TRITON_TOPK=1 PYTHONPATH=$ov/mods" "python3 $B/scripts/test_tritontopk.py $ov/dec/ops_rocm_aiter_mla_sparse.py"
echo "$(date -u +%m%d-%T) topk-prefill-1gpu tests in $O: $(grep -c ' rc=0 ' "$O/summary.txt")/6 rc=0" >> logs/summary.out
exit 0
