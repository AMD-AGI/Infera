#!/usr/bin/env bash
# Assemble the crusoe run directory ($R of llying/vllm-pd-opt-20261123) for the a49 replay on AUS, from the Infera branch
# at REF_COMMIT, unchanged except:
#   1. paths (sed, listed below): crusoe run dir -> replay/R, crusoe Infera checkout -> REF, model, work dir, image tag;
#   2. replay/aus.patch: site logic (GEM / peer memory per node, host libionic, GPU-NIC map from the probe, prebuilt
#      binaries optional because the image built from REF_COMMIT already has the VMM GEM library and the router).
# Idempotent; run on the login node (copies a few hundred KB) or as the first step of a replay item.
set -euo pipefail
. /perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/site/env.sh
R=$W/replay/R
L=$REF/llying
rm -rf "$R/scripts" "$R/once" "$R/kit" "$R/overlay-v3comm" "$R/overlay-cpupin"
mkdir -p "$R"/{scripts,once,kit,logs,runs,results,work,overlay-v3comm/vllm/v1/attention/ops,overlay-v3comm/vllm/distributed/device_communicators,overlay-cpupin/vllm/v1/kv_offload/cpu}
# Scripts: the tools set, then the newer deploy scripts of vllm-pd-opt-20261123 over it (what the crusoe $R/scripts held).
cp -r "$L/vllm-pd-tools-20261117/scripts/." "$R/scripts/"
cp "$L/vllm-pd-opt-20261123/deploy/"*.sh "$R/scripts/"
cp "$L/vllm-pd-tools-20261117/kit/constraints.txt" "$R/kit/"
cp "$L/vllm-pd-opt-20261123/once/p4d4-pair12.sh" "$L/vllm-pd-opt-20261123/once/score-a49-B.env" "$R/once/"
# Overlays the crusoe $R held (inventory: TA/HANDOFF-to-main.zh-CN.md section 2, vllm-p4d4-crusoe overlay-cpupin).
cp "$L/vllm-decode-trackA-20261118/overlay-v3/rocm_aiter_mla_sparse.py" "$R/overlay-v3comm/vllm/v1/attention/ops/"
cp "$L/vllm-kf-port-20261117/overlay/vllm/v1/attention/ops/rocm_gluon_paged_mqa_logits.py" "$R/overlay-v3comm/vllm/v1/attention/ops/"
cp "$L/vllm-decode-trackA-20261118/overlay-comm/cuda_communicator.py" "$R/overlay-v3comm/vllm/distributed/device_communicators/"
cp "$L/vllm-p4d4-crusoe-20261117/overlay-cpupin/vllm/v1/kv_offload/cpu/gpu_worker.py" "$R/overlay-cpupin/vllm/v1/kv_offload/cpu/"
# 1. paths
grep -rl -E '/home/liyingli|/mnt/m2m_nobackup|infera-vllm:v0340-0c5e5e89' "$R/scripts" "$R/once" | xargs -r sed -i \
    -e "s#/home/liyingli/bench_agentx/vllm-pd-20261117#$R#g" \
    -e "s#/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52#$REF#g" \
    -e "s#/mnt/m2m_nobackup/liyingli_[a-z]*/models/GLM-5.2-MXFP4#$MODEL#g" \
    -e "s#/mnt/m2m_nobackup/liyingli_vpd/work#$R/work#g" \
    -e "s#infera-vllm:v0340-0c5e5e89#$IMAGE_V0340#g" \
    -e "s#$REF/llying/vllm-p4d4-crusoe-20261117/results#$R/results/rounds#g" \
    -e 's#^OO=\$B/results/#OO=$R/results/#'
# (the last two keep every result under replay/R/results: REF stays a clean checkout for the image build)
left=$(grep -rn -E '/home/liyingli|/mnt/m2m_nobackup' "$R/scripts" "$R/once" | grep -v -E '^\S+:[0-9]+:\s*#' || true)
[[ -z "$left" ]] || { echo "crusoe paths left:"; echo "$left"; }
# 2. site logic
[ -s "$W/replay/aus.patch" ] && patch -s -p1 -d "$R" < "$W/replay/aus.patch"
# InferenceX at the recipe's pinned commit (agentx.sh checks it; cloned from the local clone, submodule from GitHub).
IXR=$R/work/cache/InferenceX
if [[ ! -d "$IXR/.git" ]]; then
    git clone -q "$IX" "$IXR" && git -C "$IXR" checkout -q --detach 918524ff94045b3f091115f1051c22a8588edf2b
    git -C "$IXR" remote set-url origin https://github.com/SemiAnalysisAI/InferenceX.git
    git -C "$IXR" submodule update -q --init --recursive
fi
echo "assembled $R from $(git -C "$REF" rev-parse --short HEAD)"
