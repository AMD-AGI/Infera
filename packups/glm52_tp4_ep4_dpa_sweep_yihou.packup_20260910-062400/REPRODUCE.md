# Reproduce the ten-point TP4/EP4 DPA sweep

## Prerequisites
An existing authorized allocation with four idle MI355X GPUs on one node, Docker/Spur access and shared model access. Do not request/cancel allocations. Never use234/036/249 for this task; coordinate resource changes. Historical056/job130737 is not guaranteed current. Do not copy the historical cleanup operations: those had specific user authorization, not general permission to stop other workloads.

External inputs (not copied):
- Image archive `/shared_nfs/yihou/playground/glm52_mix_repro_20260908/image/rocm-llm-bench.tar.zst`;23912216852bytes; SHA256 `a6d047de9e3791b3c819f9035b0c3a4f632663b5cf8f85c3147b4f5b8204f9a2`.
- Weights `/shared_nfs/models/GLM-5.2-MXFP4`, mounted read-only.
- Host GPU driver and Docker daemon. No dataset, inference API key or HF token is required. Registry credentials only if pulling/rebuilding, provided by your approved credential provider, never logged.

All commands below run in one Bash login shell. Required local tools: Python3, Bash, Git, zstd, sha256sum, Spur/squeue and Docker client on the allocated node. Runtime packages are inside the image.

## 1. Validate and restore into a new Git-traceable run folder

```bash
set -euo pipefail
export KIT=/home/yihou/dev/git/infera.dev.yihou.sglang.bench.fast.script/packups/glm52_tp4_ep4_dpa_sweep_yihou.packup_20260910-062400
(cd "$KIT" && sha256sum -c MANIFEST.sha256)
PYTHONDONTWRITEBYTECODE=1 python3 "$KIT/scripts/audit_packup_yihou.py"
export RUN_ROOT=/home/yihou/dev/git/infera.dev.yihou.sglang.bench.fast.script/reruns/tp4_ep4_yihou_$(date -u +%Y%m%d-%H%M%S)
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
cp -a "$KIT/scripts/original/." "$RUN_ROOT/"
mkdir -p "$RUN_ROOT/logs"
# run_decode.sh uses git rev-parse/diff. RUN_ROOT above is inside an existing repo;
# for a standalone external directory, initialize a Git repo and make a baseline
# commit with your own configured identity before invoking that launcher.
git -C "$RUN_ROOT" rev-parse HEAD
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s "$RUN_ROOT/tests" -v
PYTHONDONTWRITEBYTECODE=1 python3 "$RUN_ROOT/scripts/test_run_decode_yihou.py"
```

The copied scripts/bench are byte-identical. Per-run bench_snapshot and hashes also preserve untracked files; Git diff alone does not. Do not edit the archived evidence or reuse archived output directories.

## 2. Choose existing resources and load the exact image

```bash
squeue -u "$USER" -o '%i %u %T %N %M %l'
read -r -p 'Existing authorized job ID: ' JOB_ID
read -r -p 'Its single node: ' NODE
export JOB_ID NODE
[[ "$JOB_ID" =~ ^[0-9]+$ && "$NODE" =~ ^crsuse2-m2m-[0-9]+$ ]]
[[ "$NODE" != crsuse2-m2m-234 && "$NODE" != crsuse2-m2m-036 && "$NODE" != crsuse2-m2m-249 ]]
[[ "$(squeue -j "$JOB_ID" -h -o '%u %T %N')" == "$USER RUNNING $NODE" ]]
remote() { local cmd; printf -v cmd '%q ' "$@"; spur exec "$JOB_ID" bash -lc "$cmd"; }
[[ "$(remote hostname)" == "$NODE" ]]
remote docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
remote docker info --format 'Root={{.DockerRootDir}} Status={{json .DriverStatus}}'
export IMAGE=sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d
export ARCHIVE=/shared_nfs/yihou/playground/glm52_mix_repro_20260908/image/rocm-llm-bench.tar.zst
printf '%s  %s\n' a6d047de9e3791b3c819f9035b0c3a4f632663b5cf8f85c3147b4f5b8204f9a2 "$ARCHIVE" | sha256sum -c -
zstd -t "$ARCHIVE"
# First confirm real daemon/containerd backing storage has sufficient capacity.
if ! remote docker image inspect "$IMAGE" >/dev/null 2>&1; then
    remote bash -lc 'set -euo pipefail; zstd -dc /shared_nfs/yihou/playground/glm52_mix_repro_20260908/image/rocm-llm-bench.tar.zst | docker load' | tee "$RUN_ROOT/logs/image_load.log"
fi
[[ "$(remote docker image inspect -f '{{.Id}}' "$IMAGE")" == "$IMAGE" ]]
```

Do not infer containerd space from namespace root df; do not prune shared images/change daemon storage. If archive unavailable, the included original Dockerfile at `references/optimization_stack/packup-scripts__repo__Dockerfile` pins SGLang/AITER but has a floating FROM. A cold build with base digest `lmsysorg/sglang@sha256:6d68cd19206716cb3f1e31e2ad89cd0852d7ae614a792773c30a4277f8955c72` is an explicitly different, unvalidated binary variant; it requires network/package access. The supplied launchers deliberately require the measured image ID, so do not claim a rebuilt image is identical or silently substitute it.

## 3. Create the container and verify environment

```bash
export CONTAINER=yihou-tp4ep4-repro-$(date -u +%Y%m%d-%H%M%S)
export GPU_DEVICES=0,1,2,3
export OUTPUT_ROOT="$RUN_ROOT/iterations"
bash "$RUN_ROOT/scripts/create_container_yihou.sh"
remote docker exec "$CONTAINER" rocm-smi --showuse --showmemuse --showpids --showdriverversion
remote docker exec "$CONTAINER" python3 -c 'import torch; print(torch.__version__,torch.cuda.device_count(),torch.cuda.get_device_name(0)); assert torch.cuda.device_count()==4'
[[ "$(remote docker exec "$CONTAINER" git -C /sglang rev-parse HEAD)" == 402df1e1e453e1e85ec0f5ac4052d36598cc691a ]]
[[ "$(remote docker exec "$CONTAINER" git -C /aiter rev-parse HEAD)" == 2c71811b32c8ce2e1266aedaec199df7d90f597d ]]
printf '%s  %s\n' 8b46225f7afd7181735c2dd97b925bcb70dfebb72b1a105d4ac0be7583a5c726 /shared_nfs/models/GLM-5.2-MXFP4/config.json \
 fd42188894abe9196fb70a2113c4fd1d0569b29a307a89c0e18e200111456fc6 /shared_nfs/models/GLM-5.2-MXFP4/model.safetensors.index.json | sha256sum -c -
```

Confirm chosen GPUs are idle and no conflicting workload exists; do not blindly stop another task. Retain fresh CPU/RAM/GPU/driver/package/source/image information under RUN_ROOT/logs. Historical logs cannot prove current health. Model config/index hashes do not cover every weight file.

## 4. Smoke both modes, then run ten full points

```bash
COMMON=(--tp-size 4 --ep-size 4 --batch-size 4 --input-len 1024 --output-len 32 \
  --warmup-steps 3 --max-steps 4 --enable-aiter-allreduce-fusion \
  --enable-fused-qk-norm-rope --mem-fraction-static 0.85)
bash "$RUN_ROOT/scripts/run_decode.sh" smoke_off_yihou "${COMMON[@]}"
bash "$RUN_ROOT/scripts/run_decode.sh" smoke_on_yihou "${COMMON[@]}" --enable-dp-attention
bash "$RUN_ROOT/scripts/run_tp4_ep4_sweep_yihou.sh"
python3 "$RUN_ROOT/scripts/collect_sweep_yihou.py" "$OUTPUT_ROOT" --output "$RUN_ROOT/summary.csv"
```

Use plain Python through the launcher, not torchrun. The sweep runs offC4/8/16/20/24 then onC4/8/16/20/24 sequentially. Every full point has input70000/output10000/expected acceptance3.61,warmup10,seed1234. DPAon uses four attention replicas, localC/4. The command stops on a failed point; fix and rerun deliberately with a new output directory, never overwrite evidence. First cold smoke took891s; hot full points153–217s; ten full points totaled1850s. Allow longer cold compilation rather than restarting on silence alone.

Each result must have `complete=true`, global useful outputs C*10000, final accounting lengths80000 and four ranks with target/draft/extension graph executes equal to verify iterations. Full per-rank local result aggregation must count one representative per DP shard. Result counters include actual graph execution, not just available runner objects.

**Backend caveat:** all DPAon points fall back from FlyDSL sparse MLA due to64heads; DPAoffC20/C24 verify falls back due to120/144rows>96. Do not 'fix' this silently when comparing this measured stack.4 indexer markers/rank set are still expected. See report for actual fallback evidence.

## 5. Preserve and stop only your idle container

```bash
remote docker top "$CONTAINER" -eo pid,ppid,etime,comm
# Only after all benchmark processes have exited:
[[ "$(remote docker inspect -f '{{index .Config.Labels "owner"}}' "$CONTAINER")" == "$USER" ]]
remote docker stop -t 300 "$CONTAINER"
```

No rm/prune, allocation cancellation or original-file deletion is part of reproduction. The retained container keeps its image-keyed AITER cache. This recipe was syntax/CPU-audited during packaging, not rerun on GPUs.
