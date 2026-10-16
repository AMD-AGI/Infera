#!/usr/bin/env bash
# Every value this kit binds or depends on, each in a ${X:=default} form so a
# caller can re-point it without editing a file and the kit runs with none of
# them set. Sourced by deploy.sh, wait_ready.sh, teardown.sh and verify.sh.
#
# Shape: Qwen3.6-27B (Qwen3_5ForConditionalGeneration: dense, hybrid
# linear/full attention) on SGLang through infera, MIX (aggregated) mode, TP1,
# one MI300X (gfx942), behind the infera kv-aware router, etcd for discovery.
# Adapted from the infera repository's examples/sglang_mix_glm5.3 (env.sh,
# engine/up.sh, engine/worker.sh): the same container -> etcd -> worker -> router
# sequence, GLM-only engine flags removed, every shared-host identifier made a
# parameter.

KIT_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_DIR="$(dirname "$KIT_SCRIPTS")"

# ---- the contracted parameters (deploy_kit.layout.yaml runtime_contract) ----
# Unique per invocation. Every container name and the ownership label carry it,
# which is what lets teardown.sh remove exactly what this invocation created.
: "${E2E_KIT_RUN_TAG:=r16aea3}"
# First port of a contiguous band of SIX: router, worker, etcd client, etcd
# peer, kv-events publisher, kv-snapshot. 18101 is the band this kit was proven on.
: "${E2E_KIT_PORT_BASE:=18101}"
# This invocation's workdir: logs, staged launch scripts, handshake. LOCAL disk
# (a root_squash network home silently stops the engine writing its logs).
: "${E2E_KIT_WORK_ROOT:=/var/tmp/yihou/demoladder/l2a/work/${E2E_KIT_RUN_TAG}}"
# Appended LAST to the engine worker's argv, so it overrides an earlier flag.
: "${E2E_KIT_ENGINE_EXTRA_ARGS:=}"
# Space-separated K=V list exported into the worker process only (not the
# container, not the router).
: "${E2E_KIT_ENGINE_EXTRA_ENV:=}"
# Appended LAST to the router's argv (e.g. --enable-profiling).
: "${E2E_KIT_ROUTER_EXTRA_ARGS:=}"
# Device indices (rocm-smi / HIP numbering), comma-separated. 6 is the card the
# operator named for this run (E2E_GPU_DEVICES=6). A caller naming cards is
# obeyed exactly; deploy.sh refuses rather than substitutes if one is busy.
: "${E2E_KIT_GPU_DEVICES:=6}"
# Largest decode batch captured in a CUDA graph. 32 = the heaviest downstream
# consumer's load: stage 2's closed-loop trace replay is capped at
# E2E_MAX_CONC=32 (shared.yaml `max_conc:-32`) and measured 25.42 effective
# decode concurrency on rung 2e; it also covers check_deploy_serves's conc-16
# load. Above the ceiling decode runs eager (measured 4.6x slower). If a
# consumer raises its concurrency above 32 it must raise this too. notes.md.
: "${E2E_KIT_CUDA_GRAPH_MAX_BS:=32}"

# ---- site / model facts (defaults are this run's; override for another host) --
: "${E2E_KIT_NODE_IP:=127.0.0.1}"
: "${E2E_KIT_IMAGE:=infera:sglang-gfx942-glm52}"
: "${E2E_KIT_ETCD_IMAGE:=quay.io/coreos/etcd:v3.5.14}"
: "${E2E_KIT_MODEL_PATH:=/data/yihou_hf_cache/models/Qwen3.6-27B}"
: "${E2E_KIT_SERVED_NAME:=Qwen3.6-27B}"
: "${E2E_KIT_TP:=1}"
: "${E2E_KIT_CTX:=32768}"
: "${E2E_KIT_MEM_FRACTION:=0.85}"
# Model-specific flag groups; the literal `none` means "no flags".
# Qwen3.6 has no DSA path; it thinks before answering, so it needs the qwen3
# reasoning parser or the chain of thought lands in `content`.
: "${E2E_KIT_DSA_ARGS:=none}"
: "${E2E_KIT_PARSER_ARGS:=--reasoning-parser qwen3}"
# Container names keep `yihou` (the ownership marker on this shared host); the
# run tag varies.
: "${E2E_KIT_CONTAINER_PREFIX:=yihou_l2a}"
: "${E2E_KIT_CONTAINER:=${E2E_KIT_CONTAINER_PREFIX}_${E2E_KIT_RUN_TAG}}"
: "${E2E_KIT_ETCD_CONTAINER:=${E2E_KIT_CONTAINER}_etcd}"
: "${E2E_KIT_LABEL_KEY:=infera_e2e_run}"
# Readiness budget, sized from what was read: 51 GB of bf16 weights from local
# NVMe (ext4) plus mamba/KV pool allocation, graph capture and AITER JIT
# warm-up, measured in notes.md. 1800 s is several times that -- slack for a
# cold page cache, not a target. Reading weights over NFS needs far more.
: "${E2E_KIT_READY_TIMEOUT:=1800}"
# Roots a later stage must be able to `docker exec` into and measure in, mounted
# read-write at the same path. Space-separated. Default: the run root this kit
# sits in (derived from the kit's own path, up to runs/<id>) and the run's
# node-local scratch root. lib_mounts.sh says which mount forms are allowed.
_kit_run_root=""
case "$KIT_DIR" in
  */runs/*/*) _r="${KIT_DIR%%/runs/*}"; _rest="${KIT_DIR#*/runs/}"; _kit_run_root="$_r/runs/${_rest%%/*}" ;;
esac
: "${E2E_KIT_MEASURE_ROOTS:=${_kit_run_root} /var/tmp/yihou/demoladder/l2a/scratch}"

# ---- derived purely from the parameters above (each still pinnable) ----------
: "${KIT_ROUTER_PORT:=$((E2E_KIT_PORT_BASE + 0))}"
: "${KIT_WORKER_PORT:=$((E2E_KIT_PORT_BASE + 1))}"
: "${KIT_ETCD_PORT:=$((E2E_KIT_PORT_BASE + 2))}"
: "${KIT_ETCD_PEER_PORT:=$((E2E_KIT_PORT_BASE + 3))}"
: "${KIT_KV_EVENTS_PORT:=$((E2E_KIT_PORT_BASE + 4))}"
: "${KIT_KV_SNAPSHOT_PORT:=$((E2E_KIT_PORT_BASE + 5))}"
KIT_ALL_PORTS="$KIT_ROUTER_PORT $KIT_WORKER_PORT $KIT_ETCD_PORT $KIT_ETCD_PEER_PORT $KIT_KV_EVENTS_PORT $KIT_KV_SNAPSHOT_PORT"
: "${KIT_ROUTER_URL:=http://${E2E_KIT_NODE_IP}:${KIT_ROUTER_PORT}}"
: "${KIT_ENGINE_URL:=http://${E2E_KIT_NODE_IP}:${KIT_WORKER_PORT}}"
: "${KIT_LOG_DIR:=${E2E_KIT_WORK_ROOT}/logs}"
: "${KIT_LAUNCH_DIR:=${E2E_KIT_WORK_ROOT}/launch}"
: "${KIT_HANDSHAKE:=${E2E_KIT_WORK_ROOT}/deployment.json}"
: "${KIT_LABEL:=${E2E_KIT_LABEL_KEY}=${E2E_KIT_RUN_TAG}}"

case "$E2E_KIT_CONTAINER" in *yihou*) ;; *)
  echo "env.sh: container name '$E2E_KIT_CONTAINER' lacks the 'yihou' ownership marker; refusing" >&2
  return 2 2>/dev/null || exit 2 ;;
esac
