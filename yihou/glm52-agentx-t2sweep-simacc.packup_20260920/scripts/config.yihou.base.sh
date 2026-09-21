#!/usr/bin/env bash
# Purpose: 1P1D same-rail C64 run on crsuse2-m2m-135 (prefill) + -138 (decode).
# Usage: pass CONFIG=<this file> TOPOLOGY=<workspace>/topology.yihou.tsv to the
#   bench scripts. Everything not set here is inherited from config.sh.
# Artifacts: none; this file only defines shell variables.

# Only the values that differ from config.sh are set here. They use ":=" so a
# KEY=VALUE argument on the command line still wins: the bench scripts export
# those before sourcing this file.

: "${IMAGE:=infera-sglang:v0519-yihou-0917-nextnfix-hicache}"
: "${SGLANG_BASE_IMAGE:=lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260917}"

: "${CONTROL_NODE:=crsuse2-m2m-135}"
: "${BUILDER_NODE:=crsuse2-m2m-135}"

# Every container this task creates carries "yihou", so a stray container can
# always be attributed and is safe to remove under the deletion rule.
: "${CONTAINER_PREFIX:=glm52-pd-yihou-agentx}"

# Same-rail KV transfer: constrain Decode to the Prefill effective DP rank so
# Mooncake never crosses rails (issue.md 3.1). launch.sh translates 1/0 to the
# true/false that the router's clap ArgAction::Set requires.
: "${PD_DP_RANK_AFFINITY:=1}"

# P4D4 on GPUs 2,3,4,5 of each node, not P8D8 on all eight.
#
# Two constraints pick this set:
#  - crsuse2-m2m-135 GPU[1] is held by a root-owned Kubernetes pod
#    (pod5d84e491, "vllm serve Qwen3-32B", up 22 days) occupying 96.94 GB of
#    309.22 GB. Live workload, no sudo, no kubectl. Avoiding GPU[1] keeps
#    mem_fraction_static at the validated 0.85 instead of shrinking the KV
#    budget to squeeze in beside it.
#  - crsuse2-m2m-135's ionic_7 is defective: no netdev (enP3p0s12 absent) and
#    its GID at index 1 is all-zero. Rail 0200 is therefore missing on that
#    node. Avoiding GPU7 avoids that dead rail.
# GPUs 2,3,4,5 satisfy both and leave four healthy dedicated rails.
: "${PREFILL_GPU_DEVICES:=2,3,4,5}"
: "${DECODE_GPU_DEVICES:=2,3,4,5}"
: "${PREFILL_TP:=4}"
: "${PREFILL_DP:=4}"
: "${DECODE_TP:=4}"
: "${DECODE_DP:=4}"

# Same-rail KV transfer at the Mooncake layer -- pin each GPU to ONE HCA.
#
# This is the fix for the real cause of the cross-rail failures, and it is
# separate from the router-level PD_DP_RANK_AFFINITY above. Both are needed:
# the router guarantees Prefill rank i hands off to Decode rank i, and this
# guarantees rank i then transfers over the rail that rank i's GPU owns.
#
# Measured, first-hand: with the shared-list form below, preflight's per-GPU
# VRAM transfers FAIL on GPUs 4-7 in both directions between 135 and 138,
# because Mooncake's auto-discovery lets the two ends pick DIFFERENT HCAs from
# the NUMA-local pool -- and these rails are physically isolated (ionic_i can
# only reach ionic_i), so a mismatch is unreachable, not merely slow. Pinning
# both ends to one NIC made all 8 GPUs pass, verified, in both directions
# (results/yihou-1p1d-c64/preflight/pin-ionic4-b/).
#
# sglang's parse_ib_device_config() accepts three forms: a shared list
# ("ionic_0, ionic_1, ..."), a per-GPU JSON mapping, or a path to a JSON file.
# The shared list is what leaves the choice to auto-discovery. The JSON keys
# are the LOCAL (visible) device index, so under HIP_VISIBLE_DEVICES=2,3,4,5
# key 0 is physical GPU2. The GPU<->NIC pairing is the one proven in
# infera.glm52.view/rdma.survey.8node.packup_20260917 (GPU_n <-> ionic_n).
#
# ssh_run quotes arguments with printf %q, so the inline JSON survives the ssh
# hop and the docker run argument array intact.
# Assigned with an explicit test, not "${VAR:=...}": a bare closing brace inside
# a parameter expansion terminates it early and silently truncates the JSON.
if [[ -z "${RDMA_DEVICE:-}" ]]; then
    RDMA_DEVICE='{"0":"ionic_2","1":"ionic_3","2":"ionic_4","3":"ionic_5"}'
fi

# MC_TE_FILTERS defaults to $RDMA_DEVICE in config.sh, which would hand the
# transport engine a JSON blob where it expects a device list. Set it plainly.
: "${MC_TE_FILTERS:=ionic_2,ionic_3,ionic_4,ionic_5}"

# Half the GPUs, so half the concurrency of the C64 alignment point: this run
# Concurrency is not a config knob -- agentx_bench.sh takes it as a required
# CONC=<n> argument -- so this file says nothing about it.

# The corrected MTP configuration. Custom all-reduce on the decode leg corrupts
# the speculative path on this stack: without this flag the first token is
# correct and every later token degenerates, at accept len 1.25. Established by
# a five-round single-variable A/B on 2026-09-18; evidence and reproduction in
# yihou/glm52-mtp-garbled-decode-rootcause.packup_20260918/. Decode leg only --
# prefill's output was shown correct with custom all-reduce left on.
#
# Delivered through DECODE_EXTRA_ARGS, an escape hatch in engine.sh, because
# sglang exposes no positive/negative pair for this and the only other code path
# that disables it (--enable-deterministic-inference) changes much more besides.
# Single dash, not ":=", so a config that sets this to the EMPTY string keeps it
# empty instead of having the default reassigned. T2 needs exactly that: see
# config.yihou.full.sh.
: "${DECODE_EXTRA_ARGS=--disable-custom-all-reduce}"

# Which repo-level config to inherit from. config.sh is the validated baseline
# (HiCache off, max-running 64, AgentX 1200 s / warmup 1 == "fast mode");
# config.full.sh is the feature target (prefill HiCache on, 128, 3600 s / warmup
# 10 == "full mode"). Set by the per-test config that sources this file.
: "${YIHOU_BASE_CONFIG:=config.sh}"
case "$YIHOU_BASE_CONFIG" in
    config.sh | config.full.sh) ;;
    *) echo "unexpected YIHOU_BASE_CONFIG: $YIHOU_BASE_CONFIG" >&2; return 2 ;;
esac
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/$YIHOU_BASE_CONFIG"
