#!/usr/bin/env bash
# Purpose: T1 -- AgentX CONC=40 in fast mode with REAL MTP acceptance, on the
#   corrected MTP configuration. The headline is spec_accept_length, not
#   throughput.
# Usage: CONFIG=<this file> TOPOLOGY=<workspace>/topology.yihou.tsv
# Artifacts: none; this file only defines shell variables.
#
# FAST MODE IS ALREADY WHAT config.sh DOES. InferenceX's AIPERF_EXPERIMENTAL_FAST
# (benchmarks/benchmark_lib.sh:1979) sets exactly two values: duration 1200 and
# warmup_requests_per_lane 1. config.sh:121,123 already set AGENTX_DURATION=1200
# and AGENTX_WARMUP_REQUESTS_PER_LANE=1, and agentx_env.py writes both into
# runtime.env. So sourcing config.sh IS fast mode; the env var is not plumbed
# through and setting it would change nothing.
#
# REAL ACCEPTANCE. config.sh uses ${DECODE_SIMULATE_ACC_LEN-3.61} -- a SINGLE
# dash -- so an empty-but-set value survives and disables simulation. With ":-"
# this line would silently do nothing.
: "${DECODE_SIMULATE_ACC_LEN=}"

# PR #37152 is in the image but INERT here: config.sh has PREFILL_HICACHE=0, so
# no HiCache copy kernel runs. It is carried anyway so T1 and T2 differ in the
# benchmark configuration and not in the binary.
: "${CONTAINER_PREFIX:=glm52-pd-yihou-agentx-fast}"

: "${YIHOU_BASE_CONFIG:=config.sh}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.yihou.base.sh"
