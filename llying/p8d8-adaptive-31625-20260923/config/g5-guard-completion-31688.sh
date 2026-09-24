#!/usr/bin/env bash
# New allocation, optimization only; retain the historical A0 reference.
source /perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923/config/g1-guard-completion.sh
ALLOCATION_JOB_ID=31688
RUN_ID=g5-guard-completion-31688
RUN="$TRACE_RUNTIME/runs/$RUN_ID"
CONTAINER_PREFIX=llying-adaptive-31688
DIAG_SOURCE_PREFILL="$RUN_ID"
DIAG_SOURCE_DECODE="$RUN_ID"
SERVICE_RUN_ID="$RUN_ID"
AITER_JIT_CACHE_ROOT=/tmp/aiter-jit-100078-g5-31688
SSH_OPTS='-F /dev/null -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/bench-agentx-known-hosts'
export ALLOCATION_JOB_ID RUN_ID RUN CONTAINER_PREFIX DIAG_SOURCE_PREFILL DIAG_SOURCE_DECODE SERVICE_RUN_ID AITER_JIT_CACHE_ROOT SSH_OPTS
