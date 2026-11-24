#!/usr/bin/env bash
# Exercise smoke.sh's nic_tx / gem_procs helpers against a live deployment (read-only). Run on the node.
KIT=/perf_apps/liyingli/bench_agentx/upstream/Infera-vllm-1p1d-dsv41/examples/vllm_1p1d_dsv41_flash
export CTR=${CTR:-llying-v1p1d-pd} PREFILL_NODE=$(hostname -s)
source "$KIT/common.sh"
sed -n '/^nic_tx()/,/^gem_procs/p' "$KIT/engine/smoke.sh" | sed '$d' > /tmp/smoke_helpers.sh
sed -n '/^gem_procs()/,/echo \\\$n/p' "$KIT/engine/smoke.sh" >> /tmp/smoke_helpers.sh
bash -n /tmp/smoke_helpers.sh || { cat /tmp/smoke_helpers.sh; exit 1; }
source /tmp/smoke_helpers.sh
echo "nic_tx bytes: $(nic_tx)"
echo "decode (4,5) GEM procs: $(gem_procs "$PREFILL_NODE" 4,5)"
echo "prefill (0,1) GEM procs: $(gem_procs "$PREFILL_NODE" 0,1)"
