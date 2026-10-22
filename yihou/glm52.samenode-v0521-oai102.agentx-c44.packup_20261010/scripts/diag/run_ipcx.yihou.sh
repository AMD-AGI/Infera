#!/usr/bin/env bash
# Cross-container HIP IPC matrix on one node (plain ssh). usage: run_ipcx.yihou.sh <node> [exp_gpu=4] [other_gpu=0]
set -uo pipefail
node=${1:?node}; EG=${2:-4}; OG=${3:-0}
D=/shared_nfs/yihou_glm52/v0521dbg_20261010
W=/mnt/m2m_nobackup/yihou_v0521dbg/ipctest/$(date -u +%Y%m%d-%H%M%S)
SSH=(ssh -o BatchMode=yes -o ClearAllForwardings=yes "$node")
declare -A IMG=([v0521]=infera-sglang:v0521-yihou-1009 [v0519]=infera-sglang:v0519-yihou-0917-nextnfix-hicache)
AGENT="-e HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2 -e HSA_ENABLE_DEBUG=1 -e ROCM_DEBUG_AGENT_OPTIONS=--disable-linux-signals"
COMMON="--device /dev/kfd --device /dev/dri --group-add video --network host --ipc host --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined -v $D:$D:ro"
echo "workdir $node:$W"
printf "%-6s %-7s %-5s %-6s | %-60s | %s\n" image mode agent impGPU importer exporter
for im in v0521 v0519; do for mode in default legacy; do for ag in off on; do for ig in other same; do
  c="$im.$mode.$ag.$ig"; cell=$W/$c
  env="-e HIP_VISIBLE_DEVICES"; [[ $mode == legacy ]] && L="-e HSA_ENABLE_IPC_MODE_LEGACY=1" || L=""
  [[ $ag == on ]] && A="$AGENT" || A=""
  igpu=$([[ $ig == same ]] && echo $EG || echo $OG)
  "${SSH[@]}" "mkdir -p $cell && \
    docker run -d --rm --name yihou-ipc-exp-$c $COMMON -v $cell:$cell $L $A -e HIP_VISIBLE_DEVICES=$EG --entrypoint bash ${IMG[$im]} -c 'python3 $D/diag/ipc_xcontainer.yihou.py exp $cell > $cell/exp.log 2>&1' >/dev/null && \
    for i in \$(seq 150); do [ -f $cell/handle.bin ] && break; grep -q RESULT $cell/exp.log 2>/dev/null && break; sleep 0.2; done; \
    [ -f $cell/handle.bin ] && timeout 120 docker run --rm --name yihou-ipc-imp-$c $COMMON -v $cell:$cell $L $A -e HIP_VISIBLE_DEVICES=$igpu --entrypoint bash ${IMG[$im]} -c 'python3 $D/diag/ipc_xcontainer.yihou.py imp $cell' > $cell/imp.log 2>&1; \
    [ -f $cell/done ] || echo nohandle > $cell/done; \
    for i in \$(seq 100); do grep -q 'RESULT exp' $cell/exp.log 2>/dev/null && break; sleep 0.2; done" 
  r=$("${SSH[@]}" "cat $cell/imp.log $cell/exp.log 2>/dev/null" | strings)
  ir=$(echo "$r" | grep -aE "RESULT imp" | head -1); [[ -z $ir ]] && ir="(no importer result)"
  ie=$(echo "$r" | grep -a "^\[imp" | grep -av "rc=0 " | head -1 | sed -E 's/^\[[^]]*\] //')
  er=$(echo "$r" | grep -aE "RESULT exp" | head -1)" $(echo "$r" | grep -a "^PROOF" | sed -E 's/PROOF exp //; s/ handle16=.*//')"
  printf "%-6s %-7s %-5s %-6s | %-60s | %s\n" $im $mode $ag $ig "${ir#RESULT } ${ie}" "${er#RESULT }"
done; done; done; done
"${SSH[@]}" "docker ps -a --format '{{.Names}}' | grep yihou-ipc || true"
