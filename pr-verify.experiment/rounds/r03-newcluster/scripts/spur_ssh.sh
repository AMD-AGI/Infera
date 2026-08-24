#!/usr/bin/env bash
# SSH_CMD shim for the spur scheduler, for examples/sglang_1p1d_glm5.2/engine/up.sh.
#
# up.sh reaches each node as `$SSH_CMD <node> <command>` and says so at line 18:
# "On a scheduler that blocks ssh to compute nodes, set SSH_CMD to whatever runs a
# command on a node." That is this script.
#
# ssh to a compute node is refused here -- each node's sshd carries an
# `AllowUsers ubuntu root` whitelist, so a normal user is rejected at preauth with a
# misleading "Permission denied (publickey)". `spur exec <jobid>` goes through spurd
# and works.
#
# Two things this has to fix up beyond the name->jobid mapping:
#   * `spur exec` runs with HOME=/opt/spur, which is NOT writable -- docker then dies
#     with "mkdir /opt/spur/.docker: permission denied".
#   * it runs with pwd=/, so anything relative has to be made absolute by the caller.
#
# Usage:  SSH_CMD="/path/to/spur_ssh.sh" ...  (node names are looked up in NODE_MAP)
set -uo pipefail

# node -> spur job id. Override with NODE_MAP="node1=111,node2=222".
NODE_MAP="${NODE_MAP:-crsuse2-m2m-237=58799,crsuse2-m2m-106=58800}"

node="$1"; shift

jobid=""
IFS=',' read -ra _pairs <<< "$NODE_MAP"
for _p in "${_pairs[@]}"; do
  if [ "${_p%%=*}" = "$node" ]; then jobid="${_p#*=}"; break; fi
done
[ -n "$jobid" ] || { echo "spur_ssh: no job id for node '$node' in NODE_MAP=$NODE_MAP" >&2; exit 2; }

exec spur exec "$jobid" bash -c "export HOME=${REMOTE_HOME:-/home/yihou}; $*"
