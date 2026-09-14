#!/usr/bin/env bash
# Check only the nodes and GPUs selected by topology.tsv.
set -euo pipefail
COMPONENT=check-nodes
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/lib/common.sh"

load_config "$@"
init_ssh
start_log
print_topology
check_nodes_idle
log "PASS: all selected nodes are ready"
