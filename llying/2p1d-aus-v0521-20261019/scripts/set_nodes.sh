#!/usr/bin/env bash
# Point the kit at a new pair of AUS nodes: P = both prefills + control (etcd, router, client), D = decode.
# Rewrites config/topology.2p1d.aus.tsv (data IP = the node's fenic address), CONTROL_NODE / BUILDER_NODE in
# config/config.3p1d.sh, and the node names in the start_*.sh launchers and tune_watch.py.
# Usage: set_nodes.sh <P node> <D node>
set -euo pipefail
P="${1:?prefill/control node}"; D="${2:?decode node}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
ip_of() { ssh -o BatchMode=yes -o ConnectTimeout=10 -n "$1" "ip -4 -o addr show fenic | awk '{print \$4}' | cut -d/ -f1"; }
pip="$(ip_of "$P")"; dip="$(ip_of "$D")"
[[ -n "$pip" && -n "$dip" ]] || { echo "cannot read fenic IPs ($P='$pip' $D='$dip')" >&2; exit 1; }
printf 'role\tnode\tdata_ip\nprefill\t%s\t%s\nprefill\t%s\t%s\ndecode\t%s\t%s\n' "$P" "$pip" "$P" "$pip" "$D" "$dip" \
    > "$R/config/topology.2p1d.aus.tsv"
sed -i -E "s#^CONTROL_NODE=[^ ]+#CONTROL_NODE=$P#; s#^BUILDER_NODE=[^ ]+#BUILDER_NODE=$P#" "$R/config/config.3p1d.sh"
sed -i -E "s#smci355-ccs-aus-n[0-9]+-[0-9]+#$P#g" "$HERE/start_queue.sh" "$HERE/start_sampler.sh" "$HERE/start_hang_watch.sh"
python3 - "$HERE/tune_watch.py" "$P" "$D" <<'PY'
import re, sys
path, p, d = sys.argv[1:]
s = open(path).read()
s = re.sub(r'^CONTROL = .*$', f'CONTROL = "{p}"', s, flags=re.M)
s = re.sub(r'^NODES = .*$', f'NODES = {{"{p}": "0,1,2,3,4,5,6,7", "{d}": "0,1,2,3,4,5,6,7"}}', s, flags=re.M)
open(path, "w").write(s)
PY
echo "P=$P ($pip) D=$D ($dip)"
cat "$R/config/topology.2p1d.aus.tsv"
grep -n "^CONTROL_NODE=\|^BUILDER_NODE=" "$R/config/config.3p1d.sh"
