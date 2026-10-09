#!/usr/bin/env bash
# Start the isolated target8 deployment after component and RDMA gates pass.
set -euo pipefail
W=/home/yihou/dev/git/infera.glm52.pd/bench/glm5p2_pd/results/yihou-expert8-pd
ROUND="${1:-$W/rounds/002-bringup}"
[[ "$ROUND" == "$W/"* ]] || exit 2
[[ -s "$W/model/config.json" && -s "$W/patches/mounts.yihou.tsv" ]] || {
    printf 'model overlay or source mount manifest missing\n' >&2; exit 1;
}
python3 - "$W" <<'PY'
import json,sys
from pathlib import Path
w=Path(sys.argv[1])
c=json.loads((w/'model/config.json').read_text())
assert c['n_routed_experts']==c['num_experts_per_tok']==8
assert c.get('yihou_synthetic_expert8') is True
assert (w/'rounds/001-component/expert8/tests.json').is_file()
r=json.loads((w/'rounds/000-research/rdma-tmpfs/summary.json').read_text())
assert len(r)==8 and all(x['verified'] and x['target_rc']==x['initiator_rc']==0 for x in r)
print('config and RDMA evidence gates PASS')
PY
mkdir -p "$ROUND"
bash "$W/scripts/bench-harness/launch.sh" \
    "CONFIG=$W/scripts/config.yihou.expert8.sh" \
    "TOPOLOGY=$W/scripts/topology.yihou.138-136.tsv" \
    "OUT_DIR=$ROUND/launch"
