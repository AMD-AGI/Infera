#!/usr/bin/env bash
# Task v0521cap2 (ON v3-026): ofg2h3 stack is up but its A1 client failed (broken uv-python in reused cache key 026,
# moved aside to *.bad1-20261015). Continue on the LIVE fresh stack: A1 C56 (fresh) -> flush -> A2 C56, then flush -> C64
# (tag A2 -> out agentx-c64-A2).
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness; E=$H/env.ofg2h3-v3026.yihou.sh
bash $H/cont_bench.yihou.sh $E 56 56 || exit 1
bash $H/cont_bench.yihou.sh $E - 64
