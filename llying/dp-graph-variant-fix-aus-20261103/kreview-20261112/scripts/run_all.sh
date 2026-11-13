#!/usr/bin/env bash
# Run ana2.py on every capture dir (skipping nested duplicates); text report + json per capture.
K=/perf_apps/liyingli/bench_agentx/kreview-20261112
C=$K/out/compact
mkdir -p $K/out/ana2
python3 - <<PY > $K/out/ana2/captures.txt
import sys; sys.path.insert(0, "$K/scripts"); import kl
for c in kl.captures("$C"): print(c)
PY
run() { c="$1"; tag=$(echo "${c#$C/}" | sed 's|/traces/|__|; s|/|_|g'); python3 $K/scripts/ana2.py "$c" --json $K/out/ana2/$tag.json > $K/out/ana2/$tag.txt 2>&1 || echo "FAIL $tag"; }
export -f run; export K C
cat $K/out/ana2/captures.txt | xargs -P ${NPROC:-4} -I{} bash -c 'run {}'
ls $K/out/ana2/*.txt | wc -l
