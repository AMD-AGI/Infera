# Sourced at the top of every queue item: copy the scripts tree to node-local /tmp and continue there, so a git
# pull --rebase on the shared checkout (which replaces files) cannot hit a running bash with "Stale file handle".
# Results still go to $W (absolute paths in site/env.sh); only the scripts are read from the snapshot.
S=$(mktemp -d "/tmp/lyl-snap-${ITEM:-item}.XXXXXX")
tar -C /perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127 --exclude=./results --exclude=./replay/R --exclude='./replay/R-*' \
    --exclude=./replay/.base --exclude=./logs --exclude=./cache --exclude=./holder/q-any --exclude=./holder/q-each \
    --exclude='./holder/q-smci*' --exclude=./holder/done --exclude=./holder/running --exclude=./holder/logs \
    --exclude=./holder/parked --exclude='./holder/done-each-*' -cf - . | tar -C "$S" -xf -
cd "$S" || exit 1
echo "scripts snapshot $S ($(git -C /perf_apps/liyingli/bench_agentx/Infera-p4d4-aus rev-parse --short HEAD))"
