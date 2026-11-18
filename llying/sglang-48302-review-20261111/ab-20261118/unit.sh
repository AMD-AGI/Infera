#!/usr/bin/env bash
# sglang#48302 unit tests at the PR commit (src-pr), CPU only, in the nightly image with PYTHONPATH on the
# snapshot (the test tree comes from the checkout at the same commit).
# usage: unit.sh <node>
set -uo pipefail
AB=/perf_apps/liyingli/sgl48302-ab-20261118
SRC=/perf_apps/liyingli/bench_agentx/upstream/sglang-main
IMG=/apps/liyingli/squash/lmsysorg_sglang-rocm_v0.5.23-rocm720-mi35x-20261117_sha256_e016f33423250be30bde9e0888d159b67b01e00944de1edc11f36dd1f3dd0269.sqsh
OUT=$AB/unit-$(date -u +%m%dT%H%M).log
node=$1
echo "node $node $(date -u +%FT%TZ) checkout $(git -C $SRC log -1 --format='%h %s') dirty=$(git -C $SRC status --short --untracked-files=no | wc -l) src-pr $(cat $AB/src-pr/SOURCE_REV)" | tee "$OUT"
srun --jobid "${JOB:-36659}" --overlap --nodes 1 --ntasks 1 --nodelist "$node" \
    --container-image "$IMG" --no-container-entrypoint --no-container-mount-home \
    --container-mounts /perf_apps:/perf_apps --container-workdir "$SRC/test" \
    bash -c "export GPU_ARCHS=gfx950 PYTHONPATH=$AB/src-pr/python:/tmp/extra; mkdir -p /tmp/extra
for i in 1 2 3 4 5 6 7 8; do
  m=\$(python -c 'import sglang.srt.managers.scheduler, sglang.srt.batch_overlap.two_batch_overlap, sglang.test.test_utils' 2>&1 | grep -oP \"No module named .\\K[^.']+\" | head -1)
  [ -z \"\$m\" ] && break; echo \"installing \$m\"; pip install -q --no-deps --target /tmp/extra \"\$m\" 2>&1 | tail -1
done
for t in registered/unit/layers/test_graph_variants_dp_consistency.py \
         registered/unit/batch_overlap/test_tbo_filter_batch_marker.py \
         registered/unit/managers/test_schedule_batch_out_of_place.py \
         registered/unit/spec/test_dp_spec_prefill_coordination.py \
         registered/unit/layers/moe/test_aiter_megamoe_rank_sync.py; do
  echo \"=== \$t\"; python -m pytest -q -p no:cacheprovider \"\$t\" 2>&1 | grep -vE 'Warning|warnings.warn|^ *\$' | tail -40
done" >> "$OUT" 2>&1
echo "done $(date -u +%FT%TZ)" >> "$OUT"
