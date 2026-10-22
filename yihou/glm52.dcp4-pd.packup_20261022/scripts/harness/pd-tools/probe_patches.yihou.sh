#!/usr/bin/env bash
# Probe which Infera v0521 sglang recipe patches apply to a given sglang tree.
# Runs inside a throwaway container (--rm, --network none); mutates only the container fs
# (or a scratch copy when SGLANG_DIR points at a copied tree).
# Usage: PATCHES=/patches SGLANG_DIR=/sgl-workspace/sglang bash probe_patches.yihou.sh
set -uo pipefail
P=${PATCHES:-/patches}
S=${SGLANG_DIR:-/sgl-workspace/sglang}
cd "$S" || exit 2
echo "== sglang tree $S"; python3 -c "import sglang;print('version',sglang.__version__, sglang.__file__)" 2>&1 | tail -1
git -C "$S" log --oneline -1 2>/dev/null
run() { local name=$1; shift; local out rc; out=$("$@" 2>&1); rc=$?; echo "-- $name rc=$rc"; echo "$out" | tail -6 | sed 's/^/     /'; }
run "01 patch_dsa_indexer_hip_dp_padded_rows.py" python3 $P/sglang_dsa/patch_dsa_indexer_hip_dp_padded_rows.py
for d in sglang_dsa/dsa_dp_sync.diff sglang_dsa/v0521/dsa_page_table_rows.diff; do
  if patch -p1 --fuzz=0 --batch --forward --dry-run < $P/$d >/tmp/o 2>&1; then echo "-- $d: APPLIES fuzz0"; patch -p1 --fuzz=0 --batch --forward -s < $P/$d
  elif patch -p1 -R --fuzz=0 --batch --dry-run < $P/$d >/dev/null 2>&1; then echo "-- $d: ALREADY APPLIED (reverse applies)"
  else echo "-- $d: DOES NOT APPLY"; sed 's/^/     /' /tmp/o | head -12; fi
done
run "nextn patch_glm_nextn_fused_shared_experts_arch.py" python3 $P/sglang_dsa/v0521/patch_glm_nextn_fused_shared_experts_arch.py
run "disagg v0521/patch_prefill_nonfinal_chunk_wait_event.py" python3 $P/sglang_disagg/v0521/patch_prefill_nonfinal_chunk_wait_event.py
run "disagg v0519/patch_pd_disable_implicit_rocm_rejection_sampling.py" python3 $P/sglang_disagg/v0519/patch_pd_disable_implicit_rocm_rejection_sampling.py
for f in $P/sglang_responses/v0519/*.py; do run "responses $(basename $f)" python3 $f; done
run "rocm patch_hicache_rocm_staged_write_back.py" python3 $P/sglang_rocm/patch_hicache_rocm_staged_write_back.py
echo "== upstream markers"
SRT=$S/python/sglang/srt
grep -l "_spec_input_cuda_graph_compatible" -r $SRT --include=*.py | head -3
grep -l "can_run_dp_draft_cuda_graph" -r $SRT --include=*.py | head -3
grep -n "kv_chunk.wait_event.synchronize()" $SRT/disaggregation/mooncake/conn.py | head -2
echo "== git diff --stat after probes"; git -C "$S" diff --stat 2>/dev/null | tail -15
