#!/usr/bin/env bash
# Assemble runtime/ (B4's TRACE_RUNTIME layout) from unmodified copies of the
# committed aus kits, and record each file's source and SHA256 in
# runtime/MANIFEST.tsv. Copies, not symlinks: several files are read inside
# containers that mount only runtime/.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
LLYING="$(cd "$HERE/.." && pwd)"
RT="$HERE/runtime"
CAMPAIGN=router-capacity-campaign-20260928
TRACE=p8d8-c80-c112-tracing-aus-20260922
files=(
    # dest                                   source (relative to llying/)
    scripts/transition_two_node.py            $CAMPAIGN/scripts/transition_two_node.py
    scripts/preflight_placement.py            $CAMPAIGN/scripts/preflight_placement.py
    scripts/placement_answer_probe.py         $CAMPAIGN/scripts/placement_answer_probe.py
    scripts/radix_gate.py                     $CAMPAIGN/scripts/radix_gate.py
    scripts/start_performance_router.py       $CAMPAIGN/scripts/start_performance_router.py
    scripts/run_placement_performance.py      $CAMPAIGN/scripts/run_placement_performance.py
    scripts/capture_placement.py              $CAMPAIGN/scripts/capture_placement.py
    scripts/analyze.py                        $CAMPAIGN/scripts/analyze.py
    scripts/analyze_sessions.py               $CAMPAIGN/scripts/analyze_sessions.py
    scripts/analyze_runtime.py                $CAMPAIGN/scripts/analyze_runtime.py
    scripts/analyze_decode_prefix.py          $CAMPAIGN/scripts/analyze_decode_prefix.py
    scripts/analyze_balance.py                $CAMPAIGN/scripts/analyze_balance.py
    scripts/compare_c80_runs.py               $CAMPAIGN/scripts/compare_c80_runs.py
    scripts/compare_matched_requests.py       $CAMPAIGN/scripts/compare_matched_requests.py
    scripts/bench-harness/agentx_bench.sh     $CAMPAIGN/scripts/bench-harness/agentx_bench.sh
    scripts/bench-harness/engine.sh           $CAMPAIGN/scripts/bench-harness/engine.sh
    scripts/bench-harness/tools/agentx_env.py $CAMPAIGN/scripts/bench-harness/tools/agentx_env.py
    scripts/bench-harness/tools/campaign_topology.py $CAMPAIGN/scripts/bench-harness/tools/campaign_topology.py
    scripts/bench-harness/tools/topology.py   $CAMPAIGN/scripts/bench-harness/tools/topology.py
    scripts/bench-harness/tools/ensure_inferencex.py $TRACE/scripts/bench-harness/tools/ensure_inferencex.py
    scripts/patch_client.py                   $TRACE/scripts/patch_client.py
    scripts/pin_chunk8k_dataset.py            $TRACE/scripts/pin_chunk8k_dataset.py
    scripts/otlp_jsonl_collector.py           $TRACE/scripts/otlp_jsonl_collector.py
    scripts/sample_engine_metrics.py          $TRACE/scripts/sample_engine_metrics.py
    scripts/sample_node_runtime.py            $TRACE/scripts/sample_node_runtime.py
    scripts/validate_and_pin_client.py        p8d8-adaptive-31625-20260923/scripts/validate_and_pin_client.py
    scripts/validate_chunk8k_runtime.py       $CAMPAIGN/scripts/validate_chunk8k_runtime.py
    config/client-constraints.txt             r1r4-4k-20260924/config/client-constraints.txt
    docker/aus_diag.py                        $TRACE/docker/aus_diag.py
)
rm -rf "$RT/scripts" "$RT/config/client-constraints.txt" "$RT/docker/aus_diag.py"
mkdir -p "$RT"
printf 'dest\tsource\tsha256\n' > "$RT/MANIFEST.tsv"
for ((i = 0; i < ${#files[@]}; i += 2)); do
    dest="$RT/${files[i]}" source="$LLYING/${files[i+1]}"
    mkdir -p "$(dirname "$dest")"
    cp -p "$source" "$dest"
    printf '%s\t%s\t%s\n' "${files[i]}" "llying/${files[i+1]}" "$(sha256sum < "$dest" | cut -d' ' -f1)" >> "$RT/MANIFEST.tsv"
done
echo "assembled $(( ${#files[@]} / 2 )) files into $RT"
