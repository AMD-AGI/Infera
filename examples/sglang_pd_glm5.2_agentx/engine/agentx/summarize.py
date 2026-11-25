# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""One Markdown row per AgentX point, from the run directories agentx.sh wrote
(<run>/agentx/agentx_conc<C>.json and the AIPerf export): throughput per GPU,
requests, interactivity, ITL, TTFT, E2E and the prefix-cache hit (API-reported,
and the trace's theoretical hit in parentheses).

    python3 engine/agentx/summarize.py $WORK_DIR/runs/<run> ...
"""

import json
import sys
from pathlib import Path

print(
    "| point | tok/s/GPU | out tok/s/GPU | req/s | done / dropped | intvty p50 / p90 "
    "| ITL p50 / p90 ms | TTFT p50 / p90 / mean s | E2E p90 s | cache hit |"
)
print("|---|---|---|---|---|---|---|---|---|---|")
for run in map(Path, sys.argv[1:]):
    try:
        agg = json.loads(next((run / "agentx").glob("agentx_conc*.json")).read_text())
        exp = json.loads(
            (run / "agentx" / "aiperf_artifacts" / "profile_export_aiperf.json").read_text()
        )
    except (StopIteration, OSError) as e:
        print(f"| {run.name} | no result ({e.__class__.__name__}) |")
        continue
    r, lat = agg["request_metrics"], agg["request_metrics"]["latency"]
    acct = agg["request_accounting"]
    print(
        f"| C{agg['conc']} | {r['throughput']['per_gpu']['total_tput_tps']:,.0f} "
        f"| {r['throughput']['per_gpu']['output_tput_tps']:.1f} | {r['qps']['mean']:.3f} "
        f"| {agg['num_requests_successful']:,} / {acct.get('records_error_dropped', 0)} "
        f"| {lat['intvty']['p50']:.1f} / {lat['intvty']['p90']:.1f} "
        f"| {lat['itl']['p50'] * 1e3:.1f} / {lat['itl']['p90'] * 1e3:.1f} "
        f"| {lat['ttft']['p50']:.2f} / {lat['ttft']['p90']:.2f} / {lat['ttft']['mean']:.2f} "
        f"| {lat['e2el']['p90']:.1f} "
        f"| {exp['overall_usage_prompt_cache_read_pct']['avg']:.2f}% "
        f"({exp['theoretical_prefix_cache_hit']['avg']:.2f}%) |"
    )
