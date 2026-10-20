"""Print one Markdown row per AgentX point from results/c<C>/ (agentx_conc<C>.json and the AIPerf
export): throughput per GPU, requests, interactivity, ITL, TTFT, E2E and the API prefix-cache hit.
  python3 summarize_points.py results/c176 results/c128 ..."""

import json
import sys
from pathlib import Path

for d in map(Path, sys.argv[1:]):
    agg = json.loads(next(d.glob("agentx_conc*.json")).read_text())
    exp = json.loads((d / "profile_export_aiperf.json").read_text())
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
        f"| {exp['overall_usage_prompt_cache_read_pct']['avg']:.2f}%"
        f"（{exp['theoretical_prefix_cache_hit']['avg']:.2f}%） |"
    )
