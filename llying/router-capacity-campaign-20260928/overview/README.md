# Comparison artifacts

`metrics.csv` and `metrics.json` retain units, node groups, quality status, and source paths. The Markdown table and PNG/SVG are generated from those same records. Missing or unvalidated points are listed as pending; B3 recovered client results are explicitly marked.

Regenerate after archiving a completed point:

```bash
bash /perf_apps/liyingli/bench_agentx/router-capacity-20260928/scripts/render_campaign_matrix.sh /home/liyingli/code/bench_agentx/Infera/llying/router-capacity-campaign-20260928 --plot
```

The wrapper isolates the system matplotlib/NumPy pair. It does not install packages or change benchmark dependencies. Without `--plot`, `campaign_matrix.py` only needs Python's standard library.
