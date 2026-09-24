"""Plot observed per-rank KV occupancy; gaps over 5 s remain gaps."""
import csv
import datetime
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
report = json.loads((OUT/'comparison.json').read_text())
roots = {
    'A0': Path('/perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923/runs/a0-guard-decode'),
    'R2': Path('/perf_apps/liyingli/bench_agentx/r2-4k-31705-31706-20260924/runs/r2-on-4k-31705-31706'),
}
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, sharey=True)
with (OUT/'decode-occupancy.csv').open('w') as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(['run','elapsed_seconds',*[f'rank{i}' for i in range(8)]])
    for ax, (label, root) in zip(axes, roots.items()):
        start = report['runs'][label]['start_ns']
        xs, lows, highs, means = [], [], [], []
        for line in (root/'sampling/engine.jsonl').open():
            row = json.loads(line)
            if row.get('endpoint')!='decode' or row.get('error'):
                continue
            ns = int(datetime.datetime.fromisoformat(row['captured_at']).timestamp()*1e9)
            t = (ns-start)/1e9
            if not 0<=t<2580:
                continue
            ranks = {s['labels']['dp_rank']: s['value'] for s in row.get('series',[])
                     if s['metric']=='sglang:token_usage' and 'dp_rank' in s['labels']}
            if len(ranks)!=8:
                continue
            v = [ranks[str(i)] for i in range(8)]
            writer.writerow([label,t,*v])
            if xs and t/60-xs[-1]>5/60:
                xs.append(float('nan'));lows.append(float('nan'));highs.append(float('nan'));means.append(float('nan'))
            xs.append(t/60);lows.append(min(v)*100);highs.append(max(v)*100);means.append(sum(v)/8*100)
        ax.fill_between(xs, lows, highs, alpha=.3, color='#4c78a8', label='Range across 8 ranks')
        ax.plot(xs, means, color='#e45756', linewidth=1, label='Mean across 8 ranks')
        ax.axhline(90, color='gray', linewidth=.8, linestyle='--')
        ax.set(title=label, ylabel='Decode KV occupancy (%)', ylim=(0,100), xlim=(0,43))
        ax.grid(alpha=.2)
axes[0].legend(loc='upper left', ncol=2)
axes[1].set_xlabel('Minutes from first profiling request')
fig.suptitle('Similar total KV use; narrower instantaneous rank spread with R2')
fig.tight_layout()
fig.savefig(OUT/'decode-occupancy.png', dpi=160)
