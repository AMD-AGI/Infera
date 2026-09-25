"""Plot cumulative profiling output throughput from recorded client logs."""
import argparse
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('candidate',type=Path)
p.add_argument('output',type=Path)
a=p.parse_args()
base=Path('/perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923/runs')
runs={'A0':base/'a0-guard-decode','G0 (R1)':base/'g0-guard-completion','R1+R4':a.candidate}
fig,ax=plt.subplots(figsize=(9,4.5))
series={}
for label,root in runs.items():
    points=[];elapsed=None
    for line in (root/'c80/runner.log').read_text(errors='replace').splitlines():
        marker=re.search(r'\[realtime ([0-9:]+) profiling\]',line)
        if marker:
            elapsed=0
            for field in marker[1].split(':'):elapsed=elapsed*60+int(field)
        match=re.search(r'tput_out=([0-9,.]+)/s',line)
        if match and elapsed is not None and elapsed<=3600:
            points.append({'seconds':elapsed,'output_tokens_s_gpu':float(match[1].replace(',',''))/16})
            elapsed=None
    series[label]=points
    ax.plot([x['seconds']/60 for x in points],[x['output_tokens_s_gpu'] for x in points],label=label)
ax.set(xlabel='Profiling minutes',ylabel='Cumulative output tokens/s/GPU',xlim=(0,60),title='C80, effective chunk 4K: client progress')
ax.grid(alpha=.25);ax.legend();fig.tight_layout()
a.output.mkdir(parents=True,exist_ok=True)
fig.savefig(a.output/'throughput-progress.png',dpi=160)
(a.output/'throughput-progress.json').write_text(json.dumps(series,indent=2)+'\n')
