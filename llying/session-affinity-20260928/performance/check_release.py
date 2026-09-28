import datetime,json,shlex,subprocess
from pathlib import Path
root=Path('/perf_apps/liyingli/bench_agentx/session-affinity-31999-20260928');run=root/'runs/session-affinity-31999-performance';ssh=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile=/tmp/bench-agentx-known-hosts'];out={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'nodes':{}}
for node in ['smci355-ccs-aus-n10-29','smci355-ccs-aus-n03-33']:
 p=subprocess.run(ssh+[node,shlex.join(['rocm-smi','--showmeminfo','vram','--showuse','--json'])],text=True,capture_output=True,timeout=20)
 d=json.loads(p.stdout);cards={}
 for k,v in d.items():
  if k.startswith('card'):cards[k]={'vram_percent':100*float(v['VRAM Total Used Memory (B)'])/float(v['VRAM Total Memory (B)']),'gpu_use':v.get('GPU use (%)')}
 f=subprocess.run(ssh+[node,'fuser /dev/kfd'],text=True,capture_output=True,timeout=20)
 out['nodes'][node]={'cards':cards,'kfd_returncode':f.returncode,'kfd_stdout':f.stdout,'kfd_stderr':f.stderr}
 print(node,[round(v['vram_percent'],3) for v in cards.values()],'KFD',f.stdout.strip() or 'none')
out['all_released']=all(len(v['cards'])==8 and max(c['vram_percent'] for c in v['cards'].values())<1 and v['kfd_returncode']==1 for v in out['nodes'].values())
(run/'resource-release-latest.json').write_text(json.dumps(out,indent=2)+'\n');print('ALL_RELEASED',out['all_released'])
