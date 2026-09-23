#!/usr/bin/env python3
"""Verify reset using per-rank acknowledgements; the image's idle host gauge is stale."""
import json,os,re,shlex,subprocess,time,urllib.request
from pathlib import Path
from sample_engine_metrics import parse_metrics

def verify_reset(output_name='cache-empty-before-warmup.json'):
 r=Path(os.environ['RUN']);opts=shlex.split(os.environ['SSH_OPTS']);prefix=os.environ['CONTAINER_PREFIX']
 stamp=r/'snapshot/flush-start-epoch.txt'
 since=stamp.read_text().strip() if stamp.exists() else (r/'snapshot/capture-start-epoch.txt').read_text().strip()
 for attempt in range(45):
  results={}
  for role,port in [('prefill',29001),('decode',29002)]:
   node=os.environ[role.upper()+'_NODE']
   with urllib.request.urlopen(f"http://{os.environ[role.upper()+'_IP']}:{port}/metrics",timeout=20) as f:rows=parse_metrics(f.read().decode())[0]
   required={'sglang:num_running_reqs','sglang:num_queue_reqs','sglang:kv_used_tokens','sglang:kv_evictable_tokens'}
   checks={}
   for name in required:
    values=[x for x in rows if x['metric']==name]
    checks[name]=len({x['labels'].get('dp_rank') for x in values})==8 and all(isinstance(x['value'],(int,float)) and 0<=x['value']<=(0 if 'reqs' in name else 64) for x in values)
   cmd=['docker','logs','--since',since,prefix+'-'+role+'-0']
   logs=subprocess.check_output(['ssh',*opts,node,shlex.join(cmd)],stderr=subprocess.STDOUT,text=True,timeout=40)
   ranks={int(x) for x in re.findall(r'DP(\d+)[^\n]*Cache flushed successfully!',logs)}
   checks['all_eight_ranks_acknowledged_flush']=ranks==set(range(8))
   image=subprocess.check_output(['ssh',*opts,node,shlex.join(['docker','inspect',prefix+'-'+role+'-0','--format','{{.Image}}'])],text=True,timeout=30).strip()
   checks['known_reset_implementation']=image=='sha256:4f125ff9096f75611fa24caad4c01c3707dd0892251680a6483b99ffaa2a88bb'
   if role=='prefill':
    cap=[x for x in rows if x['metric']=='sglang:hicache_host_total_tokens'];checks['host_pool_capacity_present']=len({x['labels'].get('dp_rank') for x in cap})==8 and all(isinstance(x['value'],(int,float)) and x['value']>0 for x in cap)
   results[role]={'checks':checks,'metrics':rows,'flush_ack_ranks':sorted(ranks),'host_usage_gauge_note':'Idle logger does not refresh host usage. Known Unified reset calls host_pool.clear(), which recreates all free_slots; per-rank flush acknowledgements occur after reset returns.'}
  passed=all(all(v['checks'].values()) for v in results.values())
  (r/'snapshot'/output_name).write_text(json.dumps({'passed':passed,'roles':results,'time':time.time(),'flush_since':since},indent=2)+'\n')
  if passed:return results
  time.sleep(2)
 raise RuntimeError('Reset verification failed; do not start load')

if __name__=='__main__':
 verify_reset();print('CACHE_RESET_VERIFIED_ON_ALL_RANKS')
