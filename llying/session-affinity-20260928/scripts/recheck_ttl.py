"""Recheck saved TTL observations after stripping ANSI log decoration; no requests sent."""
import json,re
from pathlib import Path
run=Path('/perf_apps/liyingli/bench_agentx/session-affinity-31999-20260928/runs/session-affinity-31999-smoke');tag='both-2-293d1c42'
x=json.loads((run/f'{tag}-raw.json').read_text());rows=x['requests'];pmap={r['rid']:r for r in x['prefill']};dmap={r['rid']:r for r in x['decode']};metrics=x['metrics_after']
for r in rows:
 if not r['cancel']:assert pmap[r['rid']]['room']==dmap[r['rid']]['room']
for session in ['repeat','concurrent']:
 ids=[r['rid'] for r in rows if r['session']==session and not r['rid'].endswith('-18')]
 for m in [pmap,dmap]:assert len({m[i]['dp_rank'] for i in ids})==1
assert all(v==0 for k,v in metrics.items() if '_active{' in k)
log=re.sub(r'\x1b\[[0-9;]*m','',(run/f'{tag}-router.log').read_text());(run/f'{tag}-router.txt').write_text(log)
for role in ['prefill','decode']:
 assert metrics[f'infera_router_session_selected_total{{role="{role}"}}']>=12
 assert re.search(fr'role={role.capitalize()} .*reason="expired"',log)
policy=[l for l in log.splitlines() if 'pick policy="kv-aware"' in l]
assert all(re.search(r'picked=\S+#dp[0-7] ',l) for l in policy)
hits=[l for l in policy if 'role=Prefill' in l and re.search(r'cache_hits=[1-9][0-9]* ',l)];assert hits
report={'passed':True,'binary_sha256':'7e221373acd5d2605fcda8d94e6bc4f69b4e774f355520a3b89035d6d8f19d6b','mode':'both','ttl':2,'requests':len(rows),'paired_completed':sum(not r['cancel'] for r in rows),'cancelled_intentionally':1,'cache_hit_observations':len(hits),'metrics':metrics,'raw_file':f'{tag}-raw.json','validation_note':'Initial checker did not strip ANSI escape codes around reason=; saved data rechecked, no requests repeated.','assignments':[{'rid':r['rid'],'session':r['session'],'p':pmap.get(r['rid'],{}).get('dp_rank'),'d':dmap.get(r['rid'],{}).get('dp_rank')} for r in rows]}
(run/f'{tag}-result.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['passed','ttl','requests','paired_completed','cache_hit_observations']}))
