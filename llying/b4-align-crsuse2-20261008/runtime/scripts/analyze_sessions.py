"""Join client session IDs to actual engine ranks, without treating P/D pairing as affinity proof."""
import argparse,collections,json
from analyze import rank_name
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();run=a.run
engine={role:{} for role in ['prefill','decode']}
for role,records in engine.items():
 for f in (run/'diagnostics'/role).glob('*.jsonl'):
  for line in f.open():
   try:r=json.loads(line)
   except ValueError:continue
   if r.get('event')=='request_summary' and r.get('rid'):
    records[str(r['rid']).split('_')[-1]]=r
phases=collections.defaultdict(lambda: {'requests':0,'paired':0,'cancelled':0,'missing_p':0,'missing_d':0,'room_mismatch':0,'sessions':{}})
paths=list((run/'c80').rglob('profile_export.jsonl'));assert len(paths)==1,paths
for line in paths[0].open():
 if not line.endswith('\n'):break
 r=json.loads(line);m=r['metadata'];phase=phases[m['benchmark_phase']];phase['requests']+=1
 if m.get('was_cancelled'):phase['cancelled']+=1
 rid=m.get('x_request_id');session=m.get('x_correlation_id')
 ps,ds=engine['prefill'].get(rid),engine['decode'].get(rid)
 phase['missing_p']+=ps is None;phase['missing_d']+=ds is None
 if ps and ds:
  phase['paired']+=1;phase['room_mismatch']+=ps.get('room')!=ds.get('room')
 if not session:continue
 entry=phase['sessions'].setdefault(session,{'requests':0,'p_ranks':set(),'d_ranks':set(),'p_sequence':[]})
 entry['requests']+=1
 if ps:
  entry['p_ranks'].add(rank_name(ps));entry['p_sequence'].append((m['request_start_ns'],rank_name(ps)))
 if ds:entry['d_ranks'].add(rank_name(ds))
report={}
for name,d in phases.items():
 entries=d.pop('sessions');multi=[e for e in entries.values() if len(e['p_sequence'])>=2]
 d.update(sessions=len(entries),sessions_with_multiple_p_records=len(multi),p_single_rank_sessions=sum(len(e['p_ranks'])==1 for e in multi),p_multiple_rank_sessions=sum(len(e['p_ranks'])>1 for e in multi),p_rank_switches=sum(sum(x[1]!=y[1] for x,y in zip(sorted(e['p_sequence']),sorted(e['p_sequence'])[1:])) for e in entries.values()),d_multiple_rank_sessions=sum(len(e['d_ranks'])>1 for e in entries.values()),p_migration_examples=[{'session':sid,'requests':e['requests'],'p_ranks':sorted(e['p_ranks'])} for sid,e in entries.items() if len(e['p_ranks'])>1][:20])
 report[name]=d
out=run/'analysis';out.mkdir(exist_ok=True);(out/'session-affinity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
