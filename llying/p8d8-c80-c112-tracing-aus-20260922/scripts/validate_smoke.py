#!/usr/bin/env python3
"""Require paired P/D summaries, actual ranks, admission and OTLP stage spans."""
import collections
import json
import sys
import urllib.request
import subprocess
from pathlib import Path

root = Path(sys.argv[1])
live={}
for role,node in (('prefill','smci355-ccs-aus-n01-33'),('decode','smci355-ccs-aus-n02-21')):
    container=json.loads(subprocess.check_output(['ssh','-F','/dev/null','-o','BatchMode=yes',
        '-o','UserKnownHostsFile=/tmp/agentx_known_hosts',node,'docker','inspect',f'llying-aus-trace-{role}-0'],text=True))[0]
    env=dict(x.split('=',1) for x in container['Config']['Env'] if '=' in x)
    assert env.get('AUS_DIAG_ROLE')==role
    assert env.get('SGLANG_TRACE_ASYNC')=='1'
    assert env.get('SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK')=='1'
    argv=container['Config']['Cmd']
    assert '--enable-trace' in argv and '--enable-request-time-stats-logging' in argv
    assert argv[argv.index('--max-running-requests')+1]=='256'
    assert ('--enable-hierarchical-cache' in argv)==(role=='prefill')
    live[role]=container
(root/'live-containers.json').write_text(json.dumps(live,indent=2)+'\n')
rows=[]
for path in (root/'diagnostics').glob('*/*.jsonl'):
    for line in path.read_text().splitlines():
        rows.append(json.loads(line))
rooms=collections.defaultdict(set)
counts=collections.Counter()
for row in rows:
    counts[(row['role'], row['event'])]+=1
    if row['event']=='request_summary':
        assert row['dp_rank'] is not None, row
        assert row['room'], row
        assert 'aus-smoke-' in row['rid'], row
        rooms[row['room']].add(row['role'])
paired=[room for room, roles in rooms.items() if roles=={'prefill','decode'}]
assert len(paired)>=8, f'only {len(paired)} paired rooms'
assert counts['decode','admission']>=8, counts
spans=[json.loads(x) for x in (root/'traces/spans.jsonl').read_text().splitlines()]
names=collections.Counter(x.get('name') for x in spans)
for stage in ('prefill_waiting','prefill_forward','decode_bootstrap','decode_transferred'):
    assert names[stage]>=8, (stage,names)
result=dict(paired_requests=len(paired), event_counts={str(k):v for k,v in counts.items()}, span_names=dict(names))
(root/'smoke-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
# Clear the small smoke prefixes before the fresh C80 workload. Keep the
# collector running so flush boundaries remain part of the same trace capture.
for role,ip,port in (('prefill','10.235.192.136',29001),('decode','10.235.192.128',29002)):
    request=urllib.request.Request(f'http://{ip}:{port}/flush_cache',data=b'',method='POST')
    with urllib.request.urlopen(request,timeout=30) as response:
        (root/f'smoke-flush-{role}.txt').write_bytes(response.read())
