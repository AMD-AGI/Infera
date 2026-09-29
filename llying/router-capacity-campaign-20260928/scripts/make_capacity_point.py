"""Freeze the next capacity layout after the feature and backend decisions."""
import argparse,json,shlex,sys
from pathlib import Path

ROOT=Path('/perf_apps/liyingli/bench_agentx/router-capacity-20260928')
sys.path.insert(0,str(ROOT/'scripts/bench-harness/tools'))
from campaign_topology import load

p=argparse.ArgumentParser();p.add_argument('case',choices=['b5','b6','b7']);p.add_argument('--previous',type=Path,required=True);p.add_argument('--restart-prefill',action='store_true');a=p.parse_args()
for name in ['selected-p.sh','selected-d.sh','selected-backend.sh']:
    assert (ROOT/'config'/name).exists(),f'feature decision missing: {name}'
previous=a.previous.resolve();assert previous.parent==ROOT/'runs'
assert 'COMPLETE_REVIEW_PENDING' in (previous/'STATUS').read_text()
assert json.loads((previous/'review-ready.json').read_text())['checks_passed']
old=load(previous/'placement-resolved.json')
command=json.loads((previous/'snapshot/router-launch-command.json').read_text())
old_prefix=command[command.index('--name')+1].removesuffix('-router')
ptp=4 if a.case=='b7' else 8;dtp=4 if a.case=='b5' else 8
name={'b5':'p8d4','b6':'2p8d8','b7':'4p4d8'}[a.case]
run_id=f'campaign-{a.case}-{name}';prefix=f'llying-campaign-{a.case}'
assert not any((ROOT/'runs'/r).exists() for r in [run_id,run_id+'-real'])
primary=next(w for w in old if w['role']=='prefill')
decode=next(w for w in old if w['role']=='decode')
hosts=[(primary['node'],primary['ip'],primary['allocation_job'])]
if a.case!='b5':
    extra=json.loads((ROOT/'extra-node.json').read_text());assert extra['state']=='PREPARED',extra
    hosts.append((extra['node'],extra['ip'],extra['job']))
rows=[]

def row(role,node,ip,job,gpus,port):
    index=len(rows);tp=ptp if role=='prefill' else dtp
    instance=sum(w['role']==role for w in rows)
    container=f'{prefix}-real-{role}-{instance}'
    if role=='prefill' and not a.restart_prefill:
        candidates=[w for w in old if w['role']==role and w['node']==node and w['gpu_ids']==gpus and w['tp']==tp]
        if candidates:container=candidates[0]['container']
    return dict(role=role,node=node,data_ip=ip,allocation_job=job,gpu_ids=gpus,tp=tp,dp=tp,engine_port=port,bootstrap_port=28990+index,kv_port=25540+index,snapshot_port=28810+index,container=container)

for host_index,(node,ip,job) in enumerate(hosts):
    for group in range(8//ptp):
        rows.append(row('prefill',node,ip,job,list(range(group*ptp,(group+1)*ptp)),29001+host_index*2+group*256))
rows.append(row('decode',decode['node'],decode['ip'],decode['allocation_job'],list(range(dtp)),29002))
# Preserve reused workers' public control ports as well as their engine port.
for w in rows:
    prior=next((o for o in old if (o['node'],o['container'])==(w['node'],w['container'])),None)
    if prior:
        for key in ['engine_port','bootstrap_port','kv_port','snapshot_port']:w[key]=prior[key]
real_top=ROOT/'config'/f'{a.case}-real.json';sim_top=ROOT/'config'/f'{a.case}.json'
real_top.write_text(json.dumps(rows,indent=2)+'\n');load(real_top)
sim=[dict(w) for w in rows]
for w in sim:
    if w['role']=='decode':w['container']=f'{prefix}-decode-0'
sim_top.write_text(json.dumps(sim,indent=2)+'\n');load(sim_top)
config=f'''#!/usr/bin/env bash
source {ROOT}/config/recovery-32054-base.sh
source "$TRACE_RUNTIME/config/selected-p.sh"
source "$TRACE_RUNTIME/config/selected-d.sh"
source "$TRACE_RUNTIME/config/selected-backend.sh"
source "$TRACE_RUNTIME/config/placement-ports.sh"
export ALLOCATION_JOB_ID={primary['allocation_job']} PREFILL_ALLOCATION_JOB_ID={primary['allocation_job']} DECODE_ALLOCATION_JOB_ID={decode['allocation_job']}
export RUN_ID={run_id}-real
export RUN=$TRACE_RUNTIME/runs/$RUN_ID
export CONTAINER_PREFIX={prefix}-real
export PREVIOUS_RUN={shlex.quote(str(previous))}
export OLD_PREFIX={old_prefix}
export TOPOLOGY={real_top}
export PREFILL_TP={ptp} PREFILL_DP={ptp} PREFILL_CHUNK_SIZE={ptp*4096}
export PREFILL_MAX_RUNNING={ptp*32} PREFILL_GRAPH_MAX_BS={ptp*32}
export DECODE_TP={dtp} DECODE_DP={dtp} DECODE_CHUNK_SIZE={dtp*4096}
export DECODE_MAX_RUNNING=256 DECODE_GRAPH_MAX_BS=256
export PERSIST_JIT_CACHE=1
export DECODE_DIAG_SOURCE=$TRACE_RUNTIME/docker/decode_prefix_diag.py
if [[ "$DECODE_RADIX" == 1 ]]; then
 export DECODE_EXTRA_ARGS="$DECODE_EXTRA_ARGS --disaggregation-decode-enable-radix-cache"
 export DECODE_EXTRA_ENV="$DECODE_EXTRA_ENV SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1"
fi
export DECODE_SIMULATE_ACC_LEN=''
'''
(ROOT/'config'/f'{a.case}-real.sh').write_text(config)
(ROOT/'config'/f'{a.case}.sh').write_text(f'''#!/usr/bin/env bash
source {ROOT}/config/{a.case}-real.sh
export RUN_ID={run_id}
export RUN=$TRACE_RUNTIME/runs/$RUN_ID
export CONTAINER_PREFIX={prefix}
export PREVIOUS_RUN=$TRACE_RUNTIME/runs/{run_id}-real
export OLD_PREFIX={prefix}-real
export TOPOLOGY={sim_top}
export DECODE_SIMULATE_ACC_LEN=3.61
''')
print(json.dumps({'case':a.case,'real_config':str(ROOT/'config'/f'{a.case}-real.sh'),'performance_config':str(ROOT/'config'/f'{a.case}.sh'),'gpus':sum(len(w['gpu_ids']) for w in rows),'reused_prefill':[w['container'] for w in rows if w['role']=='prefill' and any(o['container']==w['container'] for o in old)]}))
