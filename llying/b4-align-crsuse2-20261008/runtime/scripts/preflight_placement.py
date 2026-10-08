"""Validate the selected multi-worker recipe and establish an empty GPU cache start."""
import json,os,re,sys,time,urllib.request
from pathlib import Path
import transition_two_node as ops

sys.path.insert(0,str(Path(__file__).parent/'bench-harness/tools'))
from agentx_env import worker_url

E=os.environ;RUN=Path(E['RUN']);rows=json.loads((RUN/'placement-resolved.json').read_text())
expected_urls={w['url'] for w in rows}
data=ops.get('http://'+E['PREFILL_IP']+':28000/v1/workers')
workers=data if isinstance(data,list) else data.get('workers',data.get('data',[]))
assert {worker_url(w) for w in workers}==expected_urls
discovered={worker_url(w):w for w in workers}
for w in rows:
    description=discovered[w['url']]
    assert description['dp_size']==w['dp'] and description['disagg_mode']==w['role']
    if w['role']=='prefill':assert description['kv_block_size']==64


def flush(url):
    with urllib.request.urlopen(urllib.request.Request(url+'/flush_cache',data=b'',method='POST'),timeout=30) as f:return f.read().decode()


infos={};flushes={}
for w in rows:
    d=ops.get(w['url']+'/get_server_info');role=w['role'];upper=role.upper();infos[w['instance']]=d
    expect={'tp_size':w['tp'],'dp_size':w['dp'],'ep_size':1,'chunked_prefill_size':4096,'mem_fraction_static':.85,'kv_cache_dtype':'fp8_e4m3','enable_dp_attention':True,'dcp_size':1,'enable_hierarchical_cache':role=='prefill','dsa_prefill_backend':E['DSA_PREFILL_BACKEND'],'dsa_decode_backend':E['DSA_DECODE_BACKEND'],'max_running_requests':int(E[upper+'_MAX_RUNNING']),'random_seed':823508857 if role=='prefill' else 19197414}
    if role=='decode':expect.update(disaggregation_decode_enable_radix_cache=E.get('DECODE_RADIX','0')=='1',speculative_algorithm='EAGLE',speculative_num_steps=5,speculative_eagle_topk=1,speculative_num_draft_tokens=6)
    else:expect.update(speculative_algorithm=None,hicache_ratio=1.5,hicache_write_policy='write_through',hicache_io_backend='kernel',hicache_mem_layout='page_first')
    for key,value in expect.items():assert d[key]==value,(w['instance'],key,d[key],value)
    assert json.loads(d['json_model_override_args'])['index_share_for_mtp_iteration'] is False
    c=json.loads(ops.remote(w['node'],['docker','inspect',w['container']]))[0]
    env=dict(v.split('=',1) for v in c['Config']['Env'] if '=' in v)
    assert list(map(int,env['HIP_VISIBLE_DEVICES'].split(',')))==w['gpu_ids']
    assert env.get('SGLANG_DSA_FUSE_TOPK',env.get('SGLANG_NSA_FUSE_TOPK','1'))=='1'
    assert env.get('SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK')=='1'
    if role=='decode':assert env.get('SGLANG_SIMULATE_ACC_LEN')=='3.61'
    flushes[w['instance']]=flush(w['url'])
empty={}
for w in rows:
    for _ in range(30):
        with urllib.request.urlopen(w['url']+'/metrics',timeout=10) as f:raw=f.read().decode()
        metrics={key:[float(x) for x in re.findall(r'^sglang:'+key+r'(?:\{[^}]*\})? ([0-9.eE+-]+)$',raw,re.M)] for key in ['kv_used_tokens','kv_evictable_tokens']}
        if all(len(v)==w['dp'] and not any(v) for v in metrics.values()):break
        time.sleep(1)
    else:raise RuntimeError(f"{w['instance']} did not empty GPU cache: {metrics}")
    empty[w['instance']]=metrics
(RUN/'snapshot/cache-empty-before-warmup.json').write_text(json.dumps({'passed':True,'pools':empty,'flush_replies':flushes},indent=2)+'\n')
(RUN/'preflight.json').write_text(json.dumps({'passed':True,'workers':[{k:w[k] for k in ['instance','role','node','gpu_ids','tp','dp']} for w in rows],'fusion':'on','index_share':False,'concurrency':80},indent=2)+'\n')
print('PLACEMENT_PREFLIGHT_PASSED',len(rows),flush=True)
