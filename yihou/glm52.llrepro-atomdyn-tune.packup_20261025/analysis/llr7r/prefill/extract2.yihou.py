# Per-stage compact extraction for cross-stage pipeline analysis. Usage: extract2.yihou.py <trace.json.gz> <out.json>
import gzip, json, sys, collections, re
f=sys.argv[1]
d=json.load(gzip.open(f)); ev=d['traceEvents']; base_ns=d.get('baseTimeNanoseconds',0)
X=[e for e in ev if e.get('ph')=='X']; del ev
W=lambda ts: base_ns/1e3+ts   # absolute epoch microseconds
GPU=[e for e in X if e.get('cat') in('kernel','gpu_memcpy','gpu_memset')]
RT={e['args'].get('correlation'):e for e in X if e.get('cat')=='cuda_runtime' and 'correlation' in e.get('args',{})}
main_tid=collections.Counter(e['tid'] for e in RT.values()).most_common(1)[0][0]
def sweep(intervals, queries):
    intervals.sort(key=lambda x:(x[0],-x[1])); queries.sort(); res={}; st=[]; i=0
    for ts,key in queries:
        while i<len(intervals) and intervals[i][0]<=ts:
            s,e_,n=intervals[i]
            while st and st[-1][1]<=s: st.pop()
            st.append((s,e_,n)); i+=1
        while st and st[-1][1]<ts: st.pop()
        res[key]=[x[2] for x in st if x[0]<=ts<=x[1]]
    return res
q=[(c['ts'],k) for k,c in RT.items() if c['tid']==main_tid]
py=[(e['ts'],e['ts']+e['dur'],e['name']) for e in X if e.get('cat')=='python_function' and e['tid']==main_tid]
pystack=sweep(py,q)
ops=[e for e in X if e.get('cat')=='cpu_op']
opl=[(e['ts'],e['ts']+e['dur'],i) for i,e in enumerate(ops) if e['tid']==main_tid]
opstack=sweep(opl,list(q))
KC=[('attn_sparse_mla',r'_sparse_mla_fwd'),('attn_misc',r'fused_qk_rope_cat_and_cache_mla|kn_entry_2c_sbhd|_batched_gemm_a8w8|fused_qk_rmsnorm'),
    ('indexer_logits',r'mqa_logits'),('indexer_other',r'topk_packed|hadamard|indexer_k_quant|cp_gather_indexer|act_quant_kernel|topk_transform|coop_topk'),
    ('moe',r'moe|gemm2_a4w4|mxfp4|fused_append_shared_experts|dynamic_per_group_scaled_quant|topk_reg_kernel|act_and_mul'),
    ('dense_gemm',r'hgemm|Cijk_|_gemm_a16_w16'),('norm',r'rmsnorm|Layernorm'),('nccl',r'nccl'),
    ('hicache_kernel',r'transfer_kernel_impl|hicache_transfer'),('scan/fill/copy/repeat(dsa metadata)',r'rocprim|FillFunc|compute_cuda_kernel|direct_copy|elementwise_kernel_manual_unroll'),('other',r'.')]
KC=[(n,re.compile(r)) for n,r in KC]
def kcat(e):
    if e['cat']=='gpu_memcpy': return 'memcpy_'+e['args'].get('kind','?')
    if e['cat']=='gpu_memset': return 'memset'
    for n,r in KC:
        if r.search(e['name']): return n
def ctx(corr):
    s=' | '.join(pystack.get(corr) or [])
    if re.search(r'_draft_extend_for_prefill',s): return 'draft'
    if re.search(r'backup_from_device|_execute_kv_backup',s): return 'hicache_backup'
    if re.search(r'load_back|load_to_device',s): return 'hicache_load'
    if re.search(r'send_kv_chunk|maybe_send_cached_prefix',s): return 'kv_send'
    if re.search(r'send_tensor_dict',s): return 'pp_send'
    if re.search(r'recv_tensor_dict',s): return 'pp_recv'
    if re.search(r'forward_batch_generation',s): return 'fwd'
    return 'other'
G=[]
for e in GPU:
    c=e['args'].get('correlation')
    G.append(dict(t=W(e['ts']),d=e['dur'],s=e['args'].get('stream'),k=kcat(e),x=ctx(c),n=e['name'][:60],b=e['args'].get('bytes',0) or 0))
steps=sorted([dict(t=W(e['ts']),d=e['dur'],n=e['name']) for e in X if e.get('cat')=='gpu_user_annotation' and e['name'].startswith('step[')],key=lambda z:z['t'])
csteps=sorted([dict(t=W(e['ts']),d=e['dur'],n=e['name']) for e in X if e.get('cat')=='user_annotation' and e['name'].startswith('step[')],key=lambda z:z['t'])
PYN=r'\): (_pp_recv_dict_from_prev_stage|_pp_recv_proxy_tensors|_pp_send_dict_to_next_stage|_pp_recv_pyobj_from_prev_stage|process_batch_result_disagg_prefill|recv_requests|_pp_launch_batch|synchronize|_pp_commit_send_output\w*|send_kv_chunk|_draft_extend_for_prefill|run_batch|resolve_forward_inputs|_execute_kv_backup|load_back|event_loop_pp_disagg_prefill|process_input_requests|get_next_disagg_prefill_batch\w*|_pp_process_batch_result|checkpoint_disagg_prefill|process_disagg_prefill_inflight_queue)$'
PM=[dict(t=W(e['ts']),d=e['dur'],n=e['name'].split('): ')[-1]) for e in X if e.get('cat')=='python_function' and e['tid']==main_tid and re.search(PYN,e['name'])]
PO=[dict(t=W(e['ts']),d=e['dur'],n=e['name'].split('): ')[-1],tid=e['tid']) for e in X if e.get('cat')=='python_function' and e['tid']!=main_tid and re.search(r'\): (batch_transfer_sync|send_kvcache|_send_kvcache_generic|transfer_worker)$',e['name'])]
# hicache cpu_op shapes
hs=[]
for e in ops:
    if 'transfer_kv' in e['name'] or 'hicache' in e['name'].lower():
        hs.append(dict(t=W(e['ts']),n=e['name'],a=e.get('args',{})))
json.dump(dict(main_tid=main_tid,G=G,steps=steps,csteps=csteps,PM=PM,PO=PO,hs=hs[:400]),open(sys.argv[2],'w'))
print('ok',len(G),len(steps),len(PM),len(PO),len(hs))
