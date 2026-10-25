# Per-batch hand-off anatomy across stages. Usage: anatomy.yihou.py <dir> <batch idx...>
import json, sys
D=sys.argv[1]; S=[json.load(open(f'{D}/ex2.pp{i}.json')) for i in range(4)]
def batches(s):
    st=s['steps']; bl=[]; i=0
    while i<len(st):
        e=st[i]; dr=None
        if i+1<len(st) and 'DRAFT' in st[i+1]['n'] and st[i+1]['t']-e['t']<1e6: dr=st[i+1]; i+=1
        bl.append(dict(t0=e['t'],end=max(e['t']+e['d'],dr['t']+dr['d']) if dr else e['t']+e['d'])); i+=1
    return bl
B=[batches(s) for s in S]
SHOW={'synchronize','_pp_recv_dict_from_prev_stage','_pp_recv_pyobj_from_prev_stage','_pp_recv_proxy_tensors','recv_requests','_pp_commit_send_output_work_and_preprocess_output_tensors','_pp_send_dict_to_next_stage','process_batch_result_disagg_prefill','checkpoint_disagg_prefill','process_disagg_prefill_inflight_queue','run_batch','send_kv_chunk','_execute_kv_backup','load_back','_pp_launch_batch','get_next_disagg_prefill_batch_to_run','_pp_process_batch_result','process_input_requests'}
for j in map(int,sys.argv[2:]):
    T=B[0][j]['t0']
    print('\n=== batch %d (times rel. PP0 GPU start of batch %d, ms) ==='%(j,j))
    for k in range(4):
        print(' PP%d GPU step b%d: %.1f -> %.1f ; next b%d GPU start %.1f'%(k,j,(B[k][j]['t0']-T)/1e3,(B[k][j]['end']-T)/1e3,j+1,(B[k][j+1]['t0']-T)/1e3))
        lo=B[k][j]['end']-30e3; hi=B[k][j+1]['t0']+2e3
        ev=[(p['t'],p['t']+p['d'],'cpu '+p['n']) for p in S[k]['PM'] if p['n'] in SHOW and p['t']<hi and p['t']+p['d']>lo and p['d']>200]
        ev+=[(g['t'],g['t']+g['d'],'gpu nccl %s s%s'%(g['x'],g['s'])) for g in S[k]['G'] if g['k']=='nccl' and g['t']<hi and g['t']+g['d']>lo and g['d']>200]
        ev+=[(g['t'],g['t']+g['d'],'gpu %s %s'%(g['k'],g['x'])) for g in S[k]['G'] if g['k'].startswith('memcpy') and g['t']<hi and g['t']+g['d']>lo and g['d']>1000]
        for a,b,n in sorted(ev): print('    %8.1f %8.1f (%6.1f) %s'%((a-T)/1e3,(b-T)/1e3,(b-a)/1e3,n))
