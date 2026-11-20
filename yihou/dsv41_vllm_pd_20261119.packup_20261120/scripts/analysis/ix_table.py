import json,sys
d=json.load(open(sys.argv[1]))
rows=[]
for r in d:
    m=r['metrics']
    ngpu = r['num_prefill_gpu']+r['num_decode_gpu'] if r['disagg'] else r['num_decode_gpu']
    topo = (f"P{r['prefill_num_workers']}x(tp{r['prefill_tp']}ep{r['prefill_ep']}{'dpa' if r['prefill_dp_attention'] else ''}) D{r['decode_num_workers']}x(tp{r['decode_tp']}ep{r['decode_ep']}{'dpa' if r['decode_dp_attention'] else ''})"
            if r['disagg'] else f"tp{r['decode_tp']}ep{r['decode_ep']}{'dpa' if r['decode_dp_attention'] else ''} x{r['decode_num_workers'] or 1}")
    rows.append((r['hardware'],r['framework'],r['disagg'],topo,ngpu,r['conc'],m.get('tput_per_gpu'),m.get('output_tput_per_gpu'),m.get('p90_intvty'),m.get('median_ttft'),m.get('p90_ttft'),m.get('dcp_size'),m.get('kv_offloading'),m.get('server_gpu_cache_hit_rate'),r['date'],r['workflow_run_id'],r['image'].split('@')[0],m.get('spec_algorithm')))
rows.sort(key=lambda x:(x[0],x[1],-(x[6] or 0)))
print('hw|fw|disagg|topo|gpus|conc|tput/gpu|out/gpu|p90intvty|ttft_med|ttft_p90|dcp|offload|gpu_hit|date|run|image|spec')
for x in rows:
    print('|'.join(str(round(v,1)) if isinstance(v,float) else str(v) for v in x))
