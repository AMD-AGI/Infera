import json, sys
rows = json.load(open(sys.argv[1]))
want_hw = set(sys.argv[2].split(','))
fws = set(sys.argv[3].split(','))
keys = ['tput_per_gpu','total_tput_tps','input_tput_tps','output_tput_tps','output_tput_per_gpu','mean_qps','total_requests_completed','duration_seconds',
        'mean_input_tokens','median_input_tokens','mean_output_tokens_actual','total_prompt_tokens','total_generation_tokens',
        'server_gpu_cache_hit_rate','server_cpu_cache_hit_rate','server_external_cache_hit_rate','theoretical_cache_hit_rate','gpu_kv_cache_usage_pct',
        'median_ttft','p90_ttft','mean_ttft','median_itl','p90_itl','mean_tpot','p90_intvty','median_intvty','allocated_cpu_dram_gb','avg_total_gpu_power_w','mean_e2el','dcp_size','kv_offloading']
out=[]
for r in rows:
    if r['hardware'] not in want_hw or r['framework'] not in fws: continue
    m=r['metrics']
    d={k:r.get(k) for k in ['id','hardware','framework','disagg','prefill_tp','prefill_ep','prefill_dp_attention','prefill_num_workers','decode_tp','decode_ep','decode_dp_attention','decode_num_workers','num_prefill_gpu','num_decode_gpu','conc','image','offload_mode','date','run_url']}
    for k in r:
        if k not in d and k not in ('metrics','workers','power_audit','power_invalid_reasons'): d[k]=r[k]
    d.update({k:m.get(k) for k in keys})
    d['workers']=[(w.get('role'),w.get('num_gpus'),len(w.get('hosts',[]))) for w in (r.get('workers') or [])]
    out.append(d)
json.dump(out, open(sys.argv[4],'w'), indent=1)
for d in sorted(out, key=lambda x:(x['hardware'],x['framework'],x['disagg'],-x['tput_per_gpu'])):
    g=(d['num_prefill_gpu'] or 0)+(d['num_decode_gpu'] or 0) if d['disagg'] else d['num_decode_gpu']
    print(d['id'],d['hardware'],d['framework'],'PD' if d['disagg'] else 'agg', f"P{d['prefill_num_workers']}xTP{d['prefill_tp']}EP{d['prefill_ep']}{'dpa' if d['prefill_dp_attention'] else ''}", f"D{d['decode_num_workers']}xTP{d['decode_tp']}EP{d['decode_ep']}{'dpa' if d['decode_dp_attention'] else ''}", 'gpus',d['num_prefill_gpu'],d['num_decode_gpu'],'c',d['conc'],
      'tpg',round(d['tput_per_gpu']), 'qps',d['mean_qps'], 'req',d['total_requests_completed'],'dur',d['duration_seconds'],'isl',round(d['mean_input_tokens'] or 0),'osl',round(d['mean_output_tokens_actual'] or 0),
      'hitG',round(d['server_gpu_cache_hit_rate'] or 0,3),'hitC',round(d['server_cpu_cache_hit_rate'] or 0,3),'hitE',round(d['server_external_cache_hit_rate'] or 0,3),'theo',d['theoretical_cache_hit_rate'],'kvuse',d['gpu_kv_cache_usage_pct'],
      'ttft50',d['median_ttft'],'ttft90',d['p90_ttft'],'itl50',d['median_itl'],'itl90',d['p90_itl'],'p90i',round(d['p90_intvty'] or 0,1),'dram',d['allocated_cpu_dram_gb'],'pw',d['avg_total_gpu_power_w'],'date',d.get('date'))
