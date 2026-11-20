import json, sys
rows={d['id']:d for d in json.load(open(sys.argv[1]))}
ids=sys.argv[2].split(',')
print('id|hw|mode|GPUs P+D|conc|conc/GPU|tput/GPU|qps|qps/GPU|ISL+OSL|cycle s (conc/qps)|e2el s|TTFT mean s|TPOT mean ms|p90i|inflight/lane|cycle-e2el s|out tok/s per decode-or-agg GPU|hitG|hitC|theo|kvuse')
for i in ids:
    d=rows[i]; g=(d['num_prefill_gpu']+d['num_decode_gpu']) if d['disagg'] else d['num_decode_gpu']
    dg=d['num_decode_gpu']
    q=d['mean_qps']; t=d['mean_input_tokens']+d['mean_output_tokens_actual']
    print('|'.join(str(x) for x in [i,d['hardware'],'PD' if d['disagg'] else 'agg',f"{d['num_prefill_gpu'] if d['disagg'] else 0}+{dg}" if d['disagg'] else g,d['conc'],round(d['conc']/g,1),round(d['tput_per_gpu']),round(q,2),round(q/g,3),round(t),round(d['conc']/q,1),round(d['mean_e2el'],1),round(d['mean_ttft'],2),round(d['mean_tpot']*1000,1),round(d['p90_intvty'],1),round(q*d['mean_e2el']/d['conc'],3),round(d['conc']/q-d['mean_e2el'],1),round(q*d['mean_output_tokens_actual']/dg),d['server_gpu_cache_hit_rate'] and round(d['server_gpu_cache_hit_rate'],3),d['server_cpu_cache_hit_rate'] and round(d['server_cpu_cache_hit_rate'],3),d['theoretical_cache_hit_rate'],d['gpu_kv_cache_usage_pct'] and round(d['gpu_kv_cache_usage_pct'],3)]))
