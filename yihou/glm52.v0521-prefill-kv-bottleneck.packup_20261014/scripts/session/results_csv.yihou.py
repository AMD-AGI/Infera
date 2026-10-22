import json,glob,csv,sys,os
meta={ # round: node, job, leg variant, prefill experts, decode experts, prefill KV/rank, decode KV/rank
 'q1':   ('crsuse2-m2m-137','(plain ssh)','reference (real 256 experts both legs)',256,256,2433088,2762176),
 'x8d':  ('crsuse2-m2m-026','211170','decode-only expert8 [INVALID: GPU3 HBM ECC]',256,8,2433088,3670016),
 'x8p':  ('crsuse2-m2m-274','211171','prefill-only expert8, KV cap 3670016',8,256,3670016,2762176),
 'x8d2': ('crsuse2-m2m-206','211763','decode-only expert8 (rerun of x8d)',256,8,2433088,3670016),
 'x8pc': ('crsuse2-m2m-206','211763','prefill-only expert8, KV capped to 2433088',8,256,2433088,2762176),
 'x128p':('crsuse2-m2m-085','211169','prefill-only 128 experts (truncated), no KV cap',128,256,3318400,2762176),
}
rows=[]
for f in sorted(glob.glob(sys.argv[1]+'/*/agentx-c*/agentx_conc*.json')):
    r=f.split('/')[-3]; tag=f.split('/')[-2].split('-')[-1]
    j=json.load(open(f)); rm=j.get('request_metrics',{}); sm=j.get('server_metrics',{})
    g=lambda d,*ks: (lambda x: x)(__import__('functools').reduce(lambda a,k: a.get(k) if isinstance(a,dict) else None, ks, d))
    m=meta[r]
    ra=j.get('request_accounting',{})
    rows.append(dict(round=r, run=tag, node=m[0], job=m[1], variant=m[2], conc=j.get('conc'),
      prefill_routed_experts=m[3], decode_routed_experts=m[4], prefill_kv_tok_per_rank=m[5], decode_kv_tok_per_rank=m[6],
      score_total_tput_per_gpu=g(rm,'throughput','per_gpu','total_tput_tps'),
      qps_mean=g(rm,'qps','mean'), isl_mean=g(rm,'tokens','input','mean'), osl_mean=g(rm,'tokens','output_actual','mean'),
      ttft_mean_s=g(rm,'latency','ttft','mean'), ttft_p50_s=g(rm,'latency','ttft','p50'), ttft_p95_s=g(rm,'latency','ttft','p95'),
      tpot_mean_ms=round(g(rm,'latency','tpot','mean')*1000,2) if g(rm,'latency','tpot','mean') else None, e2el_mean_s=g(rm,'latency','e2el','mean'),
      gpu_cache_hit=g(sm,'cache','gpu_cache_hit_rate'), cpu_cache_hit=g(sm,'cache','cpu_cache_hit_rate'), overall_cache_hit=g(sm,'cache','overall_cache_hit_rate'),
      prompt_tokens_total=g(sm,'tokens','prompt_total'), prompt_tokens_computed=g(sm,'tokens','prompt_by_source','computed'),
      cpu_pool_tokens=g(sm,'kv_cache','cpu_total_tokens'),
      ok=j.get('num_requests_successful'), err_dropped=ra.get('records_error_dropped'), error_categories=json.dumps(ra.get('error_categories',{})),
      duration_s=g(rm,'throughput','duration_seconds'), json=os.path.relpath(f,sys.argv[1]+'/..')))
w=csv.DictWriter(sys.stdout,fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(x) for x in rows]
