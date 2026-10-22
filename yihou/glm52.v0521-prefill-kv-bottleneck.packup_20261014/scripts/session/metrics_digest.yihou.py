import gzip,sys,statistics as st,collections
def q(v,p):
    v=sorted(v); return v[min(len(v)-1,int(p*len(v)))] if v else float('nan')
print("run\trole\tmax_total_num_tokens\ttoken_usage_max_perDP\ttoken_usage_p90_perDP\tqueue_reqs_mean\tqueue_reqs_max\ttransfer_q_max\tprealloc_q_max\tinflight_q_mean\tretracted_max\thost_used_frac_max")
for f in sys.argv[1:]:
    d=collections.defaultdict(list)
    for line in gzip.open(f,'rt'):
        p=line.rstrip('\n').split('\t')
        if len(p)<5 or p[2]=='-': continue
        try: v=float(p[4])
        except: continue
        d[(p[1],p[2],p[3])].append(v)
    run=f.replace('.tsv.gz','')
    for role in ('prefill','decode'):
        dps=sorted({k[1] for k in d if k[0]==role})
        g=lambda m:[d[(role,dp,m)] for dp in dps]
        mt=[int(max(x)) if x else 0 for x in g('max_total_num_tokens')]
        tu=g('token_usage')
        qr=[v for x in g('num_queue_reqs') for v in x]
        tq=[v for x in g('num_decode_transfer_queue_reqs') for v in x]
        pq=[v for x in g('num_decode_prealloc_queue_reqs') for v in x]
        iq=[v for x in g('num_prefill_inflight_queue_reqs') for v in x]
        rt=[v for x in g('num_retracted_reqs') for v in x]
        hu=g('hicache_host_used_tokens'); ht=g('hicache_host_total_tokens')
        hf=max([max(a)/max(b) for a,b in zip(hu,ht) if a and b and max(b)>0] or [float('nan')])
        print("\t".join(map(str,[run,role,"/".join(map(str,sorted(set(mt)))),
          "/".join("%.2f"%max(x) for x in tu if x),"/".join("%.2f"%q(x,.9) for x in tu if x),
          "%.2f"%(st.mean(qr) if qr else float('nan')),"%d"%(max(qr) if qr else -1),
          "%d"%(max(tq) if tq else -1),"%d"%(max(pq) if pq else -1),"%.2f"%(st.mean(iq) if iq else float('nan')),
          "%d"%(max(rt) if rt else -1),"%.3f"%hf])))
