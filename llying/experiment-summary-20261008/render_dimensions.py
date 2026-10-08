"""Render batch, cache and queue comparisons from archived evidence."""
import csv


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(row)+' |' for row in rows])


def render_dimensions(rows, dimensions, out):
    def value(row, metric, stat='mean', digits=2):
        x=dimensions[row['old_code']]['stats'].get(metric,{}).get(stat)
        return '—' if x is None else f'{x:,.{digits}f}'
    def triple(row, metric, digits=2):
        return ' / '.join(value(row,metric,s,digits) for s in ['mean','p50','p90'])
    def name(row):
        return row['name']+(' †' if row['old_code']=='B3*' else '')
    batch=[];cache=[];queues=[];work=[];coverage=[]
    for r in rows:
        d=dimensions[r['old_code']];m=r['metrics']
        batch.append([name(r),triple(r,'prefill_batch_new_tokens',0),value(r,'prefill_batch_sequences'),
                      value(r,'prefill_batch_full_pct'),triple(r,'decode_logged_batch_requests'),
                      triple(r,'decode_running_total'),value(r,'prefill_effective_compute_tps',digits=0)])
        cache.append([name(r),triple(r,'decode_active_pct'),triple(r,'decode_resident_pct'),
                      value(r,'decode_evictable_pct'),value(r,'decode_free_pct'),
                      value(r,'decode_max_rank_active_pct','p90'),value(r,'decode_hot_with_spare_pct'),
                      value(r,'decode_prealloc_total')+' / '+value(r,'decode_prealloc_total','p90')])
        queues.append([name(r),f"{m['prefill_queue_ms_mean']/1000:.3f}",f"{m['prefill_queue_ms_p99']/1000:.3f}",
                       f"{m['prefill_forward_envelope_ms_mean']/1000:.3f}",
                       value(r,'prefill_queue4_other0_pct'),value(r,'decode_transfer_total')+' / '+value(r,'decode_transfer_total','p90'),
                       f"{m['decode_alloc_wait_ms_mean']:.2f}",f"{m['decode_generation_ms_mean']/1000:.3f}"])
        work.append([name(r),f"{m['completed']:,}",f"{m['mean_input_tokens']:,.0f}",f"{m['mean_output_tokens']:.2f}",
                     f"{m['output_tps_per_gpu']:.2f}",value(r,'prefill_effective_compute_tps',digits=0),f"{m['cancelled']}/{m['errors']}"])
        counts=d['samples'];s=d['stats']
        coverage.append([name(r),str(s.get('prefill_batch_new_tokens',{}).get('n',0)),str(s.get('decode_logged_batch_requests',{}).get('n',0)),
                         f"{counts.get('prefill_complete',0)}/{counts.get('prefill_scrapes',0)}",f"{counts.get('decode_complete',0)}/{counts.get('decode_scrapes',0)}",
                         f"{len(d['logged_ranks'].get('prefill',[]))}/{len(d['logged_ranks'].get('decode',[]))}"])
    inserts={
        'BATCH_TABLE':table(['实验','P有效新tokens/本地batch<br>均值 / p50 / p90','P本地batch序列数均值','P满chunk日志占比 %','D运行本地batch请求数<br>均值 / p50 / p90','D采样running总数（8rank）<br>均值 / p50 / p90','P有效新计算 tok/s'],batch),
        'CACHE_TABLE':table(['实验','D不可淘汰KV %<br>均值 / p50 / p90','D resident KV %<br>均值 / p50 / p90','可淘汰 %<br>均值','空闲块 %<br>均值','最热rank占用 %<br>p90','一热一空样本 %','D预分配队列<br>均值 / p90'],cache),
        'QUEUE_TABLE':table(['实验','P排队均值 s','P排队p99 s','P forward均值 s','P局部积压样本 %','D等待KV请求数<br>均值 / p90','D分配等待均值 ms','D生成窗口均值 s'],queues),
        'WORK_TABLE':table(['实验','完成请求','实际输入 tok/请求','实际输出 tok/请求','Output tok/s/GPU','P有效新计算 tok/s','取消/错误'],work),
        'COVERAGE_TABLE':table(['实验','P batch日志数','D周期日志数','P完整/尝试采样','D完整/尝试采样','日志P/D rank覆盖（各8）'],coverage),
    }
    keys=sorted({key for d in dimensions.values() for key in d['stats']})
    with (out/'dimensions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,lineterminator='\n',fieldnames=['name','old_code','reference','run']+
                             [k+'_'+s for k in keys for s in ['n','mean','p50','p90','p99','max','mean_delta_pct']])
        writer.writeheader()
        by_name={r['name']:r for r in rows}
        for r in rows:
            d=dimensions[r['old_code']];ref=by_name.get(r['reference'])
            base=dimensions[ref['old_code']]['stats'] if ref else {}
            record=dict(name=r['name'],old_code=r['old_code'],reference=r['reference'],run=d['run'])
            for k,stats in d['stats'].items():
                for stat,v in stats.items():
                    record[k+'_'+stat]=v
                b=base.get(k,{}).get('mean')
                if b:
                    record[k+'_mean_delta_pct']=(stats['mean']/b-1)*100
            writer.writerow(record)
    text=(out/'DIMENSION-ANALYSIS.zh-CN.md').read_text()
    for k,v in inserts.items():
        text=text.replace('{{'+k+'}}',v)
    assert '{{' not in text
    return text
