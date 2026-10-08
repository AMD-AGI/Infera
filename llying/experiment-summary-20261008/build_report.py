import json, hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[2]
out=root/'llying/experiment-summary-20261008'
old='llying/p8d8-c80-c112-tracing-aus-20260922/'
ad='llying/p8d8-adaptive-31625-20260923/analysis/'
new='llying/router-capacity-campaign-20260928/'
dimensions=json.loads((out/'dimensions.json').read_text())
rows=[]
specs=[
('原始 C80','历史 C80','原策略；4K；C80','',old+'results/main-20260922/final/analysis/summary.json','80','n01-33 / n02-21','基线'),
('并发增至112','历史 C112','原策略；4K；C112','原始 C80',old+'results/main-20260922/final/analysis/summary.json','112','n01-33 / n02-21','负收益；并发增加40%'),
('同节点4K基线','4K复测','原策略；4K；C80','',old+'analysis/baseline4k-31625/final/summary.json','80','n03-33 / n02-21','8K的同节点参照；实际后测'),
('chunk增至8K','8K','原策略；8K；C80','同节点4K基线',old+'analysis/chunk8k-31625/final/summary.json','80','n03-33 / n02-21','负收益；同节点顺序对照'),
('释放策略基线','A0','原策略；P记账等D结束释放','',ad+'a0-final/summary.json','80','n10-29 / n02-21','R1的同进程参照'),
('P完成即释放','G0 / R1','原评分；P完成释放；无会话绑定','释放策略基线',ad+'g0-final/summary.json','80','n10-29 / n02-21','正收益；同节点同进程'),
('P KV扩至340万','C1','原策略；P GPU KV=340万/rank；host固定','释放策略基线',ad+'c1-fixed-host-31688/evidence/analysis/summary.json','80','n02-29 / n02-33','未见端到端收益；跨节点'),
('扩容后P完成即释放','C1G1','P GPU KV=340万/rank；P完成释放','P KV扩至340万','llying/experiment-summary-20261008/c1g1-source-extract.json','80','n02-29 / n02-33','正收益；同节点同容量'),
('有效工作量评分','R1+R4','P完成释放；有效输入工作量评分；无绑定','P完成即释放','llying/r1r4-4k-20260924/results/summary.json','80','n05-21 / n05-29','明显负收益；跨节点历史对照'),
('增加P会话绑定','P affinity / Base','原评分；P完成释放；仅P绑定；D radix关','P完成即释放','llying/session-affinity-20260928/performance/summary.json','80','n10-29 / n03-33','TTFT改善、吞吐小增；D换节点'),
('Dynamo式首次评分','B1','P完成释放；P绑定；新P首次评分；D radix关','增加P会话绑定',new+'b1/summary.json','80','n10-29 / n03-33','负收益；后续关闭新评分'),
('仅增加D radix','B2','原评分；P完成释放；仅P绑定；D radix开','增加P会话绑定',new+'b2/summary.json','80','n10-29 / n03-33','TTFT p90改善、p50略升；吞吐未增'),
('D radix再加D绑定','B3*','原评分；P完成释放；P/D绑定；D radix开','仅增加D radix',new+'b3-interrupted/partial-summary.json','80','n10-29 / n03-33','复用/流量改善；吞吐因果不足；恢复数据'),
('新节点组合基线','RB','原评分；P完成释放；P/D绑定；D radix开；TileLang','',new+'rb/summary.json','80','n04-33 / n05-21','换节点后重建基线；不计优化'),
('组合切换Triton','B4','原评分；P完成释放；P/D绑定；D radix开；Triton','新节点组合基线',new+'b4/summary.json','80','n04-33 / n05-21','正收益；同节点单轮对照'),
]
for name,code,config,ref,src,conc,nodes,verdict in specs:
 p=Path(src) if src.startswith('/') else root/src
 data=json.loads(p.read_text()); a=data['points'][conc]; phase=a['phases']['profiling']; h=a.get('headline'); cache=phase['cache']
 m={}
 if h:
  q=h['request_metrics']; lat=q['latency']; t=q['throughput']
  for kind in ['ttft','e2el','itl']:
   for stat in ['mean','p50','p90','p95']:
    m[kind+'_'+stat+('_ms' if kind=='itl' else '_s')]=lat[kind][stat]*(1000 if kind=='itl' else 1)
  for kind in ['intvty','e2e_norm_intvty']:
   for stat in ['p50','p90']:
    m[kind+'_'+stat+'_tps']=lat[kind][stat]
  m.update(output_tps=t['output']['tokens_per_second'],input_tps=t['input']['tokens_per_second'],total_tps_per_gpu=t['per_gpu']['total_tput_tps'],output_tps_per_gpu=t['per_gpu']['output_tput_tps'],mean_input_tokens=q['tokens']['input']['mean'],mean_output_tokens=q['tokens']['output_actual']['mean'])
  account=a['runner_accounting']['profiling'];m.update({k:account[k] for k in ['sent','completed','cancelled','errors']})
 else:
  rec=json.loads((root/new/'b3-interrupted/RECOVERED-CLIENT.json').read_text())
  m.update(output_tps=rec['output_tokens_per_second'],output_tps_per_gpu=rec['output_tokens_per_second_per_gpu'],ttft_mean_s=rec['ttft_mean_s'],ttft_p95_s=rec['ttft_p95_s'],itl_mean_ms=rec['positive_itl_mean_s']*1000,mean_input_tokens=rec['mean_input_tokens'],mean_output_tokens=rec['mean_output_tokens'],completed=rec['completed'],cancelled=rec['cancelled'],errors=rec['runner_errors'],sent=rec['completed']+rec['cancelled'])
 if code=='B3*':
  m.update({k:v for k,v in dimensions[code]['recovered_client_metrics'].items() if k!='client_count'})
 for stage in ['prefill/queue_ms','prefill/forward_envelope_ms','decode/alloc_wait_ms','decode/transfer_wait_ms','decode/generation_ms']:
  for stat in ['mean','p50','p90','p99']:
   m[stage.replace('/','_')+'_'+stat]=phase['stages_ms'][stage][stat]
 for label,key in [('p_miss_pct','miss_tokens'),('p_device_hit_pct','cached_device'),('p_host_hit_pct','cached_host')]:m[label]=cache[key]/cache['input_tokens']*100
 m['paired_requests']=phase['coverage']['paired']
 prefix={'B2':'b2/decode-local-prefix.json','B3*':'b3-interrupted/partial-decode-prefix.json','RB':'rb/decode-local-prefix.json','B4':'b4/decode-local-prefix.json'}.get(code)
 if prefix:
  d=json.loads((root/new/prefix).read_text()); m['d_local_prefix_pct']=d['prefix_tokens_last_allocation']/d['input_tokens_last_allocation']*100;m['d_prefix_observed_requests']=d['requests_with_local_prefix_event']
 rows.append(dict(name=name,old_code=code,config=config,reference=ref,concurrency=int(conc),gpus=16,nodes_p_d=nodes,verdict=verdict,source=src,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),metrics=m))
(out/'data.json').write_text(json.dumps({'scope':'2026-09-22至2026-09-29已保存的实验；2026-10-08汇总','rows':rows},ensure_ascii=False,indent=2)+'\n')

import csv
import html
from markdown_it import MarkdownIt

by_name={r['name']:r for r in rows}
metric_keys=list(dict.fromkeys(k for r in rows for k in r['metrics']))
metadata=['name','old_code','config','reference','concurrency','gpus','nodes_p_d','verdict','source','source_sha256']
with (out/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
 fields=metadata+[x for k in metric_keys for x in (k,k+'_delta_pct')]
 writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
 for r in rows:
  record={k:r[k] for k in metadata};base=by_name.get(r['reference'],{}).get('metrics',{})
  for k in metric_keys:
   value=r['metrics'].get(k);ref=base.get(k);record[k]=value
   if value is not None and ref not in (None,0):record[k+'_delta_pct']=(value/ref-1)*100
  writer.writerow(record)

columns=[('total_tps_per_gpu','Total token/s/GPU',2),('output_tps_per_gpu','Output token/s/GPU',2),('ttft_p50_s','TTFT p50 s',3),('ttft_p90_s','TTFT p90 s',3),('itl_p50_ms','ITL p50 ms',2),('itl_p90_ms','ITL p90 ms',2),('p_miss_pct','P miss %',3),('d_local_prefix_pct','D本地复用 %',2),('intvty_p50_tps','Interactivity p50 tok/s',2),('intvty_p90_tps','Interactivity p90 tok/s',2)]
headers=['实验（旧代号）','实际配置','比较基线']+[c[1] for c in columns]+['取消/错误','判断']
def cell(r,key,digits):
 value=r['metrics'].get(key)
 if value is None:return '—'
 label=f'{value:,.{digits}f}';ref=by_name.get(r['reference'],{}).get('metrics',{}).get(key)
 if ref not in (None,0):
  change=f'{value-ref:+.2f} pp' if key.endswith('_pct') else f'{(value/ref-1)*100:+.2f}%'
  label+=f' ({change})'
 if r['old_code']=='B3*' and key in ['prefill_queue_ms_mean','p_miss_pct','d_local_prefix_pct']:label+=' †'
 return label
values=[]
for r in rows:
 values.append([r['name']+' ('+r['old_code']+')',r['config'],r['reference'] or '独立基线']+[cell(r,k,d) for k,_,d in columns]+[f"{r['metrics']['cancelled']}/{r['metrics']['errors']}",r['verdict']])
def md_table(head,items):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(row)+' |' for row in items])
table=md_table(headers,values)
notes='\n\n† D绑定恢复轮：客户端总体指标完整恢复，阶段/P cache仅为已配对9,954条子集，D prefix为9,955条子集；不可与其他行的全体覆盖等同。未开启D radix的行填“—”，不把未测诊断当作数值0。\n\n'
nodes=md_table(['实验','P / D节点（省略smci355-ccs-aus-前缀）','并发','GPU数','已配对请求'],[[r['name'],r['nodes_p_d'],str(r['concurrency']),str(r['gpus']),str(r['metrics']['paired_requests'])] for r in rows])
from render_dimensions import render_dimensions
intro=(out/'INTRO.zh-CN.md').read_text()
legacy=(out/'ANALYSIS.zh-CN.md').read_text()
analysis=render_dimensions(rows, dimensions, out)+'\n\n各实验的匹配请求、成本模型和详细机制另见 [逐项补充分析](ANALYSIS.zh-CN.md)。\n\n'+legacy[legacy.index('## 未完成与不能据此回答的问题'):]
report=intro+'\n'+table+notes+analysis+'\n\n## 节点与请求关联附表\n\n'+nodes+'\n'
(out/'REPORT.zh-CN.md').write_text(report)
# Keep the standalone table readable without a Markdown renderer or external assets.
head=''.join('<th>'+html.escape(h)+'</th>' for h in headers)
body=''.join('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in row)+'</tr>' for row in values)
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PD与Router实验总报告</title><style>body{font-family:system-ui,sans-serif;margin:24px;color:#172333;line-height:1.65}h1{font-size:25px}.table{overflow:auto;max-height:78vh;border:1px solid #ccd5df}table{border-collapse:separate;border-spacing:0;font-size:13px}th,td{padding:10px;border-bottom:1px solid #dde3ea;min-width:120px;vertical-align:top}th{position:sticky;top:0;background:#e7eef6;z-index:2}th:first-child,td:first-child{position:sticky;left:0;background:#eff4fa;min-width:170px}th:first-child{z-index:3}td:nth-child(2){min-width:270px}td:last-child{min-width:190px}tr:nth-child(even){background:#f7f9fc}pre{white-space:pre-wrap;font:inherit;max-width:1100px}a{color:#1757a6}article{max-width:1150px}article table{display:block;overflow:auto}h2{margin-top:36px}h3{margin-top:28px}</style><h1>PD与Router实验：配置、性能与原因</h1><p>2026-09-22至09-29的15个测量点；2026-10-08汇总。括号为相对指定基线的变化，pp为百分点。可横向滚动，首列和表头固定。</p><p><a href="REPORT.zh-CN.md">完整Markdown报告</a> · <a href="metrics.csv">全部指标CSV</a> · <a href="data.json">源数据与追溯</a></p>'''
md=MarkdownIt('commonmark').enable('table')
page+='<article>'+md.render(intro)+'</article><div class="table"><table><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table></div><article>'+md.render(notes+analysis+'\n\n## 节点与请求关联附表\n\n'+nodes)+'</article></html>'
(out/'REPORT.html').write_text(page)
print(f'Generated {len(rows)} rows, {len(metric_keys)} metric fields; Markdown, HTML, CSV and JSON.')
