"""Build a comparison table from validated exports, keeping recovered data explicit."""
import argparse,csv,json
from pathlib import Path

ROOT=Path('/perf_apps/liyingli/bench_agentx/router-capacity-20260928')
BASELINE=Path('/perf_apps/liyingli/bench_agentx/session-affinity-31999-20260928/runs/session-affinity-31999-performance')
CASES=[('Base','原P亲和',BASELINE,'baseline'),('B1','新P首次评分',ROOT/'runs/campaign-b1-dynamo-p','b1'),('B2','D radix',ROOT/'runs/campaign-b2-radix','b2'),('B3*','D radix + D亲和',ROOT/'runs/campaign-b3-decode-affinity','b3-interrupted'),('RB','新节点组合基线',ROOT/'runs/campaign-rb-rebaseline','rb'),('B4','Triton',ROOT/'runs/campaign-b4-triton','b4'),('B5','P8D4',ROOT/'runs/campaign-b5-p8d4','b5'),('B6','2P8D8',ROOT/'runs/campaign-b6-2p8d8','b6'),('B7','4P4D8',ROOT/'runs/campaign-b7-4p4d8','b7')]


def environment(path):
    return dict(line.split('=',1) for line in path.read_text().splitlines() if '=' in line and not line.startswith('#'))


def node_group(run):
    placement=run/'placement-resolved.json'
    if placement.exists():
        workers=json.loads(placement.read_text());p=next(w for w in workers if w['role']=='prefill');d=next(w for w in workers if w['role']=='decode')
        extra=len({w['node'] for w in workers})>2
        return p['node']+' / '+d['node']+(' + extra P' if extra else '')
    return 'smci355-ccs-aus-n10-29 / smci355-ccs-aus-n03-33'


def collect(repo):
    rows=[];pending=[]
    for case,label,run,archive in CASES:
        recovered=case=='B3*'
        source=repo/archive/'RECOVERED-CLIENT.json' if recovered else run/'c80/agentx_conc80.json'
        review=repo/archive/'REVIEW.json'
        if not source.exists() or (case!='Base' and not recovered and (not review.exists() or not json.loads(review.read_text())['checks_passed'])):
            pending.append(case);continue
        if case in ['RB','B4','B5','B6','B7'] and not (run/'review-ready.json').exists():
            pending.append(case);continue
        env=environment(run/'c80/runtime.env')
        assert env['CONC']=='80' and env['DURATION']=='3600' and env['SIMULATE_ACC_LEN']=='3.61'
        pnum=int(env['PREFILL_NUM_WORKERS']);ptp=int(env['PREFILL_TP']);dnum=int(env['DECODE_NUM_WORKERS']);dtp=int(env['DECODE_TP']);gpus=pnum*ptp+dnum*dtp
        d=json.loads(source.read_text())
        if recovered:
            output=d['output_tokens_per_second'];per_gpu=d['output_tokens_per_second_per_gpu'];mean=d['ttft_mean_s'];p95=d['ttft_p95_s'];itl=d['positive_itl_mean_s'];account=d;input_mean=d['mean_input_tokens'];output_mean=d['mean_output_tokens']
        else:
            m=d['request_metrics'];output=m['throughput']['output']['tokens_per_second'];per_gpu=m['throughput']['per_gpu']['output_tput_tps'];mean=m['latency']['ttft']['mean'];p95=m['latency']['ttft']['p95'];itl=m['latency']['itl']['mean']
            summary=json.loads((run/'analysis/summary.json').read_text());account=summary['points']['80']['runner_accounting']['profiling'];input_mean=m['tokens']['input']['mean'];output_mean=m['tokens']['output_actual']['mean']
            assert d['num_prefill_gpu']==pnum*ptp and d['num_decode_gpu']==dnum*dtp
        assert abs(output/gpus-per_gpu)<.00001
        rows.append({'case':case,'label':label,'layout':f'{pnum}P(TP{ptp})+{dnum}D(TP{dtp})','gpus_used':gpus,'backend_p':env['DSA_PREFILL_BACKEND'],'backend_d':env['DSA_DECODE_BACKEND'],'node_group':node_group(run),'completed':account['completed'],'cancelled':account['cancelled'],'errors':account['runner_errors'] if recovered else account['errors'],'mean_input_tokens':input_mean,'mean_output_tokens':output_mean,'output_tps':output,'output_tps_per_gpu':per_gpu,'ttft_mean_s':mean,'ttft_p95_s':p95,'itl_mean_ms':itl*1000,'quality':'recovered client; incomplete tail diagnostics' if recovered else 'validated export','run':str(run)})
    return rows,pending


def main():
    p=argparse.ArgumentParser();p.add_argument('report_dir',type=Path);p.add_argument('--plot',action='store_true');a=p.parse_args();rows,pending=collect(a.report_dir)
    out=a.report_dir/'overview';out.mkdir(exist_ok=True)
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'metrics.json').write_text(json.dumps({'rows':rows,'pending':pending},indent=2)+'\n')
    lines=['# C80实验对照','', '固定模拟接受长度3.61。使用GPU数用于归一化，不等同于整节点租用成本；P8D4在两台独占节点上仍有4张未用于该模型的卡。不同节点组之间不能直接归因某个feature。','', '| 点 | 改动/方案 | 组合 | 后端P/D | GPU数 | 输出token/s | 输出token/s/GPU | TTFT均值(s) | TTFT p95(s) | ITL均值(ms) | 完成/取消/错误 |','|---|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['case']} | {r['label']} | {r['layout']} | {r['backend_p']}/{r['backend_d']} | {r['gpus_used']} | {r['output_tps']:.2f} | {r['output_tps_per_gpu']:.2f} | {r['ttft_mean_s']:.3f} | {r['ttft_p95_s']:.3f} | {r['itl_mean_ms']:.3f} | {r['completed']}/{r['cancelled']}/{r['errors']} |")
    lines+=['','B3*为从完整客户端记录恢复的结果，最终导出和末尾诊断被抢占打断；不与完整验证的归档混淆。各点的平均输入/输出长度、节点、质量状态和原始路径在metrics.csv/json中。请求配对存在选择效应，单次试验差异不证明稳定收益。']
    if pending:lines+=['','尚未完成验证的点：'+', '.join(pending)+'。']
    (out/'COMPARISON.zh-CN.md').write_text('\n'.join(lines)+'\n')
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True);labels=[r['case'] for r in rows]
        for ax,(key,title) in zip(axes.flat,[('output_tps','Output tokens/s (higher is better)'),('output_tps_per_gpu','Output tokens/s/GPU (higher is better)'),('ttft_mean_s','Mean TTFT, seconds (lower is better)'),('ttft_p95_s','P95 TTFT, seconds (lower is better)')]):
            bars=ax.bar(labels,[r[key] for r in rows],color=['#718096' if r['case'] in ['Base','B1','B2','B3*'] else '#2563eb' for r in rows])
            for bar,r in zip(bars,rows):
                if r['case']=='B3*':bar.set_hatch('//')
            ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.suptitle('C80 comparison — fixed simulated acceptance length 3.61\nB3*: recovered client metrics; original and replacement nodes differ',fontsize=12)
        fig.savefig(out/'comparison.png',dpi=160);fig.savefig(out/'comparison.svg');plt.close(fig)
    print(json.dumps({'completed_rows':len(rows),'pending':pending,'output':str(out)}))


if __name__=='__main__':main()
