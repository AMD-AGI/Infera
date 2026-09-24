#!/usr/bin/env python3
"""Plot observed routing; do not equate session dispersion with resident KV duplication."""
import argparse, collections, csv, hashlib, itertools, json, statistics
from pathlib import Path

def group_stats(rows, key):
    groups=collections.defaultdict(list)
    for r in rows: groups[r[key]].append(r)
    result=[]
    for ident, rs in groups.items():
        counts=collections.Counter(r['p_rank'] for r in rs)
        tokens=collections.Counter()
        for r in rs: tokens[r['p_rank']]+=r['input_tokens']
        byturn=collections.defaultdict(list)
        for r in rs: byturn[r['turn']].append(r)
        pairs=[]
        if key=='conversation':
            for t, xs in byturn.items():
                ys=byturn.get(t+1,[])
                if len(xs)==len(ys)==1: pairs.append((xs[0],ys[0]))
        result.append({'id':ident,'requests':len(rs),'ranks_used':len(counts),
            'dominant_request_share':max(counts.values())/len(rs),
            'rank_requests':[counts[i] for i in range(8)],
            'rank_input_tokens':[tokens[i] for i in range(8)],
            'consecutive_pairs':len(pairs),
            'rank_switches':sum(a['p_rank']!=b['p_rank'] for a,b in pairs),
            'ambiguous_turn_indices':sum(len(v)>1 for v in byturn.values()) if key=='conversation' else None})
    return sorted(result,key=lambda x:x['id'])

def load(run):
    rows=[]; skipped=0
    source=run/'analysis/joined-requests.jsonl'; digest=hashlib.sha256()
    with source.open('rb') as f:
        for raw in f:
            digest.update(raw); x=json.loads(raw)
            if x.get('phase')!='profiling':continue
            m=x['metadata'];p=x.get('prefill');d=x.get('decode')
            if not p or not d:skipped+=1;continue
            root=m.get('root_correlation_id') or m.get('x_correlation_id')
            if not root or not m.get('conversation_id') or not isinstance(m.get('turn_index'),int):
                raise ValueError('Missing session/conversation/turn identity')
            rank=p['dp_rank'];assert isinstance(rank,int) and 0<=rank<8
            rows.append({'rid':x['rid'],'session':root,'conversation':m['conversation_id'],
                'trace':m.get('source_trace_id'),'turn':m['turn_index'],
                'source_kind':m.get('source_kind'),'agent_depth':m.get('agent_depth'),
                'start_ns':m['request_start_ns'],'p_rank':rank,'d_rank':d['dp_rank'],
                'input_tokens':p['input_tokens'],'output_tokens':x['metrics']['output_sequence_length']['value'],
                'device_tokens':p['cached_device'],'host_tokens':p['cached_host'],
                'miss_tokens':p['input_tokens']-sum(p.get(k,0) for k in ('cached_device','cached_host','cached_storage')),
                'ttft_ms':x['metrics']['time_to_first_token']['value'],
                'p_queue_ms':p['durations_ms']['queue_ms']})
    rows.sort(key=lambda r:(r['start_ns'],r['rid']))
    sessions=group_stats(rows,'session');convs=group_stats(rows,'conversation')
    multi=[x for x in convs if x['requests']>=2];pairs=sum(x['consecutive_pairs'] for x in convs)
    switches=sum(x['rank_switches'] for x in convs)
    return {'run':str(run),'source_sha256':digest.hexdigest(),'rows':rows,'sessions':sessions,'conversations':convs,
        'summary':{'requests':len(rows),'unpaired_excluded':skipped,'sessions':len(sessions),'conversations':len(convs),
        'multi_request_conversations':len(multi),'multi_rank_conversation_fraction':sum(x['ranks_used']>1 for x in multi)/len(multi) if multi else None,
        'mean_ranks_per_multi_request_conversation':statistics.mean(x['ranks_used'] for x in multi) if multi else None,
        'consecutive_turn_pairs':pairs,'consecutive_turn_rank_switches':switches,
        'consecutive_turn_rank_switch_rate':switches/pairs if pairs else None,
        'request_weighted_dominant_rank_share':sum(x['dominant_request_share']*x['requests'] for x in convs)/len(rows),
        'rank_requests':[sum(r['p_rank']==i for r in rows) for i in range(8)]}}

def figures(cases,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap,BoundaryNorm
    labels=list(cases);colors=plt.get_cmap('tab10').colors[:8]
    fig,ax=plt.subplots(1,3,figsize=(16,4.8),constrained_layout=True)
    width=.8/len(labels)
    for i,(label,c) in enumerate(cases.items()):
        s=c['summary'];x=np.arange(8)-.4+width/2+i*width
        ax[0].bar(x,np.array(s['rank_requests'])/s['requests']*100,width,label=label)
        conv=[v for v in c['conversations'] if v['requests']>=2]
        hist=collections.Counter(v['ranks_used'] for v in conv)
        ax[1].bar(np.arange(1,9)-.4+width/2+i*width,[hist[k]/len(conv)*100 for k in range(1,9)],width,label=label)
    ax[0].set(title='Completed requests by Prefill rank',xlabel='P rank',ylabel='Share of completed requests (%)',xticks=range(8))
    ax[1].set(title='Conversation dispersion (2+ observed requests)',xlabel='Distinct P ranks used',ylabel='Share of conversations (%)',xticks=range(1,9))
    ax[2].bar(labels,[100*c['summary']['consecutive_turn_rank_switch_rate'] for c in cases.values()])
    ax[2].set(title='P rank changed between consecutive turns',ylabel='Switches / unambiguous consecutive pairs (%)')
    ax[0].legend();ax[1].legend()
    fig.suptitle('Observed routing dispersion — not simultaneous KV-cache duplication')
    fig.savefig(out/'routing-overview.png',dpi=180);fig.savefig(out/'routing-overview.pdf');plt.close(fig)
    common=set.intersection(*(set(r['conversation'] for r in c['rows']) for c in cases.values()))
    counts=collections.Counter(r['conversation'] for c in cases.values() for r in c['rows'])
    chosen=sorted(common,key=lambda k:(-counts[k],k))[:40]
    maxturn=max(r['turn'] for c in cases.values() for r in c['rows'] if r['conversation'] in chosen)
    fig,axes=plt.subplots(len(cases),1,figsize=(18,4*len(cases)),constrained_layout=True,squeeze=False)
    cmap=ListedColormap(colors);cmap.set_bad('#eeeeee');norm=BoundaryNorm(np.arange(-.5,8.5),8)
    indexes={k:i for i,k in enumerate(chosen)};duplicates={}
    for ax,(label,c) in zip(axes[:,0],cases.items()):
        a=np.full((len(chosen),maxturn+1),np.nan);seen=set();ambiguous=set()
        for r in c['rows']:
            if r['conversation'] not in indexes:continue
            key=(indexes[r['conversation']],r['turn'])
            if key in seen:ambiguous.add(key)
            seen.add(key);a[key]=r['p_rank']
        for key in ambiguous:a[key]=np.nan
        duplicates[label]=len(ambiguous)
        im=ax.imshow(a,aspect='auto',interpolation='nearest',cmap=cmap,norm=norm)
        ax.set(title=label,xlabel='Actual turn_index (grey = unobserved or ambiguous)',ylabel='Same conversations across panels',yticks=range(len(chosen)),yticklabels=[k[:8]+'/'+k[-6:] for k in chosen])
        ax.tick_params(axis='y',labelsize=6)
    fig.colorbar(im,ax=list(axes[:,0]),ticks=range(8),label='Actual Prefill DP rank',shrink=.8)
    fig.suptitle('40 common conversations with most observed requests; successful profiling requests only')
    fig.savefig(out/'conversation-turn-ranks.png',dpi=180);fig.savefig(out/'conversation-turn-ranks.pdf');plt.close(fig)
    (out/'heatmap-selection.json').write_text(json.dumps({'conversation_ids':chosen,'ambiguous_cells_hidden':duplicates},indent=2)+'\n')

def interactive(cases,out):
    data={label:[{k:r[k] for k in ['rid','session','conversation','trace','turn','start_ns','p_rank','input_tokens','device_tokens','host_tokens','miss_tokens','ttft_ms']} for r in c['rows']] for label,c in cases.items()}
    payload=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><meta charset="utf-8"><title>Session / turn routing</title>
<style>body{font:15px system-ui;margin:24px;color:#243044}select{max-width:650px;margin:8px;padding:6px}svg{width:100%;height:300px;border:1px solid #ddd}table{border-collapse:collapse;width:100%}td,th{padding:5px;border-bottom:1px solid #ddd;text-align:left}#scroll{max-height:450px;overflow:auto}.note{color:#566}</style>
<h1>Session / turn → Prefill rank</h1><p>完成且已关联的 profiling 请求。分散程度不等于同时驻留的 KV 重复率。</p>
<label>实验<select id="run"></select></label><label>分组<select id="kind"><option value="session">root session（包括分支）</option><option value="conversation">conversation（连续 turn）</option></select></label><br>
<label>Session / conversation<select id="group"></select></label><p id="stats"></p><svg id="plot" viewBox="0 0 1200 300"></svg>
<p class="note">root session 横轴是按请求发起时间排列的序号，包含并行分支；conversation 横轴是原始 turn_index。每个点可悬停查看详情。各实验完成的请求集合可能不同。</p>
<div id="scroll"><table><thead><tr><th>turn</th><th>P rank</th><th>conversation</th><th>input</th><th>device hit</th><th>host hit</th><th>miss</th><th>TTFT ms</th></tr></thead><tbody id="rows"></tbody></table></div>
<script>const data=PAYLOAD;const colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f'];
const run=document.getElementById('run'),kind=document.getElementById('kind'),group=document.getElementById('group');
for(const k of Object.keys(data))run.add(new Option(k,k));
function groups(){group.replaceChildren();const counts={};for(const r of data[run.value])counts[r[kind.value]]=(counts[r[kind.value]]||0)+1;for(const [k,n] of Object.entries(counts).sort((a,b)=>b[1]-a[1]))group.add(new Option(k+' ('+n+' requests)',k));draw()}
function draw(){const rs=data[run.value].filter(r=>r[kind.value]===group.value).sort((a,b)=>a.start_ns-b.start_ns);const hist=Array(8).fill(0);rs.forEach(r=>hist[r.p_rank]++);document.getElementById('stats').textContent=rs.length+' requests · '+hist.filter(x=>x>0).length+' P ranks · rank counts: '+hist.join(', ');const xs=rs.map((r,i)=>kind.value==='session'?i:r.turn),mx=Math.max(1,...xs);let s='';for(let k=0;k<8;k++){const y=265-k*32;s+='<line x1="55" x2="1180" y1="'+y+'" y2="'+y+'" stroke="#ddd"/><text x="10" y="'+(y+5)+'">P'+k+'</text>';}rs.forEach((r,i)=>{const x=55+xs[i]/mx*1125,y=265-r.p_rank*32;s+='<circle cx="'+x+'" cy="'+y+'" r="3.5" fill="'+colors[r.p_rank]+'"><title>turn '+r.turn+' | P'+r.p_rank+' | '+r.conversation+' | '+r.rid+'</title></circle>';});s+='<text x="55" y="295">0</text><text x="1130" y="295">'+mx+'</text>';document.getElementById('plot').innerHTML=s;const tbody=document.getElementById('rows');tbody.replaceChildren();for(const r of rs){const tr=document.createElement('tr');for(const v of [r.turn,r.p_rank,r.conversation,r.input_tokens,r.device_tokens,r.host_tokens,r.miss_tokens,r.ttft_ms.toFixed(2)]){const td=document.createElement('td');td.textContent=v;tr.append(td)}tbody.append(tr)}}
run.onchange=groups;kind.onchange=groups;group.onchange=draw;groups();</script>'''.replace('PAYLOAD',payload)
    (out/'session-routing.html').write_text(page)

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',action='append',required=True,help='LABEL=/absolute/run');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    cases={label:load(Path(path)) for label,path in (x.split('=',1) for x in a.case)}
    for label,c in cases.items():
        with (a.output/f'{label}-request-routing.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(c['rows'][0]));w.writeheader();w.writerows(c['rows'])
    summary={label:{k:v for k,v in c.items() if k!='rows'} for label,c in cases.items()}
    (a.output/'routing-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    pairmaps={}
    for label,c in cases.items():
        turns=collections.defaultdict(list)
        for r in c['rows']:turns[(r['conversation'],r['turn'])].append(r)
        pairmaps[label]={key:(xs[0],turns[(key[0],key[1]+1)][0]) for key,xs in turns.items()
            if len(xs)==1 and len(turns.get((key[0],key[1]+1),[]))==1}
    comparisons={}
    for left,right in itertools.combinations(cases,2):
        common=set(pairmaps[left]) & set(pairmaps[right]);kept=[]
        for key in common:
            a1,a2=pairmaps[left][key];b1,b2=pairmaps[right][key]
            if all(x['trace']==y['trace'] and x['output_tokens']==y['output_tokens'] and
                abs(x['input_tokens']-y['input_tokens'])<=min(8,.001*max(x['input_tokens'],y['input_tokens']))
                for x,y in [(a1,b1),(a2,b2)]):kept.append(key)
        comparisons[left+' vs '+right]={'common_consecutive_pairs':len(common),'length_matched_pairs':len(kept),
            'switch_rates':{label:sum(pairmaps[label][k][0]['p_rank']!=pairmaps[label][k][1]['p_rank'] for k in kept)/len(kept) if kept else None for label in [left,right]},
            'selection':'Both endpoints match trace identity, output length, input within 8 tokens and 0.1%; completed-request intersection only.'}
    (a.output/'common-pair-comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    figures(cases,a.output);interactive(cases,a.output)
    print(json.dumps({label:c['summary'] for label,c in cases.items()},indent=2))

if __name__=='__main__':main()
