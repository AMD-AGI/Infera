"""Run one frozen placement, preserve the engines, and export per-worker evidence."""
import datetime,json,os,signal,subprocess,sys,time
from pathlib import Path
import transition_two_node as ops

E=os.environ;ROOT=Path(E['TRACE_RUNTIME']);RUN=Path(E['RUN'])
rows=json.loads((RUN/'placement-resolved.json').read_text());observers=[];logs=[]


def state(name):
    (RUN/'STATUS').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+name+'\n')
    print(name,flush=True)


def observe(args,name):
    stream=(RUN/'logs'/name).open('w');logs.append(stream)
    p=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);observers.append(p)


def stop_observers():
    for p in observers:
        if p.poll() is None:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    for p in observers:
        try:p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            p.wait()
    for stream in logs:stream.close()
    observers.clear();logs.clear()


if __name__=='__main__':
    if (RUN/'c80-started.txt').exists():raise SystemExit('point already started')
    for d in ['logs','sampling']:(RUN/d).mkdir(exist_ok=True)
    engine=['python3',str(ROOT/'scripts/sample_engine_metrics.py')]
    for w in rows:engine+=['--endpoint',w['instance']+'='+w['url']+'/metrics']
    engine+=['--endpoint','router=http://'+E['PREFILL_IP']+':28000/metrics','--output',str(RUN/'sampling/engine.jsonl'),'--interval','2','--duration','0']
    node=['python3',str(ROOT/'scripts/sample_node_runtime.py')];seen=set()
    for w in rows:
        if w['node'] not in seen:
            node+=['--node',w['instance']+'='+w['node']];seen.add(w['node'])
    node+=['--output',str(RUN/'sampling/nodes.jsonl'),'--interval','5','--duration','0']
    capture=[sys.executable,'-c','import subprocess,time,sys\nwhile True:\n subprocess.run([sys.executable,sys.argv[1]])\n time.sleep(60)',str(ROOT/'scripts/capture_placement.py')]
    try:
        state('SAMPLING');observe(engine,'engine-sampler.log');observe(node,'node-sampler.log');observe(capture,'capture.log')
        (RUN/'monitor-pids.txt').write_text(' '.join(str(p.pid) for p in observers)+'\n')
        (RUN/'c80-started.txt').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n')
        state('BENCHMARK_C80')
        cmd=['bash',E['BENCH_DIR']+'/agentx_bench.sh','CONFIG='+E['CONFIG'],'TOPOLOGY='+E['TOPOLOGY'],'CONC=80','DURATION=3600','OUT_DIR='+str(RUN/'c80')]
        with (RUN/'logs/c80.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        (RUN/'c80-completed.txt').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n')
        state('MEASUREMENT_COMPLETE');stop_observers()
        subprocess.run([sys.executable,str(ROOT/'scripts/capture_placement.py')],check=True)
        (RUN/'final-server-info').mkdir(exist_ok=True)
        for w in rows:
            info=ops.get(w['url']+'/get_server_info')
            (RUN/f"final-server-info/{w['instance']}.json").write_text(json.dumps(info,indent=2)+'\n')
        state('ANALYZING')
        with (RUN/'logs/analysis.log').open('w') as f:
            for script in ['analyze.py','analyze_sessions.py','analyze_runtime.py','analyze_guard_lifecycle.py','analyze_decode_prefix.py','analyze_balance.py']:
                subprocess.run([sys.executable,str(ROOT/'scripts'/script),str(RUN)],stdout=f,stderr=subprocess.STDOUT,check=True)
            subprocess.run([sys.executable,str(ROOT/'scripts/compare_c80_runs.py'),E['BASELINE_RUN'],str(RUN),'--reference-label','P_session_baseline','--candidate-label',E['RUN_ID'],'--output-dir',str(RUN/'analysis/comparison-baseline')],stdout=f,stderr=subprocess.STDOUT,check=True)
        state('COMPLETE_REVIEW_PENDING')
    except BaseException:
        state('NEEDS_REVIEW_SERVICES_PRESERVED')
        raise
    finally:stop_observers()
