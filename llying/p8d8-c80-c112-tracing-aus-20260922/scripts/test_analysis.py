#!/usr/bin/env python3
"""Synthetic cohort test: warmup isolation, missing decode, and exact RID join."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as temp:
    root=Path(temp)
    (root/'c80/aiperf_artifacts').mkdir(parents=True)
    (root/'diagnostics/prefill').mkdir(parents=True)
    records=[];events=[]
    for rid,phase in [('warm','warmup'),('profile','profiling'),('missing','profiling')]:
        records.append(dict(metadata=dict(x_request_id=rid,benchmark_phase=phase,
            request_start_ns=1000000000,request_end_ns=12000000000),metrics={}))
        for role in ('prefill','decode'):
            if rid=='missing' and role=='decode':continue
            events.append(dict(event='request_summary',rid=rid,role=role,room=rid,
                wall_ns=12000000000,dp_rank=2,input_tokens=100,output_tokens=10,
                cached_device=50,cached_host=20,cached_storage=0,times=dict(
                    prefill_bootstrap_queue_entry_time=1,bootstrap_done_time=2,
                    decode_prealloc_queue_entry_time=1,decode_transfer_queue_entry_time=4,
                    wait_queue_entry_time=9,forward_entry_time=10,prefill_finished_time=11,
                    completion_time=12)))
    (root/'c80/aiperf_artifacts/profile_export.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (root/'c80/runner.log').write_text('Phase profiling (profiling) sending complete | sent=3\n'
        'Phase profiling (profiling) complete | completed=2, cancelled=1, errors=0\n')
    (root/'diagnostics/prefill/events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
    subprocess.run([sys.executable,str(Path(__file__).with_name('analyze.py')),str(root)],check=True)
    summary=json.loads((root/'analysis/summary.json').read_text())
    profile=summary['points']['80']['phases']['profiling']
    assert profile['coverage']['client_records']==2
    assert profile['coverage']['paired']==1
    assert profile['coverage']['missing_decode']==1
    assert profile['coverage']['sent_by_runner']==3
    assert profile['coverage']['unexported_requests']==1
    assert profile['coverage']['paired_fraction_of_sent']==1/3
    assert profile['stages_ms']['decode/alloc_wait_ms']['p50']==2000
    assert profile['cache']['input_tokens']==200
    assert profile['rank_totals']['prefill']=={'2':2}
    assert profile['minute_stage_ms']['0/decode/alloc_wait_ms']['p50']==2000
    assert profile['minute_prefill_work']['0/2']['miss_tokens']==60
    assert [(r['rid'],r['role']) for r in profile['incomplete_requests']]==[('missing','decode')]
    print('PASS: phase isolation, missing denominator, rank accounting, decode allocation timing')
