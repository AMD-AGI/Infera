#!/usr/bin/env python3
"""Prepare (not run) a fixed P-capacity intervention with constant host tokens."""
import argparse,json,shlex
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('control_run',type=Path);p.add_argument('control_config',type=Path);p.add_argument('case_id');p.add_argument('output',type=Path);p.add_argument('--tokens',type=int,default=3400000);a=p.parse_args()
assert a.case_id.replace('-','').isalnum()
server=json.loads((a.control_run/'launch/server-info/prefill-0.json').read_text())
assert server.get('max_total_tokens') is None, 'Use an uncapped control for this first capacity trial'
assert a.tokens>server['max_total_num_tokens'] and a.tokens%64==0
validation=json.loads((a.control_run/'case-validation.json').read_text())
host_values={int(x['value']) for x in validation['host_capacity_samples']};assert len(host_values)==1
host=host_values.pop();ratio=(host-32)/a.tokens
assert (int(a.tokens*ratio)//64+1)*64==host
mode=json.loads((a.control_run/'snapshot/router-mode-validation.json').read_text())['mode']
root=a.control_run.parent.parent
s=f'''#!/usr/bin/env bash
source {shlex.quote(str(a.control_config))}
RUN_ID={shlex.quote(a.case_id)}
RUN="$TRACE_RUNTIME/runs/$RUN_ID"
BASELINE_RUN={shlex.quote(str(a.control_run))}
PREFILL_MEM_FRACTION=0.90
PREFILL_MAX_TOTAL_TOKENS={a.tokens}
PREFILL_HICACHE_RATIO={ratio!r}
PREFILL_EXTRA_ARGS="$PREFILL_EXTRA_ARGS --max-total-tokens {a.tokens}"
EXPECTED_PREFILL_TOKENS={a.tokens}
EXPECTED_HOST_TOKENS={host}
GUARD_MODE={mode}
DIAG_SOURCE_PREFILL="$RUN_ID"
DIAG_SOURCE_DECODE="$RUN_ID"
SERVICE_RUN_ID="$RUN_ID"
export RUN_ID RUN BASELINE_RUN
'''
a.output.write_text(s)
print(json.dumps({'status':'PREPARED_NOT_LAUNCHED','gpu_tokens_before':server['max_total_num_tokens'],'gpu_tokens_after':a.tokens,'host_tokens_fixed':host,'host_ratio':ratio,'guard_mode_fixed':mode,'configuration':str(a.output)}))
