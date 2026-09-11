#!/usr/bin/env python3
"""Bounded readiness monitor; does not stop or alter the server on timeout."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('job')
p.add_argument('round')
p.add_argument('--seconds', type=int, default=2400)
a = p.parse_args()
w = Path('/apps/tas/yaoc/research/topic/infera-with-hyperloom/exp_only/Infera-yx-test/examples/glm52_fake_tp4ep4_goodput/experiments/20260910_1540_n04_c128')
r = w/'rounds'/a.round
start = time.monotonic()
while time.monotonic() - start < a.seconds:
    now = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(['spur','exec',a.job,'curl','-fsS','--max-time','5',
                             'http://127.0.0.1:31864/server_info'],capture_output=True,text=True,timeout=25)
    elapsed = round(time.monotonic()-start)
    status = {'time': now, 'round': a.round, 'job': a.job, 'elapsed_seconds': elapsed,
              'state': 'starting', 'curl_exit': result.returncode}
    if result.returncode == 0:
        try:
            info=json.loads(result.stdout)
        except json.JSONDecodeError:
            info=None
        if info is not None:
            (r/'server-info.json').write_text(json.dumps(info,indent=2)+'\n')
            status['state']='ready'
            (r/'readiness.json').write_text(json.dumps(status,indent=2)+'\n')
            (w/'state/readiness.json').write_text(json.dumps(status,indent=2)+'\n')
            print(json.dumps(status),flush=True)
            raise SystemExit(0)
    check = subprocess.run(['spur','exec',a.job,'docker','inspect','glm52-goodput-c128-20260910-1540',
                            '--format','{{.State.Running}}'],capture_output=True,text=True,timeout=25)
    if check.returncode !=0 or check.stdout.strip() != 'true':
        status.update(state='exited_or_allocation_lost',error=check.stderr[-1000:])
        (w/'state/readiness.json').write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status),flush=True)
        raise SystemExit(1)
    (w/'state/readiness.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status),flush=True)
    time.sleep(20)
raise SystemExit('Readiness timeout; server left untouched for diagnosis')
