#!/usr/bin/env python3
"""Read-only Slurm allocation journal; never treats SchedNodeList as allocated."""
import argparse
import datetime
import json
import re
import subprocess
import time
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--job',default='31526')
args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
while True:
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':args.job}
    try:
        text=subprocess.check_output(['scontrol','show','job',args.job,'-o'],text=True,timeout=15)
        result['raw']=text.strip()
        fields=dict(re.findall(r'(\w+)=([^ ]*)',text))
        result['state']=fields.get('JobState')
        result['reason']=fields.get('Reason')
        if fields.get('JobState')=='RUNNING' and fields.get('NodeList') not in (None,'','(null)'):
            nodes=subprocess.check_output(['scontrol','show','hostnames',fields['NodeList']],text=True,timeout=15).split()
            result['allocated_nodes']=nodes
            if len(nodes)==2:
                args.output.with_name('ALLOCATION_READY.json').write_text(json.dumps(result,indent=2)+'\n')
    except Exception as exc:
        result['error']=str(exc)
    with args.output.open('a') as stream:stream.write(json.dumps(result)+'\n')
    time.sleep(60)
