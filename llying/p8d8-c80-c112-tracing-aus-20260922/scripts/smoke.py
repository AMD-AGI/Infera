#!/usr/bin/env python3
"""Exercise long-context PD requests with externally assigned trace IDs."""
import argparse
import concurrent.futures
import json
import time
import urllib.request
import uuid
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

def request(index):
    trace_id, span_id = uuid.uuid4().hex, uuid.uuid4().hex[:16]
    body = dict(rid=f"aus-smoke-{index}", model="glm5.2-mxfp4", messages=[dict(role="user", content=
        "Read these measurements and report their count briefly.\n" +
        (f"measurement {index}: temperature 20 pressure 100.\n" * (4000 if index==0 else 400)))],
        max_tokens=32, temperature=0, stream=False)
    start = time.time_ns()
    req = urllib.request.Request("http://10.235.192.136:28000/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type":"application/json", "traceparent":f"00-{trace_id}-{span_id}-01"})
    with urllib.request.urlopen(req, timeout=240) as response:
        result = json.load(response)
    if not result.get("choices"):
        raise RuntimeError(f"missing choices: {result}")
    return dict(index=index, trace_id=trace_id, start_ns=start, end_ns=time.time_ns(), response=result)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    rows = list(pool.map(request, range(8)))
args.output.write_text(json.dumps(rows, indent=2) + "\n")
time.sleep(5)
