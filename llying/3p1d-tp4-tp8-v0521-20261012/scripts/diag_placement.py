#!/usr/bin/env python3
"""Send known-answer requests one at a time and attribute each to the prefill instance,
prefill DP rank and decode DP rank that served it, from the server logs.

Each request has a unique prompt length (records = base + i), so the prefill log line
"Prefill batch ... #new-token: N" (N = prompt tokens, single chunk) identifies the prefill
instance and DP rank; decode "Decode batch" lines inside the request window give the
decode DP rank. Usage:
  diag_placement.py --url ROUTER --model NAME --logs LAUNCH/server-logs --out OUT.json
                    [--n 24] [--records 100] [--decode-log decode-0.log]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kv_gate  # noqa: E402

TS = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)Z")


def lines_between(path: Path, start: str, end: str):
    out = []
    for line in path.read_text(errors="replace").splitlines():
        m = TS.match(line)
        if m and start <= m.group(1) <= end:
            out.append(line)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--logs", required=True, type=Path)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--records", type=int, default=100)
    ap.add_argument("--decode-log", default="decode-0.log")
    args = ap.parse_args()
    seed0 = int(time.time()) % 100000 * 100
    rows = []
    for i in range(args.n):
        start = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        r = kv_gate.ask(args.url, args.model, seed0 + i, args.records + i, 300)
        time.sleep(1.5)  # let docker logs flush
        end = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() + 1))
        r["window"] = [start, end]
        rows.append(r)
    time.sleep(3)
    for r in rows:
        start, end = r["window"]
        n_tok = r.get("prompt_tokens")
        r["prefill"] = []
        for p in sorted(args.logs.glob("prefill-*.log")):
            for line in lines_between(p, start, end + ".999999999"):
                m = re.search(r"DP(\d+) TP\d+\] Prefill batch, #new-seq: \d+, #new-token: (\d+)", line)
                if m and n_tok and abs(int(m.group(2)) - n_tok) <= 64:
                    r["prefill"].append(f"{p.stem}/DP{m.group(1)}")
        dec = set()
        for line in lines_between(args.logs / args.decode_log, start, end + ".999999999"):
            m = re.search(r"DP(\d+) TP\d+\] Decode batch", line)
            if m:
                dec.add(int(m.group(1)))
        r["decode_dp"] = sorted(dec)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    for r in rows:
        ok = r["verdict"] in ("exact", "contains")
        print(f"{'OK ' if ok else 'BAD'} tok={r.get('prompt_tokens')} P={','.join(r['prefill']) or '?'} "
              f"D={r['decode_dp']} expected={r['expected']} got={r.get('content', '')[:40]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
