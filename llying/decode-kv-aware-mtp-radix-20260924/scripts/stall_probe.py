#!/usr/bin/env python3
"""Reproduce the ~11 s PD handshake stall and capture stacks while it happens.

Sends long-prefix requests one at a time through the router: round 0 is cold on
both legs, later rounds reuse the same documents (prefill and decode device
hits). When a request has been in flight longer than --stall-s, dumps the
Python stacks of every SGLang/Infera process in the prefill and decode
containers with `py-spy dump --nonblocking` (does not pause the process).

Usage:
    stall_probe.py --router URL --model NAME --out-dir DIR
        --prefill-node HOST --prefill-container NAME
        --decode-node HOST --decode-container NAME
Artifacts: OUT_DIR/requests.jsonl (one line per request, UTC timestamps) and
    OUT_DIR/stacks-<request>-<role>.txt for each stalled request.
Exit: 0 always once the run completes; the analysis is separate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import threading
import time
from pathlib import Path

from l2_probe import code, document

DUMP = (
    "for p in $(pgrep -f 'sglang|infera'); do "
    "echo \"=== pid $p: $(tr '\\0' ' ' < /proc/$p/cmdline | cut -c1-160)\"; "
    "py-spy dump --nonblocking --pid $p 2>&1; done"
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def dump_stacks(node: str, container: str, path: Path) -> None:
    started = now()
    # The script travels on stdin: nesting it in ssh + docker exec quoting breaks it.
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node,
         f"docker exec -i {container} sh"],
        input=DUMP, capture_output=True, text=True, timeout=120,
    )
    path.write_text(f"# dump started {started} finished {now()}\n{proc.stdout}{proc.stderr}")


def ask(router: str, model: str, doc: int, records: int, record: int, max_tokens: int,
        pad_words: int = 0) -> dict:
    import urllib.request

    # Padding only lengthens the uncached suffix, i.e. the prefill extend length.
    pad = (" Double-check the record number before answering." * pad_words) if pad_words else ""
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "messages": [
        {"role": "system", "content": document(doc, records)},
        {"role": "user", "content": f"What is the code in record {record}? Answer briefly.{pad}"}]}
    req = urllib.request.Request(router + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        d = json.loads(resp.read())
    content = d["choices"][0]["message"].get("content") or ""
    return {"correct": code(doc, record) in content, "content": content[:80], "usage": d.get("usage")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--router", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--prefill-node", required=True)
    ap.add_argument("--prefill-container", required=True)
    ap.add_argument("--decode-node", required=True)
    ap.add_argument("--decode-container", required=True)
    ap.add_argument("--docs", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--records", type=int, default=740)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--stall-s", type=float, default=4.0)
    ap.add_argument("--first-doc", type=int, default=3000)
    ap.add_argument("--pad-words", type=int, default=0,
                    help="repeat a 7-word sentence N times after the question")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = (args.out_dir / "requests.jsonl").open("a")
    docs = list(range(args.first_doc, args.first_doc + args.docs))

    index = 0
    for rnd in range(args.rounds):
        for doc in docs:
            record = (97 * index + 13) % args.records
            result: dict = {}
            done = threading.Event()

            def worker():
                try:
                    result.update(ask(args.router, args.model, doc, args.records, record,
                                      args.max_tokens, args.pad_words))
                except Exception as e:  # noqa: BLE001 - recorded, run continues
                    result["error"] = f"{type(e).__name__}: {e}"
                done.set()

            entry = {"index": index, "round": rnd, "doc": doc, "record": record, "sent": now()}
            t0 = time.monotonic()
            threading.Thread(target=worker, daemon=True).start()
            dumps = []
            if not done.wait(args.stall_s):
                entry["stall_dump_at"] = now()
                for role, node, container in (
                    ("prefill", args.prefill_node, args.prefill_container),
                    ("decode", args.decode_node, args.decode_container),
                ):
                    path = args.out_dir / f"stacks-{index:03d}-{role}.txt"
                    t = threading.Thread(target=dump_stacks, args=(node, container, path), daemon=True)
                    t.start()
                    dumps.append(t)
            done.wait()
            entry.update(result, done=now(), elapsed_s=round(time.monotonic() - t0, 3))
            for t in dumps:
                t.join()
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(f"{index:3d} r{rnd} doc{doc} {entry['elapsed_s']:7.2f}s "
                  f"{'STALL' if 'stall_dump_at' in entry else ''} {entry.get('error', '')}", flush=True)
            index += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
