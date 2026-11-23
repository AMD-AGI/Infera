#!/usr/bin/env python3
"""Open-loop replay of a decode arrival trace against vLLM decode-only servers (DecodeBenchConnector).

  decode_replay.py --trace data/m10-decode-arrivals.jsonl.gz --start 600 --end 2100 \
      --url http://127.0.0.1:27501 --out A/replay.jsonl [--url ... --out ...] [--drain 120] [--speed 1]

Each trace record [t_s, isl, osl] is the moment a request reached the P4D4 decode in a real AgentX run (request start +
TTFT) with its prompt / output token counts. Records with start <= t_s < end are sent at (t_s - start) / speed to every
--url at once, as /v1/completions with random prompt token ids (no prefix caching on the decode, so content does not
matter), max_tokens = osl, ignore_eos, and the model's default sampling (as AgentX sends no sampling parameters).
The servers fill the prompt KV with dummy values and compute the last prompt token, like a PD decode after a KV pull.
Writes one JSON line per request per target and $out.window ("<first send epoch> <last send epoch>").
Runs inside the vLLM image (aiohttp, numpy).
"""
import argparse
import asyncio
import gzip
import json
import resource
import time

import aiohttp
import numpy as np


def load(path, start, end):
    recs = []
    with gzip.open(path, "rt") as f:
        f.readline()
        for line in f:
            t, isl, osl = json.loads(line)
            if start <= t < end:
                recs.append((t - start, int(isl), int(osl)))
    return recs


async def send(session, url, model, ids, osl, meta, out, stats):
    payload = {"model": model, "prompt": ids, "max_tokens": osl, "ignore_eos": True, "stream": False}
    loop = asyncio.get_running_loop()
    body = await loop.run_in_executor(None, lambda: json.dumps(payload).encode())
    t_sent = time.time()
    status, ctoks = None, None
    try:
        async with session.post(url + "/v1/completions", data=body, headers={"Content-Type": "application/json"}) as r:
            status = r.status
            txt = await r.read()
            if status == 200:
                ctoks = (json.loads(txt).get("usage") or {}).get("completion_tokens")
    except asyncio.CancelledError:
        status = "cancelled"
    except Exception as e:  # noqa: BLE001
        status = repr(e)[:200]
    t_done = time.time()
    stats[status == 200] += 1
    out.write(json.dumps({"t_trace": meta[0], "t_sent": round(t_sent, 3), "t_done": round(t_done, 3), "isl": meta[1],
                          "osl": osl, "status": status, "completion_tokens": ctoks}) + "\n")


async def main():
    # One connection per in-flight request: a container's default soft limit of 1024 files ran out in b0 (~30% failed).
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--end", type=float, default=1e9)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--drain", type=float, default=120, help="seconds to wait for in-flight requests after the last send")
    ap.add_argument("--url", action="append", required=True)
    ap.add_argument("--out", action="append", required=True)
    ap.add_argument("--model", default="glm5.2-mxfp4")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    assert len(a.url) == len(a.out)
    recs = load(a.trace, a.start, a.end)
    rng = np.random.default_rng(a.seed)
    base = rng.integers(1000, 150000, size=2_000_000, dtype=np.int32)
    outs = [open(p, "w", buffering=1) for p in a.out]
    stats = [{True: 0, False: 0} for _ in a.url]
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=None)
    tasks = []
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        t0 = time.time()
        print(f"{time.strftime('%H:%M:%S')} replay {len(recs)} requests, trace [{a.start}, {a.end}) s, speed {a.speed}, "
              f"{len(a.url)} target(s)", flush=True)
        last_log = t0
        for t, isl, osl in recs:
            delay = t0 + t / a.speed - time.time()
            if delay > 0:
                await asyncio.sleep(delay)
            off = int(rng.integers(0, len(base) - isl))
            ids = base[off:off + isl].tolist()
            for k, url in enumerate(a.url):
                tasks.append(asyncio.create_task(send(session, url, a.model, ids, osl, (t, isl), outs[k], stats[k])))
            if time.time() - last_log > 300:
                last_log = time.time()
                print(f"{time.strftime('%H:%M:%S')} sent {len(tasks) // len(a.url)} / {len(recs)}; ok / failed per target "
                      + " ".join(f"{s[True]}/{s[False]}" for s in stats), flush=True)
        t_last = time.time()
        for p in a.out:
            with open(p + ".window", "w") as f:
                f.write(f"{t0:.3f} {t_last:.3f}\n")
        done, pending = await asyncio.wait(tasks, timeout=a.drain) if tasks else (set(), set())
        for x in pending:
            x.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    for o in outs:
        o.close()
    print(f"{time.strftime('%H:%M:%S')} done: sent {len(recs)} per target over {t_last - t0:.0f} s; ok / failed per target "
          + " ".join(f"{s[True]}/{s[False]}" for s in stats) + f"; {len(pending)} cancelled after the drain", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
