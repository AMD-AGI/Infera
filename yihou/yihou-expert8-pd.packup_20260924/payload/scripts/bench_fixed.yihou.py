#!/usr/bin/env python3
"""Bounded synthetic PD benchmark with exact input IDs and server token usage."""
import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import aiohttp


def quantiles(values):
    values = sorted(values)
    if not values:
        return {}
    return {'mean': statistics.mean(values), **{
        f'p{p}': values[round((len(values)-1)*p/100)] for p in (50,90,99)}}


async def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    randomizer = random.Random(args.seed)
    # Exclude special/control IDs; use native token IDs so text decoding cannot change input length.
    prompts = [[randomizer.randrange(1000,100000) for _ in range(args.input_len)]
               for _ in range(args.warmup+args.requests)]
    (args.out/'config.json').write_text(json.dumps(vars(args), default=str, indent=2))
    active, peak = 0, 0
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for endpoint in ('/health','/v1/workers'):
            async with session.get(args.url+endpoint) as response:
                response.raise_for_status()
                body = await response.text()
                (args.out/(endpoint.rsplit('/',1)[-1]+'.json')).write_text(body)
                if endpoint == '/v1/workers':
                    data = json.loads(body)
                    workers = data if isinstance(data, list) else data.get('workers',data.get('data',data.get('instances',[])))
                    roles = [str(w.get('disagg_mode') or w.get('role') or w.get('mode') or '').lower() for w in workers]
                    if roles.count('prefill') != 1 or roles.count('decode') != 1:
                        raise RuntimeError(f'expected one prefill and one decode, got roles={roles}')

        async def request(index, phase):
            nonlocal active, peak
            async with semaphore:
                active += 1
                peak = max(peak, active)
                start = time.perf_counter()
                first, last = None, None
                chunks, usage, finish = 0, None, None
                gaps = []
                row = {'index':index,'phase':phase,'ok':False}
                payload = {'model':args.model,'prompt':prompts[index],
                           'max_tokens':args.output_len,'temperature':0,'ignore_eos':True,
                           'stream':True,'stream_options':{'include_usage':True}}
                try:
                    async with session.post(args.url+'/v1/completions',json=payload) as response:
                        response.raise_for_status()
                        async for raw in response.content:
                            line = raw.decode('utf-8').strip()
                            if not line.startswith('data:'):
                                continue
                            data = line[5:].strip()
                            if data == '[DONE]':
                                break
                            item = json.loads(data)
                            if item.get('error'):
                                raise RuntimeError(str(item['error']))
                            if item.get('usage'):
                                usage = item['usage']
                            for choice in item.get('choices',[]):
                                if choice.get('finish_reason'):
                                    finish = choice['finish_reason']
                                if choice.get('text'):
                                    now = time.perf_counter()
                                    if first is None:
                                        first = now
                                    if last is not None:
                                        gaps.append(now-last)
                                    last = now
                                    chunks += 1
                    elapsed = time.perf_counter()-start
                    if first is None:
                        raise RuntimeError('no non-empty generated text chunk')
                    if usage is None:
                        raise RuntimeError('missing measured server usage; requested length is not evidence')
                    if usage.get('prompt_tokens') != args.input_len or usage.get('completion_tokens') != args.output_len:
                        raise RuntimeError(f'unexpected measured lengths: {usage}')
                    row.update(ok=True,usage=usage,finish_reason=finish,latency_s=elapsed,
                               ttft_s=None if first is None else first-start,
                               tpot_s=None if first is None or last is None or args.output_len<2 else (last-first)/(args.output_len-1),
                               stream_chunks=chunks,chunk_intervals_s=gaps)
                except Exception as exc:
                    row.update(error=repr(exc),latency_s=time.perf_counter()-start,usage=usage)
                finally:
                    active -= 1
                with (args.out/'requests.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k!='chunk_intervals_s'}),flush=True)
                return row

        warm = await asyncio.gather(*(request(i,'warmup') for i in range(args.warmup)))
        if not all(r['ok'] for r in warm):
            raise RuntimeError('warmup failed; measured run not started')
        peak = 0
        started = time.perf_counter()
        rows = await asyncio.gather(*(request(i,'measured') for i in range(args.warmup,args.warmup+args.requests)))
        duration = time.perf_counter()-started
        good = [r for r in rows if r['ok']]
        summary = {'synthetic':True,'requests':len(rows),'completed':len(good),
                   'failures':len(rows)-len(good),'peak_client_inflight':peak,
                   'duration_s':duration,'input_len':args.input_len,'output_len':args.output_len,
                   'output_tokens_per_s':sum(r['usage']['completion_tokens'] for r in good)/duration,
                   'total_tokens_per_s':sum(r['usage']['prompt_tokens']+r['usage']['completion_tokens'] for r in good)/duration,
                   'latency_s':quantiles([r['latency_s'] for r in good]),
                   'ttft_s':quantiles([r['ttft_s'] for r in good if r['ttft_s'] is not None]),
                   'tpot_s':quantiles([r['tpot_s'] for r in good if r['tpot_s'] is not None]),
                   'chunk_intervals_s':quantiles([v for r in good for v in r['chunk_intervals_s']]),
                   'note':'TTFT is first non-empty text chunk; chunk intervals are not guaranteed per-token ITL.'}
        summary['pass'] = len(good)==args.requests and peak==min(args.concurrency,args.requests)
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary,indent=2),flush=True)
        if not summary['pass']:
            raise RuntimeError('benchmark acceptance failed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://10.245.157.237:28000')
    p.add_argument('--model',default='glm5.2-synthetic-expert8-yihou')
    p.add_argument('--input-len',type=int,default=4096)
    p.add_argument('--output-len',type=int,default=1024)
    p.add_argument('--concurrency',type=int,default=32)
    p.add_argument('--warmup',type=int,default=32)
    p.add_argument('--requests',type=int,default=256)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--timeout',type=float,default=1800)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    workspace=Path(__file__).resolve().parents[1]
    args.out=args.out.resolve()
    if not args.out.is_relative_to(workspace):
        p.error('output must be inside workspace')
    if min(args.input_len,args.output_len,args.concurrency,args.requests)<=0 or args.warmup<0:
        p.error('invalid positive counts')
    asyncio.run(run(args))

if __name__=='__main__':
    main()
