"""Validate measured-token enforcement and concurrency against a local mock endpoint."""
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from aiohttp import web

W=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('client',W/'scripts/bench_fixed.yihou.py')
assert spec is not None and spec.loader is not None
client=importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)

async def main():
    bad=False
    async def health(request):
        if request.path == '/v1/workers':
            return web.json_response([{'disagg_mode':'prefill'},{'disagg_mode':'decode'}])
        return web.json_response({'status':'ok'})
    async def complete(request):
        body=await request.json()
        assert len(body['prompt'])==16 and isinstance(body['prompt'][0],int)
        assert body['ignore_eos'] is True
        r=web.StreamResponse(headers={'Content-Type':'text/event-stream'})
        await r.prepare(request)
        await asyncio.sleep(.02)
        for text in ['x','y']:
            await r.write(('data: '+json.dumps({'choices':[{'text':text}]})+'\n\n').encode())
        usage={'prompt_tokens':16,'completion_tokens':1 if bad else 2}
        await r.write(('data: '+json.dumps({'choices':[],'usage':usage})+'\n\ndata: [DONE]\n\n').encode())
        return r
    app=web.Application()
    app.router.add_get('/health',health)
    app.router.add_get('/v1/workers',health)
    app.router.add_post('/v1/completions',complete)
    runner=web.AppRunner(app)
    await runner.setup()
    site=web.TCPSite(runner,'127.0.0.1',0)
    await site.start()
    port=runner.addresses[0][1]
    try:
        args=SimpleNamespace(out=W/'rounds/001-component/client-good-discovery',seed=42,input_len=16,
            warmup=2,requests=6,concurrency=3,timeout=10,url=f'http://127.0.0.1:{port}',
            model='mock',output_len=2)
        await client.run(args)
        summary=json.loads((args.out/'summary.json').read_text())
        assert summary['pass'] and summary['peak_client_inflight']==3
        bad=True
        args.out=W/'rounds/001-component/client-bad-discovery'
        try:
            await client.run(args)
        except RuntimeError as e:
            assert 'warmup failed' in str(e)
        else:
            raise AssertionError('client accepted incorrect measured length')
        print('PASS: client preserves exact IDs, bounds concurrency, rejects short generation')
    finally:
        await runner.cleanup()

asyncio.run(main())
