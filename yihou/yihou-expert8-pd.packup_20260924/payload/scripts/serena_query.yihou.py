#!/usr/bin/env python3
"""Query workspace symbols using an isolated Serena MCP subprocess."""
import asyncio
import json
import os
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

W = Path(__file__).resolve().parents[1]

async def main():
    env = dict(os.environ)
    env.update(SERENA_HOME=str(W / 'cache/serena'), TMPDIR=str(W / 'tmp'),
               XDG_CACHE_HOME=str(W / 'cache/serena-xdg'), PYTHONDONTWRITEBYTECODE='1')
    params = StdioServerParameters(
        command='/home/yihou/.local/bin/serena',
        args=['start-mcp-server', '--project', str(W), '--context', 'ide',
              '--enable-web-dashboard', 'false', '--open-web-dashboard', 'false'], env=env)
    with (W / 'rounds/000-research/serena.stderr.log').open('w') as err:
        async with stdio_client(params, errlog=err) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool('get_symbols_overview', {
                    'relative_path': 'source/mooncakeperf.reference.py', 'depth': 0})
                (W / 'rounds/000-research/serena-symbols.json').write_text(
                    json.dumps(result.model_dump(mode='json'), indent=2))
                print(result.model_dump_json())

if __name__ == '__main__':
    asyncio.run(main())
