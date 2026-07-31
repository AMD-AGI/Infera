python3 - <<'PY'
import asyncio
from infera.kvd.client import KvdClient
async def m():
    c = KvdClient("/tmp/kvd/kvd.sock", client_id="stats-probe")
    await c.connect(); print(await c.stats()); await c.close()
asyncio.run(m())
PY
