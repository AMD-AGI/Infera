// Spawner (design §7, §8.1): bump a kind's seq in requests/analysis.json (the whole file; T is its only
// writer), then launch `<pluginRoot>/server/o11y-run --kind <k>` detached. The job serves every seq up to
// the one it reads, so a launch while one runs is never lost.
import { P, detachArgv, nowIso, readDoc, writeDoc } from '../../../protocol/ts/index.ts'
import type { JobKind } from './counters.ts'

export interface SpawnCtx {
  fleet: string
  pluginRoot: string
  env: Record<string, string> // AGENT_SYS_O11Y_FLEET_DIR, CLAUDE_CONFIG_DIR, AGENT_SYS_HOME
}

const SPAWN_GAP_MS = 5000 // at most one launch per kind this often; the job loops while requests remain
let requests: any = null
const lastLaunch: Record<string, number> = {}

export async function request($: any, c: SpawnCtx, kind: JobKind, reason: string): Promise<string | null> {
  if (!requests) requests = (await readDoc($, `${c.fleet}/${P.requests}`, 'requests')).value ?? { v: 1 }
  const seq = (requests[kind]?.seq ?? 0) + 1
  requests = { ...requests, [kind]: { seq, ts: nowIso(), reason } }
  await writeDoc($, `${c.fleet}/${P.requests}`, requests)
  const now = Date.now()
  if (now - (lastLaunch[kind] ?? 0) < SPAWN_GAP_MS) return null
  lastLaunch[kind] = now
  return launch($, c, [`${c.pluginRoot}/server/o11y-run`, '--kind', kind])
}

// Detached; returns a problem line or null.
export async function launch($: any, c: SpawnCtx, argv: string[]): Promise<string | null> {
  try {
    const r = await $.process.run(detachArgv(`${c.fleet}/state/spawn.log`, argv), { env: c.env, timeoutMs: 10000 })
    return r.exitCode === 0 ? null : `${argv[0]}: exit ${r.exitCode} ${String(r.stderr).slice(0, 200)}`
  } catch (err: any) {
    return `${argv[0]}: ${err.message}`
  }
}
