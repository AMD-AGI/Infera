// Spawner (design §7, §8.1): bump a kind's seq in requests/analysis.json (the whole file; T is its only
// writer), then launch `<pluginRoot>/server/o11y-run --kind <k>` detached. The job serves every seq up to
// the one it reads, so a launch while one runs is never lost.
import { P, detachArgv, nowIso, readDoc, writeDoc } from '../../../protocol/ts/index.ts'
import type { JobKind } from './counters.ts'

// env: AGENT_SYS_O11Y_FLEET_DIR, CLAUDE_CONFIG_DIR, AGENT_SYS_HOME for the launched process
export interface SpawnCtx { fleet: string; pluginRoot: string; env: Record<string, string> }

const SPAWN_GAP_MS = 5000 // at most one launch per kind this often; the job serves every seq up to the newest
let requests: any = null, chain: Promise<unknown> = Promise.resolve() // bumps: read-modify-write of one file, one at a time
const lastLaunch: Record<string, number> = {}, deferred: Record<string, boolean> = {}

export function request($: any, c: SpawnCtx, kind: JobKind, reason: string): Promise<string | null> {
  const done = chain.then(() => bump($, c, kind, reason))
  chain = done.catch(() => {})
  return done
}

async function bump($: any, c: SpawnCtx, kind: JobKind, reason: string): Promise<string | null> {
  requests ??= (await readDoc($, `${c.fleet}/${P.requests}`, 'requests')).value ?? { v: 1 }
  requests = { ...requests, [kind]: { seq: (requests[kind]?.seq ?? 0) + 1, ts: nowIso(), reason } }
  await writeDoc($, `${c.fleet}/${P.requests}`, requests)
  const wait = lastLaunch[kind] + SPAWN_GAP_MS - Date.now()
  if (!(wait > 0)) return start($, c, kind)
  // within the gap: one launch when it ends, so a request after a job has exited is still served
  if (!deferred[kind]) (deferred[kind] = true), $.clock.after(wait, () => ((deferred[kind] = false), start($, c, kind)))
  return null
}

function start($: any, c: SpawnCtx, kind: JobKind): Promise<string | null> {
  lastLaunch[kind] = Date.now()
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
