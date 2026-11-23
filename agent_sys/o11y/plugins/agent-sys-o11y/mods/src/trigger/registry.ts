// Registry normaliser (design §7): Claude Code's session records + registry/launches.jsonl → the fleet's
// sessions. The only judge of alive, crashed, status, status_since and lineage. Also returns the live
// sessions outside the launch convention that the main should be asked about (spec §3.5).
import { P, nowIso, readDoc, readJsonl } from '../../../protocol/ts/index.ts'

export interface RegCtx {
  fleet: string
  launchDir: string
  configDir: string
  pid: number | null // this claude process
  sessionId: string
  lineage: string[] // earlier ids of this process
}

export interface Candidate {
  sid: string
  name: string
  tmux: string
  cwd: string
}

export const DEAD_RETENTION_MS = 3600_000 // a dead former main stays listed (active=false) this long

const transcripts = new Map<string, string>() // session id → history path, found once

function procStartOf(stat: string): string | null {
  // field 22 of /proc/<pid>/stat; field 2 may contain spaces, so count after the last ')'
  return stat.slice(stat.lastIndexOf(')') + 2).split(' ')[19] ?? null
}

async function aliveOf($: any, pid: number, procStart: unknown): Promise<boolean> {
  try {
    const s = await $.fs.read(`/proc/${pid}/stat`)
    return procStart == null || procStartOf(s) === String(procStart)
  } catch {
    return false
  }
}

// The tmux server a process talks to: the socket path in its TMUX variable.
async function tmuxServerOf($: any, pid: number): Promise<string | null> {
  try {
    const kv = String(await $.fs.read(`/proc/${pid}/environ`)).split('\0').find((x) => x.startsWith('TMUX='))
    return kv ? kv.slice(5).split(',')[0] : null
  } catch {
    return null
  }
}

// History location (design §4): <config dir>/projects/*/<sid>.jsonl, found once per session id; never
// derived from a slug.
async function transcriptOf($: any, configDir: string, sid: string): Promise<string | null> {
  const hit = transcripts.get(sid)
  if (hit) return hit
  try {
    for (const d of await $.fs.list(`${configDir}/projects`)) {
      const p = `${configDir}/projects/${d.name}/${sid}.jsonl`
      if (d.kind === 'directory' && (await $.fs.exists(p))) {
        transcripts.set(sid, p)
        return p
      }
    }
  } catch {}
  return null
}

function tmuxMatches(recTmux: string | null, target: unknown): boolean {
  if (!recTmux || typeof target !== 'string' || !target) return false
  // a record's tmux is "<session>:@<window>.%<pane>"; a launch target is a session name, "%<pane>" or the whole value
  const [sess, rest] = recTmux.split(':')
  return target === sess || target === recTmux || (rest ?? '').endsWith('.' + target)
}

const statusOf = (s: unknown) => (['busy', 'idle', 'waiting', 'shell'].includes(s as string) ? (s as string) : 'unknown')
const under = (dir: unknown, root: string) => typeof dir === 'string' && (dir === root || dir.startsWith(root.replace(/\/$/, '') + '/'))

// deadSince: session id → ms when a former main was first seen dead (T's private state, returned updated).
export async function normalise($: any, c: RegCtx, deadSince: Record<string, number>, now: number) {
  const dir = `${c.configDir}/sessions`
  let names: string[] = []
  try {
    names = (await $.fs.list(dir)).map((x: any) => x.name).filter((n: string) => /^\d+\.json$/.test(n))
  } catch {}
  const launches = (await readJsonl($, `${c.fleet}/${P.launches}`)).filter((l) => l && l.v === 1)
  const prev = (await readDoc($, `${c.fleet}/${P.sessions}`, 'registry.sessions')).value
  const prevByPid = new Map<number, any>((prev?.sessions ?? []).map((s: any) => [s.pid, s]))
  const prevMains = new Set<string>((prev?.sessions ?? []).filter((s: any) => s.role === 'main').map((s: any) => s.session_id))
  const isMe = (r: any) => r.sessionId === c.sessionId || (c.pid != null && r.pid === c.pid)
  const records: any[] = []
  for (const n of names) {
    try {
      const r = JSON.parse(await $.fs.read(`${dir}/${n}`))
      if (r && typeof r.pid === 'number' && typeof r.sessionId === 'string') {
        records.push(r)
        continue
      }
    } catch {}
    const p = prevByPid.get(Number(n.slice(0, -5))) // torn or malformed: keep the last good value (spec §3.4)
    if (p) records.push({ __last: p })
  }
  const me = records.find((r) => !r.__last && isMe(r))
  const myServer = c.pid ? await tmuxServerOf($, c.pid) : null
  const sessions: any[] = []
  const candidates: Candidate[] = []
  const dead: Record<string, number> = {}
  for (const r of records) {
    if (r.__last) {
      sessions.push({ ...r.__last, record_stale: true })
      continue
    }
    const alive = await aliveOf($, r.pid, r.procStart)
    const join = launches.findLast((x) => x.session_id === r.sessionId || (x.name && x.name === r.name) || tmuxMatches(r.tmux ?? null, x.tmux_target))
    let role = 'other'
    let active = false
    let parent: string | null = null
    let purpose: string | null = null
    if (isMe(r)) {
      role = 'main'
      active = true
    } else if ((join?.decision === 'main' && join.session_id === r.sessionId) || prevMains.has(r.sessionId)) {
      role = 'main' // a former main of this fleet: listed, not active, until it is gone or retained long enough
      if (!alive) {
        dead[r.sessionId] = deadSince[r.sessionId] ?? now
        if (now - dead[r.sessionId] > DEAD_RETENTION_MS) continue
      }
    } else if (join?.decision === 'child') {
      role = 'child'
      parent = join.parent_session_id ?? c.sessionId
      purpose = join.purpose ?? null
    } else if (!join && me && alive && r.tmux && under(r.cwd, c.launchDir) && (r.startedAt ?? 0) > (me.startedAt ?? 0)) {
      const server = await tmuxServerOf($, r.pid)
      if (!myServer || !server || server === myServer) candidates.push({ sid: r.sessionId, name: r.name ?? '', tmux: r.tmux, cwd: r.cwd })
    }
    if (role === 'other') continue
    const p = prevByPid.get(r.pid)
    let lineage: string[] = p && p.session_id !== r.sessionId && p.proc_start === (r.procStart ?? null) ? [...(p.lineage ?? []), p.session_id] : [...(p?.lineage ?? [])]
    if (isMe(r)) lineage = [...c.lineage]
    sessions.push({
      session_id: isMe(r) ? c.sessionId : r.sessionId, // the record can lag behind a /clear
      pid: r.pid,
      proc_start: r.procStart ?? null,
      alive,
      crashed: !alive,
      name: r.name ?? null,
      cwd: r.cwd ?? null,
      tmux: r.tmux ?? null,
      transcript_path: await transcriptOf($, c.configDir, isMe(r) ? c.sessionId : r.sessionId),
      status: statusOf(r.status),
      waiting_for: r.waitingFor ?? null,
      ...(r.statusUpdatedAt ? { status_since: nowIso(r.statusUpdatedAt) } : {}),
      role,
      parent_session_id: parent,
      lineage,
      ...(r.startedAt ? { started_at: nowIso(r.startedAt) } : {}),
      ...(r.updatedAt ? { updated_at: nowIso(r.updatedAt) } : {}),
      version: r.version ?? null,
      kind: r.kind ?? null,
      purpose,
      active,
      fleet_dir: c.fleet,
      config_dir: c.configDir,
      launch_dir: isMe(r) ? c.launchDir : null,
    })
  }
  // A former main whose record is gone: Claude Code deletes a dead process's record when another session
  // starts, so its last snapshot decides. Alive at its last snapshot means it died without session.end
  // (crashed); a clean end was written by its own trigger.
  const listed = new Set(sessions.map((s) => s.session_id))
  for (const p of prev?.sessions ?? []) {
    if (p.role !== 'main' || listed.has(p.session_id) || p.pid === c.pid) continue
    dead[p.session_id] = deadSince[p.session_id] ?? now
    if (now - dead[p.session_id] > DEAD_RETENTION_MS) continue
    sessions.push({ ...p, alive: false, active: false, crashed: p.alive === true || p.crashed === true })
  }
  return { sessions, candidates, deadSince: dead }
}
