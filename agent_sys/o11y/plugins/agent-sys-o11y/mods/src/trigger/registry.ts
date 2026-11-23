// Registry normaliser (design §7): Claude Code's session records + registry/launches.jsonl → the fleet's
// sessions. The only judge of alive, crashed, status, status_since and lineage. Also returns the live
// sessions outside the launch convention that the main should be asked about (spec §3.5).
import { P, nowIso, readDoc, readJsonl } from '../../../protocol/ts/index.ts'

export interface RegCtx { fleet: string; launchDir: string; configDir: string; sessionId: string; lineage: string[] } // lineage: earlier ids of this process
export interface Candidate { sid: string; name: string; tmux: string; cwd: string }

export const DEAD_RETENTION_MS = 3600_000 // a dead former main stays listed (active=false) this long

const transcripts = new Map<string, string>() // session id → history path, found once

function read($: any, path: string): Promise<string | null> {
  return $.fs.read(path).catch(() => null)
}

// Alive: /proc/<pid>/stat exists and its field 22 equals the record's procStart (field 2 may hold spaces).
async function aliveOf($: any, pid: number, procStart: unknown): Promise<boolean> {
  const s = await read($, `/proc/${pid}/stat`)
  return s != null && (procStart == null || s.slice(s.lastIndexOf(')') + 2).split(' ')[19] === String(procStart))
}

// The tmux server a process talks to: the socket path in its TMUX variable.
async function tmuxServerOf($: any, pid: number | string): Promise<string | null> {
  const kv = (await read($, `/proc/${pid}/environ`))?.split('\0').find((x) => x.startsWith('TMUX='))
  return kv ? kv.slice(5).split(',')[0] : null
}

// History location (design §4): <config dir>/projects/*/<sid>.jsonl, found once per session id; never from a slug.
async function transcriptOf($: any, configDir: string, sid: string): Promise<string | null> {
  if (transcripts.has(sid)) return transcripts.get(sid)!
  for (const d of await $.fs.list(`${configDir}/projects`).catch(() => [])) {
    const p = `${configDir}/projects/${d.name}/${sid}.jsonl`
    if (d.kind === 'directory' && (await $.fs.exists(p))) return transcripts.set(sid, p), p
  }
  return null
}

// A record's tmux is "<session>:@<window>.%<pane>"; a launch target is a session name, "%<pane>" or the whole value.
function tmuxMatches(recTmux: string | null, target: unknown): boolean {
  if (!recTmux || typeof target !== 'string' || !target) return false
  const [sess, rest] = recTmux.split(':')
  return target === sess || target === recTmux || (rest ?? '').endsWith('.' + target)
}

const statusOf = (s: unknown) => (['busy', 'idle', 'waiting', 'shell'].includes(s as string) ? (s as string) : 'unknown')
const under = (dir: unknown, root: string) => typeof dir === 'string' && (dir === root || dir.startsWith(root.replace(/\/$/, '') + '/'))

// deadSince: session id → ms when a former main was first seen dead (T's private state, returned updated).
export async function normalise($: any, c: RegCtx, deadSince: Record<string, number>, now: number) {
  const dir = `${c.configDir}/sessions`
  const names: string[] = (await $.fs.list(dir).catch(() => [])).map((x: any) => x.name).filter((n: string) => /^\d+\.json$/.test(n))
  const launches = (await readJsonl($, `${c.fleet}/${P.launches}`)).filter((l) => l && l.v === 1)
  const prev: any[] = (await readDoc($, `${c.fleet}/${P.sessions}`, 'registry.sessions')).value?.sessions ?? []
  const prevByPid = new Map<number, any>(prev.map((s) => [s.pid, s]))
  const prevMains = new Set(prev.filter((s) => s.role === 'main').map((s) => s.session_id))
  const pid = Number((await read($, '/proc/self/stat'))?.split(' ')[0]) || null // this claude process
  const isMe = (r: any) => r.sessionId === c.sessionId || r.pid === pid
  const dead: Record<string, number> = {}
  const retained = (sid: string) => now - (dead[sid] = deadSince[sid] ?? now) <= DEAD_RETENTION_MS
  const records: any[] = []
  for (const n of names) {
    let r: any = null
    try {
      r = JSON.parse((await read($, `${dir}/${n}`)) ?? '')
    } catch {}
    if (r && typeof r.pid === 'number' && typeof r.sessionId === 'string') records.push(r)
    else if (prevByPid.has(Number(n.slice(0, -5)))) records.push({ __last: prevByPid.get(Number(n.slice(0, -5))) }) // torn: last good value (spec §3.4)
  }
  const me = records.find((r) => !r.__last && isMe(r))
  const myServer = pid ? await tmuxServerOf($, pid) : null
  const sessions: any[] = []
  const candidates: Candidate[] = []
  for (const r of records) {
    if (r.__last) {
      sessions.push({ ...r.__last, record_stale: true })
      continue
    }
    const alive = await aliveOf($, r.pid, r.procStart)
    const join = launches.findLast((x) => x.session_id === r.sessionId || (x.name && x.name === r.name) || tmuxMatches(r.tmux ?? null, x.tmux_target))
    const mine = isMe(r)
    let role = mine ? 'main' : 'other'
    if (!mine && prevMains.has(r.sessionId)) {
      if (!alive && !retained(r.sessionId)) continue
      role = 'main' // a former main of this fleet: listed, not active
    } else if (!mine && join?.decision === 'child') role = 'child'
    else if (!mine && !join && me && alive && r.tmux && under(r.cwd, c.launchDir) && (r.startedAt ?? 0) > (me.startedAt ?? 0)) {
      const server = await tmuxServerOf($, r.pid)
      if (!myServer || !server || server === myServer) candidates.push({ sid: r.sessionId, name: r.name ?? '', tmux: r.tmux, cwd: r.cwd })
    }
    if (role === 'other') continue
    const p = prevByPid.get(r.pid)
    const sid = mine ? c.sessionId : r.sessionId // the record can lag behind a /clear
    const lineage = mine ? [...c.lineage] : p && p.session_id !== sid && p.proc_start === (r.procStart ?? null) ? [...(p.lineage ?? []), p.session_id] : [...(p?.lineage ?? [])]
    sessions.push({
      session_id: sid, pid: r.pid, proc_start: r.procStart ?? null, alive, crashed: !alive,
      name: r.name ?? null, cwd: r.cwd ?? null, tmux: r.tmux ?? null,
      messaging_socket: typeof r.messagingSocketPath === 'string' ? r.messagingSocketPath : null,
      transcript_path: await transcriptOf($, c.configDir, sid),
      status: statusOf(r.status), waiting_for: r.waitingFor ?? null, ...(r.statusUpdatedAt ? { status_since: nowIso(r.statusUpdatedAt) } : {}),
      role, parent_session_id: role === 'child' ? (join.parent_session_id ?? c.sessionId) : null, lineage,
      ...(r.startedAt ? { started_at: nowIso(r.startedAt) } : {}), ...(r.updatedAt ? { updated_at: nowIso(r.updatedAt) } : {}),
      version: r.version ?? null, kind: r.kind ?? null, purpose: role === 'child' ? (join.purpose ?? null) : null,
      active: mine, fleet_dir: c.fleet, config_dir: c.configDir, launch_dir: mine ? c.launchDir : null,
    })
  }
  // A former main whose record is gone: Claude Code deletes a dead process's record when another session
  // starts, so its last snapshot decides. Alive then means it died without session.end (crashed); a clean
  // end was written by its own trigger.
  const listed = new Set(sessions.map((s) => s.session_id))
  for (const p of prev) {
    if (p.role === 'main' && !listed.has(p.session_id) && p.pid !== pid && retained(p.session_id)) {
      sessions.push({ ...p, alive: false, active: false, crashed: p.alive === true || p.crashed === true })
    }
  }
  return { sessions, candidates, deadSince: dead }
}
