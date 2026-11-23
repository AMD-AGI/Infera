// T — the trigger mod of plugin agent-sys-o11y (design §7): activation, the root pin, the tick, and the
// wiring of the registry normaliser, the schedule, the counters, the spawner and the collector health.
// Active only in a fleet's main session. Imports only the protocol layer and its own modules.
import { P, configDirOf, derivedFleetDir, loadConfig, newId, nowIso, pinPath, prefixOf, readDoc, readEnv, writeDoc, type ConfigState, type Env } from '../../../protocol/ts/index.ts'
import { normalise, type RegCtx } from './registry.ts'
import { dueKick, fresh, isPending, kickText, pickCandidate, questionText, type Pending, type PromptKind, type Schedule } from './schedule.ts'
import { count, type Counts, type JobKind } from './counters.ts'
import { request, type SpawnCtx } from './spawn.ts'
import { sample, sweep } from './otel_health.ts'

const SKILL_RE = /^(infera-agent-sys|agent-sys-[\w-]+)$/
const REGISTRY_TICK_MS = 30_000

// state/trigger.json: seq numbers the trigger_fire records; asked lists the sessions already asked about.
interface State { v: 1; session_id: string; seq: number; schedule: Schedule; next_slow_at: number; counts: Counts; pending: Pending | null
  retry: { kind: PromptKind; text: string; subject?: string } | null; asked: string[]; dead_since: Record<string, number> }

let sessionId = ''
let vetoed = false
let active = false
let env: Env = {}
let reg: RegCtx | null = null
let spawnCtx: SpawnCtx | null = null
let cfgState: ConfigState | null = null
let st: State | null = null
let pendingLive = false // false: the pending entry was restored from disk and its promise is gone
let ticker: { cancel(): void } | null = null
let lastRegistry = ''
let chain: Promise<unknown> = Promise.resolve()

const cfg = () => cfgState!.cfg
const abs = (rel: string) => `${reg!.fleet}/${rel}`
const serial = (fn: () => Promise<unknown>) => (chain = chain.then(fn).catch(() => {})) // writes go one at a time

function skillName(raw: unknown): string | null {
  const bare = typeof raw === 'string' ? (raw.replace(/^\//, '').split(/\s/)[0].split(':').pop() ?? '') : ''
  return SKILL_RE.test(bare) ? bare : null
}

function saveState($: any) {
  const snap = JSON.parse(JSON.stringify(st))
  return serial(() => writeDoc($, abs(P.triggerState), snap))
}

function fire($: any, kind: PromptKind, outcome: string, x: { detail?: string; text?: string; subject?: string } = {}) {
  const seq = ++st!.seq
  const rec = {
    v: 1, id: newId('trig'), ts: nowIso(), seq, kind, outcome, session_id: sessionId,
    ...(x.detail ? { detail: x.detail.slice(0, 500) } : {}), ...(x.text ? { text: x.text } : {}), ...(x.subject ? { subject_session_id: x.subject } : {}),
    counters: { tool_calls: st!.counts.tool_calls, turns: st!.counts.turns, kicks: st!.schedule.kicks },
    writer: { session_id: sessionId, role: 'main', source: 'detected' },
  }
  return serial(() => writeDoc($, abs(`${P.triggers}/${String(seq).padStart(8, '0')}.json`), rec))
}

async function reloadConfig($: any) {
  const before = cfgState?.cfg.kick.interval_s
  cfgState = await loadConfig($, abs(P.config('trigger')), 'config.trigger', cfgState)
  if (cfgState.report) $.ui.log(`o11y: config/trigger.json not applied, the previous config stays: ${cfgState.report}`)
  // a new kick interval applies now, not after the kick scheduled with the old one
  if (st && before !== undefined && cfg().kick.interval_s !== before) st.schedule.next_kick_at = Math.min(st.schedule.next_kick_at, Date.now() + cfg().kick.interval_s * 1000)
}

async function submit($: any, kind: PromptKind, text: string, subject?: string, detail?: string) {
  st!.pending = { kind, submitted_at: Date.now(), ...(subject ? { subject } : {}) }
  pendingLive = true
  // not awaited: it resolves when the new turn starts
  $.prompt.submit({ text }).then(
    (r: any) => r?.drop && ((st!.retry = { kind, text, subject }), fire($, kind, 'refused', { detail: String(r.drop), subject })),
    (err: any) => fire($, kind, 'failed', { detail: String(err?.message ?? err), subject }),
  ).finally(() => ((st!.pending = null), saveState($)))
  await fire($, kind, 'submitted', { text, subject, detail })
}

async function prompts($: any, now: number, sessions: any[], candidates: any[]) {
  const due = dueKick(st!.schedule, cfg(), now)
  if (due) {
    st!.schedule = due.next
    if (isPending(st!.pending, pendingLive, now, cfg())) await fire($, due.kind, 'skipped_pending', { detail: `${st!.pending!.kind} pending since ${nowIso(st!.pending!.submitted_at)}` })
    else await submit($, due.kind, kickText(due.kind as 'kick' | 'packup', cfg(), cfg().task_book.startsWith('/') ? cfg().task_book : `${reg!.launchDir}/${cfg().task_book}`, sessions))
  }
  const c = pickCandidate(candidates, st!.asked)
  if (c && !isPending(st!.pending, pendingLive, now, cfg())) {
    st!.asked.push(c.sid)
    await submit($, 'question', questionText(c), c.sid)
  }
}

// Requests run on the clock: work started inside an event handler is aborted by Esc.
function requestLater($: any, kinds: JobKind[], reason: string) {
  for (const k of kinds) $.clock.after(1, () => request($, spawnCtx!, k, reason).then((p) => p && $.ui.log(`o11y: ${p}`, { to: 'debug' })))
}

async function tick($: any) {
  if (!active || !st) return
  try {
    await followClear($)
    const now = Date.now()
    await reloadConfig($)
    const { sessions, candidates, deadSince } = await normalise($, reg!, st.dead_since, now)
    st.dead_since = deadSince
    await writeRegistry($, sessions)
    await prompts($, now, sessions, candidates)
    if (now >= st.next_slow_at) {
      while (st.next_slow_at <= now) st.next_slow_at += cfg().log.slow_interval_s * 1000
      requestLater($, ['log'], 'timer')
    }
    const url = await $.env.get('AGENT_SYS_O11Y_OTEL_HEALTH_URL')
    if (url) await sample($, spawnCtx!, url).then(() => sweep($, spawnCtx!))
    await saveState($)
  } catch (err: any) {
    $.ui.log(`o11y: trigger tick: ${err.message}`, { to: 'debug' })
  }
}

async function writeRegistry($: any, sessions: any[]) {
  const body = JSON.stringify(sessions)
  if (body === lastRegistry) return
  lastRegistry = body
  await serial(() => writeDoc($, abs(P.sessions), { v: 1, generated_at: nowIso(), main_session_id: sessionId, sessions }))
}

function writePin($: any) {
  return writeDoc($, pinPath(env, sessionId), { v: 1, session_id: sessionId, fleet_dir: reg!.fleet, launch_dir: reg!.launchDir, pinned_at: nowIso(), lineage: reg!.lineage })
}

async function activate($: any, by: string, pin: any | null) {
  if (active || vetoed) return
  active = true
  const launchDir = pin?.launch_dir ?? String(await $.session.root())
  const fleet = pin?.fleet_dir ?? (await derivedFleetDir(env, launchDir))
  const configDir = configDirOf(env)
  reg = { fleet, launchDir, configDir, sessionId, lineage: pin?.lineage ?? [] }
  spawnCtx = { fleet, pluginRoot: $.plugin.root, env: { AGENT_SYS_O11Y_FLEET_DIR: fleet, CLAUDE_CONFIG_DIR: configDir, AGENT_SYS_HOME: prefixOf(env) } }
  await writePin($)
  await reloadConfig($)
  const s = (await readDoc($, abs(P.triggerState))).value
  const now = Date.now()
  st = s?.v === 1 && s.session_id === sessionId
    ? { pending: null, retry: null, asked: [], dead_since: {}, ...s }
    : { v: 1, session_id: sessionId, seq: s?.seq ?? 0, schedule: fresh(now, cfg()), next_slow_at: now + cfg().log.slow_interval_s * 1000, counts: { tool_calls: 0, turns: 0 }, pending: null, retry: null, asked: s?.asked ?? [], dead_since: s?.dead_since ?? {} }
  pendingLive = false
  ticker?.cancel()
  ticker = $.clock.every(REGISTRY_TICK_MS, () => tick($))
  await tick($) // the registry is written at once: the door and the display key on it
  $.ui.log(`o11y active (${by}); fleet ${fleet}`)
}

// /clear gives a new session id in the same process without a new session.start: carry the pin over.
async function followClear($: any) {
  const sid = await $.session.id()
  if (!sid || sid === sessionId) return
  const old = sessionId
  sessionId = sid
  if (!active || !reg) return
  reg = { ...reg, sessionId: sid, lineage: [...reg.lineage, old] }
  st!.session_id = sid
  await writePin($)
}

function onSkill($: any, name: unknown, by: string) {
  const s = skillName(name)
  if (s && !active && !vetoed) $.clock.after(1, () => activate($, `${by}${s}`, null))
}

export function register(on: any) {
  on('session.start', async ($: any, e: any, next: any) => {
    sessionId = await $.session.id()
    const flag = await $.env.get('AGENT_SYS_O11Y')
    vetoed = flag !== undefined && flag !== 'true'
    if (!vetoed) {
      env = await readEnv($)
      const pin = (await readDoc($, pinPath(env, sessionId), 'session_pin')).value // re-arm after a reload
      if (flag === 'true' || pin) await activate($, flag === 'true' ? 'env' : 're-armed', pin)
    }
    return next(e)
  })

  on('tool.call', async ($: any, e: any, next: any) => {
    const r = await next(e) // count calls that ran past every hook
    if (e.agentId || vetoed) return r
    if (!active) onSkill($, e.tool === 'Skill' ? e.skill : null, 'skill ')
    else {
      const c = count(st!.counts, 'tool', cfg())
      st!.counts = c.counts
      requestLater($, c.kinds, c.reason)
    }
    return r
  })

  on('command.run', async ($: any, e: any, next: any) => {
    if (e.origin?.kind === 'composer') onSkill($, e.command, 'skill /')
    return next(e)
  })

  on('turn.start', async ($: any, e: any, next: any) => {
    if (active && st?.pending && !pendingLive && !e.agentId) st.pending = null
    return next(e)
  })

  on('turn.complete', async ($: any, e: any, next: any) => {
    if (!active || !st || e.agentId) return next(e)
    const c = count(st.counts, 'turn', cfg())
    st.counts = c.counts
    requestLater($, c.kinds, c.reason)
    // a refused prompt is retried once, after a turn the user did not abort
    const r = st.retry
    if (r && !e.isAborted) {
      st.retry = null
      $.clock.after(1, () => submit($, r.kind, r.text, r.subject, 'retry after refusal'))
    }
    return next(e)
  })

  on('session.end', async ($: any, e: any, next: any) => {
    if (active && st && e.reason !== 'clear' && e.reason !== 'resume') {
      ticker?.cancel()
      ticker = null
      // a clean end, written by this session itself: the next main tells it from a crash (registry.ts)
      const ended = JSON.parse(lastRegistry || '[]').map((s: any) => (s.session_id === sessionId ? { ...s, alive: false, active: false, crashed: false } : s))
      lastRegistry = ''
      await writeRegistry($, ended)
      if (await $.env.get('AGENT_SYS_O11Y_OTEL_HEALTH_URL')) void sweep($, spawnCtx!)
    }
    if (active && st) void saveState($)
    return next(e)
  })
}
