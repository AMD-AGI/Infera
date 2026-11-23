// T — the trigger mod of the agent-sys-o11y plugin (design §7). Active only in a fleet's main session.
// Writes, under the fleet root: registry/sessions.json, triggers/<seq>.json, queue/<id>.json,
// state/trigger.json, state/active/<sid>.json. Reads: config/trigger.json, registry/launches.jsonl and Claude
// Code's <config dir>/sessions/*.json. Imports only the protocol layer.
import {
  LEDGER_TOOL,
  P,
  SCHEMAS,
  configDirOf,
  detachArgv,
  fleetDirOf,
  newId,
  nowIso,
  prefixOf,
  readDoc,
  readEnv,
  readJsonl,
  sanitize,
  slug,
  withDefaults,
  writeDoc,
} from '../../../protocol/ts/index'

const SKILL_RE = /^(infera-agent-sys|agent-sys-[\w-]+)$/
const MARK = { kick: '[o11y automated kick] ', packup: '[o11y automated packup] ', question: '[o11y automated question] ' }
type PromptKind = keyof typeof MARK

interface Ctx {
  fleet: string // the fleet root
  launchDir: string
  configDir: string
  prefix: string
  pluginRoot: string
  pid: number | null // this claude process, from /proc/self/stat
}

interface Pending {
  kind: PromptKind
  submitted_at: number
  about?: string // the session a question is about
}

interface TriggerState {
  v: 1
  session_id: string
  seq: number
  next_kick_at: number
  next_slow_at: number
  kicks: number
  kicks_since_packup: number
  last_packup_at: number
  tool_calls: number
  turns: number
  pending: Pending | null
  retry: { kind: PromptKind; text: string; about?: string } | null // refused by Esc; re-sent after the next completed turn
  asked: string[] // session ids already asked about (spec §3.5)
}

let sessionId = ''
let myLineage: string[] = [] // earlier ids of this process (/clear), from state/active
let vetoed = false
let active = false
let ctx: Ctx | null = null
let cfg: any = withDefaults(SCHEMAS['config.trigger'], { v: 1 })
let cfgMtime = -1
let st: TriggerState | null = null
let pendingIsLive = false // false: the pending entry came from disk (reload), its promise is gone
let ticker: { cancel(): void } | null = null
let tickMs = 0
let sidWatch: { cancel(): void } | null = null
const lastSpawn: Record<string, number> = {}
let lastRegistry = ''
let collectorUp: boolean | null = null
let writeChain: Promise<unknown> = Promise.resolve()

function abs(rel: string): string {
  return `${ctx!.fleet}/${rel}`
}

function writer() {
  return { session_id: sessionId, role: 'main', source: 'detected' }
}

// Writes of state and records go one at a time.
function serial(fn: () => Promise<unknown>): Promise<unknown> {
  writeChain = writeChain.then(fn).catch(() => {})
  return writeChain
}

function skillName(raw: unknown): string | null {
  if (typeof raw !== 'string') return null
  const bare = raw.replace(/^\//, '').split(/\s/)[0].split(':').pop() ?? ''
  return SKILL_RE.test(bare) ? bare : null
}

// ---- config and state ----

async function loadConfig($: any) {
  const r = await readDoc($, abs(P.config('trigger')), 'config.trigger')
  if (r.mtimeMs === cfgMtime) return
  cfgMtime = r.mtimeMs
  if (r.problem) $.ui.log(`o11y: trigger config ignored: ${r.problem}`, { to: 'debug' })
  const before = cfg.kick.interval_s
  cfg = withDefaults(SCHEMAS['config.trigger'], r.problem ? { v: 1 } : (r.value ?? { v: 1 }))
  // A new kick interval applies now, not after the kick already scheduled with the old one.
  if (st && cfg.kick.interval_s !== before) st.next_kick_at = Math.min(st.next_kick_at, Date.now() + cfg.kick.interval_s * 1000)
}

async function loadState(now: number, $: any) {
  const s = (await readDoc($, abs(P.triggerState))).value
  if (s && s.v === 1 && s.session_id === sessionId) {
    st = { pending: null, retry: null, asked: [], ...s }
    pendingIsLive = false
    return
  }
  st = {
    v: 1,
    session_id: sessionId,
    seq: s?.seq ?? 0, // trigger records never reuse a number
    next_kick_at: now + cfg.kick.interval_s * 1000,
    next_slow_at: now + cfg.log.slow_interval_s * 1000,
    kicks: 0,
    kicks_since_packup: 0,
    last_packup_at: now,
    tool_calls: 0,
    turns: 0,
    pending: null,
    retry: null,
    asked: s?.asked ?? [],
  }
}

function saveState($: any) {
  const snapshot = JSON.parse(JSON.stringify(st))
  return serial(() => writeDoc($, abs(P.triggerState), snapshot))
}

function fire($: any, kind: string, outcome: string, extra: { detail?: string; text?: string; subject?: string } = {}) {
  const seq = ++st!.seq
  const rec = {
    v: 1,
    id: newId('trig'),
    ts: nowIso(),
    seq,
    kind,
    outcome,
    session_id: sessionId,
    ...(extra.detail ? { detail: extra.detail.slice(0, 500) } : {}),
    ...(extra.text ? { text: extra.text } : {}),
    ...(extra.subject ? { subject_session_id: extra.subject } : {}),
    counters: { tool_calls: st!.tool_calls, turns: st!.turns, kicks: st!.kicks },
    writer: writer(),
  }
  return serial(() => writeDoc($, abs(`${P.triggers}/${String(seq).padStart(8, '0')}.json`), rec))
}

// ---- registry (tick step 1) ----

function procStartOf(stat: string): string | null {
  // field 22 of /proc/<pid>/stat; field 2 may contain spaces, so count after the last ')'
  return stat.slice(stat.lastIndexOf(')') + 2).split(' ')[19] ?? null
}

async function aliveOf($: any, pid: number, procStart: string | null): Promise<boolean> {
  try {
    const s = await $.fs.read(`/proc/${pid}/stat`)
    return procStart == null || procStartOf(s) === String(procStart)
  } catch {
    return false
  }
}

// The tmux server a process talks to: the socket path in its TMUX variable, or null.
async function tmuxServerOf($: any, pid: number): Promise<string | null> {
  try {
    const env: string = await $.fs.read(`/proc/${pid}/environ`)
    const kv = env.split('\0').find((x) => x.startsWith('TMUX='))
    return kv ? kv.slice(5).split(',')[0] : null
  } catch {
    return null
  }
}

function statusOf(s: unknown): string {
  return ['busy', 'idle', 'waiting', 'shell'].includes(s as string) ? (s as string) : 'unknown'
}

function tmuxMatches(recTmux: string | null, target: string | undefined): boolean {
  if (!recTmux || !target) return false
  // a registry tmux value is "<session>:@<window>.%<pane>"; a launch target is a session name, "%<pane>" or the full value
  const [sess, rest] = recTmux.split(':')
  return target === sess || target === recTmux || (rest ?? '').endsWith('.' + target)
}

interface Candidate {
  sid: string
  name: string
  tmux: string
  cwd: string
}

async function refreshRegistry($: any): Promise<{ sessions: any[]; candidates: Candidate[] }> {
  const dir = `${ctx!.configDir}/sessions`
  let names: string[] = []
  try {
    names = (await $.fs.list(dir)).map((x: any) => x.name).filter((n: string) => /^\d+\.json$/.test(n))
  } catch {}
  const launches = (await readJsonl($, abs(P.launches))).filter((l) => l && l.v === 1)
  const prev = (await readDoc($, abs(P.sessions))).value
  const prevByPid = new Map<number, any>((prev?.sessions ?? []).map((s: any) => [s.pid, s]))
  const records: any[] = []
  for (const n of names) {
    const pid = Number(n.slice(0, -5))
    try {
      const r = JSON.parse(await $.fs.read(`${dir}/${n}`))
      if (r && typeof r.pid === 'number' && typeof r.sessionId === 'string') records.push(r)
      else throw new Error('shape')
    } catch {
      // torn or empty: keep the last good entry, marked (spec §3.4)
      const p = prevByPid.get(pid)
      if (p) records.push({ __stale: p })
    }
  }
  // this process is the main whatever id its record shows (the record can lag behind a /clear)
  const isMe = (r: any) => r.sessionId === sessionId || (ctx!.pid != null && r.pid === ctx!.pid)
  const me = records.find(isMe)
  const myServer = ctx!.pid ? await tmuxServerOf($, ctx!.pid) : null
  const out: any[] = []
  const candidates: Candidate[] = []
  for (const r of records) {
    if (r.__stale) {
      out.push({ ...r.__stale, last_good: true })
      continue
    }
    let role = 'other'
    let parent: string | null = null
    let purpose: string | null = null
    if (isMe(r)) role = 'main'
    else {
      const l = launches.findLast(
        (x) => x.session_id === r.sessionId || (x.name && x.name === r.name) || tmuxMatches(r.tmux ?? null, x.tmux_target),
      )
      if (l && l.role === 'child') {
        role = 'child'
        parent = l.parent_session_id ?? sessionId
        purpose = l.purpose ?? null
      } else if (!l && me && r.tmux && typeof r.cwd === 'string' && (r.cwd === ctx!.launchDir || r.cwd.startsWith(ctx!.launchDir + '/')) && (r.startedAt ?? 0) > (me.startedAt ?? 0)) {
        // a live session outside the convention: same tmux server, under the fleet dir, started after the main
        const server = await tmuxServerOf($, r.pid)
        if (!myServer || !server || server === myServer) {
          if (await aliveOf($, r.pid, r.procStart ?? null)) candidates.push({ sid: r.sessionId, name: r.name ?? '', tmux: r.tmux, cwd: r.cwd })
        }
      }
    }
    if (role === 'other') continue // sessions outside the fleet are not listed
    const alive = await aliveOf($, r.pid, r.procStart ?? null)
    const p = prevByPid.get(r.pid)
    const lineage: string[] = role === 'main' ? [...myLineage] : p ? [...(p.lineage ?? [])] : []
    if (p && p.session_id !== r.sessionId && p.proc_start === (r.procStart ?? null) && !lineage.includes(p.session_id)) {
      lineage.push(p.session_id) // same process, new id: /clear
    }
    const tp = r.cwd ? `${ctx!.configDir}/projects/${await slug(String(r.cwd))}/${r.sessionId}.jsonl` : null
    out.push({
      session_id: role === 'main' ? sessionId : r.sessionId,
      pid: r.pid,
      proc_start: r.procStart ?? null,
      alive,
      crashed: !alive,
      name: r.name ?? null,
      cwd: r.cwd ?? null,
      tmux: r.tmux ?? null,
      transcript_path: tp && (await $.fs.exists(tp)) ? tp : null,
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
      active: role === 'main',
      fleet_dir: ctx!.fleet,
      config_dir: ctx!.configDir,
      launch_dir: role === 'main' ? ctx!.launchDir : null,
    })
  }
  const body = JSON.stringify(out)
  if (body !== lastRegistry) {
    lastRegistry = body
    const doc = { v: 1, generated_at: nowIso(), main_session_id: sessionId, sessions: out }
    await serial(() => writeDoc($, abs(P.sessions), doc))
  }
  return { sessions: out, candidates }
}

// ---- automated prompts (tick step 2, spec §4.1–4.3, §3.5) ----

function render(template: string, vars: Record<string, string>): string {
  return template.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? vars[k] : m))
}

function childrenText(sessions: any[]): string {
  const max = cfg.kick.max_value_chars
  const kids = sessions.filter((s) => s.role === 'child' && s.alive)
  if (!kids.length) return 'none'
  return kids.map((s) => `${sanitize(s.name ?? s.session_id, max)}${s.tmux ? ` (tmux ${sanitize(s.tmux.split(':')[0], max)})` : ''}`).join(', ')
}

function isPending(now: number): boolean {
  const p = st!.pending
  if (!p) return false
  if (pendingIsLive) return true
  // From disk after a reload, its promise gone: it blocks until the next turn starts (turn.start clears it).
  // One interval is not enough: the next kick falls due exactly one interval after this one was queued. The
  // cap of three intervals only guards against a lost turn.start.
  return now - p.submitted_at < 3 * cfg.kick.interval_s * 1000
}

function submit($: any, kind: PromptKind, text: string, about?: string) {
  st!.pending = { kind, submitted_at: Date.now(), ...(about ? { about } : {}) }
  pendingIsLive = true
  // Not awaited: it resolves only when the new turn starts (mods api.md).
  $.prompt.submit({ text }).then(
    (r: any) => {
      st!.pending = null
      if (r && r.drop) {
        st!.retry = { kind, text, ...(about ? { about } : {}) }
        void fire($, kind, 'refused', { detail: String(r.drop) })
      }
      void saveState($)
    },
    (err: any) => {
      st!.pending = null
      void fire($, kind, 'failed', { detail: String(err?.message ?? err) })
      void saveState($)
    },
  )
}

async function scheduleKicks($: any, now: number, sessions: any[]) {
  if (!cfg.enabled || !cfg.kick.enabled || now < st!.next_kick_at) return
  // absolute schedule: a missed interval is not replayed
  const step = cfg.kick.interval_s * 1000
  while (st!.next_kick_at <= now) st!.next_kick_at += step
  const usePackup =
    cfg.packup.enabled &&
    (st!.kicks_since_packup + 1 >= cfg.packup.every_n_kicks || now - st!.last_packup_at >= cfg.packup.interval_s * 1000)
  const kind: PromptKind = usePackup ? 'packup' : 'kick'
  if (isPending(now)) {
    await fire($, kind, 'skipped_pending', { detail: `${st!.pending!.kind} pending since ${nowIso(st!.pending!.submitted_at)}` })
    return
  }
  const max = cfg.kick.max_value_chars
  const book = cfg.task_book.startsWith('/') ? cfg.task_book : `${ctx!.launchDir}/${cfg.task_book}`
  const vars = { task_book: sanitize(book, max), children: childrenText(sessions), ledger_tool: LEDGER_TOOL }
  const text = MARK[kind] + render(usePackup ? cfg.packup.template : cfg.kick.template, vars)
  st!.kicks++
  if (usePackup) {
    st!.kicks_since_packup = 0
    st!.last_packup_at = now
  } else st!.kicks_since_packup++
  submit($, kind, text)
  await fire($, kind, 'submitted', { text })
}

async function askAboutUnregistered($: any, now: number, candidates: Candidate[]) {
  if (!cfg.question.enabled || !candidates.length || isPending(now)) return
  const c = candidates.find((x) => !st!.asked.includes(x.sid))
  if (!c) return
  const max = cfg.kick.max_value_chars
  const vars = { name: sanitize(c.name || c.sid, max), tmux: sanitize(c.tmux, max), cwd: sanitize(c.cwd, max) }
  const text = MARK.question + render(cfg.question.template, vars)
  st!.asked.push(c.sid)
  submit($, 'question', text, c.sid)
  await fire($, 'question', 'submitted', { text, subject: c.sid })
}

// ---- analysis requests ----

async function enqueue($: any, kind: 'display' | 'log', reason: string) {
  if (!ctx || !st) return
  const id = newId(kind)
  const item = { v: 1, id, ts: nowIso(), kind, reason, session_id: sessionId, counters: { tool_calls: st.tool_calls, turns: st.turns }, writer: writer() }
  await serial(() => writeDoc($, abs(`${P.queue}/${id}.json`), item))
  const fk = kind === 'display' ? 'refresh' : 'log'
  await fire($, fk, 'enqueued', { detail: reason })
  // at most one spawn per kind every 5 s; the analyzer drains everything queued and re-scans before it exits
  const now = Date.now()
  if (now - (lastSpawn[kind] ?? 0) < 5000) return
  lastSpawn[kind] = now
  const argv = [`${ctx.pluginRoot}/server/py.sh`, '-m', 'o11y_analyze', '--kind', kind]
  try {
    const r = await $.process.run(detachArgv(abs('state/analyze.spawn.log'), argv), {
      env: { AGENT_SYS_O11Y_FLEET_DIR: ctx.fleet, CLAUDE_CONFIG_DIR: ctx.configDir, AGENT_SYS_HOME: ctx.prefix },
      timeoutMs: 10000,
    })
    if (r.exitCode !== 0) await fire($, fk, 'failed', { detail: r.stderr.slice(0, 300) })
  } catch (err: any) {
    await fire($, fk, 'failed', { detail: err.message })
  }
}

// Work started inside an event handler is aborted with its event (Esc), so requests run on the clock.
function later($: any, kind: 'display' | 'log', reason: string) {
  $.clock.after(1, () => enqueue($, kind, reason))
}

async function detach($: any, argv: string[]) {
  try {
    await $.process.run(detachArgv(abs('state/spawn.log'), argv), {
      env: { AGENT_SYS_O11Y_FLEET_DIR: ctx!.fleet, CLAUDE_CONFIG_DIR: ctx!.configDir, AGENT_SYS_HOME: ctx!.prefix },
      timeoutMs: 10000,
    })
  } catch {}
}

// ---- optional OTel collector (design §10): health per tick, one error per outage ----

async function checkCollector($: any) {
  const url = await $.env.get('AGENT_SYS_O11Y_OTEL_HEALTH_URL')
  if (!url) return
  let up = false
  try {
    up = (await $.http.fetch(url)).ok === true
  } catch {}
  if (up === false && collectorUp !== false) {
    await detach($, [
      `${ctx!.pluginRoot}/server/py.sh`, '-m', 'o11y_core', '--session-id', sessionId, '--role', 'main', 'event', 'report',
      '--severity', 'ERROR', '--name', 'o11y.otel.collector_down', '--body', `collector at ${sanitize(url, 200)} is not healthy`, '--source', 'detector',
    ])
  }
  collectorUp = up
}

// ---- tick ----

async function tick($: any) {
  if (!ctx || !st) return
  try {
    await watchSessionId($) // follow a /clear before writing the registry
    const now = Date.now()
    await loadConfig($)
    const { sessions, candidates } = await refreshRegistry($)
    await scheduleKicks($, now, sessions)
    await askAboutUnregistered($, now, candidates)
    if (cfg.enabled && now >= st.next_slow_at) {
      while (st.next_slow_at <= now) st.next_slow_at += cfg.log.slow_interval_s * 1000
      await enqueue($, 'log', 'timer')
    }
    await checkCollector($)
    await saveState($)
    if (cfg.registry_interval_s * 1000 !== tickMs) startTicker($)
  } catch (err: any) {
    $.ui.log(`o11y: trigger tick: ${err.message}`, { to: 'debug' })
  }
}

function startTicker($: any) {
  if (ticker) ticker.cancel()
  tickMs = cfg.registry_interval_s * 1000
  ticker = $.clock.every(tickMs, () => tick($))
}

// ---- activation (design §7) ----

async function selfPid($: any): Promise<number | null> {
  try {
    return Number((await $.fs.read('/proc/self/stat')).split(' ')[0]) || null
  } catch {
    return null
  }
}

async function makeCtx($: any): Promise<Ctx> {
  const env = await readEnv($)
  const launchDir = String(await $.session.root())
  return {
    fleet: await fleetDirOf(env, launchDir),
    launchDir,
    configDir: configDirOf(env),
    prefix: prefixOf(env),
    pluginRoot: $.plugin.root,
    pid: await selfPid($),
  }
}

async function activate($: any, by: string, lineage: string[] = []) {
  if (active || vetoed) return
  active = true
  myLineage = lineage
  ctx ??= await makeCtx($)
  await writeDoc($, abs(P.active(sessionId)), { v: 1, session_id: sessionId, by, at: nowIso(), lineage })
  await loadConfig($)
  await loadState(Date.now(), $)
  startTicker($)
  await tick($)
  $.ui.log(`o11y active (${by}); fleet ${ctx.fleet}`)
}

// /clear and /resume give a new session id in the same process, without a new session.start.
async function watchSessionId($: any) {
  const sid = await $.session.id()
  if (!sid || sid === sessionId) return
  const old = sessionId
  sessionId = sid
  if (!active || !ctx) return
  const prev = (await readDoc($, abs(P.active(old)))).value
  myLineage = [...(prev?.lineage ?? myLineage), old]
  await writeDoc($, abs(P.active(sid)), { v: 1, session_id: sid, by: `cleared from ${old}`, at: nowIso(), lineage: myLineage })
  st = { ...st!, session_id: sid }
  await tick($)
}

export function register(on: any) {
  on('session.start', async ($: any, e: any, next: any) => {
    sessionId = await $.session.id()
    const env = await $.env.get('AGENT_SYS_O11Y')
    vetoed = env !== undefined && env !== 'true'
    if (sidWatch) sidWatch.cancel()
    sidWatch = $.clock.every(5000, () => watchSessionId($))
    if (!vetoed) {
      ctx = await makeCtx($)
      const armed = (await readDoc($, abs(P.active(sessionId)))).value
      if (env === 'true') await activate($, 'env')
      else if (armed) await activate($, `re-armed (${armed.by})`, armed.lineage ?? [])
    }
    return next(e)
  })

  on('tool.call', async ($: any, e: any, next: any) => {
    const r = await next(e) // count calls that ran past every hook
    if (e.agentId || vetoed) return r
    if (!active) {
      if (e.tool === 'Skill') {
        const s = skillName(e.skill)
        if (s) $.clock.after(1, () => activate($, `skill ${s}`))
      }
      return r
    }
    st!.tool_calls++
    if (st!.tool_calls % cfg.refresh.every_n_tool_calls === 0) later($, 'display', 'tool_calls')
    if (st!.tool_calls % cfg.log.every_n_tool_calls === 0) later($, 'log', 'tool_calls')
    return r
  })

  on('command.run', async ($: any, e: any, next: any) => {
    if (!active && !vetoed && e.origin?.kind === 'composer') {
      const s = skillName(e.command)
      if (s) $.clock.after(1, () => activate($, `skill /${s}`))
    }
    return next(e)
  })

  on('turn.start', async ($: any, e: any, next: any) => {
    // a pending entry restored from disk has no promise left; the next turn start ends it
    if (active && st?.pending && !pendingIsLive && !e.agentId) st.pending = null
    return next(e)
  })

  on('turn.complete', async ($: any, e: any, next: any) => {
    if (!active || !st || e.agentId) return next(e)
    st.turns++
    // a refused prompt waits for the user's next completed turn; an aborted turn is not that
    if (st.retry && !e.isAborted) {
      const r = st.retry
      st.retry = null
      $.clock.after(1, async () => {
        submit($, r.kind, r.text, r.about)
        await fire($, r.kind, 'submitted', { text: r.text, detail: 'retry after refusal', ...(r.about ? { subject: r.about } : {}) })
      })
    }
    if (st.turns % cfg.refresh.every_n_turns === 0) later($, 'display', 'turns')
    if (st.turns % cfg.log.every_n_turns === 0) later($, 'log', 'turns')
    return next(e)
  })

  on('session.end', async ($: any, e: any, next: any) => {
    if (active && st) {
      void saveState($)
      if (e.reason !== 'clear' && e.reason !== 'resume') {
        if (ticker) ticker.cancel()
        ticker = null
        // the collector sweep outlives this hook's 1.5 s budget, so it is detached
        const launcher = `${ctx!.pluginRoot}/bin/o11y-launch`
        if (await $.fs.exists(launcher)) void detach($, [launcher, '--sweep'])
      }
    }
    return next(e)
  })
}
