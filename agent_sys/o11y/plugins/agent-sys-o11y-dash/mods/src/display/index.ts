// D — the display mod of the agent-sys-o11y-dash plugin (design §9). It resolves the fleet root as the
// trigger does, is active only while registry/sessions.json lists this session as an alive, active main, and
// writes nothing but config/*.json (forms). Imports only the protocol layer.
import { P, SCHEMAS, fleetDirOf, nowIso, readDoc, readEnv, readJsonl, validate, withDefaults, writeDoc } from '../../../protocol/ts/index'

const PANE = 'o11y'
const MENU = 'o11y-menu'
const FORM = 'o11y-cfg'
const TABS = ['general', 'detail', 'kernelforge', 'log'] as const
type Tab = (typeof TABS)[number]
const FACTS = ['general', 'detail', 'kernelforge', 'usage'] as const
const SIZE = { narrow: { columns: 46, rows: 16 }, wide: { columns: 96, rows: 40 } }

const COMMANDS = [
  { name: 'o11y', description: 'o11y: menu (g/d/k/l tabs, m mode, b band, t/a/v config)' },
  { name: 'o11y-general', description: 'o11y: general tab' },
  { name: 'o11y-detail', description: 'o11y: detail tab (runs, plan, machines)' },
  { name: 'o11y-kf', description: 'o11y: KernelForge tab' },
  { name: 'o11y-log', description: 'o11y: log tab' },
  { name: 'o11y-mode', description: 'o11y: switch NARROW / WIDE', argumentHint: '[narrow|wide]' },
  { name: 'o11y-band', description: 'o11y: band above the prompt on/off', argumentHint: '[on|off]' },
  { name: 'o11y-cfg-trigger', description: 'o11y: edit trigger config (kick, packup, frequencies)' },
  { name: 'o11y-cfg-analysis', description: 'o11y: edit analysis config (model, effort, lookback)' },
  { name: 'o11y-cfg-display', description: 'o11y: edit display config' },
]

let root: string | null = null // the fleet root; null when vetoed
let vetoReason = ''
let sessionId = ''
let active = false
let registryKey = ''
let cfg: any = withDefaults(SCHEMAS['config.display'], { v: 1 })
let cfgMtime = -1
let tab: Tab = 'general'
let mode: 'narrow' | 'wide' = 'narrow'
let bandOn = true
let isOpen = false
const docs: Record<string, { value: any; problem: string | null; mtimeMs: number }> = {}
let sessions: any[] = []
let sessionsMtime = -1
let events: any[] = []
let eventsSize = -1
let lastFire: any = null
let lastFireName = ''
let polls = 0
let poller: { cancel(): void } | null = null
let pollMs = 0
let formTarget: 'trigger' | 'analysis' | 'display' = 'trigger'
let formDraft: any = {}
let formNote = ''

function abs(rel: string): string {
  return `${root}/${rel}`
}

// ---- data polling ----

// Active iff the registry lists this session as an alive, active main (design §9).
async function checkActive($: any) {
  sessionId = (await $.session.id()) || sessionId
  const sp = abs(P.sessions)
  const m = (await $.fs.exists(sp)) ? (await $.fs.stat(sp)).mtimeMs : 0
  const key = `${m}:${sessionId}`
  if (key === registryKey) return
  registryKey = key
  const reg = (await readDoc($, sp, 'registry.sessions')).value
  sessions = reg?.sessions ?? []
  sessionsMtime = m
  const mine = sessions.some((x) => x.session_id === sessionId && x.role === 'main' && x.alive && x.active === true)
  if (mine && !active) {
    active = true
    await poll($)
    mode = cfg.mode
    tab = cfg.tab
    bandOn = cfg.band
    await openPane($, false) // opened unasked: placed only in a wide enough terminal; the band always shows
    $.ui.invalidate('ui.render')
  } else if (!mine && active) {
    active = false
    await $.ui.close({ id: PANE })
    isOpen = false
    $.ui.invalidate('ui.render')
  }
}

async function poll($: any) {
  if (!root) return
  let changed = false
  try {
    await checkActive($)
    if (!active) return
    const c = await readDoc($, abs(P.config('display')), 'config.display')
    if (c.mtimeMs !== cfgMtime) {
      cfgMtime = c.mtimeMs
      cfg = withDefaults(SCHEMAS['config.display'], c.problem ? { v: 1 } : (c.value ?? { v: 1 }))
      if (c.problem) docs['config'] = { value: null, problem: c.problem, mtimeMs: c.mtimeMs }
      else delete docs['config']
      changed = true
    }
    for (const f of FACTS) {
      const path = abs(P.facts(f))
      const exists = await $.fs.exists(path)
      const m = exists ? (await $.fs.stat(path)).mtimeMs : 0
      if (docs[f]?.mtimeMs === m && docs[f] !== undefined) continue
      const r = await readDoc($, path, `facts.${f}`)
      // An invalid document counts as absent (PROTOCOL.md §3); show its problem with a root-relative path.
      docs[f] = r.problem ? { ...r, value: null, problem: 'invalid ' + r.problem.replace(root + '/', '') } : r
      changed = true
    }
    const ep = abs(P.events)
    const es = (await $.fs.exists(ep)) ? (await $.fs.stat(ep)).size : 0
    if (es !== eventsSize) {
      eventsSize = es
      events = (await readJsonl($, ep, 256 * 1024)).filter((x) => x && x.v === 1 && x.severity_text)
      changed = true
    }
    // The newest trigger record (listed every 5th poll: the directory only grows)
    if (polls++ % 5 === 0 && (await $.fs.exists(abs(P.triggers)))) {
      const names = (await $.fs.list(abs(P.triggers))).map((x: any) => x.name).filter((n: string) => /^\d+\.json$/.test(n))
      const newest = names.sort().pop() ?? ''
      if (newest && newest !== lastFireName) {
        lastFireName = newest
        lastFire = (await readDoc($, abs(`${P.triggers}/${newest}`), 'trigger_fire')).value
        changed = true
      }
    }
  } catch (err: any) {
    $.ui.log(`o11y display poll: ${err.message}`, { to: 'debug' })
  }
  if (cfg.poll_ms !== pollMs) startPoller($)
  if (changed) $.ui.invalidate('ui.render')
}

function startPoller($: any) {
  if (poller) poller.cancel()
  pollMs = cfg.poll_ms
  poller = $.clock.every(pollMs, () => poll($))
}

// ---- formatting ----

function num(x: any, digits = 0): string {
  if (typeof x !== 'number' || !isFinite(x)) return '-'
  return x.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits })
}

function pct(x: any): string {
  return typeof x === 'number' && isFinite(x) ? `${x >= 0 ? '+' : ''}${x.toFixed(1)}%` : '-'
}

function tok(n: any): string {
  if (typeof n !== 'number') return '-'
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}k` : String(n)
}

function dur(s: any): string {
  if (typeof s !== 'number' || !isFinite(s)) return '-'
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  return h ? `${h}h${String(m).padStart(2, '0')}m` : `${m}m`
}

function ago(iso: any): string {
  const t = Date.parse(iso)
  return isNaN(t) ? '-' : dur(Math.max(0, (Date.now() - t) / 1000))
}

function staleOf(name: string): string | null {
  const d = docs[name]?.value
  if (!d) return docs[name]?.problem ? 'invalid' : 'no data'
  const age = (Date.now() - Date.parse(d.generated_at)) / 1000
  return age > d.stale_after_s ? `stale ${dur(age)}` : null
}

// ---- tab bodies: each returns lines [text, color?] ----

type Line = [string, string?]

function generalLines(wide: boolean): Line[] {
  const g = docs.general?.value
  const u = docs.usage?.value
  if (!g) return [[docs.general?.problem ?? 'no facts/general.json yet', 'yellow']]
  const out: Line[] = []
  const t = g.task
  if (t) {
    if (wide) out.push([`${t.title}${t.phase ? ` [${t.phase}]` : ''}`])
    out.push([`${t.metric.name} (${t.metric.unit}, ${t.metric.direction} is better)`])
    out.push([`base ${num(t.baseline.value)}${t.baseline.label && wide ? ` (${t.baseline.label})` : ''} · target ${t.target ? num(t.target.value) : '-'}`])
  } else out.push(['task: not recorded yet (ledger kind=task)', 'yellow'])
  const b = g.best
  if (b) {
    out.push([`best ${num(b.value)}  ${pct(b.vs_baseline_pct)} vs base · ${b.of_target_pct == null ? '-' : b.of_target_pct.toFixed(1) + '%'} of target`, 'green'])
    if (wide) out.push([`  runs ${b.run_ids.join(', ')}${b.stack ? ` · ${b.stack}` : ''}${b.at ? ` · ${ago(b.at)} ago` : ''}`])
  }
  const s = g.spend
  out.push([`tokens ${tok(s.tokens_new)} new + ${tok(s.tokens_cache_read)} cache reads · wall ${dur(s.wall_s)}${s.sessions ? ` · ${s.sessions} sessions` : ''}`])
  const p = g.progress
  out.push([`progress ${pct(p.gain_pct)} · ${pct(p.per_hour_pct)}/h · ${pct(p.per_mtok_pct)}/Mtok`])
  if (wide) {
    out.push([`  abs ${num(p.gain_abs, 1)} · ${num(p.per_hour_abs, 1)}/h · ${num(p.per_mtok_abs, 1)}/Mtok`])
    if (u) {
      const k = u.fleet.tokens
      out.push([`  in ${tok(k.input)} · cache w ${tok(k.cache_creation)} r ${tok(k.cache_read)} · out ${tok(k.output)}`])
    }
    if (cfg.show_sessions) {
      out.push(['sessions:'])
      for (const x of sessions) {
        const ut = u?.sessions?.find((y: any) => y.session_id === x.session_id)?.tokens?.new
        out.push([`  ${x.role === 'main' ? '*' : ' '} ${x.name ?? x.session_id.slice(0, 8)} ${x.alive ? x.status : 'DEAD'}${x.waiting_for ? ` (${x.waiting_for})` : ''} · ${tok(ut)}`, x.alive ? undefined : 'red'])
      }
    }
    for (const pr of g.problems ?? []) out.push([`! ${pr}`, 'yellow'])
  }
  return out
}

function detailLines(wide: boolean): Line[] {
  const d = docs.detail?.value
  if (!d) return [[docs.detail?.problem ?? 'no facts/detail.json yet', 'yellow']]
  const out: Line[] = []
  for (const r of d.current_runs) {
    const m = (r.machines ?? []).map((x: any) => (typeof x === 'string' ? x : x.node)).join(',')
    out.push([`▶ ${r.run_id} ${r.theme}${r.point ? ` [${r.point}]` : ''}${m ? ` @${m}` : ''}`, 'cyan'])
    if (wide && r.hypothesis) out.push([`    hyp: ${r.hypothesis}`])
  }
  if (!d.current_runs.length) out.push(['no run in progress', 'dim'])
  out.push([`last ${Math.min(cfg.recent_runs_n, d.recent_runs.length)} runs:`])
  for (const r of d.recent_runs.slice(0, cfg.recent_runs_n)) {
    const c = r.new_best ? 'green' : r.status === 'failed' || r.status === 'rejected' ? 'red' : undefined
    out.push([`  ${r.status.padEnd(8)} ${num(r.value)} ${pct(r.delta_pct)} ${r.run_id} ${r.theme}`, c])
    if (wide && r.conclusion) out.push([`      ${r.conclusion}`])
  }
  out.push([`next: ${d.next.length ? d.next[0].theme : '-'}`])
  if (wide) for (const n of d.next.slice(1)) out.push([`      ${n.theme}${n.blocked_on ? ` (blocked: ${n.blocked_on})` : ''}`])
  out.push([`machines (${d.machines.length}):`])
  for (const m of wide ? d.machines : d.machines.slice(0, 4)) {
    out.push([`  ${m.node}${m.cluster && wide ? `/${m.cluster}` : ''} → ${m.runs ?? 'idle'}${wide && m.gpu_util != null ? ` · gpu ${m.gpu_util}%` : ''}`])
  }
  if (wide) for (const pr of d.problems ?? []) out.push([`! ${pr}`, 'yellow'])
  return out
}

function kfLines(wide: boolean): Line[] {
  const k = docs.kernelforge?.value
  if (!k) return [[docs.kernelforge?.problem ?? 'no facts/kernelforge.json yet', 'yellow']]
  const out: Line[] = []
  for (const h of k.hot_ops.slice(0, wide ? 3 : 1)) {
    out.push([`hot ops (${h.role}${h.scope && wide ? `, ${h.scope}` : ''}):`])
    for (const o of h.ops.slice(0, wide ? 12 : 5)) out.push([`  ${o.share_pct.toFixed(1).padStart(5)}%  ${o.name}`])
  }
  out.push([`campaigns (${k.campaigns.length}):`])
  for (const c of k.campaigns) {
    const col = c.status === 'running' ? 'cyan' : c.status === 'stale' || c.problem ? 'yellow' : undefined
    out.push([`  ${c.operator_ids.join(',')} ${c.status} ${pct(c.gain_pct)} (${pct(c.gain_pct_per_hour)}/h) ${dur(c.age_s)}`, col])
    if (wide) out.push([`      ${c.campaign_root}${c.iterations != null ? ` · it ${c.iterations} kept ${c.kept ?? '-'}` : ''}${c.problem ? ` · ${c.problem}` : ''}`])
  }
  out.push(['ops:'])
  for (const o of wide ? k.ops : k.ops.slice(0, 6)) {
    out.push([`  ${o.op_id} ${o.stage}${o.e2e_verdict ? ` · e2e ${o.e2e_verdict} ${pct(o.e2e_delta_pct)}` : ''}`, o.e2e_verdict ? 'green' : undefined])
  }
  if (wide) for (const pr of k.problems ?? []) out.push([`! ${pr}`, 'yellow'])
  return out
}

const LEVEL = { INFO: 9, WARN: 13, ERROR: 17 } as Record<string, number>
const COLOR = { INFO: undefined, WARN: 'yellow', ERROR: 'red' } as Record<string, string | undefined>

function logLines(wide: boolean): Line[] {
  const min = LEVEL[cfg.log_min_level] ?? 9
  const ev = events.filter((x) => (x.severity_number ?? LEVEL[x.severity_text] ?? 9) >= min).slice(-cfg.log_lines)
  if (!ev.length) return [['no events yet', 'dim']]
  return ev.map((x): Line => {
    const t = (x.ts ?? '').slice(11, 19)
    const who = wide && x.attributes?.['session.name'] ? ` ${x.attributes['session.name']}` : ''
    return [`${t} ${x.severity_text.padEnd(5)}${wide ? ` ${x.event_name}` : ''}${who} ${x.body}`, COLOR[x.severity_text]]
  })
}

// ---- drawing ----

function linesOf(t: Tab, wide: boolean): Line[] {
  if (t === 'general') return generalLines(wide)
  if (t === 'detail') return detailLines(wide)
  if (t === 'kernelforge') return kfLines(wide)
  return logLines(wide)
}

function factFor(t: Tab): string | null {
  return t === 'log' ? null : t
}

async function openPane($: any, focus: boolean) {
  const size = SIZE[mode]
  const r = await $.ui.open(focus ? { id: PANE, title: 'o11y', focus: true, ...size } : { id: PANE, title: 'o11y', ...size })
  isOpen = true
  return r
}

async function dRenderPane($: any, e: any, next: any) {
  if (e.requestId === MENU) return renderMenu($, e)
  if (e.requestId === FORM) return renderForm($, e)
  if (e.requestId !== PANE || !active) return next(e)
  const { Box, Text, Button } = $.ui.resolve(e)
  const wide = mode === 'wide'
  const tabRow = Box({
    flexDirection: 'row',
    columnGap: 2,
    children: [
      ...TABS.map((t, i) =>
        Button({
          key: `tab-${t}`,
          label: t === 'kernelforge' && !wide ? 'kf' : t,
          hotkey: String(i + 1),
          plain: true,
          dimColor: t !== tab,
          onPress: () => selectTab($, t),
        }),
      ),
      Button({ key: 'tab-mode', label: wide ? 'narrow' : 'wide', hotkey: 'm', plain: true, dimColor: true, onPress: () => setMode($, wide ? 'narrow' : 'wide') }),
    ],
  })
  const f = factFor(tab)
  const stale = f ? staleOf(f) : null
  const meta = f && docs[f]?.value ? `${stale ? '⚠ ' + stale : 'updated ' + ago(docs[f].value.generated_at) + ' ago'}` : ''
  const body = linesOf(tab, wide).map(([s, color], i) =>
    Text({ key: `l${i}`, wrap: 'truncate-end', ...(color === 'dim' ? { dimColor: true } : color ? { color } : {}), children: [s] }),
  )
  return Box({
    flexDirection: 'column',
    children: [tabRow, ...(meta ? [Text({ wrap: 'truncate-end', ...(stale ? { color: 'yellow' } : { dimColor: true }), children: [meta] })] : []), ...body],
  })
}

async function dRenderBand($: any, e: any, next: any) {
  if (!active || !bandOn || !cfg.band) return next(e)
  const { Text } = $.ui.resolve(e)
  const g = docs.general?.value
  const parts: string[] = ['o11y']
  if (g?.task && g.best) parts.push(`${g.task.metric.name} ${num(g.best.value)} ${pct(g.best.vs_baseline_pct)} (tgt ${g.task.target ? num(g.task.target.value) : '-'})`)
  else if (g?.task) parts.push(`${g.task.metric.name} base ${num(g.task.baseline.value)}`)
  if (g) parts.push(`tok ${tok(g.spend.tokens_new)}+${tok(g.spend.tokens_cache_read)}c`)
  const d = docs.detail?.value
  if (d?.current_runs?.length) parts.push(`▶ ${d.current_runs[0].run_id}`)
  const bad = [...events].reverse().find((x) => x.severity_text === 'WARN' || x.severity_text === 'ERROR')
  if (bad) parts.push(`${bad.severity_text} ${ago(bad.ts)} ago: ${bad.body}`)
  if (lastFire) parts.push(`${lastFire.kind} ${ago(lastFire.ts)} ago`)
  const st = staleOf('general')
  if (st) parts.push(`(${st})`)
  return Text({ wrap: 'truncate-end', ...(bad?.severity_text === 'ERROR' ? { color: 'red' } : {}), children: [parts.join(' · ')] })
}

function selectTab($: any, t: Tab) {
  tab = t
  $.ui.invalidate('ui.render')
}

async function setMode($: any, m: 'narrow' | 'wide') {
  mode = m
  await openPane($, false)
  $.ui.invalidate('ui.render')
}

// ---- menu ----

async function renderMenu($: any, e: any) {
  const { Box, Text, Button } = $.ui.resolve(e)
  const items: [string, string][] = [
    ['g', 'general'], ['d', 'detail'], ['k', 'kernelforge'], ['l', 'log'],
    ['m', mode === 'wide' ? 'narrow' : 'wide'], ['b', bandOn ? 'band off' : 'band on'],
    ['t', 'cfg trigger'], ['a', 'cfg analysis'], ['v', 'cfg display'], ['x', 'close o11y'],
  ]
  const btn = ([k, label]: [string, string]) =>
    Button({ key: `menu-${k}`, label, hotkey: k, plain: true, onPress: () => menuPick($, k) })
  return Box({
    flexDirection: 'column',
    children: [
      Text({ dimColor: true, children: ['o11y: press a key, Esc to cancel'] }),
      Box({ flexDirection: 'row', columnGap: 2, children: items.slice(0, 5).map(btn) }),
      Box({ flexDirection: 'row', columnGap: 2, children: items.slice(5).map(btn) }),
    ],
  })
}

async function menuPick($: any, k: string) {
  await $.ui.close({ id: MENU })
  const tabs: Record<string, Tab> = { g: 'general', d: 'detail', k: 'kernelforge', l: 'log' }
  if (tabs[k]) return showTab($, tabs[k])
  if (k === 'm') return setMode($, mode === 'wide' ? 'narrow' : 'wide')
  if (k === 'b') {
    bandOn = !bandOn
    $.ui.invalidate('ui.render')
    return
  }
  if (k === 't') return openForm($, 'trigger')
  if (k === 'a') return openForm($, 'analysis')
  if (k === 'v') return openForm($, 'display')
  if (k === 'x') {
    await $.ui.close({ id: PANE })
    isOpen = false
  }
}

async function showTab($: any, t: Tab) {
  tab = t
  await openPane($, false)
  $.ui.invalidate('ui.render')
}

// ---- config forms: generated from the config schema ----

interface Field {
  path: string[]
  schema: any
}

function fieldsOf(schema: any, at: string[] = []): Field[] {
  const out: Field[] = []
  for (const [k, ps] of Object.entries<any>(schema.properties ?? {})) {
    if (k === 'v' || k === 'updated_at') continue
    if (ps.type === 'object' && ps.properties) out.push(...fieldsOf(ps, [...at, k]))
    else out.push({ path: [...at, k], schema: ps })
  }
  return out
}

function getAt(obj: any, path: string[]): any {
  return path.reduce((o, k) => (o == null ? undefined : o[k]), obj)
}

function setAt(obj: any, path: string[], value: any) {
  let o = obj
  for (const k of path.slice(0, -1)) o = o[k] ??= {}
  o[path[path.length - 1]] = value
}

async function openForm($: any, target: 'trigger' | 'analysis' | 'display') {
  if (!active) return
  formTarget = target
  const r = await readDoc($, abs(P.config(target)))
  formDraft = r.value && typeof r.value === 'object' ? r.value : { v: 1 }
  formNote = r.problem ? `current file invalid: ${r.problem}` : `config/${target}.json`
  await $.ui.open({ id: FORM, title: `o11y config: ${target}`, focus: true, closeOnEscape: true, rows: 30, columns: 90 })
}

function parseField(f: Field, raw: string): { ok: boolean; value?: any; problem?: string } {
  const t = Array.isArray(f.schema.type) ? f.schema.type[0] : f.schema.type
  if (t === 'number' || t === 'integer') {
    const n = Number(raw)
    if (raw.trim() === '' || !isFinite(n) || (t === 'integer' && !Number.isInteger(n))) return { ok: false, problem: `${f.path.join('.')}: not a ${t}` }
    return { ok: true, value: n }
  }
  if (t === 'array') {
    try {
      return { ok: true, value: JSON.parse(raw) }
    } catch {
      return { ok: false, problem: `${f.path.join('.')}: not a JSON array` }
    }
  }
  return { ok: true, value: raw }
}

async function saveField($: any, f: Field, value: any) {
  const draft = JSON.parse(JSON.stringify(formDraft))
  setAt(draft, f.path, value)
  draft.v = 1
  draft.updated_at = nowIso()
  const probs = validate(SCHEMAS[`config.${formTarget}`], draft)
  if (probs.length) {
    formNote = `not saved: ${probs[0]}`
  } else {
    await writeDoc($, abs(P.config(formTarget)), draft)
    formDraft = draft
    formNote = `saved ${f.path.join('.')} = ${JSON.stringify(value).slice(0, 60)}`
  }
  $.ui.invalidate('ui.render')
}

async function renderForm($: any, e: any) {
  const { Box, Text, Input, Select } = $.ui.resolve(e)
  const schema = SCHEMAS[`config.${formTarget}`]
  const full = withDefaults(schema, formDraft)
  const rows = fieldsOf(schema).map((f, i) => {
    const key = `cfg-${formTarget}-${f.path.join('.')}`
    const label = f.path.join('.')
    const cur = getAt(full, f.path)
    const isBool = f.schema.type === 'boolean'
    const choices = isBool ? [true, false] : Array.isArray(f.schema.enum) ? f.schema.enum.filter((x: any) => x !== null) : null
    if (choices) {
      return Select({
        key,
        label,
        value: String(cur),
        ...(i === 0 ? { autoFocus: true } : {}),
        options: choices.map((c: any) => ({ value: String(c), label: String(c) })),
        onSelect: (v: string) => saveField($, f, isBool ? v === 'true' : v),
      })
    }
    return Input({
      key,
      label,
      value: f.schema.type === 'array' ? JSON.stringify(cur ?? []) : String(cur ?? ''),
      submitLabel: 'save',
      ...(i === 0 ? { autoFocus: true } : {}),
      onSubmit: (raw: string) => {
        const p = parseField(f, raw)
        if (!p.ok) {
          formNote = `not saved: ${p.problem}`
          $.ui.invalidate('ui.render')
          return
        }
        return saveField($, f, p.value)
      },
    })
  })
  return Box({
    flexDirection: 'column',
    children: [Text({ dimColor: true, wrap: 'truncate-end', children: [`${formNote} · Tab/Enter to edit, Esc to close`] }), ...rows],
  })
}

// ---- commands and hooks ----

async function command($: any, name: string, args: string): Promise<any> {
  if (!active) {
    const why = vetoReason || `this session is not the active main of fleet ${root} (it becomes one with AGENT_SYS_O11Y=true or on first agent_sys skill use; the display follows within one trigger tick)`
    return { text: `o11y display inactive: ${why}`, isTextDisplayOnly: true }
  }
  const a = (args ?? '').trim()
  switch (name) {
    case 'o11y':
      await $.ui.open({ id: MENU, title: 'o11y menu', focus: true, closeOnEscape: true, holdToasts: true, rows: 4 })
      break
    case 'o11y-general':
      await showTab($, 'general')
      break
    case 'o11y-detail':
      await showTab($, 'detail')
      break
    case 'o11y-kf':
      await showTab($, 'kernelforge')
      break
    case 'o11y-log':
      await showTab($, 'log')
      break
    case 'o11y-mode':
      await setMode($, a === 'narrow' || a === 'wide' ? a : mode === 'wide' ? 'narrow' : 'wide')
      break
    case 'o11y-band':
      bandOn = a === 'on' ? true : a === 'off' ? false : !bandOn
      $.ui.invalidate('ui.render')
      break
    case 'o11y-cfg-trigger':
      await openForm($, 'trigger')
      break
    case 'o11y-cfg-analysis':
      await openForm($, 'analysis')
      break
    case 'o11y-cfg-display':
      await openForm($, 'display')
      break
  }
  return {}
}

export function register(on: any) {
  on('session.start', async ($: any, e: any, next: any) => {
    sessionId = await $.session.id()
    for (const c of COMMANDS) {
      try {
        await $.command.register({ ...c, immediate: true })
      } catch (err: any) {
        $.ui.log(`o11y: /${c.name} not registered: ${err.message}`)
      }
    }
    const env = await $.env.get('AGENT_SYS_O11Y')
    if (env !== undefined && env !== 'true') {
      vetoReason = `AGENT_SYS_O11Y=${env}; the display runs only in a fleet's main session`
      return next(e) // vetoed: no root, no polling
    }
    root = await fleetDirOf(await readEnv($), String(await $.session.root()))
    startPoller($)
    return next(e)
  })

  for (const c of COMMANDS) {
    on('command.run', { command: c.name }, async ($: any, e: any) => command($, e.command, e.args))
  }

  on('ui.render', { component: 'Pane' }, async ($: any, e: any, next: any) => dRenderPane($, e, next))
  on('ui.render', { component: 'AbovePrompt' }, async ($: any, e: any, next: any) => dRenderBand($, e, next))

  on('ui.close', async ($: any, e: any, next: any) => {
    if (e.id === PANE) isOpen = false
    return next(e)
  })
}
