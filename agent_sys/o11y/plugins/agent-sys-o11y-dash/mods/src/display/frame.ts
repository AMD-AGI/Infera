// The display frame (design §9): fleet root, activation from the registry, polling of the files D reads,
// the pane (tab row, age / stale line, the open tab's renderer), the band, and the menu.
import { P, fleetDirOf, loadConfig, readDoc, readEnv, readJsonl, type ConfigState } from '../../../protocol/ts/index.ts'
import { ago, dur, num, pct, tok, type Line } from './fmt.ts'
import { general } from './general.ts'
import { detail } from './detail.ts'
import { kernelforge } from './kernelforge.ts'
import { LEVELS, log } from './log.ts'
import { openForm } from './forms.ts'

export const PANE = 'o11y'
export const MENU = 'o11y-menu'
export const TABS = ['general', 'detail', 'kernelforge', 'log'] as const
export type Tab = (typeof TABS)[number]
const FACTS = ['general', 'detail', 'kernelforge'] as const
const SIZE = { narrow: { columns: 46, rows: 16 }, wide: { columns: 96, rows: 40 } }

let root: string | null = null
let sessionId = ''
let active = false
let registryKey = ''
let cfgState: ConfigState | null = null
let tab: Tab = 'general'
let mode: 'narrow' | 'wide' = 'narrow'
let bandOn = true
let minLevel: string = 'INFO'
const docs: Record<string, { value: any; problem: string | null; mtimeMs: number }> = {}
let sessions: any[] = []
let events: any[] = []
let eventsSize = -1
let lastFire: any = null
let lastFireName = ''
let polls = 0
let poller: { cancel(): void } | null = null
let pollMs = 0

const abs = (rel: string) => `${root}/${rel}`
const cfg = () => cfgState!.cfg

export const isActive = () => active
export const fleetRoot = () => root

export async function start($: any) {
  sessionId = await $.session.id()
  root = (await fleetDirOf($, await readEnv($), sessionId)).fleet
  cfgState = await loadConfig($, abs(P.config('display')), 'config.display', null)
  startPoller($)
}

function startPoller($: any) {
  if (poller) poller.cancel()
  pollMs = cfg().poll_ms
  poller = $.clock.every(pollMs, () => poll($))
}

// Active iff the registry lists this session as an alive, active main (design §9).
async function checkActive($: any) {
  const sid = (await $.session.id()) || sessionId
  if (sid !== sessionId || (!active && polls % 5 === 0)) {
    sessionId = sid
    root = (await fleetDirOf($, await readEnv($), sessionId)).fleet // the pin appears at activation
  }
  const sp = abs(P.sessions)
  const m = (await $.fs.exists(sp)) ? (await $.fs.stat(sp)).mtimeMs : 0
  const key = `${root}:${m}:${sessionId}`
  if (key === registryKey) return
  registryKey = key
  sessions = (await readDoc($, sp, 'registry.sessions')).value?.sessions ?? []
  const mine = sessions.some((x) => x.session_id === sessionId && x.role === 'main' && x.alive && x.active === true)
  if (mine && !active) {
    active = true
    await poll($)
    mode = cfg().mode
    await openPane($) // opened unasked: placed only in a wide enough terminal; the band always shows
  } else if (!mine && active) {
    active = false
    await $.ui.close({ id: PANE })
  }
  $.ui.invalidate('ui.render')
}

async function poll($: any) {
  if (!root) return
  polls++
  let changed = false
  try {
    await checkActive($)
    if (!active) return
    const before = cfgState!.mtimeMs
    cfgState = await loadConfig($, abs(P.config('display')), 'config.display', cfgState)
    if (cfgState.report) $.ui.log(`o11y: config/display.json not applied, the previous config stays: ${cfgState.report}`)
    changed ||= cfgState.mtimeMs !== before
    for (const f of FACTS) {
      const path = abs(P.facts(f))
      const m = (await $.fs.exists(path)) ? (await $.fs.stat(path)).mtimeMs : 0
      if (docs[f] && docs[f].mtimeMs === m) continue
      const r = await readDoc($, path, `facts.${f}`) // an invalid document counts as absent; its problem is shown
      docs[f] = r.problem ? { ...r, problem: 'invalid ' + r.problem.replace(root + '/', '') } : r
      changed = true
    }
    const ep = abs(P.events)
    const es = (await $.fs.exists(ep)) ? (await $.fs.stat(ep)).size : 0
    if (es !== eventsSize) {
      eventsSize = es
      events = (await readJsonl($, ep, 256 * 1024)).filter((x) => x && x.v === 1 && typeof x.severity_text === 'string')
      changed = true
    }
    if (polls % 5 === 0 && (await $.fs.exists(abs(P.triggers)))) {
      // the newest trigger record; the directory only grows, so it is listed every 5th poll
      const newest = (await $.fs.list(abs(P.triggers))).map((x: any) => x.name).filter((n: string) => /^\d+\.json$/.test(n)).sort().pop() ?? ''
      if (newest && newest !== lastFireName) {
        lastFireName = newest
        lastFire = (await readDoc($, abs(`${P.triggers}/${newest}`), 'trigger_fire')).value
        changed = true
      }
    }
  } catch (err: any) {
    $.ui.log(`o11y: display poll: ${err.message}`, { to: 'debug' })
  }
  if (cfg().poll_ms !== pollMs) startPoller($)
  if (changed) $.ui.invalidate('ui.render')
}

function staleOf(name: string): string | null {
  const d = docs[name]?.value
  if (!d) return docs[name]?.problem ? 'invalid' : 'no data'
  const age = (Date.now() - Date.parse(d.generated_at)) / 1000
  return age > d.stale_after_s ? `stale ${dur(age)}` : null
}

function linesOf(t: Tab, wide: boolean): Line[] {
  if (t === 'log') return log(events, wide, cfg().log_lines, minLevel)
  const d = docs[t]
  if (!d?.value) return [[d?.problem ?? `no facts/${t}.json yet`, 'yellow']]
  if (t === 'general') return general(d.value, wide, sessions)
  if (t === 'detail') return detail(d.value, wide, cfg().recent_runs_n)
  return kernelforge(d.value, wide)
}

export async function openPane($: any) {
  await $.ui.open({ id: PANE, title: 'o11y', ...SIZE[mode] })
}

export async function showTab($: any, t: Tab) {
  tab = t
  await openPane($)
  $.ui.invalidate('ui.render')
}

export async function setMode($: any, m?: string) {
  mode = m === 'narrow' || m === 'wide' ? m : mode === 'wide' ? 'narrow' : 'wide'
  await openPane($)
  $.ui.invalidate('ui.render')
}

export function setBand($: any, arg?: string) {
  bandOn = arg === 'on' ? true : arg === 'off' ? false : !bandOn
  $.ui.invalidate('ui.render')
}

function text(Text: any, [s, color]: Line, key: string) {
  return Text({ key, wrap: 'truncate-end', ...(color === 'dim' ? { dimColor: true } : color ? { color } : {}), children: [s] })
}

export async function renderPane($: any, e: any) {
  const { Box, Text, Button } = $.ui.resolve(e)
  const wide = mode === 'wide'
  const btn = (key: string, label: string, hotkey: string, dim: boolean, onPress: () => unknown) => Button({ key, label, hotkey, plain: true, dimColor: dim, onPress })
  const row = Box({
    flexDirection: 'row',
    columnGap: 2,
    children: [
      ...TABS.map((t, i) => btn(`tab-${t}`, t === 'kernelforge' && !wide ? 'kf' : t, String(i + 1), t !== tab, () => showTab($, t))),
      btn('tab-mode', wide ? 'narrow' : 'wide', 'm', true, () => setMode($)),
      ...(tab === 'log' ? [btn('tab-level', 'level', 'f', true, () => cycleLevel($))] : []),
    ],
  })
  const stale = tab === 'log' ? null : staleOf(tab)
  const meta: Line | null = tab === 'log' ? null : stale ? [`⚠ ${stale}`, 'yellow'] : [`updated ${ago(docs[tab].value.generated_at)} ago`, 'dim']
  const body = linesOf(tab, wide).map((l, i) => text(Text, l, `l${i}`))
  return Box({ flexDirection: 'column', children: [row, ...(meta ? [text(Text, meta, 'meta')] : []), ...body] })
}

function cycleLevel($: any) {
  minLevel = LEVELS[(LEVELS.indexOf(minLevel as any) + 1) % LEVELS.length]
  $.ui.invalidate('ui.render')
}

export function renderBand($: any, e: any) {
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

export const bandShown = () => active && bandOn

export async function openMenu($: any) {
  await $.ui.open({ id: MENU, title: 'o11y menu', focus: true, closeOnEscape: true, holdToasts: true, rows: 4 })
}

export async function renderMenu($: any, e: any) {
  const { Box, Text, Button } = $.ui.resolve(e)
  const items: [string, string][] = [
    ['g', 'general'], ['d', 'detail'], ['k', 'kernelforge'], ['l', 'log'], ['m', mode === 'wide' ? 'narrow' : 'wide'],
    ['b', bandOn ? 'band off' : 'band on'], ['t', 'cfg trigger'], ['a', 'cfg analysis'], ['v', 'cfg display'], ['x', 'close o11y'],
  ]
  const b = ([k, label]: [string, string]) => Button({ key: `menu-${k}`, label, hotkey: k, plain: true, onPress: () => menuPick($, k) })
  return Box({
    flexDirection: 'column',
    children: [
      Text({ dimColor: true, children: ['o11y: press a key, Esc to cancel'] }),
      Box({ flexDirection: 'row', columnGap: 2, children: items.slice(0, 5).map(b) }),
      Box({ flexDirection: 'row', columnGap: 2, children: items.slice(5).map(b) }),
    ],
  })
}

async function menuPick($: any, k: string) {
  await $.ui.close({ id: MENU })
  const tabs: Record<string, Tab> = { g: 'general', d: 'detail', k: 'kernelforge', l: 'log' }
  const forms: Record<string, 'trigger' | 'analysis' | 'display'> = { t: 'trigger', a: 'analysis', v: 'display' }
  if (tabs[k]) return showTab($, tabs[k])
  if (forms[k]) return openForm($, root!, forms[k])
  if (k === 'm') return setMode($)
  if (k === 'b') return setBand($)
  if (k === 'x') await $.ui.close({ id: PANE })
}
