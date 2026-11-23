// Display-side acceptance criteria (spec §12; design §12 map), run against a fake mods API.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fakeApi, hooksOf, texts } from './fake_api.ts'

const PREFIX = '/prefix'
const FLEET = `${PREFIX}/state/o11y/fleets/-work-proj`
let instance = 0
const iso = (msAgo: number) => new Date(Date.now() - msAgo).toISOString()
const header = (msAgo: number) => ({ v: 1, generated_at: iso(msAgo), stale_after_s: 900, sources: ['ledger/runs.jsonl'] })

function world(env: Record<string, string> = {}) {
  const f = fakeApi(mkdtempSync(join(tmpdir(), 'o11y-dash-')), { HOME: '/home/u', AGENT_SYS_HOME: PREFIX, ...env })
  f.put(`${PREFIX}/state/o11y/sessions/sid-main.json`, { v: 1, session_id: 'sid-main', fleet_dir: FLEET, role: 'main', pinned_at: iso(0) })
  f.put(`${FLEET}/facts/general.json`, {
    ...header(1000),
    task: { task_id: 't', title: 'GLM tput', metric: { name: 'tput', unit: 'tok/s', direction: 'higher' }, baseline: { value: 100 }, target: { value: 200 } },
    best: { value: 150, run_ids: ['r1'], vs_baseline_pct: 50, of_target_pct: 75 },
    spend: { tokens_new: 2_000_000, tokens_cache_read: 9_000_000, wall_s: 3600 },
    progress: { gain_abs: 50, gain_pct: 50, per_hour_pct: 50, per_mtok_pct: 25 },
  })
  f.put(`${FLEET}/facts/detail.json`, {
    ...header(1000),
    current_runs: [{ run_id: 'r2', theme: 'EP8 decode', point: 'explore', machines: ['m135'] }],
    recent_runs: [{ run_id: 'r1', theme: 'fused moe', status: 'kept', value: 150, delta_pct: 3, new_best: true }],
    next: [{ theme: 'MTP depth 3' }],
    machines: [{ node: 'm135', runs: 'r2 decode' }],
  })
  f.put(`${FLEET}/facts/kernelforge.json`, {
    ...header(2 * 3600_000), // older than stale_after_s
    hot_ops: [{ profile_id: 'p', role: 'decode', ops: [{ name: 'fused_moe', share_pct: 31.2 }] }],
    campaigns: [{ campaign_root: '/r', operator_ids: ['fused_moe'], status: 'running', age_s: 600, gain_pct: 7, gain_pct_per_hour: 42 }],
    ops: [{ op_id: 'fused_moe', stage: 'campaign' }],
  })
  f.put(`${FLEET}/logs/events.jsonl`, JSON.stringify({ v: 1, ts: iso(0), severity_text: 'WARN', severity_number: 13, event_name: 'o11y.machine.lost', body: 'm137 reclaimed', attributes: {} }) + '\n')
  return f
}

const activeRegistry = (f: any, sid = 'sid-main') =>
  f.put(`${FLEET}/registry/sessions.json`, { v: 1, generated_at: iso(0), main_session_id: sid, sessions: [{ session_id: sid, pid: 1, alive: true, role: 'main', status: 'idle', active: true }] })

// The shipped single-file bundle: each import is a fresh instance of all display state, as a plugin reload is.
// protocol.test.ts checks the bundle is built from the current sources.
async function load() {
  return hooksOf((await import(`../../plugins/agent-sys-o11y-dash/mods/dist/display.js?i=${++instance}`)).register)
}
const pane = async (f: any, emit: any) => texts(await emit(f.$, 'ui.render', { component: 'Pane', requestId: 'o11y' }))
const band = async (f: any, emit: any) => texts(await emit(f.$, 'ui.render', { component: 'AbovePrompt' }))

test('criterion 17: the band appears within one poll of the registry listing the active main; each tab shows its contents, age and stale mark', async () => {
  const f = world()
  const { emit } = await load()
  await emit(f.$, 'session.start')
  await f.tick()
  assert.deepEqual(await band(f, emit), [], 'not active before the registry lists it')
  activeRegistry(f)
  await f.tick() // one poll
  const b = (await band(f, emit)).join(' ')
  assert.match(b, /tput 150 \+50\.0% \(tgt 200\)/)
  assert.match(b, /WARN .* m137 reclaimed/)
  assert.match((await pane(f, emit)).join('\n'), /updated \dm ago[\s\S]*tokens 2\.0M new \+ 9\.0M cache reads/)
  await emit(f.$, 'command.run', { command: 'o11y-detail', args: '' })
  assert.match((await pane(f, emit)).join('\n'), /▶ r2 EP8 decode \[explore\] @m135[\s\S]*next: MTP depth 3/)
  await emit(f.$, 'command.run', { command: 'o11y-kf', args: '' })
  const kf = (await pane(f, emit)).join('\n')
  assert.match(kf, /⚠ stale 2h00m/)
  assert.match(kf, /31\.2%  fused_moe/)
  await emit(f.$, 'command.run', { command: 'o11y-log', args: '' })
  assert.match((await pane(f, emit)).join('\n'), /WARN  m137 reclaimed/)
})

test('criterion 4 (display side): a vetoed session never shows the display', async () => {
  const f = world({ AGENT_SYS_O11Y: 'child' })
  activeRegistry(f)
  const { emit } = await load()
  await emit(f.$, 'session.start')
  await f.tick()
  assert.deepEqual(await band(f, emit), [])
  const r = await emit(f.$, 'command.run', { command: 'o11y', args: '' })
  assert.match(r.text, /AGENT_SYS_O11Y=child/)
})

test('criterion 27 (display side): an invalid facts file shows its problem, the others still render, and it recovers', async () => {
  const f = world()
  activeRegistry(f)
  const { emit } = await load()
  await emit(f.$, 'session.start')
  f.put(`${FLEET}/facts/general.json`, '{"v": 1, "generated_at": ')
  await f.tick()
  assert.match((await pane(f, emit)).join('\n'), /invalid facts\/general\.json/)
  await emit(f.$, 'command.run', { command: 'o11y-detail', args: '' })
  assert.match((await pane(f, emit)).join('\n'), /r2 EP8 decode/)
  f.put(`${FLEET}/facts/general.json`, { ...header(0), task: null, best: null, spend: { tokens_new: 1, tokens_cache_read: 0, wall_s: 1 }, progress: { gain_abs: null, gain_pct: null } })
  await f.tick()
  await emit(f.$, 'command.run', { command: 'o11y-general', args: '' })
  assert.match((await pane(f, emit)).join('\n'), /task: not recorded yet/)
})

test('criterion 18 (dispatch): every sub-key command reaches its tab, mode, band toggle or form', async () => {
  const f = world()
  activeRegistry(f)
  const opened: any[] = []
  f.$.ui.open = async (o: any) => (opened.push(o), { isPlaced: true })
  const { emit } = await load()
  await emit(f.$, 'session.start')
  await f.tick()
  const snippet = (await import('../../plugins/agent-sys-o11y-dash/keybindings.snippet.json', { with: { type: 'json' } })).default
  const commands = Object.values(snippet.bindings[0].bindings).map((a: any) => a.replace('command:', ''))
  assert.equal(commands.length, 10)
  for (const c of commands) {
    const r = await emit(f.$, 'command.run', { command: c, args: '' })
    assert.deepEqual(r, {}, c)
  }
  assert.ok(opened.some((o) => o.id === 'o11y-menu'))
  for (const t of ['trigger', 'analysis', 'display']) assert.ok(opened.some((o) => o.id === 'o11y-cfg' && o.title.endsWith(t)), t)
  assert.ok(opened.some((o) => o.id === 'o11y' && o.columns === 96), 'wide mode reopened the pane wider')
})

test('config forms: a value outside the schema is refused; a valid one is written whole', async () => {
  const f = world()
  activeRegistry(f)
  const { emit } = await load()
  await emit(f.$, 'session.start')
  await f.tick()
  await emit(f.$, 'command.run', { command: 'o11y-cfg-trigger', args: '' })
  const form = await emit(f.$, 'ui.render', { component: 'Pane', requestId: 'o11y-cfg' })
  const field = form.children.find((c: any) => c.key === 'cfg-trigger-kick.interval_s')
  await field.onSubmit('5')
  assert.match(texts(await emit(f.$, 'ui.render', { component: 'Pane', requestId: 'o11y-cfg' }))[0], /not saved: \/kick\/interval_s: must be >= 60/)
  await field.onSubmit('120')
  assert.equal(f.get(`${FLEET}/config/trigger.json`).kick.interval_s, 120)
})
