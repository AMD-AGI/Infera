// Trigger-side acceptance criteria (spec §12; design §12 map), run deterministically against a fake mods API
// with controlled time. Tests load the shipped single-file bundle: each import (with a query) is a fresh
// instance of all trigger state, as a plugin reload is.
import { test, mock } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, readdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fakeApi, hooksOf, procStat } from './fake_api.ts'

const PREFIX = '/prefix'
const LAUNCH = '/work/proj'
const CFG = '/cfg'
const FLEET = `${PREFIX}/state/o11y/fleets/-work-proj`
const MAIN_PID = 4242
let instance = 0

// A main session's world: its Claude record, /proc entries, a fast config.
function world(env: Record<string, string> = { AGENT_SYS_O11Y: 'true' }) {
  const f = fakeApi(mkdtempSync(join(tmpdir(), 'o11y-trig-')), { HOME: '/home/u', AGENT_SYS_HOME: PREFIX, CLAUDE_CONFIG_DIR: CFG, ...env })
  f.put(`/proc/self/stat`, procStat(MAIN_PID, '100'))
  f.put(`/proc/${MAIN_PID}/stat`, procStat(MAIN_PID, '100'))
  f.put(`/proc/${MAIN_PID}/environ`, 'TMUX=/tmp/tmux-1/default,1,0\0HOME=/home/u\0')
  f.put(`${CFG}/sessions/${MAIN_PID}.json`, { pid: MAIN_PID, sessionId: 'sid-main', cwd: LAUNCH, procStart: '100', startedAt: 1000, status: 'idle', tmux: 'main:@1.%1' })
  f.put(`${FLEET}/config/trigger.json`, { v: 1, kick: { interval_s: 60 }, packup: { every_n_kicks: 2 }, refresh: { every_n_tool_calls: 2, every_n_turns: 1 }, log: { every_n_tool_calls: 4, every_n_turns: 2, slow_interval_s: 600 } })
  return f
}

async function load() {
  const mod = await import(`../../plugins/agent-sys-o11y/mods/dist/trigger.js?i=${++instance}`)
  return hooksOf(mod.register)
}

const fires = (f: ReturnType<typeof world>) =>
  readdirSync(f.path(`${FLEET}/triggers`)).sort().map((n) => f.get(`${FLEET}/triggers/${n}`))

test('criterion 1: the kick is submitted at its due tick; the recorded text is the submitted text and starts with the marker', async () => {
  mock.timers.enable({ apis: ['Date'], now: 1_000_000 })
  try {
    const f = world()
    const { emit } = await load()
    await emit(f.$, 'session.start')
    mock.timers.tick(59_000)
    await f.tick()
    assert.equal(f.prompts.length, 0)
    mock.timers.tick(1_000)
    await f.tick()
    assert.equal(f.prompts.length, 1)
    const rec = fires(f).find((r) => r.kind === 'kick')
    assert.equal(rec.text, f.prompts[0])
    assert.ok(f.prompts[0].startsWith('[o11y automated kick] '))
  } finally {
    mock.timers.reset()
  }
})

test('criterion 2: with a prompt queued, a newly due kick is skipped and recorded — also after a reload', async () => {
  mock.timers.enable({ apis: ['Date'], now: 2_000_000 })
  try {
    const f = world()
    f.$.prompt.submit = async ({ text }: any) => (f.prompts.push(text), new Promise(() => {})) // queued forever
    let { emit } = await load()
    await emit(f.$, 'session.start')
    mock.timers.tick(60_000)
    await f.tick()
    mock.timers.tick(60_000)
    await f.tick()
    assert.deepEqual(fires(f).map((r) => r.outcome), ['submitted', 'skipped_pending'])
    ;({ emit } = await load()) // reload: a new module instance, the pending entry only on disk
    f.timers.length = 0
    await emit(f.$, 'session.start')
    mock.timers.tick(60_000)
    await f.tick()
    assert.equal(f.prompts.length, 1, 'never two queued')
    assert.equal(fires(f).at(-1).outcome, 'skipped_pending')
    await emit(f.$, 'turn.start', {}) // the queued prompt's turn starts: the restored entry ends
    mock.timers.tick(60_000)
    await f.tick()
    assert.equal(fires(f).at(-1).outcome, 'submitted')
  } finally {
    mock.timers.reset()
  }
})

test('criterion 3: every 2nd automated kick (config) is a packup; a question does not count', async () => {
  mock.timers.enable({ apis: ['Date'], now: 3_000_000 })
  try {
    const f = world()
    const { emit } = await load()
    await emit(f.$, 'session.start')
    const kinds: string[] = []
    for (let i = 0; i < 4; i++) {
      mock.timers.tick(60_000)
      if (i === 1) {
        // an unregistered session appears: one question in between
        f.put(`/proc/777/stat`, procStat(777, '5'))
        f.put(`/proc/777/environ`, 'TMUX=/tmp/tmux-1/default,1,0\0')
        f.put(`${CFG}/sessions/777.json`, { pid: 777, sessionId: 'sid-x', cwd: LAUNCH, procStart: '5', startedAt: 2000, status: 'idle', tmux: 'x:@2.%2', name: 'x' })
      }
      await f.tick()
    }
    for (const r of fires(f)) kinds.push(r.kind)
    assert.deepEqual(kinds.filter((k) => k !== 'question'), ['kick', 'packup', 'kick', 'packup'])
    assert.equal(kinds.filter((k) => k === 'question').length, 1)
  } finally {
    mock.timers.reset()
  }
})

test('criterion 3: a packup comes earlier when the packup interval elapses before the N-th kick', async () => {
  mock.timers.enable({ apis: ['Date'], now: 3_500_000 })
  try {
    const f = world()
    f.put(`${FLEET}/config/trigger.json`, { v: 1, kick: { interval_s: 60 }, packup: { every_n_kicks: 10, interval_s: 150 } })
    const { emit } = await load()
    await emit(f.$, 'session.start')
    for (let i = 0; i < 4; i++) {
      mock.timers.tick(60_000)
      await f.tick()
    }
    assert.deepEqual(fires(f).map((r) => r.kind), ['kick', 'kick', 'packup', 'kick'])
  } finally {
    mock.timers.reset()
  }
})

test('criterion 4: a child (or any non-true activation value) never activates, even after an agent_sys skill', async () => {
  for (const value of ['child', 'false', '0']) {
    const f = world({ AGENT_SYS_O11Y: value })
    const { emit } = await load()
    await emit(f.$, 'session.start')
    await emit(f.$, 'tool.call', { tool: 'Skill', skill: 'agent-sys-prior-work' })
    await emit(f.$, 'command.run', { command: 'infera-agent-sys', origin: { kind: 'composer' } })
    await f.flushAfter()
    await f.tick()
    assert.equal(f.prompts.length, 0, value)
    assert.equal(f.timers.filter((t) => t.every).length, 0, `${value}: no tick`)
    assert.ok(!f.runs.some((r) => r[0] === 'mv'), `${value}: nothing written`)
  }
})

test('criterion 5: no activation variable → main on the first agent_sys skill use (model or user); stays main after /clear', async () => {
  for (const how of ['model', 'user']) {
    const f = world({})
    const { emit } = await load()
    await emit(f.$, 'session.start')
    await emit(f.$, 'tool.call', { tool: 'Skill', skill: 'plain-skill' })
    await f.flushAfter()
    assert.equal(f.timers.filter((t) => t.every).length, 0, 'a non-agent_sys skill does not activate')
    if (how === 'model') await emit(f.$, 'tool.call', { tool: 'Skill', skill: 'agent-sys-demo' })
    else await emit(f.$, 'command.run', { command: 'infera-agent-sys', origin: { kind: 'composer' } })
    await f.flushAfter()
    assert.equal(f.get(`${PREFIX}/state/o11y/sessions/sid-main.json`).fleet_dir, FLEET, how)
    assert.equal(f.get(`${FLEET}/registry/sessions.json`).sessions.find((s: any) => s.active).session_id, 'sid-main')
    f.session.id = 'sid-cleared' // /clear: new id, same process
    await f.tick()
    const pin = f.get(`${PREFIX}/state/o11y/sessions/sid-cleared.json`)
    assert.deepEqual(pin.lineage, ['sid-main'])
    const reg = f.get(`${FLEET}/registry/sessions.json`)
    assert.equal(reg.main_session_id, 'sid-cleared')
    assert.equal(reg.sessions.find((s: any) => s.active).session_id, 'sid-cleared')
  }
})

test('criterion 24: one question per unregistered session; none after a not_in_fleet answer, or no answer', async () => {
  mock.timers.enable({ apis: ['Date'], now: 4_000_000 })
  try {
    const f = world()
    const { emit } = await load()
    await emit(f.$, 'session.start')
    const add = (pid: number, sid: string, server = '/tmp/tmux-1/default') => {
      f.put(`/proc/${pid}/stat`, procStat(pid, '7'))
      f.put(`/proc/${pid}/environ`, `TMUX=${server},9,0\0`)
      f.put(`${CFG}/sessions/${pid}.json`, { pid, sessionId: sid, cwd: `${LAUNCH}/sub`, procStart: '7', startedAt: 5000, status: 'idle', tmux: `${sid}:@3.%3`, name: sid })
    }
    add(801, 'sid-a')
    add(802, 'sid-other-server', '/tmp/tmux-1/other')
    for (let i = 0; i < 3; i++) await f.tick()
    const q = () => fires(f).filter((r) => r.kind === 'question')
    assert.equal(q().length, 1)
    assert.equal(q()[0].subject_session_id, 'sid-a')
    assert.match(q()[0].text, /session_id="sid-a"/)
    // a declined session is never asked
    f.put(`${FLEET}/registry/launches.jsonl`, JSON.stringify({ v: 1, decision: 'not_in_fleet', session_id: 'sid-b', purpose: 'x' }) + '\n')
    add(803, 'sid-b')
    for (let i = 0; i < 3; i++) await f.tick()
    assert.equal(q().length, 1)
  } finally {
    mock.timers.reset()
  }
})

test('criterion 27 (mods side): a torn Claude record keeps the last good value; a crashed former main stays listed', async () => {
  const f = world()
  const { emit } = await load()
  f.put(`/proc/900/stat`, procStat(900, '9'))
  f.put(`${FLEET}/registry/launches.jsonl`, JSON.stringify({ v: 1, decision: 'child', name: 'kf', purpose: 'kf campaign' }) + '\n{"torn":')
  f.put(`${CFG}/sessions/900.json`, { pid: 900, sessionId: 'sid-kf', cwd: LAUNCH, procStart: '9', startedAt: 3000, status: 'busy', name: 'kf' })
  await emit(f.$, 'session.start')
  const child = () => f.get(`${FLEET}/registry/sessions.json`).sessions.find((s: any) => s.session_id === 'sid-kf')
  assert.equal(child().role, 'child')
  assert.equal(child().purpose, 'kf campaign')
  f.put(`${CFG}/sessions/900.json`, '{"pid": 900, "sessi') // torn mid-write
  await f.tick()
  assert.equal(child().record_stale, true)
  assert.equal(child().status, 'busy')
  // a former main whose record vanished while alive at the last snapshot: crashed
  const reg = f.get(`${FLEET}/registry/sessions.json`)
  reg.sessions.push({ ...reg.sessions[0], session_id: 'sid-old', pid: 555, role: 'main', active: false, alive: true })
  f.put(`${FLEET}/registry/sessions.json`, reg)
  await f.tick()
  const old = f.get(`${FLEET}/registry/sessions.json`).sessions.find((s: any) => s.session_id === 'sid-old')
  assert.equal(old.crashed, true)
  assert.equal(old.active, false)
})

test('criterion 6 (trigger side): counting, timing and registering make no model request', async () => {
  mock.timers.enable({ apis: ['Date'], now: 5_000_000 })
  try {
    const f = world()
    let fetches = 0
    f.$.http.fetch = async () => (fetches++, { ok: false })
    f.put(`${FLEET}/config/trigger.json`, { v: 1, kick: { interval_s: 3600 }, refresh: { every_n_tool_calls: 1000, every_n_turns: 1000 }, log: { every_n_tool_calls: 1000, every_n_turns: 1000, slow_interval_s: 86_400 } })
    const { emit } = await load()
    await emit(f.$, 'session.start')
    for (let i = 0; i < 100; i++) {
      await emit(f.$, 'tool.call', { tool: 'Bash' })
      if (i % 10 === 9) await emit(f.$, 'turn.complete', {})
      mock.timers.tick(30_000)
      await f.tick()
    }
    await f.flushAfter()
    assert.equal(f.prompts.length, 0, 'no prompt')
    assert.equal(fetches, 0, 'no HTTP request')
    assert.deepEqual([...new Set(f.runs.map((r) => r[0]))], ['mv'], 'no process but the atomic renames')
    assert.equal(f.get(`${FLEET}/state/trigger.json`).counts.tool_calls, 100)
  } finally {
    mock.timers.reset()
  }
})

test('child pins: every registered child (launched by name or adopted by session_id) is pinned to the fleet, also after its /clear; such a pin never activates', async () => {
  const f = world()
  const { emit } = await load()
  f.put(`/proc/900/stat`, procStat(900, '9'))
  f.put(`/proc/901/stat`, procStat(901, '8'))
  f.put(`${FLEET}/registry/launches.jsonl`, [
    { v: 1, decision: 'child', name: 'kf', purpose: 'kf campaign' },
    { v: 1, decision: 'child', session_id: 'sid-ad', purpose: 'adopted' },
  ].map((r) => JSON.stringify(r)).join('\n') + '\n')
  f.put(`${CFG}/sessions/900.json`, { pid: 900, sessionId: 'sid-kf', cwd: LAUNCH, procStart: '9', startedAt: 3000, status: 'idle', name: 'kf' })
  f.put(`${CFG}/sessions/901.json`, { pid: 901, sessionId: 'sid-ad', cwd: LAUNCH, procStart: '8', startedAt: 3100, status: 'idle' })
  await emit(f.$, 'session.start')
  for (const sid of ['sid-kf', 'sid-ad']) {
    const pin = f.get(`${PREFIX}/state/o11y/sessions/${sid}.json`)
    assert.equal(pin.fleet_dir, FLEET, sid)
    assert.equal(pin.role, 'child', sid)
  }
  f.put(`${CFG}/sessions/900.json`, { pid: 900, sessionId: 'sid-kf2', cwd: LAUNCH, procStart: '9', startedAt: 3000, status: 'idle', name: 'kf' }) // its /clear
  await f.tick()
  assert.deepEqual(f.get(`${PREFIX}/state/o11y/sessions/sid-kf2.json`).lineage, ['sid-kf'])

  const c = world({}) // the child's own trigger: no activation variable, its pin from the main
  c.put(`${PREFIX}/state/o11y/sessions/sid-main.json`, { v: 1, session_id: 'sid-main', fleet_dir: FLEET, role: 'child', pinned_at: '2026-11-23T00:00:00Z' })
  const child = await load()
  await child.emit(c.$, 'session.start')
  await child.emit(c.$, 'tool.call', { tool: 'Skill', skill: 'agent-sys-prior-work' })
  await c.flushAfter()
  assert.equal(c.timers.filter((t) => t.every).length, 0, 'no tick')
  assert.ok(!c.runs.some((r) => r[0] === 'mv'), 'nothing written')
})

test('requests: refresh every-N → extract + facts, log every-N → log; all bumps kept (spec §4.4, §4.5)', async () => {
  const f = world()
  const { emit } = await load()
  await emit(f.$, 'session.start')
  for (let i = 0; i < 4; i++) await emit(f.$, 'tool.call', { tool: 'Bash' })
  await emit(f.$, 'tool.call', { tool: 'Bash', agentId: 'sub-1' }) // a subagent's call is not counted
  await f.flushAfter()
  const req = f.get(`${FLEET}/requests/analysis.json`)
  assert.equal(req.extract.seq, 2)
  assert.equal(req.facts.seq, 2)
  assert.equal(req.log.seq, 1)
  assert.ok(f.runs.some((r) => r.includes('/plugins/agent-sys-o11y/server/o11y-run') && r.includes('extract')))
})
