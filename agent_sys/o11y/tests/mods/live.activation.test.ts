// Live acceptance criteria 5, 24 and 2 (spec §12; design §12 map): activation without the variable, the
// question about an unregistered session, and the skip of a kick due while one is queued. Same variables and
// isolation as live.test.ts.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, mkdirSync, readdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { makeRun, readJson, skipReason, sleep, waitFor } from './live_harness.ts'

const skip = skipReason()
const never = { every_n_tool_calls: 1_000_000, every_n_turns: 1_000_000 }
const TRIGGER = { v: 1, kick: { interval_s: 60 }, packup: { every_n_kicks: 2 }, refresh: never, log: { ...never, slow_interval_s: 86_400 } }

// A project skill whose name makes the trigger activate (agent-sys-*).
function addSkill(proj: string) {
  mkdirSync(join(proj, '.claude', 'skills', 'agent-sys-demo'), { recursive: true, mode: 0o700 })
  writeFileSync(join(proj, '.claude', 'skills', 'agent-sys-demo', 'SKILL.md'), '---\nname: agent-sys-demo\ndescription: Demo agent_sys skill for the o11y live tests. Use it when asked to.\n---\n\nReply with exactly: DEMO-SKILL-OK\n')
}
const fires = (fleet: string) => {
  const d = join(fleet, 'triggers')
  return existsSync(d) ? readdirSync(d).sort().map((n) => readJson(join(d, n))).filter(Boolean) : []
}

test('live: criteria 5 (user-invoked skill, /clear), 24 and 2 in a session started without AGENT_SYS_O11Y', { skip, timeout: 20 * 60_000 }, async (t) => {
  const run = makeRun({ flag: null, trigger: TRIGGER })
  t.diagnostic(`run directory ${run.dir}`)
  addSkill(run.proj)
  t.after(run.kill)
  const s = run.start()
  const sid0 = await waitFor('the session record', () => run.sidOf(s.pid), 60_000)
  const pin = (sid: string) => readJson(join(run.prefix, 'state', 'o11y', 'sessions', `${sid}.json`))

  await t.test('criterion 5: dormant until the user invokes an agent_sys skill, then main', async () => {
    await sleep(35_000) // more than one registry tick
    assert.equal(existsSync(join(run.fleet, 'registry', 'sessions.json')), false, 'nothing registered before the skill')
    run.type(s.name, '/agent-sys-demo')
    await waitFor('the registry to list the session as main', () => run.registry()?.main_session_id === sid0, 60_000)
    assert.equal(pin(sid0)?.role, 'main')
  })

  await t.test('criterion 5: still main after /clear, the pin carried over', async () => {
    await waitFor('idle', () => run.status(sid0) === 'idle', 120_000)
    run.type(s.name, '/clear')
    const sid1 = await waitFor('a new session id', () => (run.sidOf(s.pid) !== sid0 ? run.sidOf(s.pid) : undefined), 30_000)
    await waitFor('the registry to follow /clear', () => run.registry()?.main_session_id === sid1, 60_000)
    assert.deepEqual(pin(sid1)?.lineage, [sid0])
  })

  await t.test('criterion 24: one question about a hand-started session; none after it', async () => {
    const h = run.start()
    const hsid = await waitFor('the hand-started session record', () => run.sidOf(h.pid), 60_000)
    const asked = () => fires(run.fleet).filter((r) => r.kind === 'question' && r.subject_session_id === hsid)
    await waitFor('a question about it', () => asked().length > 0, 150_000)
    await sleep(75_000) // two more registry ticks
    assert.equal(asked().length, 1)
    assert.match(asked()[0].text, new RegExp(`session_id="${hsid}"`))
  })

  await t.test('criterion 2: a kick due while one is queued is skipped and recorded; never two queued', async () => {
    const main = run.registry()!.main_session_id
    await waitFor('idle', () => run.status(main) === 'idle', 180_000)
    const before = fires(run.fleet).length
    run.type(s.name, 'Run the shell command `python3 -c "import time; time.sleep(150)"` with the Bash tool in the foreground (run_in_background false) and wait for it, then reply DONE.')
    await waitFor('a skipped_pending record', () => fires(run.fleet).slice(before).some((r) => r.outcome === 'skipped_pending'), 240_000)
    const during = fires(run.fleet).slice(before).filter((r) => r.kind !== 'question')
    const firstSkip = during.findIndex((r) => r.outcome === 'skipped_pending')
    assert.equal(during.slice(0, firstSkip).filter((r) => r.outcome === 'submitted').length, 1, 'exactly one queued before the skip')
  })
})

test('live: criterion 5 (model-invoked skill) in a session started without AGENT_SYS_O11Y', { skip, timeout: 10 * 60_000 }, async (t) => {
  const run = makeRun({ flag: null, trigger: TRIGGER })
  t.diagnostic(`run directory ${run.dir}`)
  addSkill(run.proj)
  t.after(run.kill)
  const s = run.start()
  const sid = await waitFor('the session record', () => run.sidOf(s.pid), 60_000)
  await sleep(5_000)
  run.type(s.name, 'Use the agent-sys-demo skill now (Skill tool).')
  await waitFor('the registry to list the session as main', () => run.registry()?.main_session_id === sid, 120_000)
})
