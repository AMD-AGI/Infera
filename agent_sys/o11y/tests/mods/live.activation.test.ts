// Live acceptance criteria 5, 24 and 2 (spec §12; design §12 map): activation without the variable, the
// question about an unregistered session, and the skip of a kick due while one is queued. Same variables and
// isolation as live.test.ts.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { makeRun, readJson, skipReason, sleep, transcriptTexts, waitFor } from './live_harness.ts'

const skip = skipReason()
const never = { every_n_tool_calls: 1_000_000, every_n_turns: 1_000_000 }
const TRIGGER = { v: 1, kick: { interval_s: 60 }, packup: { every_n_kicks: 2 }, refresh: never, log: { ...never, slow_interval_s: 86_400 } }

// A project skill whose name makes the trigger activate (agent-sys-*).
function addSkill(proj: string) {
  mkdirSync(join(proj, '.claude', 'skills', 'agent-sys-demo'), { recursive: true, mode: 0o700 })
  writeFileSync(join(proj, '.claude', 'skills', 'agent-sys-demo', 'SKILL.md'), '---\nname: agent-sys-demo\ndescription: Demo agent_sys skill for the o11y live tests. Use it when asked to.\n---\n\nReply with exactly: DEMO-SKILL-OK\n')
}
// Every line of a session's transcript.
function transcriptLines(cfg: string, sid: string): any[] {
  const projects = join(cfg, 'projects')
  for (const d of existsSync(projects) ? readdirSync(projects) : []) {
    const p = join(projects, d, `${sid}.jsonl`)
    if (existsSync(p)) return readFileSync(p, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l))
  }
  return []
}
const fires = (fleet: string) => {
  const d = join(fleet, 'triggers')
  return existsSync(d) ? readdirSync(d).sort().map((n) => readJson(join(d, n))).filter(Boolean) : []
}

test('live: criteria 5 (user-invoked skill, /clear), 24 (with adoption) and 2 in a session started without AGENT_SYS_O11Y', { skip, timeout: 20 * 60_000 }, async (t) => {
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

    // The main answers "child" through the door; the adopted session then gets the door's tools (PR #252 F4).
    const main = run.registry()!.main_session_id
    await waitFor('the main idle', () => run.liveStatus(s.pid) === 'idle', 180_000)
    run.type(s.name, `Call the session_register tool with decision child, session_id ${hsid} and purpose "live adoption test", then reply DONE.`)
    await waitFor('the registry to list it as a child', () => (readJson(join(run.fleet, 'registry', 'sessions.json'))?.sessions ?? []).some((x: any) => x.session_id === hsid && x.role === 'child'), 120_000)
    await sleep(35_000) // the door's membership poll (30 s) and its tools/list_changed
    run.type(h.name, 'Call the fleet_status tool of the o11y MCP server now, then reply with exactly: STATUS-SEEN')
    // fleet_status answered it as a child of the fleet (the tool was listed to it)
    const answered = () => transcriptLines(run.cfg, hsid).some((x: any) => x.type === 'user' && Array.isArray(x.message?.content) &&
      x.message.content.some((b: any) => b.type === 'tool_result' && !b.is_error &&
        /"ok": true[\s\S]*"role": "child"/.test((Array.isArray(b.content) ? b.content : []).map((c: any) => c.text ?? '').join(''))))
    await waitFor('fleet_status to answer the adopted session as a child', answered, 180_000)
    await waitFor('its reply', () => transcriptTexts(run.cfg, hsid).some((x) => x.trim() === 'STATUS-SEEN'), 60_000)
    assert.equal(run.registry()!.main_session_id, main)
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
