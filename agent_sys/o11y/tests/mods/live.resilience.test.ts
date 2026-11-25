// Live acceptance criterion 22 (spec §12; design §12 map), the parts live.test.ts does not cover: a crash of
// the hooks worker caused by the package, and the package's plugins installed but disabled. In both the
// session must keep working. Same variables and isolation as live.test.ts.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { HERE, PLUGIN_NAMES, makeRun, skipReason, sleep, transcriptTexts, waitFor } from './live_harness.ts'

const skip = skipReason()
const replied = (cfg: string, sid: string, word: string) => transcriptTexts(cfg, sid).some((x) => x.trim() === word)

test('live: criterion 22 — the display mod blocks the hooks worker; the session keeps working', { skip, timeout: 10 * 60_000 }, async (t) => {
  // A copy of the package whose display bundle never returns from ui.render once a fault file exists.
  const faulty = makeRun({ copy: true })
  t.diagnostic(`run directory ${faulty.dir}`)
  t.after(faulty.kill)
  const fault = join(faulty.dir, 'FAULT')
  const bundle = join(faulty.plugins, 'agent-sys-o11y-dash', 'mods', 'dist', 'display.js')
  const src = readFileSync(bundle, 'utf8')
  const at = src.indexOf('function register(on) {')
  assert.ok(at > 0, 'register() in the display bundle')
  const hook = `\n  on("ui.render", async ($, e, next) => {\n    if (await $.fs.exists(${JSON.stringify(fault)})) for (;;) {}\n    return next(e);\n  });`
  writeFileSync(bundle, src.slice(0, at + 23) + hook + src.slice(at + 23))
  const s = faulty.start()
  const sid = await waitFor('the registry to list the main', () => faulty.registry()?.main_session_id, 90_000)
  await waitFor('the band', () => /o11y · /.test(faulty.screen(s.name)), 20_000)
  writeFileSync(fault, '1')
  await waitFor('Claude Code to unload the blocking mod', () => /crashed the hooks worker|hooks worker are off/.test(readFileSync(s.debug, 'utf8')), 120_000)
  faulty.type(s.name, 'Reply with exactly: LIVE-OK-W')
  await waitFor('a reply after the worker crash', () => replied(faulty.cfg, sid, 'LIVE-OK-W'), 120_000)
  assert.ok(existsSync(`/proc/${s.pid}`), 'the session is alive')
})

test('live: criterion 22 — the package installed but its mods plugins disabled; the session works and nothing is written', { skip, timeout: 10 * 60_000 }, async (t) => {
  const run = makeRun({ pluginDirs: false })
  t.diagnostic(`run directory ${run.dir}`)
  run.claude('plugin', 'marketplace', 'add', HERE)
  for (const p of PLUGIN_NAMES) run.claude('plugin', 'install', `${p}@agent-sys-local`)
  for (const p of ['agent-sys-o11y', 'agent-sys-o11y-dash']) run.claude('plugin', 'disable', `${p}@agent-sys-local`)
  t.after(run.kill)
  const s = run.start()
  const sid = await waitFor('the session record', () => run.sidOf(s.pid), 60_000)
  await sleep(5_000)
  run.type(s.name, 'Reply with exactly: LIVE-OK-D')
  await waitFor('a reply', () => replied(run.cfg, sid, 'LIVE-OK-D'), 120_000)
  await sleep(35_000) // more than one registry tick
  assert.equal(existsSync(join(run.fleet, 'registry')), false, 'no registry written')
  assert.equal(existsSync(join(run.fleet, 'triggers')), false, 'no automated prompt')
  assert.doesNotMatch(run.screen(s.name), /o11y · /, 'no band')
})
