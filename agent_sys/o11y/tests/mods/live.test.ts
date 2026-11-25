// Live acceptance criteria (spec §12; design §12 map) in a real, isolated interactive Claude Code session
// driven through tmux. Skipped unless all of these are given:
//   O11Y_LIVE_DIR  an existing directory you own, mode 0700, not a symlink (/var/tmp/<user>.o11y_live, made with
//                  `mkdir -m 700`); every file of the run is created in a new subdirectory and nothing is deleted
//   CLAUDE_BIN     the Claude Code binary (absolute path)
//   ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN (plus ANTHROPIC_BASE_URL / ANTHROPIC_CUSTOM_HEADERS if needed)
// Optional: O11Y_LIVE_MODEL (default haiku), O11Y_LIVE_VENV (the o11y venv; without it the door's MCP server
// runs on python3 from PATH). The session runs on its own tmux server and its own CLAUDE_CONFIG_DIR, so no
// user setting, plugin or tmux session is touched; only processes carrying this run's marker are signalled.
// Put O11Y_LIVE_DIR on local disk outside any project (Claude Code reads the settings, hooks and CLAUDE.md of a
// project it lies in) and not directly under $HOME: /var/tmp/<user>.o11y_live, mode 0700.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { HERE, PLUGIN_NAMES, makeRun, readJson, skipReason, sleep, transcriptTexts, waitFor } from './live_harness.ts'

const skip = skipReason()

test('live: criteria 1, 17, 18, 22, 29 and 19 in one isolated fleet session', { skip, timeout: 20 * 60_000 }, async (t) => {
  const run = makeRun()
  t.diagnostic(`run directory ${run.dir}`)
  // An installed copy of every plugin, so that criterion 29 can see the development copies override it.
  run.claude('plugin', 'marketplace', 'add', HERE)
  for (const p of PLUGIN_NAMES) run.claude('plugin', 'install', `${p}@agent-sys-local`)
  const s = run.start()
  t.after(run.kill)
  try {
    const sid = await waitFor('the registry to list the main', () => run.registry()?.main_session_id, 90_000)

    await t.test('criterion 17: the band appears within one registry period plus one poll', async () => {
      await waitFor('the band', () => /o11y · tput 150 \+50\.0% \(tgt 200\)/.test(run.screen(s.name)), 15_000 + 5_000)
    })

    await t.test('criterion 1: the kick is submitted at its due time; the history holds the recorded text', async () => {
      const kick = await waitFor('a submitted kick', () => {
        const dir = join(run.fleet, 'triggers')
        return existsSync(dir) && readdirSync(dir).map((n) => readJson(join(dir, n))).find((r) => r?.kind === 'kick' && r.outcome === 'submitted')
      }, 150_000)
      assert.ok(kick.text.startsWith('[o11y automated kick] '))
      // Claude Code records a plugin's prompt wrapped in its own attribution lines
      await waitFor('the kick in the history', () => transcriptTexts(run.cfg, sid).some((x) => x.includes(kick.text)), 30_000)
      const cfg = readJson(join(run.fleet, 'config', 'trigger.json'))
      writeFileSync(join(run.fleet, 'config', 'trigger.json'), JSON.stringify({ ...cfg, kick: { interval_s: 3600 } }))
    })

    const chords: [string, RegExp][] = [
      ['0', /── config in effect/], ['g', /tokens 2\.0M new \+ 9\.0M cache reads/], ['d', /next: MTP depth 3/], ['k', /31\.2%\s+fused_moe/], ['l', /level ≥/],
      ['t', /config\/trigger\.json/], ['a', /config\/analysis\.json/], ['v', /config\/display\.json/], ['q', /o11y: press a key/],
    ]
    const chordRound = async (when: string) => {
      for (const [k, want] of chords) {
        await waitFor(`a quiet screen before ctrl+q ${k}`, () => !/o11y: press a key|Tab\/Enter to edit/.test(run.screen(s.name)), 5_000, 300).catch(() => run.keys(s.name, 'Escape'))
        run.keys(s.name, 'C-q')
        run.keys(s.name, k)
        await waitFor(`ctrl+q ${k} (${when})`, () => want.test(run.screen(s.name)), 5_000, 300, () => run.screen(s.name))
        if ('tavq'.includes(k)) run.keys(s.name, 'Escape')
      }
      for (const want of [/\bnarrow\b/, /\bwide\b/]) {
        run.keys(s.name, 'C-q')
        run.keys(s.name, 'm')
        await waitFor(`ctrl+q m (${when})`, () => want.test(run.screen(s.name)), 5_000, 300, () => run.screen(s.name))
      }
      run.keys(s.name, 'C-q')
      run.keys(s.name, 'b')
      await waitFor(`ctrl+q b hides the band (${when})`, () => !/o11y · tput/.test(run.screen(s.name)), 5_000, 300)
      run.keys(s.name, 'C-q')
      run.keys(s.name, 'b')
      await waitFor(`ctrl+q b shows the band (${when})`, () => /o11y · tput/.test(run.screen(s.name)), 5_000, 300)
    }

    await t.test('criterion 18: every chord reaches its target while idle', async () => {
      await waitFor('the session to be idle', () => run.status(sid) === 'idle', 120_000)
      await chordRound('idle')
    })

    await t.test('criterion 18: every chord reaches its target while busy', async () => {
      run.type(s.name, 'Run the shell command `python3 -c "import time; time.sleep(90)"` with the Bash tool in the foreground (run_in_background false) and wait for it, then reply DONE.')
      await waitFor('the session to be busy', () => run.status(sid) === 'busy', 60_000)
      await chordRound('busy')
      assert.equal(run.status(sid), 'busy', 'still busy after the round')
    })

    await t.test('criterion 22: killing the MCP server (and any analysis job) leaves the session working', async () => {
      await waitFor('the busy turn to end', () => run.status(sid) === 'idle', 150_000)
      const victims = [...run.procs()].filter(([, c]) => /o11y_door\.serve|o11y-run|o11y_job/.test(c))
      assert.ok(victims.some(([, c]) => c.includes('o11y_door.serve')), 'the door MCP server is running')
      for (const [pid] of victims) process.kill(pid, 'SIGKILL')
      run.type(s.name, 'Reply with exactly: LIVE-OK-22')
      await waitFor('a reply after the kills', () => transcriptTexts(run.cfg, sid).some((x) => x.includes('LIVE-OK-22') && !x.startsWith('Reply')), 120_000)
    })

    await t.test('criterion 29 (mods side): protocol files are owner-only; the development plugins override the installed ones', async () => {
      const written = [join(run.fleet, 'registry', 'sessions.json'), join(run.fleet, 'state', 'trigger.json'), join(run.prefix, 'state', 'o11y', 'sessions', `${sid}.json`), ...readdirSync(join(run.fleet, 'triggers')).map((n) => join(run.fleet, 'triggers', n))]
      for (const p of written) assert.equal(statSync(p).mode & 0o777, 0o600, p)
      for (const d of [join(run.fleet, 'registry'), join(run.fleet, 'triggers'), join(run.fleet, 'state')]) assert.equal(statSync(d).mode & 0o777, 0o700, d)
      const log = readFileSync(s.debug, 'utf8')
      for (const p of PLUGIN_NAMES) assert.ok(log.includes(`Plugin "${p}" from --plugin-dir overrides installed version`), p)
    })

    await t.test('criterion 19: after /exit no process of the package remains (an analysis run may finish its budget)', async () => {
      await waitFor('the session to be idle', () => run.status(sid) === 'idle', 120_000)
      run.type(s.name, '/exit')
      await waitFor('the main to exit', () => !existsSync(`/proc/${s.pid}`), 30_000)
      await sleep(10_000)
      const left = [...run.procs()].filter(([, c]) => !/o11y-run|o11y_job/.test(c))
      assert.deepEqual(left, [])
    })
  } finally {
    if (existsSync(`/proc/${s.pid}`)) process.kill(s.pid, 'SIGTERM')
  }

  await t.test('criterion 19: after SIGKILL of the main no process of the package remains (an analysis run may finish its budget)', async () => {
    const k = run.start()
    try {
      await waitFor('the second main', () => run.registry()?.sessions?.some((x: any) => x.role === 'main' && x.active && x.pid === k.pid), 90_000)
      process.kill(k.pid, 'SIGKILL')
      await sleep(20_000)
      const left = [...run.procs()].filter(([, c]) => !/o11y-run|o11y_job/.test(c))
      assert.deepEqual(left, [])
    } finally {
      if (existsSync(`/proc/${k.pid}`)) process.kill(k.pid, 'SIGTERM')
    }
  })
})
