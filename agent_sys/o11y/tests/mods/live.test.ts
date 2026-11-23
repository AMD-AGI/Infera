// Live acceptance criteria (spec §12; design §12 map) in a real, isolated interactive Claude Code session
// driven through tmux. Skipped unless all of these are given:
//   O11Y_LIVE_DIR  an existing directory you own; every file of the run is created in a new subdirectory and
//                  nothing is deleted
//   CLAUDE_BIN     the Claude Code binary (absolute path)
//   ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN (plus ANTHROPIC_BASE_URL / ANTHROPIC_CUSTOM_HEADERS if needed)
// Optional: O11Y_LIVE_MODEL (default haiku), O11Y_LIVE_VENV (the o11y venv; without it the door's MCP server
// runs on python3 from PATH). The session runs on its own tmux server and its own CLAUDE_CONFIG_DIR, so no
// user setting, plugin or tmux session is touched; only processes carrying this run's marker are signalled.
// Put O11Y_LIVE_DIR on local disk outside any project (Claude Code reads the settings, hooks and CLAUDE.md of a
// project it lies in) and not directly under $HOME, e.g. /var/tmp/<user>/o11y_live.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { accessSync, constants, existsSync, mkdirSync, readdirSync, readFileSync, statSync, symlinkSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'

const HERE = resolve(import.meta.dirname, '../..')
const PLUGINS = join(HERE, 'plugins')
const LIVE = process.env.O11Y_LIVE_DIR ?? ''
const BIN = process.env.CLAUDE_BIN ?? ''

function skipReason(): string | false {
  if (!LIVE.startsWith('/') || !existsSync(LIVE)) return 'O11Y_LIVE_DIR (an existing absolute directory) not set'
  try {
    accessSync(BIN, constants.X_OK)
  } catch {
    return 'CLAUDE_BIN not set or not executable'
  }
  if (!process.env.ANTHROPIC_API_KEY && !process.env.ANTHROPIC_AUTH_TOKEN) return 'no ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN'
  try {
    execFileSync('tmux', ['-V'])
  } catch {
    return 'tmux not found'
  }
  return false
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))
async function waitFor<T>(what: string, fn: () => T | undefined | false, timeoutMs: number, stepMs = 1000, context?: () => string): Promise<T> {
  const end = Date.now() + timeoutMs
  for (;;) {
    const v = fn()
    if (v) return v
    if (Date.now() > end) throw new Error(`timed out after ${timeoutMs / 1000}s waiting for ${what}${context ? `\n${context()}` : ''}`)
    await sleep(stepMs)
  }
}
const readJson = (p: string) => {
  try {
    return JSON.parse(readFileSync(p, 'utf8'))
  } catch {
    return undefined
  }
}

// One run: its own directory, tmux server, config dir, prefix and fleet.
function makeRun() {
  const id = `live-${new Date().toISOString().replace(/[^0-9]/g, '').slice(0, 14)}-${process.pid}`
  const dir = join(LIVE, id)
  const cfg = join(dir, 'cfg')
  const proj = join(dir, 'proj')
  const prefix = join(dir, 'prefix')
  const fleet = join(dir, 'fleet')
  for (const d of [cfg, proj, join(prefix, 'share', 'o11y'), join(fleet, 'config'), join(fleet, 'facts'), join(fleet, 'logs')]) mkdirSync(d, { recursive: true })
  if (process.env.O11Y_LIVE_VENV) symlinkSync(process.env.O11Y_LIVE_VENV, join(prefix, 'share', 'o11y', 'venv'))
  const key = process.env.ANTHROPIC_API_KEY ?? ''
  writeFileSync(
    join(cfg, '.claude.json'),
    JSON.stringify({ hasCompletedOnboarding: true, projects: { [proj]: { hasTrustDialogAccepted: true } }, customApiKeyResponses: { approved: key ? [key.slice(-20)] : [], rejected: [] } }),
  )
  const snippet = JSON.parse(readFileSync(join(PLUGINS, 'agent-sys-o11y-dash', 'keybindings.snippet.json'), 'utf8'))
  writeFileSync(join(cfg, 'keybindings.json'), JSON.stringify({ bindings: snippet.bindings }))
  // No analysis runs, so the fixture facts below stay what the display shows (the jobs have their own tests).
  const never = { every_n_tool_calls: 1_000_000, every_n_turns: 1_000_000 }
  writeFileSync(join(fleet, 'config', 'trigger.json'), JSON.stringify({ v: 1, kick: { interval_s: 60 }, packup: { every_n_kicks: 2 }, refresh: never, log: { ...never, slow_interval_s: 86_400 } }))
  writeFileSync(join(proj, 'mission.md'), 'Live test of the o11y plugins. There is no work: answer every automated prompt with one short sentence and no tool use; do what the user asks.\n')
  const now = new Date().toISOString()
  const h = { v: 1, generated_at: now, stale_after_s: 900, sources: ['fixture'] }
  writeFileSync(join(fleet, 'facts', 'general.json'), JSON.stringify({ ...h, task: { task_id: 't', title: 'live', metric: { name: 'tput', unit: 'tok/s', direction: 'higher' }, baseline: { value: 100 }, target: { value: 200 } }, best: { value: 150, run_ids: ['r1'], vs_baseline_pct: 50, of_target_pct: 75 }, spend: { tokens_new: 2_000_000, tokens_cache_read: 9_000_000, wall_s: 3600 }, progress: { gain_abs: 50, gain_pct: 50 } }))
  writeFileSync(join(fleet, 'facts', 'detail.json'), JSON.stringify({ ...h, current_runs: [{ run_id: 'r2', theme: 'EP8 decode' }], recent_runs: [], next: [{ theme: 'MTP depth 3' }], machines: [] }))
  writeFileSync(join(fleet, 'facts', 'kernelforge.json'), JSON.stringify({ ...h, hot_ops: [{ profile_id: 'p', role: 'decode', ops: [{ name: 'fused_moe', share_pct: 31.2 }] }], campaigns: [], ops: [] }))
  writeFileSync(join(fleet, 'logs', 'events.jsonl'), JSON.stringify({ v: 1, ts: now, severity_text: 'WARN', severity_number: 13, event_name: 'o11y.machine.lost', body: 'm137 reclaimed', attributes: {} }) + '\n')

  const env: Record<string, string> = { ...(process.env as Record<string, string>) }
  for (const k of Object.keys(env)) if (k.startsWith('CLAUDE_CODE_') || k === 'CLAUDECODE' || k === 'CLAUDE_PID' || k === 'CLAUDE_EFFORT' || k === 'TMUX' || k === 'TMUX_PANE') delete env[k]
  Object.assign(env, {
    O11Y_LIVE_RUN: id, // the marker: every process of the run inherits it
    CLAUDE_CONFIG_DIR: cfg, DISABLE_AUTOUPDATER: '1', CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: '1', CLAUDE_CODE_ENABLE_TELEMETRY: '0',
    AGENT_SYS_HOME: prefix, AGENT_SYS_O11Y: 'true', AGENT_SYS_O11Y_FLEET_DIR: fleet,
  })
  const tmux = (...a: string[]) => execFileSync('tmux', ['-S', join(dir, 'tmux.sock'), ...a], { env, encoding: 'utf8' })
  let n = 0
  return {
    id, dir, cfg, proj, prefix, fleet, env, tmux,
    claude: (...a: string[]) => execFileSync(BIN, a, { env, cwd: proj, encoding: 'utf8' }),
    start() {
      const name = `s${++n}`
      const debug = join(dir, `debug.${name}.txt`)
      const argv = [BIN, '--model', process.env.O11Y_LIVE_MODEL ?? 'haiku', '--permission-mode', 'dontAsk', '--allowedTools', 'Bash(python3 -c:*)', '--debug-file', debug,
        ...['agent-sys-o11y-door', 'agent-sys-o11y', 'agent-sys-o11y-dash'].flatMap((p) => ['--plugin-dir', join(PLUGINS, p)])]
      tmux('new-session', '-d', '-s', name, '-x', '220', '-y', '50', '-c', proj, ...argv)
      return { name, debug, pid: Number(tmux('display', '-p', '-t', name, '#{pane_pid}').trim()) }
    },
    screen: (name: string) => {
      try {
        return tmux('capture-pane', '-p', '-t', name)
      } catch {
        return ''
      }
    },
    keys: (name: string, ...k: string[]) => tmux('send-keys', '-t', name, ...k),
    type: (name: string, text: string) => (tmux('send-keys', '-t', name, '-l', text), tmux('send-keys', '-t', name, 'Enter')),
    // Processes of this run (by the marker in their environment), except the tmux server: pid → cmdline.
    procs() {
      const out = new Map<number, string>()
      for (const p of readdirSync('/proc').filter((x) => /^\d+$/.test(x))) {
        try {
          if (!readFileSync(`/proc/${p}/environ`, 'latin1').includes(`O11Y_LIVE_RUN=${id}\0`)) continue
          const cmd = readFileSync(`/proc/${p}/cmdline`, 'utf8').split('\0').join(' ').trim()
          if (!cmd.startsWith('tmux')) out.set(Number(p), cmd)
        } catch {}
      }
      return out
    },
    registry: () => readJson(join(fleet, 'registry', 'sessions.json')),
    status: (sid: string) => readJson(join(fleet, 'registry', 'sessions.json'))?.sessions?.find((x: any) => x.session_id === sid)?.status,
  }
}

// User-visible text of a transcript (user prompts and assistant replies), one entry per message.
function transcriptTexts(cfg: string, sid: string): string[] {
  const projects = join(cfg, 'projects')
  if (!existsSync(projects)) return []
  for (const d of readdirSync(projects)) {
    const p = join(projects, d, `${sid}.jsonl`)
    if (!existsSync(p)) continue
    return readFileSync(p, 'utf8').split('\n').filter(Boolean).flatMap((l) => {
      try {
        const c = JSON.parse(l).message?.content
        return [typeof c === 'string' ? c : Array.isArray(c) ? c.filter((b: any) => b.type === 'text').map((b: any) => b.text).join('') : '']
      } catch {
        return []
      }
    })
  }
  return []
}

const skip = skipReason()

test('live: criteria 1, 17, 18, 22, 29 and 19 in one isolated fleet session', { skip, timeout: 20 * 60_000 }, async (t) => {
  const run = makeRun()
  t.diagnostic(`run directory ${run.dir}`)
  // An installed copy of every plugin, so that criterion 29 can see the development copies override it.
  run.claude('plugin', 'marketplace', 'add', HERE)
  for (const p of ['agent-sys-o11y-door', 'agent-sys-o11y', 'agent-sys-o11y-dash']) run.claude('plugin', 'install', `${p}@agent-sys-local`)
  const s = run.start()
  t.after(() => {
    try {
      run.tmux('kill-server')
    } catch {}
  })
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
      ['g', /tokens 2\.0M new \+ 9\.0M cache reads/], ['d', /next: MTP depth 3/], ['k', /31\.2%\s+fused_moe/], ['l', /level ≥/],
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
      for (const p of ['agent-sys-o11y-door', 'agent-sys-o11y', 'agent-sys-o11y-dash']) assert.ok(log.includes(`Plugin "${p}" from --plugin-dir overrides installed version`), p)
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
