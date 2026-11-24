// Harness of the live acceptance tests: one isolated run per makeRun() (own directory, tmux server,
// CLAUDE_CONFIG_DIR, prefix and fleet), driven through tmux. See live.test.ts for the variables it needs.
import { execFileSync } from 'node:child_process'
import { accessSync, constants, cpSync, existsSync, lstatSync, mkdirSync, readdirSync, readFileSync, symlinkSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'

export const HERE = resolve(import.meta.dirname, '../..')
export const PLUGINS = join(HERE, 'plugins')
export const PLUGIN_NAMES = ['agent-sys-o11y-door', 'agent-sys-o11y', 'agent-sys-o11y-dash']
const LIVE = process.env.O11Y_LIVE_DIR ?? ''
const BIN = process.env.CLAUDE_BIN ?? ''

export function skipReason(): string | false {
  if (!LIVE.startsWith('/') || !existsSync(LIVE)) return 'O11Y_LIVE_DIR (an existing absolute directory) not set'
  const st = lstatSync(LIVE)
  if (!st.isDirectory() || st.uid !== process.getuid!() || (st.mode & 0o077) !== 0) return 'O11Y_LIVE_DIR must be a directory you own with mode 0700 (not a symlink)'
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

export const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))
export async function waitFor<T>(what: string, fn: () => T | undefined | false, timeoutMs: number, stepMs = 1000, context?: () => string): Promise<T> {
  const end = Date.now() + timeoutMs
  for (;;) {
    const v = fn()
    if (v) return v
    if (Date.now() > end) throw new Error(`timed out after ${timeoutMs / 1000}s waiting for ${what}${context ? `\n${context()}` : ''}`)
    await sleep(stepMs)
  }
}
export const readJson = (p: string) => {
  try {
    return JSON.parse(readFileSync(p, 'utf8'))
  } catch {
    return undefined
  }
}

export interface RunOpts {
  flag?: string | null // AGENT_SYS_O11Y for every session of the run; null: unset
  trigger?: object // config/trigger.json (default: kick 60 s, packup every 2, no analysis runs)
  copy?: boolean // load a copy of this tree's plugins and protocol made in the run directory (`plugins` of the result)
  pluginDirs?: boolean // false: no --plugin-dir (the installed copies, if any, load)
}

// One run: its own directory, tmux server, config dir, prefix and fleet.
let runs = 0
export function makeRun(o: RunOpts = {}) {
  const id = `live-${new Date().toISOString().replace(/[^0-9]/g, '').slice(0, 14)}-${process.pid}-${++runs}`
  const dir = join(LIVE, id)
  const cfg = join(dir, 'cfg')
  const proj = join(dir, 'proj')
  const prefix = join(dir, 'prefix')
  const fleet = join(dir, 'fleet')
  const plugins = o.copy ? join(dir, 'o11y', 'plugins') : PLUGINS
  if (o.copy) for (const d of ['plugins', 'protocol']) cpSync(join(HERE, d), join(dir, 'o11y', d), { recursive: true, verbatimSymlinks: true })
  for (const d of [cfg, proj, join(prefix, 'share', 'o11y'), join(fleet, 'config'), join(fleet, 'facts'), join(fleet, 'logs')]) mkdirSync(d, { recursive: true, mode: 0o700 })
  if (process.env.O11Y_LIVE_VENV) symlinkSync(process.env.O11Y_LIVE_VENV, join(prefix, 'share', 'o11y', 'venv'))
  const key = process.env.ANTHROPIC_API_KEY ?? ''
  writeFileSync(
    join(cfg, '.claude.json'),
    JSON.stringify({ hasCompletedOnboarding: true, projects: { [proj]: { hasTrustDialogAccepted: true } }, customApiKeyResponses: { approved: key ? [key.slice(-20)] : [], rejected: [] } }),
  )
  const snippet = JSON.parse(readFileSync(join(plugins, 'agent-sys-o11y-dash', 'keybindings.snippet.json'), 'utf8'))
  writeFileSync(join(cfg, 'keybindings.json'), JSON.stringify({ bindings: snippet.bindings }))
  // No analysis runs, so the fixture facts below stay what the display shows (the jobs have their own tests).
  const never = { every_n_tool_calls: 1_000_000, every_n_turns: 1_000_000 }
  writeFileSync(join(fleet, 'config', 'trigger.json'), JSON.stringify(o.trigger ?? { v: 1, kick: { interval_s: 60 }, packup: { every_n_kicks: 2 }, refresh: never, log: { ...never, slow_interval_s: 86_400 } }))
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
    AGENT_SYS_HOME: prefix, AGENT_SYS_O11Y_FLEET_DIR: fleet,
  })
  if (o.flag === null) delete env.AGENT_SYS_O11Y
  else env.AGENT_SYS_O11Y = o.flag ?? 'true'
  const tmux = (...a: string[]) => execFileSync('tmux', ['-S', join(dir, 'tmux.sock'), ...a], { env, encoding: 'utf8' })
  let n = 0
  return {
    id, dir, cfg, proj, prefix, fleet, env, tmux, plugins,
    claude: (...a: string[]) => execFileSync(BIN, a, { env, cwd: proj, encoding: 'utf8' }),
    start(x: { env?: Record<string, string | null>; args?: string[] } = {}) {
      const name = `s${++n}`
      const debug = join(dir, `debug.${name}.txt`)
      const extra = Object.entries(x.env ?? {}).flatMap(([k, v]) => (v === null ? ['-u', k] : [`${k}=${v}`]))
      const argv = ['env', ...extra, BIN, '--model', process.env.O11Y_LIVE_MODEL ?? 'haiku', '--permission-mode', 'dontAsk', '--allowedTools', 'Bash(python3 -c:*)', '--debug-file', debug,
        ...(o.pluginDirs === false ? [] : PLUGIN_NAMES.flatMap((p) => ['--plugin-dir', join(plugins, p)])), ...(x.args ?? [])]
      tmux('new-session', '-d', '-s', name, '-x', '220', '-y', '50', '-c', proj, ...argv)
      const pid = Number(tmux('display', '-p', '-t', name, '#{pane_pid}').trim())
      return { name, debug, pid }
    },
    // The session id Claude Code records for a process (changes on /clear).
    sidOf: (pid: number): string | undefined => readJson(join(cfg, 'sessions', `${pid}.json`))?.sessionId,
    // Claude Code's own live status of a process (busy | idle | waiting); the registry's copy lags a tick.
    liveStatus: (pid: number): string | undefined => readJson(join(cfg, 'sessions', `${pid}.json`))?.status,
    kill: () => {
      try {
        tmux('kill-server')
      } catch {}
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
export function transcriptTexts(cfg: string, sid: string): string[] {
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
