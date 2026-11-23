// A fake of the mods API (`$`) over a temporary directory, for deterministic tests of the mods without Claude
// Code. Absolute paths are mapped under the directory (`/proc/...` included), so fixtures can stand in for
// Claude Code's session records and for processes. `$.process.run` performs `mv` and records everything else.
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, statSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'

export interface Fake {
  $: any
  root: string // the directory absolute paths are mapped into
  path: (abs: string) => string
  put: (abs: string, content: string | object) => void
  get: (abs: string) => any
  runs: string[][]
  prompts: string[]
  logs: string[]
  timers: { ms: number; fn: () => unknown; every: boolean }[]
  tick: () => Promise<void> // run every `every` timer once
  flushAfter: () => Promise<void> // run and drop the `after` timers
  env: Record<string, string>
  session: { id: string; root: string }
}

export function fakeApi(root: string, env: Record<string, string> = {}, session = { id: 'sid-main', root: '/work/proj' }): Fake {
  const path = (abs: string) => join(root, abs)
  const runs: string[][] = []
  const prompts: string[] = []
  const logs: string[] = []
  const timers: Fake['timers'] = []
  const el = (type: string) => (props: any) => ({ type, ...props })
  const $ = {
    plugin: { root: '/plugins/agent-sys-o11y', name: 'agent-sys-o11y' },
    env: { get: async (k: string) => env[k] },
    session: { id: async () => session.id, root: async () => session.root, cwd: async () => session.root },
    fs: {
      exists: async (p: string) => existsSync(path(p)),
      stat: async (p: string) => {
        const s = statSync(path(p))
        return { kind: s.isDirectory() ? 'directory' : 'file', size: s.size, mtimeMs: s.mtimeMs, isLink: false }
      },
      read: async (p: string, o?: { offset: number; length: number }) => {
        const b = readFileSync(path(p))
        return (o ? b.subarray(o.offset, o.offset + o.length) : b).toString('utf8')
      },
      write: async (p: string, text: string) => {
        mkdirSync(dirname(path(p)), { recursive: true, mode: 0o700 })
        writeFileSync(path(p), text, { mode: 0o600 })
      },
      list: async (p: string) =>
        readdirSync(path(p), { withFileTypes: true }).map((d) => ({ name: d.name, kind: d.isDirectory() ? 'directory' : 'file', size: 0, isLink: false })),
    },
    process: {
      run: async (argv: string[]) => {
        runs.push(argv)
        if (argv[0] === 'mv') renameSync(path(argv[3]), path(argv[4]))
        return { exitCode: 0, stdout: '', stderr: '' }
      },
    },
    clock: {
      now: async () => Date.now(),
      every: (ms: number, fn: () => unknown) => (timers.push({ ms, fn, every: true }), { cancel: () => timers.splice(timers.findIndex((t) => t.fn === fn), 1) }),
      after: (ms: number, fn: () => unknown) => (timers.push({ ms, fn, every: false }), { cancel: () => {} }),
    },
    prompt: { submit: async ({ text }: { text: string }) => (prompts.push(text), {}) },
    ui: {
      log: (t: string) => logs.push(t),
      invalidate: () => {},
      open: async () => ({ isPlaced: true }),
      close: async () => {},
      resolve: () => ({ Box: el('Box'), Text: el('Text'), Button: el('Button'), Input: el('Input'), Select: el('Select') }),
    },
    http: { fetch: async () => ({ ok: false }) },
    command: { register: async () => {} },
  }
  return {
    $, root, path, runs, prompts, logs, timers, env, session,
    put: (abs, content) => {
      mkdirSync(dirname(path(abs)), { recursive: true })
      writeFileSync(path(abs), typeof content === 'string' ? content : JSON.stringify(content))
    },
    get: (abs) => JSON.parse(readFileSync(path(abs), 'utf8')),
    tick: async () => {
      for (const t of timers.filter((t) => t.every)) await t.fn()
    },
    flushAfter: async () => {
      for (const t of timers.filter((t) => !t.every)) {
        timers.splice(timers.indexOf(t), 1)
        await t.fn()
      }
    },
  }
}

// A /proc/<pid>/stat line whose field 22 (process start) is `start`.
export const procStat = (pid: number, start: string) => `${pid} (2.1.325) S 1 ${pid} ${pid} 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 ${start} 0 0`

// Text of every Text element in a rendered tree, in order.
export function texts(node: any): string[] {
  if (!node || typeof node !== 'object') return []
  const own = node.type === 'Text' ? [(node.children ?? []).filter((c: any) => typeof c === 'string').join('')] : []
  return [...own, ...(node.children ?? []).flatMap((c: any) => texts(c))]
}

// Collect a mod's hooks: register(on) → hooks[event] = [{ matcher, fn }]; emit runs them like the engine.
export function hooksOf(register: (on: any) => void) {
  const hooks: Record<string, { matcher: any; fn: any }[]> = {}
  register((event: string, a: any, b?: any) => (hooks[event] ??= []).push(b ? { matcher: a, fn: b } : { matcher: null, fn: a }))
  const emit = async ($: any, event: string, e: any = {}) => {
    let out: any
    for (const h of hooks[event] ?? []) {
      if (h.matcher && !Object.entries(h.matcher).every(([k, v]) => e[k] === v)) continue
      out = await h.fn($, e, async (x: any) => x)
    }
    return out
  }
  return { hooks, emit }
}
