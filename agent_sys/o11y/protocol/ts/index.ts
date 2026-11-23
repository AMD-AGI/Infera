// The mods' protocol layer (design §2, §4, §5): the bundled JSON Schemas, a validator for the subset the
// mods use, schema defaults, the whole-file write rule, and fleet-root / slug resolution. The trigger and
// display mods import only this module; esbuild bundles it into each.
import configTrigger from '../schemas/config.trigger.v1.schema.json' with { type: 'json' }
import configAnalysis from '../schemas/config.analysis.v1.schema.json' with { type: 'json' }
import configDisplay from '../schemas/config.display.v1.schema.json' with { type: 'json' }
import factsGeneral from '../schemas/facts.general.v1.schema.json' with { type: 'json' }
import factsDetail from '../schemas/facts.detail.v1.schema.json' with { type: 'json' }
import factsKernelforge from '../schemas/facts.kernelforge.v1.schema.json' with { type: 'json' }
import factsUsage from '../schemas/facts.usage.v1.schema.json' with { type: 'json' }
import registrySessions from '../schemas/registry.sessions.v1.schema.json' with { type: 'json' }
import registryLaunch from '../schemas/registry.launch.v1.schema.json' with { type: 'json' }
import triggerFire from '../schemas/trigger_fire.v1.schema.json' with { type: 'json' }
import queueItem from '../schemas/queue.v1.schema.json' with { type: 'json' }
import event from '../schemas/event.v1.schema.json' with { type: 'json' }

export const SCHEMAS: Record<string, any> = {
  'config.trigger': configTrigger,
  'config.analysis': configAnalysis,
  'config.display': configDisplay,
  'facts.general': factsGeneral,
  'facts.detail': factsDetail,
  'facts.kernelforge': factsKernelforge,
  'facts.usage': factsUsage,
  'registry.sessions': registrySessions,
  'registry.launch': registryLaunch,
  trigger_fire: triggerFire,
  queue: queueItem,
  event,
}

// Paths under the fleet root (design §5)
export const P = {
  config: (name: string) => `config/${name}.json`,
  sessions: 'registry/sessions.json',
  launches: 'registry/launches.jsonl',
  events: 'logs/events.jsonl',
  triggers: 'triggers',
  queue: 'queue',
  facts: (tab: string) => `facts/${tab}.json`,
  triggerState: 'state/trigger.json',
  active: (sid: string) => `state/active/${sid}.json`,
}

export const LEDGER_TOOL = 'mcp__plugin_agent-sys-o11y_o11y__ledger_append'

// ---- locations (design §4) ----

// Claude Code's project-directory rule (protocol/test_vectors/slug.json): every code point outside
// [A-Za-z0-9] becomes '-'; a name over 200 characters keeps its first 200 and appends '-' and the first 12 hex
// digits of the SHA-256 of the directory's UTF-8 bytes.
export async function slug(dir: string): Promise<string> {
  const name = dir.replace(/[^A-Za-z0-9]/gu, '-')
  if (name.length <= 200) return name
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(dir)))
  const hex = Array.from(digest, (b) => b.toString(16).padStart(2, '0')).join('')
  return `${name.slice(0, 200)}-${hex.slice(0, 12)}`
}

export interface Env {
  AGENT_SYS_O11Y_FLEET_DIR?: string
  AGENT_SYS_HOME?: string
  HOME?: string
  CLAUDE_CONFIG_DIR?: string
}

function trimSlash(p: string): string {
  return p.length > 1 ? p.replace(/\/+$/, '') : p
}

// The prefix rule of env_mgr/prefix.py: $AGENT_SYS_HOME, else $HOME/.infera_agent_sys. AGENT_SYS_HOME must be
// absolute and canonical (a mod cannot resolve symlinks); a relative value is ignored.
export function prefixOf(env: Env): string {
  const h = env.AGENT_SYS_HOME
  if (h && h.startsWith('/')) return trimSlash(h)
  return `${trimSlash(env.HOME || '/tmp')}/.infera_agent_sys`
}

// The launch directory is used as given (no trimming), as o11y_core.fleet does.
export async function fleetDirOf(env: Env, launchDir: string): Promise<string> {
  const f = env.AGENT_SYS_O11Y_FLEET_DIR
  if (f && f.startsWith('/')) return f
  return `${prefixOf(env)}/state/o11y/fleets/${await slug(launchDir)}`
}

export function configDirOf(env: Env): string {
  const c = env.CLAUDE_CONFIG_DIR
  return c ? trimSlash(c) : `${trimSlash(env.HOME || '/tmp')}/.claude`
}

// Read the variables above through the mods API (names must be literal strings).
export async function readEnv($: any): Promise<Env> {
  return {
    AGENT_SYS_O11Y_FLEET_DIR: await $.env.get('AGENT_SYS_O11Y_FLEET_DIR'),
    AGENT_SYS_HOME: await $.env.get('AGENT_SYS_HOME'),
    HOME: await $.env.get('HOME'),
    CLAUDE_CONFIG_DIR: await $.env.get('CLAUDE_CONFIG_DIR'),
  }
}

// ---- small helpers ----

export function nowIso(ms?: number): string {
  return new Date(ms ?? Date.now()).toISOString()
}

export function newId(prefix: string): string {
  const t = new Date().toISOString().replace(/[-:.]/g, '').replace('T', '-').slice(0, 15)
  return `${prefix}-${t}-${Math.random().toString(36).slice(2, 8)}`
}

// Interpolated values in automated prompts: no control characters or line breaks, length-capped.
export function sanitize(value: unknown, max: number): string {
  const s = String(value ?? '').replace(/[\u0000-\u001f\u007f-\u009f\u2028\u2029]+/g, ' ').trim()
  return s.length > max ? s.slice(0, max - 1) + '…' : s
}

// ---- validation: JSON Schema subset ----

function typeOf(x: unknown): string {
  if (x === null) return 'null'
  if (Array.isArray(x)) return 'array'
  if (typeof x === 'number') return Number.isInteger(x) ? 'integer' : 'number'
  return typeof x
}

function resolve(root: any, node: any): any {
  if (node && typeof node.$ref === 'string' && node.$ref.startsWith('#/$defs/')) {
    return resolve(root, root.$defs[node.$ref.slice(8)])
  }
  return node
}

// type, enum, const, required, properties, additionalProperties:false, items, minimum, maximum, minLength,
// minItems, pattern and local $ref. Conditional keywords (allOf/if/then) are left to o11y_core.
export function validate(schema: any, value: unknown, root: any = schema, at = ''): string[] {
  const s = resolve(root, schema)
  if (!s || typeof s !== 'object') return []
  const where = at || '/'
  const t = typeOf(value)
  if (s.type) {
    const types = Array.isArray(s.type) ? s.type : [s.type]
    if (!types.some((ty: string) => ty === t || (ty === 'number' && t === 'integer'))) {
      return [`${where}: expected ${types.join('|')}, got ${t}`]
    }
  }
  const out: string[] = []
  if (s.const !== undefined && value !== s.const) out.push(`${where}: must be ${JSON.stringify(s.const)}`)
  if (s.enum && !s.enum.includes(value)) out.push(`${where}: must be one of ${s.enum.join(', ')}`)
  if (typeof value === 'number') {
    if (typeof s.minimum === 'number' && value < s.minimum) out.push(`${where}: must be >= ${s.minimum}`)
    if (typeof s.maximum === 'number' && value > s.maximum) out.push(`${where}: must be <= ${s.maximum}`)
  }
  if (typeof value === 'string') {
    if (typeof s.minLength === 'number' && value.length < s.minLength) out.push(`${where}: too short`)
    if (typeof s.pattern === 'string' && !new RegExp(s.pattern).test(value)) out.push(`${where}: does not match ${s.pattern}`)
  }
  if (t === 'object') {
    const obj = value as Record<string, unknown>
    for (const k of s.required ?? []) if (!(k in obj)) out.push(`${at}/${k}: required`)
    for (const [k, v] of Object.entries(obj)) {
      const ps = s.properties?.[k]
      if (ps) out.push(...validate(ps, v, root, `${at}/${k}`))
      else if (s.additionalProperties === false) out.push(`${at}/${k}: unknown key`)
    }
  }
  if (t === 'array') {
    const arr = value as unknown[]
    if (typeof s.minItems === 'number' && arr.length < s.minItems) out.push(`${where}: needs ${s.minItems}+ items`)
    if (s.items) arr.forEach((v, i) => out.push(...validate(s.items, v, root, `${at}/${i}`)))
  }
  return out
}

// Fill schema defaults into an object, recursively for nested objects.
export function withDefaults(schema: any, value: any, root: any = schema): any {
  const s = resolve(root, schema)
  if (!s || s.type !== 'object' || !s.properties) return value
  const out: any = { ...(value ?? {}) }
  for (const [k, ps0] of Object.entries<any>(s.properties)) {
    const ps = resolve(root, ps0)
    if (out[k] === undefined && ps.default !== undefined) out[k] = JSON.parse(JSON.stringify(ps.default))
    if (ps.type === 'object' && ps.properties) out[k] = withDefaults(ps, out[k], root)
  }
  return out
}

// ---- file rules (design §5) ----

export interface ReadResult {
  value: any
  problem: string | null
  mtimeMs: number
}

// A whole-file document; a missing file is value=null with no problem. An invalid one keeps its value and
// names the problem; callers treat it as absent.
export async function readDoc($: any, path: string, schemaName?: string): Promise<ReadResult> {
  try {
    if (!(await $.fs.exists(path))) return { value: null, problem: null, mtimeMs: 0 }
    const st = await $.fs.stat(path)
    const value = JSON.parse(await $.fs.read(path))
    const probs = schemaName ? validate(SCHEMAS[schemaName], value) : []
    return { value, problem: probs.length ? `${path}: ${probs[0]}` : null, mtimeMs: st.mtimeMs }
  } catch (err: any) {
    return { value: null, problem: `${path}: ${err.message}`, mtimeMs: 0 }
  }
}

// Whole-file write: tmp in the same directory, then rename. A mod has no rename, so the rename is `mv -f`
// with both paths as arguments. $.fs.write creates parent directories (0700) and files (0600).
export async function writeDoc($: any, path: string, value: unknown): Promise<void> {
  const tmp = `${path}.tmp.${Math.random().toString(36).slice(2, 8)}`
  await $.fs.write(tmp, JSON.stringify(value, null, 1) + '\n')
  const r = await $.process.run(['mv', '-f', '--', tmp, path], { timeoutMs: 10000 })
  if (r.exitCode !== 0) throw new Error(`mv ${tmp}: ${r.stderr.trim()}`)
}

// The valid, newline-terminated lines of a JSONL file, newest last, from at most the last maxBytes.
export async function readJsonl($: any, path: string, maxBytes = 1 << 20): Promise<any[]> {
  if (!(await $.fs.exists(path))) return []
  const st = await $.fs.stat(path)
  const offset = Math.max(0, st.size - maxBytes)
  const text: string = offset > 0 ? await $.fs.read(path, { offset, length: st.size - offset }) : await $.fs.read(path)
  const lines = text.split('\n')
  if (offset > 0) lines.shift()
  lines.pop() // after the last '\n': empty, or a partial line
  const out: any[] = []
  for (const l of lines) {
    if (!l) continue
    try {
      out.push(JSON.parse(l))
    } catch {}
  }
  return out
}

// Start a detached one-shot process: `sh -c` only redirects stdio (a mod cannot); every value, the log path
// included, is an argument, never part of the script.
export const DETACH_SCRIPT = 'log=$1; shift; setsid -f "$@" >>"$log" 2>&1 </dev/null'

export function detachArgv(log: string, argv: string[]): string[] {
  return ['sh', '-c', DETACH_SCRIPT, 'sh', log, ...argv]
}
