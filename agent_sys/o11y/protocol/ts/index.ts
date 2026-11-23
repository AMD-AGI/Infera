// The mods' protocol layer (design §2, §4, §5): bundled JSON Schemas, a validator for a declared keyword
// subset, schema defaults, the config reader rule, the whole-file write rule, the fleet root (pin first),
// the slug, and sanitisation of session strings. The trigger and display mods import only this module.
import common from '../schemas/common.v1.schema.json' with { type: 'json' }
import configTrigger from '../schemas/config.trigger.v1.schema.json' with { type: 'json' }
import configAnalysis from '../schemas/config.analysis.v1.schema.json' with { type: 'json' }
import configDisplay from '../schemas/config.display.v1.schema.json' with { type: 'json' }
import factsGeneral from '../schemas/facts.general.v1.schema.json' with { type: 'json' }
import factsDetail from '../schemas/facts.detail.v1.schema.json' with { type: 'json' }
import factsKernelforge from '../schemas/facts.kernelforge.v1.schema.json' with { type: 'json' }
import registrySessions from '../schemas/registry.sessions.v1.schema.json' with { type: 'json' }
import triggerFire from '../schemas/trigger_fire.v1.schema.json' with { type: 'json' }
import requests from '../schemas/requests.v1.schema.json' with { type: 'json' }
import otelHealth from '../schemas/otel_health.v1.schema.json' with { type: 'json' }
import sessionPin from '../schemas/session_pin.v1.schema.json' with { type: 'json' }

// The schemas the mods validate. Others (event, registry.launch) are read leniently by field.
export const SCHEMAS: Record<string, any> = {
  'config.trigger': configTrigger,
  'config.analysis': configAnalysis,
  'config.display': configDisplay,
  'facts.general': factsGeneral,
  'facts.detail': factsDetail,
  'facts.kernelforge': factsKernelforge,
  'registry.sessions': registrySessions,
  trigger_fire: triggerFire,
  requests,
  otel_health: otelHealth,
  session_pin: sessionPin,
}

// Keywords validate() enforces or may ignore as annotations. A schema above using anything else fails a test.
// `format` is allowed only beside an explicit `pattern`, which carries the check.
export const SUPPORTED_KEYWORDS = [
  '$schema', '$id', '$ref', '$defs', 'title', 'description', 'default', 'type', 'enum', 'const', 'required',
  'properties', 'additionalProperties', 'items', 'minimum', 'exclusiveMinimum', 'minLength', 'maxLength',
  'minItems', 'pattern', 'format', 'allOf',
]

export const P = {
  config: (name: string) => `config/${name}.json`,
  sessions: 'registry/sessions.json',
  launches: 'registry/launches.jsonl',
  events: 'logs/events.jsonl',
  triggers: 'triggers',
  requests: 'requests/analysis.json',
  otelHealth: 'otel/health.json',
  facts: (tab: string) => `facts/${tab}.json`,
  triggerState: 'state/trigger.json',
}

// ---- locations (design §4) ----

export interface Env {
  AGENT_SYS_O11Y_FLEET_DIR?: string
  AGENT_SYS_HOME?: string
  HOME?: string
  CLAUDE_CONFIG_DIR?: string
}

const trim = (p: string) => (p.length > 1 ? p.replace(/\/+$/, '') : p)

// Claude Code's project-directory rule (protocol/test_vectors/slug.json): every code point outside
// [A-Za-z0-9] becomes '-'; over 200 characters keeps 200 + '-' + 12 hex of SHA-256 of the UTF-8 bytes.
export async function slug(dir: string): Promise<string> {
  const name = dir.replace(/[^A-Za-z0-9]/gu, '-')
  if (name.length <= 200) return name
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(dir)))
  return `${name.slice(0, 200)}-${Array.from(digest, (b) => b.toString(16).padStart(2, '0')).join('').slice(0, 12)}`
}

// env_mgr/prefix.py: $AGENT_SYS_HOME (absolute and canonical; a mod cannot resolve symlinks), else
// $HOME/.infera_agent_sys.
export function prefixOf(env: Env): string {
  const h = env.AGENT_SYS_HOME
  return h && h.startsWith('/') ? trim(h) : `${trim(env.HOME || '/tmp')}/.infera_agent_sys`
}

export const pinPath = (env: Env, sid: string) => `${prefixOf(env)}/state/o11y/sessions/${sid}.json`

export function configDirOf(env: Env): string {
  return env.CLAUDE_CONFIG_DIR ? trim(env.CLAUDE_CONFIG_DIR) : `${trim(env.HOME || '/tmp')}/.claude`
}

// The root without a pin: AGENT_SYS_O11Y_FLEET_DIR, else <prefix>/state/o11y/fleets/<slug(launch dir)>.
export async function derivedFleetDir(env: Env, launchDir: string): Promise<string> {
  const f = env.AGENT_SYS_O11Y_FLEET_DIR
  if (f && f.startsWith('/')) return f
  return `${prefixOf(env)}/state/o11y/fleets/${await slug(launchDir)}`
}

export async function readEnv($: any): Promise<Env> {
  return {
    AGENT_SYS_O11Y_FLEET_DIR: await $.env.get('AGENT_SYS_O11Y_FLEET_DIR'),
    AGENT_SYS_HOME: await $.env.get('AGENT_SYS_HOME'),
    HOME: await $.env.get('HOME'),
    CLAUDE_CONFIG_DIR: await $.env.get('CLAUDE_CONFIG_DIR'),
  }
}

// The fleet root of a session: its pin first (design §4), then the derivation.
export async function fleetDirOf($: any, env: Env, sid: string): Promise<{ fleet: string; pinned: any | null }> {
  const pin = (await readDoc($, pinPath(env, sid), 'session_pin')).value
  if (pin) return { fleet: pin.fleet_dir, pinned: pin }
  return { fleet: await derivedFleetDir(env, String(await $.session.root())), pinned: null }
}

// ---- small helpers ----

export const nowIso = (ms?: number) => new Date(ms ?? Date.now()).toISOString()

export function newId(prefix: string): string {
  const t = new Date().toISOString().replace(/[-:.]/g, '').replace('T', '-').slice(0, 15)
  return `${prefix}-${t}-${Math.random().toString(36).slice(2, 8)}`
}

// Session-controlled strings in automated prompts: allowlist charset (everything else becomes '_'), a fixed
// cap, and only as quoted data after a fixed sentence (`key="value"`). Fails closed: a non-string or
// non-number becomes ''.
export const SANITIZE_MAX = 200
export function sanitize(value: unknown): string {
  if (typeof value !== 'string' && typeof value !== 'number') return ''
  const s = String(value).replace(/[^A-Za-z0-9._:/@%+-]/g, '_')
  return s.length > SANITIZE_MAX ? s.slice(0, SANITIZE_MAX) : s
}
export const quoted = (key: string, value: unknown) => `${key}="${sanitize(value)}"`

// ---- validation ----

function typeOf(x: unknown): string {
  if (x === null) return 'null'
  if (Array.isArray(x)) return 'array'
  if (typeof x === 'number') return Number.isInteger(x) ? 'integer' : 'number'
  return typeof x
}

// A local ref resolves in the current document; "common.v1.schema.json#/$defs/x" in common.v1.
function deref(node: any, root: any): [any, any] {
  while (node && typeof node.$ref === 'string') {
    const [file, frag] = node.$ref.split('#')
    if (file) root = file === 'common.v1.schema.json' ? common : null
    if (!root || !frag?.startsWith('/$defs/')) throw new Error(`unresolvable $ref ${node.$ref}`)
    node = root.$defs[frag.slice(7)]
  }
  return [node, root]
}

export function validate(schema: any, value: unknown, root: any = schema, at = ''): string[] {
  const [s, r] = deref(schema, root)
  if (!s || typeof s !== 'object') return []
  const where = at || '/'
  const out: string[] = []
  for (const sub of s.allOf ?? []) out.push(...validate(sub, value, r, at))
  const t = typeOf(value)
  if (s.type) {
    const types = Array.isArray(s.type) ? s.type : [s.type]
    if (!types.some((ty: string) => ty === t || (ty === 'number' && t === 'integer'))) return [...out, `${where}: expected ${types.join('|')}, got ${t}`]
  }
  if (s.const !== undefined && value !== s.const) out.push(`${where}: must be ${JSON.stringify(s.const)}`)
  if (s.enum && !s.enum.includes(value)) out.push(`${where}: must be one of ${s.enum.join(', ')}`)
  if (typeof value === 'number') {
    if (typeof s.minimum === 'number' && value < s.minimum) out.push(`${where}: must be >= ${s.minimum}`)
    if (typeof s.exclusiveMinimum === 'number' && value <= s.exclusiveMinimum) out.push(`${where}: must be > ${s.exclusiveMinimum}`)
  }
  if (typeof value === 'string') {
    const n = [...value].length
    if (typeof s.minLength === 'number' && n < s.minLength) out.push(`${where}: shorter than ${s.minLength}`)
    if (typeof s.maxLength === 'number' && n > s.maxLength) out.push(`${where}: longer than ${s.maxLength}`)
    if (typeof s.pattern === 'string' && !new RegExp(s.pattern, 'u').test(value)) out.push(`${where}: does not match ${s.pattern}`)
  }
  if (t === 'object') {
    const obj = value as Record<string, unknown>
    for (const k of s.required ?? []) if (!(k in obj)) out.push(`${at}/${k}: required`)
    for (const [k, v] of Object.entries(obj)) {
      if (s.properties?.[k]) out.push(...validate(s.properties[k], v, r, `${at}/${k}`))
      else if (s.additionalProperties === false) out.push(`${at}/${k}: unknown key${didYouMean(k, Object.keys(s.properties ?? {}))}`)
    }
  }
  if (t === 'array') {
    const arr = value as unknown[]
    if (typeof s.minItems === 'number' && arr.length < s.minItems) out.push(`${where}: needs ${s.minItems}+ items`)
    if (s.items) arr.forEach((v, i) => out.push(...validate(s.items, v, r, `${at}/${i}`)))
  }
  return out
}

function distance(a: string, b: string): number {
  const d = Array.from({ length: b.length + 1 }, (_, i) => i)
  for (let i = 1; i <= a.length; i++) {
    let prev = d[0]
    d[0] = i
    for (let j = 1; j <= b.length; j++) {
      const cur = d[j]
      d[j] = Math.min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1))
      prev = cur
    }
  }
  return d[b.length]
}

function didYouMean(key: string, known: string[]): string {
  const best = known.map((k) => [k, distance(key, k)] as const).sort((x, y) => x[1] - y[1])[0]
  return best && best[1] <= Math.max(2, key.length / 3) ? ` (did you mean "${best[0]}"?)` : ''
}

export function withDefaults(schema: any, value: any, root: any = schema): any {
  const [s, r] = deref(schema, root)
  if (!s || s.type !== 'object' || !s.properties) return value
  const out: any = { ...(value ?? {}) }
  for (const [k, ps0] of Object.entries<any>(s.properties)) {
    const [ps] = deref(ps0, r)
    if (out[k] === undefined && ps.default !== undefined) out[k] = JSON.parse(JSON.stringify(ps.default))
    if (ps.type === 'object' && ps.properties) out[k] = withDefaults(ps, out[k], r)
  }
  return out
}

// ---- files (design §5) ----

export interface ReadResult {
  value: any
  problem: string | null
  mtimeMs: number
}

// A whole-file document. Missing: value null, no problem. Invalid: value null and the problem.
export async function readDoc($: any, path: string, schemaName?: string): Promise<ReadResult> {
  try {
    if (!(await $.fs.exists(path))) return { value: null, problem: null, mtimeMs: 0 }
    const st = await $.fs.stat(path)
    const value = JSON.parse(await $.fs.read(path))
    const probs = schemaName ? validate(SCHEMAS[schemaName], value) : []
    return probs.length ? { value: null, problem: `${path}: ${probs[0]}`, mtimeMs: st.mtimeMs } : { value, problem: null, mtimeMs: st.mtimeMs }
  } catch (err: any) {
    return { value: null, problem: `${path}: ${err.message}`, mtimeMs: 0 }
  }
}

// Spec §8.4: an invalid config (an unknown key included) is reported once with a did-you-mean and the last
// valid config stays in effect (else the schema defaults).
export interface ConfigState {
  cfg: any
  mtimeMs: number
  reported: string | null // the problem last reported
  report: string | null // a problem to report now (once per distinct problem)
}
export async function loadConfig($: any, path: string, name: string, prev: ConfigState | null): Promise<ConfigState> {
  const base = prev ?? { cfg: withDefaults(SCHEMAS[name], { v: 1 }), mtimeMs: -1, reported: null, report: null }
  const r = await readDoc($, path, name)
  if (r.mtimeMs === base.mtimeMs) return { ...base, report: null }
  if (r.problem) {
    const again = base.reported === r.problem
    return { cfg: base.cfg, mtimeMs: r.mtimeMs, reported: r.problem, report: again ? null : r.problem }
  }
  return { cfg: withDefaults(SCHEMAS[name], r.value ?? { v: 1 }), mtimeMs: r.mtimeMs, reported: null, report: null }
}

// Whole file: tmp in the same directory, then rename (`mv -f`, both paths as arguments: a mod has no
// rename). $.fs.write creates files 0600 and directories 0700.
export async function writeDoc($: any, path: string, value: unknown): Promise<void> {
  const tmp = `${path}.tmp.${Math.random().toString(36).slice(2, 8)}`
  await $.fs.write(tmp, JSON.stringify(value, null, 1) + '\n')
  const r = await $.process.run(['mv', '-f', '--', tmp, path], { timeoutMs: 10000 })
  if (r.exitCode !== 0) throw new Error(`mv ${tmp}: ${r.stderr.trim()}`)
}

// The complete lines of a JSONL file that parse, newest last, from at most the last maxBytes.
export async function readJsonl($: any, path: string, maxBytes = 1 << 20): Promise<any[]> {
  if (!(await $.fs.exists(path))) return []
  const st = await $.fs.stat(path)
  const offset = Math.max(0, st.size - maxBytes)
  const text: string = offset > 0 ? await $.fs.read(path, { offset, length: st.size - offset }) : await $.fs.read(path)
  const lines = text.split('\n')
  if (offset > 0) lines.shift()
  lines.pop()
  const out: any[] = []
  for (const l of lines) {
    try {
      if (l) out.push(JSON.parse(l))
    } catch {}
  }
  return out
}

// A detached one-shot: `sh -c` only redirects stdio; every value, the log path included, is an argument.
export const DETACH_SCRIPT = 'log=$1; shift; setsid -f "$@" >>"$log" 2>&1 </dev/null'
export const detachArgv = (log: string, argv: string[]) => ['sh', '-c', DETACH_SCRIPT, 'sh', log, ...argv]
