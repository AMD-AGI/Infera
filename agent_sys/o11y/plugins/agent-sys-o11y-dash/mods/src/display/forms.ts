// Configuration forms (spec §7.5): generated from the config schemas; a value is range- and type-checked
// against the whole document before the file is written (whole file, tmp + rename).
import { P, SCHEMAS, nowIso, readDoc, validate, withDefaults, writeDoc } from '../../../protocol/ts/index.ts'

export const FORM = 'o11y-cfg'
type Target = 'trigger' | 'analysis' | 'display'

interface Field {
  path: string[]
  schema: any
}

let root = ''
let target: Target = 'trigger'
let draft: any = { v: 1 }
let note = ''

function fieldsOf(schema: any, at: string[] = []): Field[] {
  const out: Field[] = []
  for (const [k, ps] of Object.entries<any>(schema.properties ?? {})) {
    if (k === 'v' || k === 'updated_at') continue
    if (ps.type === 'object' && ps.properties) out.push(...fieldsOf(ps, [...at, k]))
    else out.push({ path: [...at, k], schema: ps })
  }
  return out
}

const getAt = (obj: any, path: string[]) => path.reduce((o, k) => (o == null ? undefined : o[k]), obj)

function setAt(obj: any, path: string[], value: any) {
  let o = obj
  for (const k of path.slice(0, -1)) o = o[k] ??= {}
  o[path[path.length - 1]] = value
}

export async function openForm($: any, fleetRoot: string, t: Target) {
  root = fleetRoot
  target = t
  const r = await readDoc($, `${root}/${P.config(t)}`)
  draft = r.value && typeof r.value === 'object' ? r.value : { v: 1 }
  note = r.problem ? `current file unreadable: ${r.problem}` : `config/${t}.json`
  await $.ui.open({ id: FORM, title: `o11y config: ${t}`, focus: true, closeOnEscape: true, rows: 30, columns: 90 })
}

function parse(f: Field, raw: string): { value?: any; problem?: string } {
  const t = Array.isArray(f.schema.type) ? f.schema.type[0] : f.schema.type
  if (t !== 'number' && t !== 'integer') return { value: raw }
  const n = Number(raw)
  if (raw.trim() === '' || !isFinite(n) || (t === 'integer' && !Number.isInteger(n))) return { problem: `${f.path.join('.')}: not a ${t}` }
  return { value: n }
}

async function save($: any, f: Field, value: any) {
  const next = JSON.parse(JSON.stringify(draft))
  setAt(next, f.path, value)
  next.v = 1
  next.updated_at = nowIso()
  const probs = validate(SCHEMAS[`config.${target}`], next)
  if (probs.length) note = `not saved: ${probs[0]}`
  else {
    await writeDoc($, `${root}/${P.config(target)}`, next)
    draft = next
    note = `saved ${f.path.join('.')} = ${JSON.stringify(value).slice(0, 60)}`
  }
  $.ui.invalidate('ui.render')
}

export async function renderForm($: any, e: any) {
  const { Box, Text, Input, Select } = $.ui.resolve(e)
  const schema = SCHEMAS[`config.${target}`]
  const full = withDefaults(schema, draft)
  const rows = fieldsOf(schema).map((f, i) => {
    const key = `cfg-${target}-${f.path.join('.')}`
    const label = f.path.join('.')
    const cur = getAt(full, f.path)
    const isBool = f.schema.type === 'boolean'
    const choices = isBool ? [true, false] : Array.isArray(f.schema.enum) ? f.schema.enum.filter((x: any) => x !== null) : null
    if (choices) {
      return Select({
        key, label, value: String(cur), ...(i === 0 ? { autoFocus: true } : {}),
        options: choices.map((c: any) => ({ value: String(c), label: String(c) })),
        onSelect: (v: string) => save($, f, isBool ? v === 'true' : v),
      })
    }
    return Input({
      key, label, value: String(cur ?? ''), submitLabel: 'save', ...(i === 0 ? { autoFocus: true } : {}),
      onSubmit: (raw: string) => {
        const p = parse(f, raw)
        if (p.problem) {
          note = `not saved: ${p.problem}`
          $.ui.invalidate('ui.render')
          return
        }
        return save($, f, p.value)
      },
    })
  })
  return Box({ flexDirection: 'column', children: [Text({ dimColor: true, wrap: 'truncate-end', children: [`${note} · Tab/Enter to edit, Esc to close`] }), ...rows] })
}
