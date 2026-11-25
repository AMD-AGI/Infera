// The KernelForge tab (spec §7.3): hot operators and share per role, campaigns, per-operator stage. Pure.
import { dur, num, pct, type Line } from './fmt.ts'

// A campaign the ledger reports but this host cannot read (source "reported"): its reason, in brackets.
export const notReadable = (c: any) => `reported, not readable here (${String(c.problem ?? '').replace(/^reported, not readable here:\s*/, '') || 'unknown'})`

// "Forge campaigns M (r running, u not readable here)".
export function campaignsLine(k: any): string {
  const running = k.campaigns.filter((c: any) => c.status === 'running').length
  const reported = k.campaigns.filter((c: any) => c.source === 'reported').length
  return `Forge campaigns ${k.campaigns.length} (${running} running${reported ? `, ${reported} not readable here` : ''})`
}

// Hot-op profiles by role: prefill and decode always (a role without a profile is "not focused"); any other
// role under common/mix, shown only when present.
export function byRole(hot: any[]): [string, any[]][] {
  const main = ['prefill', 'decode']
  const mix = hot.filter((h) => !main.includes(h.role))
  return [...main.map((r): [string, any[]] => [r, hot.filter((h) => h.role === r)]), ...(mix.length ? [['common/mix', mix] as [string, any[]]] : [])]
}

export function kernelforge(k: any, wide: boolean): Line[] {
  const out: Line[] = []
  for (const [role, hs] of byRole(k.hot_ops)) {
    out.push([`hot ops (${role}):`])
    if (!hs.length) out.push(['  not focused', 'dim'])
    for (const h of hs.slice(0, wide ? 3 : 1)) {
      if (wide && (hs.length > 1 || h.scope)) out.push([`  ${h.profile_id}${role === 'common/mix' ? ` [${h.role}]` : ''}${h.scope ? ` (${h.scope})` : ''}`, 'dim'])
      for (const o of h.ops.slice(0, wide ? 12 : 5)) out.push([`  ${o.share_pct.toFixed(1).padStart(5)}%  ${o.name}`])
    }
  }
  out.push([`Forge campaigns (${k.campaigns.length}):`])
  if (!k.campaigns.length) out.push(['  none recorded', 'dim'])
  for (const c of k.campaigns) {
    const col = c.status === 'stale' || c.problem ? 'yellow' : undefined // cyan is for section headers only
    if (c.source === 'reported') out.push([`  ${c.operator_ids.join(',')} ${c.status}${c.iterations != null ? ` it ${c.iterations}` : ''}${c.best_speedup != null ? ` best ${num(c.best_speedup, 2)}x` : ''} · ${notReadable(c)}`, 'yellow'])
    else out.push([`  ${c.operator_ids.join(',')} ${c.status} ${pct(c.gain_pct)} (${pct(c.gain_pct_per_hour)}/h) ${dur(c.age_s)}`, col])
    if (wide) out.push([`      ${c.campaign_root}${c.iterations != null ? ` · it ${c.iterations} kept ${c.kept ?? '-'}` : ''}${c.problem ? ` · ${c.problem}` : ''}`])
  }
  out.push([`operators tracked (${k.ops.length}):`])
  for (const o of wide ? k.ops : k.ops.slice(0, 6)) {
    out.push([`  ${o.op_id} ${o.stage}${o.e2e_verdict ? ` · e2e ${o.e2e_verdict} ${pct(o.e2e_delta_pct)}` : ''}`, o.e2e_verdict ? 'green' : undefined])
  }
  if (wide) for (const pr of k.problems ?? []) out.push([`! ${pr}`, 'yellow'])
  return out
}
