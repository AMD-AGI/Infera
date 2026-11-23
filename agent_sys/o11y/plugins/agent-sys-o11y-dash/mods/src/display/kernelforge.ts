// The KernelForge tab (spec §7.3): hot operators and share, campaigns, per-operator stage. Pure.
import { dur, pct, type Line } from './fmt.ts'

export function kernelforge(k: any, wide: boolean): Line[] {
  const out: Line[] = []
  for (const h of k.hot_ops.slice(0, wide ? 3 : 1)) {
    out.push([`hot ops (${h.role}${wide && h.scope ? `, ${h.scope}` : ''}):`])
    for (const o of h.ops.slice(0, wide ? 12 : 5)) out.push([`  ${o.share_pct.toFixed(1).padStart(5)}%  ${o.name}`])
  }
  out.push([`campaigns (${k.campaigns.length}):`])
  for (const c of k.campaigns) {
    const col = c.status === 'running' ? 'cyan' : c.status === 'stale' || c.problem ? 'yellow' : undefined
    out.push([`  ${c.operator_ids.join(',')} ${c.status} ${pct(c.gain_pct)} (${pct(c.gain_pct_per_hour)}/h) ${dur(c.age_s)}`, col])
    if (wide) out.push([`      ${c.campaign_root}${c.iterations != null ? ` · it ${c.iterations} kept ${c.kept ?? '-'}` : ''}${c.problem ? ` · ${c.problem}` : ''}`])
  }
  out.push(['ops:'])
  for (const o of wide ? k.ops : k.ops.slice(0, 6)) {
    out.push([`  ${o.op_id} ${o.stage}${o.e2e_verdict ? ` · e2e ${o.e2e_verdict} ${pct(o.e2e_delta_pct)}` : ''}`, o.e2e_verdict ? 'green' : undefined])
  }
  if (wide) for (const pr of k.problems ?? []) out.push([`! ${pr}`, 'yellow'])
  return out
}
