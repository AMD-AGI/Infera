// The all tab (tab 0, the default; spec §7.3): a brief of every other tab, then the configuration in effect,
// so the user need not switch tabs. Pure.
import { dur, num, pct, tok, type Line } from './fmt.ts'
import { byRole, campaignsLine } from './kernelforge.ts'
import { nowLines } from './detail.ts'
import { machineLines } from './general.ts'
import { log } from './log.ts'

export interface Brief { general: any; detail: any; kernelforge: any; problems: Record<string, string | null>; ages: Record<string, string | null>
  events: any[]; trigger: any; analysis: any; taskBook: string }

// A section header, set apart by a blank line (a space: an empty Text takes no row) from the section before it.
function head(out: Line[], name: string, age: string | null, problem: string | null): Line[] {
  return [...(out.length ? [[' '] as Line] : []), [`── ${name}${problem ? ` · ${problem}` : age ? ` · ${age}` : ''}`, 'cyan']]
}

export function all(b: Brief): Line[] {
  const out: Line[] = []
  const g = b.general
  out.push(...head(out, 'general', b.ages.general, b.problems.general))
  if (g) {
    const t = g.task
    out.push(t ? [`${t.metric.name}: base ${num(t.baseline.value)} · best ${num(g.best?.value)} ${pct(g.best?.vs_baseline_pct)} · target ${t.target ? num(t.target.value) : '-'}`, g.best ? 'green' : undefined] : ['task: not recorded yet', 'yellow'])
    out.push([`tokens ${tok(g.spend.tokens_new)}+${tok(g.spend.tokens_cache_read)}c in ${dur(g.spend.wall_s)} · ${pct(g.progress.per_hour_pct)}/h`])
    out.push(...machineLines(g, 4))
  }
  const d = b.detail
  out.push(...head(out, 'detail', b.ages.detail, b.problems.detail))
  if (d) {
    out.push(...nowLines(d))
    const r = d.recent_runs[0]
    if (r) out.push([`last ${r.status} ${num(r.value)} ${pct(r.delta_pct)} ${r.run_id} ${r.theme}`, r.new_best ? 'green' : undefined])
    out.push([`next: ${d.next[0]?.theme ?? '-'}`])
    for (const c of d.child_next ?? []) out.push([`${c.name ?? c.session_id.slice(0, 8)} next: ${c.next[0]?.theme ?? '-'}`])
  }
  const k = b.kernelforge
  out.push(...head(out, 'KernelForge', b.ages.kernelforge, b.problems.kernelforge))
  if (k) {
    for (const [role, hs] of byRole(k.hot_ops)) {
      const ops = hs[0]?.ops.slice(0, 3).map((o: any) => `${o.name} ${o.share_pct.toFixed(0)}%`).join(' · ')
      out.push([`${role}: ${ops || 'not focused'}`, ops ? undefined : 'dim'])
    }
    out.push([campaignsLine(k)])
    out.push([`operators tracked ${k.ops.length}`])
  }
  out.push(...head(out, 'log', null, null))
  out.push(...log(b.events, false, 3, 'INFO').slice(1))
  const tc = b.trigger
  const ac = b.analysis
  out.push(...head(out, 'config in effect', null, null))
  out.push([`kick ${dur(tc.kick.interval_s)} · packup /${tc.packup.every_n_kicks} kicks or ${dur(tc.packup.interval_s)}`])
  out.push([`task book ${b.taskBook}`])
  out.push([`refresh /${tc.refresh.every_n_tool_calls} calls /${tc.refresh.every_n_turns} turns · rotate ${tc.rotate_mb} MB`])
  out.push([`log /${tc.log.every_n_tool_calls} calls /${tc.log.every_n_turns} turns · slow ${dur(tc.log.slow_interval_s)}`])
  out.push([`extract ${ac.extract.model}/${ac.extract.effort} · log ${ac.log.model}/${ac.log.effort}`])
  return out
}
