// Formatting shared by the tab renderers. Pure.
export type Line = [string, string?] // text, colour ('dim' = dimmed)

export function num(x: any, digits = 0): string {
  return typeof x === 'number' && isFinite(x) ? x.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits }) : '-'
}
export const pct = (x: any) => (typeof x === 'number' && isFinite(x) ? `${x >= 0 ? '+' : ''}${x.toFixed(1)}%` : '-')
export const tok = (n: any) => (typeof n !== 'number' ? '-' : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}k` : String(n))
export function dur(s: any): string {
  if (typeof s !== 'number' || !isFinite(s)) return '-'
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  return h ? `${h}h${String(m).padStart(2, '0')}m` : `${m}m`
}
export function ago(iso: any, now = Date.now()): string {
  const t = Date.parse(iso)
  return isNaN(t) ? '-' : dur(Math.max(0, (now - t) / 1000))
}
