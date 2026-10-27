// Ledger: the results a project measured (results/*.json of its tasks, schema infera.result/v1), against the
// baseline. The page leads with the numbers (primary metric, baseline, best and its delta; at the working point when
// the server compares series there, curve_head), then, when the project says its X metric (ledger.x), the curves (one
// Y–X line per series through its frontier, the working point and each series' delta there, with a table of the
// series), then the trajectory (one dot per result over time, its A/B decision as shape + color; without curve_head
// the baseline dashed and the best so far as a step line), then the table, which holds every value the charts show.
// The board's head shows the same numbers (LedgerChips). Which result is the best, and whether a delta is better or
// worse (its gain), the server says.
import { useState, useEffect, useRef, useCallback } from './vendor/hooks.module.js';
import { html, Icon, useApi, clock, taskHref, ErrorBox, Loading } from './ui.js';
import { t as tr } from './i18n.js';

// ---------------------------------------------------------------- numbers

// 9850 -> 9.85K, 12345.6 -> 12.3K, 31.234 -> 31.23: compact from a thousand (three significant digits), four below.
export function fmt(v) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a >= 1e6) return `${+(v / 1e6).toPrecision(3)}M`;
  if (a >= 1e3) return `${+(v / 1e3).toPrecision(3)}K`;
  return `${+v.toPrecision(4)}`;
}

// A delta against the baseline: "+15.4%" (better or worse is the caller's color), "基线" for the baseline itself.
export function deltaText(d, { baseline = false } = {}) {
  if (baseline && (d == null || d === 0)) return tr('基线');
  if (typeof d !== 'number') return '—';
  return `${d > 0 ? '+' : d < 0 ? '−' : '±'}${Math.abs(d).toFixed(Math.abs(d) >= 10 ? 1 : 2)}%`;
}

// The A/B decision of a result: its word, its mark (shape + color class) and its legend order.
const DECISIONS = {
  accept: { get label() { return tr('接受'); }, mark: 'dot', cls: 'd-accept' },
  reject: { get label() { return tr('拒绝'); }, mark: 'cross', cls: 'd-reject' },
  noise: { get label() { return tr('噪声'); }, mark: 'dot', cls: 'd-noise' },
  // noise the measurement could not resolve (its noise floor above the accept bar): left to the round's close
  unresolvable: { get label() { return tr('判不了 · 待集成'); }, mark: 'dot', cls: 'd-unresolvable' },
  rerun: { get label() { return tr('需重跑'); }, mark: 'dot', cls: 'd-rerun' },
  void: { get label() { return tr('作废'); }, mark: 'ring', cls: 'd-void' },
  none: { get label() { return tr('未判定'); }, mark: 'dot', cls: 'd-none' },
  baseline: { get label() { return tr('基线'); }, mark: 'diamond', cls: 'd-baseline' },
};
export const decisionOf = (r) => (r.status === 'void' ? 'void' : r.baseline || r.role === 'baseline' ? 'baseline'
  : r.decision === 'noise' && r.resolvable === false ? 'unresolvable' : r.decision || 'none');

// The color of a delta by its gain (the server's delta in the better direction): better, worse, or neither.
export const gainClass = (gain) => (gain > 0 ? 'ok-text' : gain < 0 ? 'warn-text' : 'muted');

// What the ledger says of a row's verdict and series, as tags: a verdict that does not hold when recomputed from its
// AgentX output (判定未核实, with the server's reasons), one judged against a baseline series that is no longer the
// project's (相对 v0 判定), a point of the baseline series another task measured (复测: drawn, not the reference), a
// measurement that is not formal (筛选: AgentX fast mode or too short; never an A/B's input or a reference).
export function rowTags(r, baseSeries = '') {
  const out = [];
  if (r.verified === false) out.push({ kind: 'unverified', cls: 'warn', text: tr('判定未核实'), title: (r.unverified || []).join('\n') });
  if (r.judged_against) {
    out.push({ kind: 'judged', cls: '', text: tr('相对 {series} 判定', { series: r.judged_against }),
      title: tr('判定用的基线臂属于 {series}，不是当前的基线系列 {base}', { series: r.judged_against, base: baseSeries || '—' }) });
  }
  if (r.repeat) out.push({ kind: 'repeat', cls: '', text: tr('复测'), title: tr('这个系列的复测（不是封版任务测的）：画在曲线上，不进 Δ 的参照') });
  if (r.screening) out.push({ kind: 'screening', cls: '', text: tr('筛选'), title: tr('不是正式测量（{why}）：留在台账里，不作判定输入，也不进参照', { why: r.screening }) });
  return out;
}
const Tags = ({ tags }) => tags.map((t) => html` <span class=${`tag ledger-tag ${t.cls}`} title=${t.title}>${t.text}</span>`);
const Delta = ({ d }) => html`<span class=${gainClass(d.gain)}>${deltaText(d.delta_pct)}</span>`;
// Whether two of the head's results (a baseline, a best) are the same one.
const same = (a, b) => !!a && !!b && a.task === b.task && a.file === b.file && a.series === b.series;

// ---------------------------------------------------------------- board chips

export function LedgerChips({ pid, ledger }) {
  const href = `#/p/${pid}/ledger`;
  if (!ledger?.count) return html`<a class="pill ledger-chip muted" href=${href} title=${tr('结果台账：任务写在 results/ 下的 infera.result 文件')}><${Icon} name="chart" />${tr('台账：暂无结果')}</a>`;
  const { baseline: b, best, primary } = ledger;
  const series = (c) => c.series != null && html` <span class="mono">${c.series}</span>`;
  return html`<a class="pill ledger-chip" href=${href} title=${tr('结果台账：主指标 {metric}', { metric: primary })}>
    <${Icon} name="chart" /><span class="muted">${primary}</span>
    ${b && html`<span>${tr('基线')}${series(b)} <b>${fmt(b.value)}</b></span>`}
    ${best && !same(best, b) && best.delta_pct != null && html`<span title=${ledger.curve_head ? tr('各系列在工作点相对基线系列的变化，取最好的一条') : ''}>${tr('最佳')}${series(best)} <b>${fmt(best.value)}</b> <${Delta} d=${best} /></span>`}</a>`;
}

// ---------------------------------------------------------------- the page

export function LedgerPage({ pid }) {
  const { data, error, reload } = useApi(`/api/projects/${pid}/ledger`, (c) => !c.project || c.project === pid);
  const [picked, setPicked] = useState(null);  // a decision to show alone, or null for all
  if (error && !data) return html`<main id="main" tabIndex="-1" class="page"><${ErrorBox} what=${tr('加载台账')} error=${error} retry=${reload} /></main>`;
  if (!data) return html`<${Loading} text=${tr('正在加载台账…')} />`;
  const present = Object.keys(DECISIONS).filter((k) => data.rows.some((r) => decisionOf(r) === k));
  const only = present.includes(picked) ? picked : null;  // a decision the results no longer have filters nothing
  const rows = data.rows.filter((r) => !only || decisionOf(r) === only);
  const extra = data.metrics.filter((k) => k !== data.primary).slice(0, 4);
  return html`<main id="main" tabIndex="-1" class="page ledger-page">
    ${error && html`<${ErrorBox} what=${tr('刷新台账')} error=${error} retry=${reload} />`}
    <header class="page-head"><div><h1>${tr('台账')}</h1>
      <p class="muted small">${tr('各任务写在工作区 results/ 下的测量结果（infera.result/v1）。台账 Δ 是主指标相对当前基线（封版任务测的点）的变化；判定 Δ 是 A/B 判定相对它自己的基线臂的变化。')}</p></div></header>
    ${!data.count ? html`<${EmptyLedger} errors=${data.errors} pid=${pid} />` : html`
      <section class="stat-row" aria-label=${tr('概览')}>
        <div class="stat"><span class="muted small">${tr('主指标')}</span><b class="mono">${data.primary || '—'}</b>
          <span class="muted small">${data.higher_is_better ? tr('越大越好') : tr('越小越好')}</span></div>
        ${data.curve_head ? html`<${CurveStats} data=${data} />` : html`
        <div class="stat"><span class="muted small">${tr('基线')}</span><b>${fmt(data.baseline?.value)}</b>
          ${data.baseline ? html`<a class="small" href=${taskHref(pid, data.baseline.task, { tab: 'docs', file: data.baseline.file })}>${data.baseline.label}</a>`
            : html`<span class="muted small">${tr('还没有基线（role: baseline）')}</span>`}
          ${data.pin_missing && html`<span class="warn-text small">${tr('ledger.baseline「{pin}」没有可用的结果', { pin: data.pin_missing })}</span>`}</div>
        <div class="stat"><span class="muted small">${tr('最佳')}</span><b>${fmt(data.best?.value)}
          ${data.best?.delta_pct != null && html` <${Delta} d=${data.best} />`}</b>
          ${data.best && html`<a class="small" href=${taskHref(pid, data.best.task, { tab: 'docs', file: data.best.file })}>${data.best.label}</a>`}</div>`}
        <div class="stat"><span class="muted small">${tr('结果')}</span><b>${data.count}</b>
          <span class="muted small">${tr('作废 {n}', { n: data.void })}${data.errors.length ? tr('，格式错误 {n}', { n: data.errors.length }) : ''}</span></div>
      </section>
      ${data.x && data.series?.length > 0 && html`<${Curves} data=${data} pid=${pid} />`}
      <section class="chart-card" aria-label=${tr('趋势')}>
        <div class="legend-row" role="group" aria-label=${tr('按判定筛选')}>
          ${present.map((k) => html`<button class="chip" aria-pressed=${only === k} onClick=${() => setPicked(only === k ? null : k)}>
            <svg width="14" height="14" aria-hidden="true"><${Mark} kind=${k} x=${7} y=${7} /></svg>${DECISIONS[k].label}</button>`)}
          ${!data.curve_head && html`<span class="legend-line"><svg width="22" height="10" aria-hidden="true"><line class="base-line" x1="1" y1="5" x2="21" y2="5" /></svg>${tr('基线')}</span>
          <span class="legend-line"><svg width="22" height="10" aria-hidden="true"><line class="best-line" x1="1" y1="5" x2="21" y2="5" /></svg>${tr('目前最佳')}</span>`}
        </div>
        <${Trajectory} data=${data} rows=${rows} pid=${pid} />
      </section>
      <${LedgerTable} pid=${pid} rows=${[...rows].reverse()} data=${data} extra=${extra} />
      ${data.errors.length > 0 && html`<${BadFiles} errors=${data.errors} pid=${pid} />`}`}
  </main>`;
}

function EmptyLedger({ errors, pid }) {
  return html`<section class="empty-state">
    <p><b>${tr('还没有结果。')}</b>${tr('任务把每个测量点写成工作区里的 results/<名字>.json，台账就会收进来。最少要有：')}</p>
    <pre>${'{"schema": "infera.result/v1", "label": "c64-baseline", "role": "baseline",\n "metrics": {"input_tput_per_gpu": 12345.6, "p90_intvty": 30.1}}'}</pre>
    <p class="muted small">${tr('skill infera-bench 会按这个格式写；A/B 的判定写在 verdict（infera-experiment 的 ab_judge.py）。')}</p>
    ${errors.length > 0 && html`<${BadFiles} errors=${errors} pid=${pid} />`}
  </section>`;
}

function BadFiles({ errors, pid }) {
  return html`<section class="bad-files"><h2><${Icon} name="warning" />${tr('格式错误的结果文件 {n}', { n: errors.length })}</h2>
    <ul>${errors.map((e) => html`<li><a class="mono small" href=${taskHref(pid, e.task, { tab: 'docs', file: e.file })}>${e.task}/${e.file}</a>
      <span class="warn-text small">${e.errors.join('; ')}</span></li>`)}</ul></section>`;
}

function LedgerTable({ pid, rows, data, extra }) {
  return html`<section class="table-wrap"><h2>${tr('结果')} <span class="muted small">${tr('新的在前')}</span></h2>
    <table class="tasks ledger">
      <thead><tr><th>${tr('时间')}</th><th>${tr('任务 / 标签')}</th><th>${tr('改动')}</th><th class="num">${data.primary}</th>
        <th class="num" title=${tr('相对当前基线：基线系列在同一并发、由封版任务测的点')}>${tr('台账 Δ')}</th>
        <th class="num" title=${tr('A/B 判定写下的变化：相对这次实验自己的基线臂')}>${tr('判定 Δ')}</th>
        ${extra.map((k) => html`<th class="num mono">${k}</th>`)}<th>${tr('判定')}</th><th>${tr('说明')}</th></tr></thead>
      <tbody>${rows.map((r) => {
        const k = decisionOf(r);
        const tags = rowTags(r, data.baseline_series);
        const vd = r.verdict_delta_pct;
        return html`<tr class=${r.status === 'void' ? 'void' : ''}>
          <td class="nowrap small">${clock(r.time)}</td>
          <td><a href=${taskHref(pid, r.task, { tab: 'docs', file: r.file })}><b>${r.label}</b></a><${Tags} tags=${tags.filter((t) => ['repeat', 'screening'].includes(t.kind))} /><div class="small muted">${r.title}</div></td>
          <td class="small">${r.change}</td>
          <td class="num"><b>${fmt(r.value)}</b></td>
          <td class=${`num ${gainClass(r.gain)}`}>${deltaText(r.delta_pct, { baseline: r.baseline && !r.repeat })}</td>
          <td class=${`num ${['accept', 'reject'].includes(r.decision) && typeof vd === 'number' ? gainClass(data.higher_is_better ? vd : -vd) : 'muted'}`}>${deltaText(vd)}</td>
          ${extra.map((m) => html`<td class="num">${fmt(r.metrics[m])}</td>`)}
          <td><span class="nowrap"><svg width="14" height="14" aria-hidden="true"><${Mark} kind=${k} x=${7} y=${7} /></svg> ${DECISIONS[k].label}</span><${Tags} tags=${tags.filter((t) => !['repeat', 'screening'].includes(t.kind))} />
            ${r.verified === false && r.unverified?.length > 0 && html`<div class="small warn-text unverified">${r.unverified[0]}${r.unverified.length > 1 ? tr('（另有 {n} 条）', { n: r.unverified.length - 1 }) : ''}</div>`}</td>
          <td class="small">${r.notes}</td></tr>`;
      })}</tbody></table></section>`;
}

// ---------------------------------------------------------------- the chart

const H = 280, M = { l: 56, r: 132, t: 16, b: 30 };
const HOUR = 3600e3, DAY = 24 * HOUR;

// The chart is drawn at the width it is shown at, so its text stays the size of the page's. A callback ref: the
// chart's element comes and goes (no points to draw, all series hidden), and the observer follows it.
function useWidth(fallback = 960) {
  const [w, setW] = useState(fallback);
  const ro = useRef(null);
  const ref = useCallback((el) => {
    ro.current?.disconnect();
    ro.current = null;
    if (!el || typeof ResizeObserver === 'undefined') return;
    ro.current = new ResizeObserver(([e]) => setW(Math.max(480, Math.round(e.contentRect.width))));
    ro.current.observe(el);
  }, []);
  useEffect(() => () => ro.current?.disconnect(), []);
  return [ref, w];
}

// Time ticks on round hours or days (local time), about n of them: [ms].
export function timeTicks(t0, t1, n = 5) {
  const step = [HOUR, 3 * HOUR, 6 * HOUR, 12 * HOUR, DAY, 2 * DAY, 7 * DAY, 30 * DAY].find((s) => (t1 - t0) / s <= n) || 30 * DAY;
  const first = new Date(t0);
  first.setHours(0, 0, 0, 0);
  const out = [];
  for (let t = first.getTime(); t <= t1; t += step) if (t >= t0) out.push(t);
  return { ticks: out, days: step >= DAY };
}

// A result's mark: a dot for a decision, a cross for a rejection, a ring for a voided result, a diamond for the baseline.
function Mark({ kind, x, y, r = 4.5 }) {
  const d = DECISIONS[kind] || DECISIONS.none;
  if (d.mark === 'cross') return html`<path class=${`mark ${d.cls}`} d=${`M${x - r},${y - r}L${x + r},${y + r}M${x + r},${y - r}L${x - r},${y + r}`} />`;
  if (d.mark === 'diamond') return html`<path class=${`mark ${d.cls}`} d=${`M${x},${y - r - 1}L${x + r + 1},${y}L${x},${y + r + 1}L${x - r - 1},${y}Z`} />`;
  return html`<circle class=${`mark ${d.cls}`} cx=${x} cy=${y} r=${r} />`;
}

// Round tick values for [lo, hi]: 1, 2 or 5 times a power of ten apart, about n of them.
export function ticks(lo, hi, n = 4) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / n, p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

// The step line's corners: the best value so far at each result in time order, as the server counts it (none with
// curve_head: results at different concurrencies do not compare).
export const bestSoFar = (rows) => rows.filter((r) => r.best_so_far != null).map((r) => ({ t: Date.parse(r.time), v: r.best_so_far }));

function Trajectory({ data, rows, pid }) {
  const [tip, setTip] = useState(null);
  const [box, W] = useWidth();
  const pts = rows.filter((r) => typeof r.value === 'number' && Number.isFinite(Date.parse(r.time)));
  if (!pts.length) return html`<p class="empty-line">${tr('没有可画的结果（缺少主指标 {metric}）。', { metric: data.primary })}</p>`;
  const times = pts.map((r) => Date.parse(r.time));
  let t0 = Math.min(...times), t1 = Math.max(...times);
  if (t1 - t0 < 3600e3) { t0 -= 1800e3; t1 += 1800e3; }
  const base = data.curve_head ? null : data.baseline?.value;
  const vals = [...pts.map((r) => r.value), ...(typeof base === 'number' ? [base] : [])];
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = (hi - lo) * 0.12 || Math.abs(hi) * 0.05 || 1;
  lo -= pad; hi += pad;
  const ys = ticks(lo, hi);
  const x = (t) => M.l + ((t - t0) / (t1 - t0)) * (W - M.l - M.r);
  const y = (v) => H - M.b - ((v - lo) / (hi - lo)) * (H - M.t - M.b);
  const steps = bestSoFar(pts);
  const path = steps.map((s, i) => `${i ? `H${x(s.t)}V${y(s.v)}` : `M${x(s.t)},${y(s.v)}`}`).join('') + (steps.length ? `H${W - M.r}` : '');
  const last = steps[steps.length - 1];
  const xt = timeTicks(t0, t1, Math.max(3, Math.floor(W / 160)));
  const show = (r, ev) => setTip({ r, x: x(Date.parse(r.time)), y: y(r.value) });
  return html`<div class="chart" ref=${box} onMouseLeave=${() => setTip(null)}>
    <svg viewBox=${`0 0 ${W} ${H}`} width=${W} height=${H} role="img" aria-label=${tr('{metric} 随时间的变化：{n} 个结果', { metric: data.primary, n: pts.length })}>
      ${ys.map((v) => html`<line class="grid" x1=${M.l} x2=${W - M.r} y1=${y(v)} y2=${y(v)} />
        <text class="tick" x=${M.l - 8} y=${y(v) + 4} text-anchor="end">${fmt(v)}</text>`)}
      ${xt.ticks.map((v) => html`<text class="tick" x=${x(v)} y=${H - 8} text-anchor="middle">${clock(new Date(v).toISOString()).slice(0, xt.days ? 5 : 11)}</text>`)}
      ${typeof base === 'number' && html`<line class="base-line" x1=${M.l} x2=${W - M.r} y1=${y(base)} y2=${y(base)} />
        <text class="end-label" x=${W - M.r + 8} y=${y(base) + 4}>${tr('基线')} ${fmt(base)}</text>`}
      ${steps.length > 0 && html`<path class="best-line" d=${path} />
        ${(typeof base !== 'number' || Math.abs(y(last.v) - y(base)) > 14) && html`<text class="end-label" x=${W - M.r + 8} y=${y(last.v) + 4}>${tr('最佳')} ${fmt(last.v)}</text>`}`}
      ${pts.map((r) => html`<g key=${`${r.task}/${r.file}`}><${Mark} kind=${decisionOf(r)} x=${x(Date.parse(r.time))} y=${y(r.value)} />
        <a href=${taskHref(pid, r.task, { tab: 'docs', file: r.file })} aria-label=${`${r.label} ${fmt(r.value)} ${deltaText(r.delta_pct, r)} ${DECISIONS[decisionOf(r)].label}`}
          onFocus=${(ev) => show(r, ev)} onBlur=${() => setTip(null)}>
          <circle class="hit" cx=${x(Date.parse(r.time))} cy=${y(r.value)} r="12" onMouseEnter=${(ev) => show(r, ev)} /></a></g>`)}
    </svg>
    ${tip && html`<div class="chart-tip" style=${{ left: `${(tip.x / W) * 100}%`, top: `${(tip.y / H) * 100}%` }}>
      <b>${fmt(tip.r.value)}</b> <span class=${gainClass(tip.r.gain)}>${deltaText(tip.r.delta_pct, tip.r)}</span>
      <div>${tip.r.label} · ${DECISIONS[decisionOf(tip.r)].label}${rowTags(tip.r, data.baseline_series).map((t) => ` · ${t.text}`).join('')}</div>
      <div class="muted">${tip.r.title} · ${clock(tip.r.time)}</div>
      ${tip.r.change && html`<div class="muted">${tip.r.change}</div>`}</div>`}
  </div>`;
}

// ---------------------------------------------------------------- the curves

// A series keeps its color: the slot of its place in the ledger's list (the baseline first, ours by time, the
// references last), whatever is hidden. Past the eight categorical slots a series is gray.
export const seriesColor = (i) => (i < 8 ? `var(--series-${i + 1})` : 'var(--ink-3)');

// The working point as words: "X ≥ 30" (≤ when a lower X is better) or "C64".
export function wpText(wp, x, xHigher = true) {
  if (wp?.x != null) return `${x} ${xHigher ? '≥' : '≤'} ${fmt(wp.x)}`;
  if (wp?.conc != null) return `C${wp.conc}`;
  return '';
}

// Labels at the right end of the curves, nudged apart vertically so that none overlaps the one above: [{y, ...}]
// sorted by y, each at least gap below the previous.
export function spreadLabels(labels, gap = 14) {
  const out = [...labels].sort((a, b) => a.y - b.y);
  for (let i = 1; i < out.length; i++) if (out[i].y - out[i - 1].y < gap) out[i] = { ...out[i], y: out[i - 1].y + gap };
  return out;
}

const CH = 340, CM = { l: 64, r: 128, t: 18, b: 40 };

function Curves({ data, pid }) {
  const [hidden, setHidden] = useState(() => new Set());
  const [tip, setTip] = useState(null);
  const [box, W] = useWidth();
  const all = data.series;
  const color = new Map(all.map((c, i) => [c.name, seriesColor(i)]));
  const shown = all.filter((c) => !hidden.has(c.name));
  const pts = shown.flatMap((c) => c.points);
  const wp = data.working_point || {};
  const xs = [...pts.map((p) => p.x), ...(wp.x != null ? [wp.x] : [])], ys = pts.map((p) => p.y);
  const span = (v) => {
    let lo = Math.min(...v), hi = Math.max(...v);
    const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.05 || 1;
    return [lo - pad, hi + pad];
  };
  const toggle = (name) => { const h = new Set(hidden); if (h.has(name)) h.delete(name); else h.add(name); setHidden(h); };
  const wpWords = wpText(wp, data.x, data.x_higher_is_better);
  const legend = html`<div class="legend-row" role="group" aria-label=${tr('显示或隐藏系列')}>
    ${all.map((c) => html`<button class="chip" aria-pressed=${!hidden.has(c.name)} onClick=${() => toggle(c.name)}
      title=${c.role === 'reference' ? tr('对标（role: reference）') : c.baseline ? tr('基线系列') : ''}>
      <svg width="22" height="10" aria-hidden="true"><line class=${`curve ${c.role === 'reference' ? 'ref' : ''}`} style=${{ stroke: color.get(c.name) }} x1="1" y1="5" x2="21" y2="5" /></svg>
      <span class="mono">${c.name}</span>${c.baseline ? html` <span class="muted">${tr('基线')}</span>` : ''}</button>`)}
    ${wpWords && html`<span class="legend-line"><svg width="22" height="10" aria-hidden="true"><line class="wp-line" x1="11" y1="0" x2="11" y2="10" /></svg>${tr('工作点')} ${wpWords}</span>`}
  </div>`;
  if (!pts.length) return html`<section class="chart-card" aria-label=${tr('曲线')}>${legend}<p class="empty-line">${tr('所有系列都已隐藏。')}</p></section>`;
  const [x0, x1] = span(xs), [y0, y1] = span(ys);
  const x = (v) => CM.l + ((v - x0) / (x1 - x0)) * (W - CM.l - CM.r);
  const y = (v) => CH - CM.b - ((v - y0) / (y1 - y0)) * (CH - CM.t - CM.b);
  const labels = shown.length <= 4 ? spreadLabels(shown.map((c) => {
    const last = c.points.filter((p) => p.frontier).reduce((a, p) => (p.x > a.x ? p : a));
    return { name: c.name, y: y(last.y), x: x(last.x) };
  })) : [];
  return html`<section class="chart-card" aria-label=${tr('曲线')}>
    <h2 class="chart-title">${tr('曲线')} <span class="muted small">${tr('{y} 对 {x}，每个系列一条线，连接它的前沿点（空心点被同系列的其他点支配，方块是别的任务测的复测）', { y: data.primary, x: data.x })}</span></h2>
    ${legend}
    <div class="chart" ref=${box} onMouseLeave=${() => setTip(null)}>
      <svg viewBox=${`0 0 ${W} ${CH}`} width=${W} height=${CH} role="img" aria-label=${tr('{y} 对 {x} 的曲线：{n} 个系列', { y: data.primary, x: data.x, n: shown.length })}>
        ${ticks(y0, y1).map((v) => html`<line class="grid" x1=${CM.l} x2=${W - CM.r} y1=${y(v)} y2=${y(v)} />
          <text class="tick" x=${CM.l - 8} y=${y(v) + 4} text-anchor="end">${fmt(v)}</text>`)}
        ${ticks(x0, x1, Math.max(3, Math.floor(W / 140))).map((v) => html`<text class="tick" x=${x(v)} y=${CH - CM.b + 16} text-anchor="middle">${fmt(v)}</text>`)}
        <text class="tick" x=${(CM.l + W - CM.r) / 2} y=${CH - 6} text-anchor="middle">${data.x} →</text>
        <text class="tick" x=${CM.l} y=${CM.t - 6} text-anchor="start">↑ ${data.primary}</text>
        ${wp.x != null && html`<line class="wp-line" x1=${x(wp.x)} x2=${x(wp.x)} y1=${CM.t} y2=${CH - CM.b} />`}
        ${shown.map((c) => {
          const front = c.points.filter((p) => p.frontier);
          const d = front.map((p, i) => `${i ? 'L' : 'M'}${x(p.x)},${y(p.y)}`).join('');
          return html`<g key=${c.name}><path class=${`curve ${c.role === 'reference' ? 'ref' : ''}`} style=${{ stroke: color.get(c.name) }} d=${d} />
            ${c.points.map((p) => (p.repeat || p.screening
              ? html`<rect class=${`pt rep ${p.screening ? 'scr' : ''}`} style=${{ stroke: color.get(c.name) }} x=${x(p.x) - 3.5} y=${y(p.y) - 3.5} width="7" height="7" />`
              : html`<circle class=${`pt ${p.frontier ? '' : 'off'}`} style=${{ [p.frontier ? 'fill' : 'stroke']: color.get(c.name) }} cx=${x(p.x)} cy=${y(p.y)} r="4" />`))}
            ${c.wp && html`<path class="wp-mark" style=${{ stroke: color.get(c.name) }}
              d=${`M${x(c.wp.x)},${y(c.wp.y) - 6}L${x(c.wp.x) + 6},${y(c.wp.y)}L${x(c.wp.x)},${y(c.wp.y) + 6}L${x(c.wp.x) - 6},${y(c.wp.y)}Z`} />`}
            ${c.points.map((p) => html`<a href=${taskHref(pid, p.task, { tab: 'docs', file: p.file })} aria-label=${`${c.name} ${p.label}: ${data.x} ${fmt(p.x)}, ${data.primary} ${fmt(p.y)}`}
              onFocus=${() => setTip({ c, p, X: x(p.x), Y: y(p.y) })} onBlur=${() => setTip(null)}>
              <circle class="hit" cx=${x(p.x)} cy=${y(p.y)} r="10" onMouseEnter=${() => setTip({ c, p, X: x(p.x), Y: y(p.y) })} /></a>`)}</g>`;
        })}
        ${labels.map((l) => html`<text class="end-label" x=${Math.min(l.x + 10, W - CM.r + 8)} y=${l.y + 4}>${l.name}</text>`)}
      </svg>
      ${tip && html`<div class="chart-tip" style=${{ left: `${(tip.X / W) * 100}%`, top: `${(tip.Y / CH) * 100}%` }}>
        <b>${fmt(tip.p.y)}</b> <span class="muted">${data.primary}</span>
        <div>${fmt(tip.p.x)} <span class="muted">${data.x}</span>${tip.p.conc != null ? ` · C${tip.p.conc}` : ''}</div>
        <div><span class="mono">${tip.c.name}</span> · ${tip.p.label}${tip.p.screening ? ` · ${tr('筛选（不是正式测量）')}` : tip.p.repeat ? ` · ${tr('复测（不进参照）')}` : tip.p.frontier ? '' : ` · ${tr('非前沿')}`}</div>
        <div class="muted">${tip.p.task} · ${clock(tip.p.time)}</div></div>`}
    </div>
    <${SeriesTable} data=${data} color=${color} wpWords=${wpWords} />
  </section>`;
}

// The head's baseline and best with curve_head: the baseline series and the best series at the working point (the
// baseline itself while no other beats it).
function CurveStats({ data }) {
  const { baseline: base, best } = data;
  const wp = wpText(data.working_point, data.x, data.x_higher_is_better);
  return html`<div class="stat"><span class="muted small">${tr('基线 · 工作点 {wp}', { wp })}</span><b>${fmt(base.value)}</b>
      <span class="small mono">${base.series}</span></div>
    <div class="stat"><span class="muted small">${tr('最佳 · 工作点 {wp}', { wp })}</span><b>${fmt(best?.value)}
      ${best?.delta_pct != null && html` <${Delta} d=${best} />`}</b>
      <span class="small mono">${best ? best.series : tr('还没有别的系列到达工作点')}</span></div>`;
}

function SeriesTable({ data, color, wpWords }) {
  return html`<table class="tasks ledger series-table">
    <thead><tr><th>${tr('系列')}</th><th>${tr('轮次')}</th><th class="num">${tr('点数')}</th>
      <th class="num">${tr('工作点 {wp}', { wp: wpWords || '—' })}</th><th class="num">Δ</th></tr></thead>
    <tbody>${data.series.map((c) => html`<tr>
      <td><svg width="14" height="10" aria-hidden="true"><line class=${`curve ${c.role === 'reference' ? 'ref' : ''}`} style=${{ stroke: color.get(c.name) }} x1="0" y1="5" x2="14" y2="5" /></svg>
        <span class="mono">${c.name}</span> <span class="muted small">${c.role === 'reference' ? tr('对标') : c.baseline ? tr('基线') : ''}</span></td>
      <td>${c.round ?? '—'}</td>
      <td class="num">${c.points.filter((p) => !p.repeat && !p.screening).length}${c.points.some((p) => p.repeat) ? html` <span class="muted small" title=${tr('这个系列的复测（不是封版任务测的）：画在曲线上，不进 Δ 的参照')}>${tr('+{n} 复测', { n: c.points.filter((p) => p.repeat).length })}</span>` : ''}${c.points.some((p) => p.screening) ? html` <span class="muted small" title=${tr('AgentX fast 模式或时长不够的点：画在曲线上，不进前沿和工作点')}>${tr('+{n} 筛选', { n: c.points.filter((p) => p.screening).length })}</span>` : ''}</td>
      <td class="num">${c.wp ? html`<b>${fmt(c.wp.y)}</b>${c.wp.interpolated ? html` <span class="muted small" title=${tr('在前沿上相邻两点之间插值')}>${tr('插值')}</span>` : ''}`
        : html`<span class="muted small" title=${tr('这个系列没有到达工作点的点')}>${tr('未到达')}</span>`}</td>
      <td class=${`num ${gainClass(c.baseline ? null : c.wp?.gain)}`}>${c.wp ? deltaText(c.wp.delta_pct, { baseline: c.baseline }) : '—'}</td></tr>`)}</tbody></table>`;
}
