// Project board: lanes (task types) × rows in the order the tasks run, the DAG between tasks, the three attention
// panels above it and the task table below. Where cards go and how edges run between them is in graph.js.
import { useState, useEffect, useRef, useMemo } from './vendor/hooks.module.js';
import { api, paths } from './api.js';
import { html, Icon, StatusMark, STATUS, STATUS_ORDER, EDGE_KINDS, statusLine, useApi, useNow, usePopover, money, ago,
  since, clock, parseHash, taskHref, closeHref, go, toast, fail, confirmDialog, ErrorBox, Loading, Holder, Problems } from './ui.js';
import { NeedsCard, ReviewCard, MarkAllRead, resumeAll, interruptedOf } from './cards.js';
import { confirmMove } from './task.js';
import { t as tr, list as listOf } from './i18n.js';
import { LedgerChips, deltaText } from './ledger.js';
import { NODE_W, laneX, laneAt, layout, runOrder, redundant } from './graph.js';

// ---------------------------------------------------------------- helpers (web/test/pure.test.js)

// All the lanes after moving the shown lane at index `from` to index `to` among the shown ones; hidden lanes keep
// their places.
export function moveLane(all, shown, from, to) {
  const moved = [...shown];
  moved.splice(to, 0, ...moved.splice(from, 1));
  let i = 0;
  return all.map((id) => (shown.includes(id) ? moved[i++] : id));
}

// Everything a task depends on and everything that depends on it, along the edges on display
// (review edges excluded: a review does not hold up the downstream).
function lineage(id, edges) {
  const flow = edges.filter((e) => e.kind !== 'review');
  const walk = (next) => {
    const seen = new Set(), stack = [id];
    while (stack.length) for (const n of next(stack.pop())) if (!seen.has(n)) { seen.add(n); stack.push(n); }
    return seen;
  };
  return { up: walk((n) => flow.filter((e) => e.to === n).map((e) => e.from)), down: walk((n) => flow.filter((e) => e.from === n).map((e) => e.to)) };
}

// ---------------------------------------------------------------- the page

// labels are looked up when shown, not when the module loads (the language can be set after, in tests)
const WINDOWS = { all: ['全部日期', 0], d3: ['近 3 天', 3], d7: ['近 7 天', 7], d30: ['近 30 天', 30], custom: ['自定义日期…', 0] };  // i18n: translated where shown
const windowLabel = (k) => tr(WINDOWS[k][0]);
const lastActive = (t) => t.since || t.updated || t.started || t.created || '';
const DAY = 86400000;

// The dates a board filter keeps, {from, to} in ms (null: open): the last n days, or the days you pick (local dates,
// both included).
export function dateRange(view, now) {
  if (view.window === 'custom') {
    return { from: view.from ? Date.parse(`${view.from}T00:00:00`) : null, to: view.to ? Date.parse(`${view.to}T23:59:59.999`) : null };
  }
  const days = WINDOWS[view.window]?.[1];
  return { from: days ? now - days * DAY : null, to: null };
}

// A task is in a date range when it was active in it: from its start (its creation, if it has not started) to its
// end (its last change once done or cancelled; now while it is not).
export function inDates(t, { from, to }, now) {
  const start = Date.parse(t.started || t.created || '') || now;
  const end = ['done', 'cancelled'].includes(t.status) ? Date.parse(t.finished || lastActive(t)) || start : now;
  return (from == null || end >= from) && (to == null || start <= to);
}

export function BoardPage({ pid, onNewTask }) {
  const { data, error, reload } = useApi(paths.board(pid), (c) => !c.project || c.project === pid);
  const selected = (parseHash().params.t || '').startsWith(`${pid}/`) ? parseHash().params.t.slice(pid.length + 1) : '';
  // every kind but 评审覆盖 is drawn unless the legend says otherwise; showAll: also the lines other edges imply
  const [view, setView] = useState({ q: '', statuses: new Set(), hiddenTypes: new Set(), collapse: null, window: null,
    from: '', to: '', kinds: new Set(['input', 'code', 'children', 'subtask']), showAll: false, zoom: 1 });
  const set = (patch) => setView((v) => ({ ...v, ...patch }));

  if (error && !data) {
    return html`<main id="main" tabIndex="-1" class="page"><${ErrorBox} what=${error.status === 404 ? tr('打开项目') : tr('加载看板')} error=${error} retry=${reload} />
      ${error.status === 404 && html`<p><a href="#/">${tr('返回首页')}</a></p>`}</main>`;
  }
  if (!data) return html`<${Loading} text=${tr('正在加载看板…')} />`;
  const { project, types, tasks, edges, machines } = data;
  const byId = new Map(tasks.map((t) => [t.id, t]));
  const titleOf = (id) => byId.get(id)?.title || id;
  // big projects start folded: finished cards collapse, and only the last week is drawn
  const collapse = view.collapse ?? tasks.length > 40;
  const win = view.window ?? (tasks.length > 60 ? 'd7' : 'all');
  const v = { ...view, window: win };

  return html`<main id="main" tabIndex="-1" class="page board-page">
    ${error && html`<${ErrorBox} what=${tr('刷新看板')} error=${error} retry=${reload} />`}
    ${project.archived && html`<div class="notice archived" role="status"><${Icon} name="archive" />
      <span>${tr('该项目已于 {time} 归档：不显示在首页，任务不会自动开始，也不能新建任务或启动运行。取消归档后恢复正常。', { time: clock(project.archived_at) })}</span>
      <${ArchiveButton} project=${project} /></div>`}
    ${tasks.length > 0 && tasks.every((t) => !t.managed) && html`<div class="notice archived" role="note">
      <span>${tr('本项目的任务都未纳入托管：agent_sys 按各任务的 progress.md 显示状态，不会运行或调度它们（卡片左侧为点线）。')}
        ${tr('需要 agent_sys 管理某个任务时，打开它并点"纳入托管"。')}</span></div>`}
    <header class="page-head">
      <div>
        <h1>${project.title}</h1>
        <div class="muted mono small">${project.path}${project.template ? ` · ${tr('模板 {template}', { template: project.template })}` : ''}</div>
        ${project.goal && html`<div class="goal"><span class="muted">${tr('目标')}</span> ${project.goal}</div>`}
      </div>
      <div class="head-actions">
        <${LedgerChips} pid=${pid} ledger=${project.ledger} />
        <${AutorunSwitch} project=${project} />
        <span class="pill" title=${tr('运行中的任务数 / 并发上限（后台等待和终端接管的任务也计入）')}>${tr('运行 {count}/{limit}', { count: data.running.count, limit: data.running.limit })}<span class="muted">${tr('，全局 {count}/{limit}', { count: data.running.global_count, limit: data.running.global_limit })}</span></span>
        <button class="btn primary" onClick=${onNewTask} disabled=${project.archived}
          title=${project.archived ? tr('项目已归档，取消归档后才能新建任务') : tr('新建任务（快捷键 n）')}><${Icon} name="plus" />${tr('新建任务')}</button>
      </div>
    </header>
    ${project.template_review && !project.archived && html`<${TemplateReview} pid=${pid} project=${project} types=${types} tasks=${tasks} edges=${edges} problems=${data.preflight || []} recheck=${reload} />`}
    ${machines.length > 0 && html`<${MachineStrip} machines=${machines} pid=${pid} />`}
    ${tasks.length > 0 && html`<${Attention} pid=${pid} tasks=${tasks} titleOf=${titleOf} />`}
    <${Toolbar} view=${v} set=${set} types=${types} tasks=${tasks} collapse=${collapse} />
    <${Canvas} pid=${pid} types=${types} tasks=${tasks} edges=${edges} view=${v} set=${set}
      selected=${selected} collapse=${collapse} byId=${byId} onNewTask=${project.archived ? null : onNewTask} />
    <${TaskTable} pid=${pid} tasks=${tasks} edges=${edges} view=${v} titleOf=${titleOf} selected=${selected}
      tagUnmanaged=${tasks.some((t) => t.managed)} />
  </main>`;
}

// A project made from a template that brings tasks waits for you (project.yaml template_review): the tasks are listed in
// the order they will run, to open and review (and edit) one by one, with the project's rules and settings; 开始运行 lets
// them start as in any other project. Starting one of them by hand does the same. What the check before a start finds is
// listed here; with an error 开始运行 is off (the server refuses it as well).
function TemplateReview({ pid, project, types, tasks, edges, problems, recheck }) {
  const [busy, setBusy] = useState(false);
  const queued = runOrder(tasks.filter((t) => t.managed && t.status === 'queued'), edges, types.map((ty) => ty.id)).order;
  const first = queued.filter((t) => t.detail?.kind === 'template_review');
  const errors = problems.filter((p) => p.level === 'error').length;
  const start = async () => {
    if (!await confirmDialog({ title: tr('开始运行「{title}」？', { title: project.title }), ok: tr('开始运行'),
      body: html`<p>${project.autorun ? tr('{tasks} 将立即开始，其余任务在上游完成后自动开始。', { tasks: listOf(first.map((t) => t.title)) })
        : tr('自动开始已关闭：开始运行后，任务仍需在各自的详情里手动开始。')}</p>` })) return;
    setBusy(true);
    api.post(`/api/projects/${pid}/start`).then(() => toast(tr('已开始运行：{title}', { title: project.title })), fail)
      .finally(() => setBusy(false));
  };
  return html`<section class="template-review" aria-labelledby="h-template-review">
    <div class="tr-head"><${Icon} name="in_review" /><h2 id="h-template-review">${tr('待审核模板：{n} 个任务尚未开始', { n: queued.length })}</h2>
      <span class="spacer"></span>
      <a class="btn small ghost" href=${`#/p/${pid}/rules`}>${tr('项目规则')}</a>
      <a class="btn small ghost" href=${`#/p/${pid}/settings`}>${tr('项目设置')}</a>
      <button class="btn primary" onClick=${start} disabled=${busy || errors > 0}
        title=${errors ? tr('开始前检查发现 {n} 个错误，改正后才能开始运行', { n: errors }) : undefined}><${Icon} name="play" />${tr('开始运行')}</button></div>
    <p class="small">${tr('这些任务由模板 {template} 预置，在你审核之前不会开始。请按顺序打开每个任务，检查 prompt、依赖、机器和验收检查（可以直接修改），再看一遍项目规则和设置（目标、机器）。确认后点"开始运行"，或在某个任务的详情里点"开始"；之后的任务按依赖自动推进。', { template: project.template || '' })}</p>
    ${problems.length ? html`<div class="tr-check" role=${errors ? 'alert' : 'status'}>
        <h3 class=${errors ? 'warn-text' : ''}>${errors ? tr('开始前检查：{errors} 个错误、{warns} 个提醒，改正错误后才能开始运行', { errors, warns: problems.length - errors })
          : tr('开始前检查：{n} 个提醒，不影响开始运行', { n: problems.length })}</h3>
        <${Problems} pid=${pid} problems=${problems} />
        <button class="btn small ghost" onClick=${recheck} title=${tr('改了检查用到的脚本或 skill 之后（这些文件的变化不会自动刷新）')}>${tr('重新检查')}</button></div>`
      : html`<div class="tr-check ok-text small"><${Icon} name="done" />${tr('开始前检查：没有发现问题')}</div>`}
    <ol class="tr-tasks">${queued.map((t) => html`<li><a href=${taskHref(pid, t.id, { tab: 'task' })}><b>${t.title}</b><span class="mono small muted">${t.id}</span></a>
      ${first.includes(t) && html`<span class="tag tr-first">${tr('第一个开始')}</span>`}</li>`)}</ol>
  </section>`;
}

function AutorunSwitch({ project }) {
  const toggle = () => api.put('/api/settings', { scope: 'project', project: project.id, values: { autorun: !project.autorun } })
    .then(() => toast(project.autorun ? tr('已关闭自动开始：待开始的任务需手动开始') : tr('已开启自动开始：上游全部完成的任务将自动开始')),
      (e) => toast(tr('保存失败：{error}', { error: e.message }), { error: true }));
  return html`<button class="switch" role="switch" aria-checked=${project.autorun} onClick=${toggle}
    title=${tr('上游任务完成后自动开始下游任务；关闭后所有任务均需手动开始')}>
    <span class="knob" aria-hidden="true"></span>${tr('自动开始')}</button>`;
}

// Archive / unarchive a project (on the home page, the archived-projects page and the project settings).
// A project with running tasks cannot be archived: what they ask later would be hidden from the home page.
export function ArchiveButton({ project, small = true }) {
  const run = async () => {
    if (project.archived) {
      api.post(`/api/projects/${project.id}/unarchive`).then(() => toast(tr('已取消归档：{title}', { title: project.title })), fail);
      return;
    }
    if (project.running > 0) {
      toast(tr('项目「{title}」中还有 {n} 个运行中的任务（含后台等待和终端接管），请先停止或等待其结束后再归档。', { title: project.title, n: project.running }), { error: true });
      return;
    }
    const c = project.counts;
    if (!await confirmDialog({ title: tr('确认归档项目「{title}」？', { title: project.title }), ok: tr('归档'),
      body: html`<p>${tr('归档后，该项目不再显示在首页，调度器也不会自动开始其中的任务。项目文件不会改动，可随时在"已归档项目"中取消归档。')}</p>
        ${c.needs_you + c.in_review > 0 && html`<p class="warn-text">${tr('该项目还有 {needs} 项待确认、{review} 项待审核，归档后将不再显示在首页。', { needs: c.needs_you, review: c.in_review })}</p>`}
        ${c.queued > 0 && html`<p class="muted">${tr('{n} 个待开始的任务将保持排队，取消归档后才会开始。', { n: c.queued })}</p>`}` })) return;
    api.post(`/api/projects/${project.id}/archive`).then(() => toast(tr('已归档：{title}', { title: project.title }), { href: '#/archived', link: tr('查看已归档项目') }), fail);
  };
  return html`<button class=${`btn ${small ? 'small' : ''} ${small && !project.archived ? 'ghost' : ''}`} onClick=${run}>
    <${Icon} name="archive" />${project.archived ? tr('取消归档') : tr('归档')}</button>`;
}

// ---------------------------------------------------------------- the three panels on top

// The three panels start folded into one line, so the graph gets the first screen. Reminders
// about tasks finished long ago are already hints only (the server), not listed here.
function Attention({ pid, tasks, titleOf }) {
  const [open, setOpen] = useState(false);
  const now = useNow(30000);
  // the same order as the home page: 待确认 then 待审核, the longest waiting first
  const needs = ['needs_you', 'in_review'].flatMap((s) => tasks.filter((t) => t.status === s).sort((a, b) => a.since.localeCompare(b.since)));
  const interrupted = interruptedOf(needs);
  const unread = tasks.filter((t) => t.status === 'done' && t.unread);
  const running = tasks.filter((t) => t.status === 'running');
  const queued = tasks.filter((t) => t.status === 'queued');
  const warnings = tasks.flatMap((t) => (t.warnings || []).map((w) => ({ t, w })));
  const item = (t, text, extra = '') => html`<li><a class="att-item" href=${taskHref(pid, t.id)}>
    <${StatusMark} t=${t} short /><b>${t.title}</b><span class="att-text">${text}</span>${extra && html`<span class="muted nowrap">${extra}</span>`}</a></li>`;
  const list = (items, empty) => (items.length ? html`<ul class="att-list">${items}</ul>` : html`<div class="empty-line">${empty}</div>`);
  return html`<section class="attention" aria-label=${tr('本项目的待处理事项')}>
    <div class="att-head"><button class="btn ghost small att-toggle" aria-expanded=${open} onClick=${() => setOpen(!open)}>
      <${Icon} name="chevron" className=${open ? 'rot90' : ''} />
      <span class=${needs.length ? 'st-needs_you' : ''}><${Icon} name="needs_you" />${tr('待处理 {n}', { n: needs.length })}</span>
      <span class=${running.length ? 'st-running' : ''}><${Icon} name="running" />${tr('运行中 {n}', { n: running.length })}${queued.length ? tr('（待开始 {n}）', { n: queued.length }) : ''}</span>
      ${unread.length > 0 && html`<span><${Icon} name="unread" />${tr('最近完成 {n}', { n: unread.length })}</span>`}
      <span class=${warnings.length ? 'warn-text' : ''}><${Icon} name="warning" />${tr('记录提醒 {n}', { n: warnings.length })}</span>
      <span class="muted">${open ? tr('收起') : tr('展开')}</span></button></div>
    ${open && html`<div class="att-grid">
      <div class="att-panel att-needs"><h2 title=${tr('待确认和待审核的任务')}><${Icon} name="needs_you" />${tr('待处理 {n}', { n: needs.length })}
          ${interrupted.length >= 2 && html`<button class="btn primary small" onClick=${() => resumeAll(interrupted)}>${tr('继续全部中断任务（{n}）', { n: interrupted.length })}</button>`}</h2>
        ${needs.length ? needs.map((t) => (t.status === 'in_review' ? html`<${ReviewCard} key=${t.id} t=${t} now=${now} here />`
          : html`<${NeedsCard} key=${t.id} t=${t} now=${now} here />`)) : html`<div class="empty-line">${tr('暂无待处理事项')}</div>`}
        ${unread.length > 0 && html`<h3><${Icon} name="unread" />${tr('最近完成 {n}（未查看，不阻塞下游）', { n: unread.length })}<${MarkAllRead} tasks=${unread} /></h3>
          ${list(unread.map((t) => item(t, t.report ? tr('报告：{report}', { report: t.report }) : '', ago(t.since, now))), '')}`}
      </div>
      <div class="att-panel"><h2><${Icon} name="running" className="busy" />${tr('运行中 {n}', { n: running.length })}${queued.length ? html`<span class="muted">${tr('（待开始 {n}）', { n: queued.length })}</span>` : ''}</h2>
        ${list([...running.map((t) => item(t, statusLine(t, titleOf), since(t.since, now))),
          ...queued.map((t) => item(t, statusLine(t, titleOf)))], tr('暂无运行中的任务'))}
      </div>
      <div class="att-panel"><h2><${Icon} name="warning" />${tr('记录提醒 {n}', { n: warnings.length })}</h2>
        ${list(warnings.map(({ t, w }) => item(t, w)), tr('记录完整，无需处理'))}
      </div>
    </div>`}
  </section>`;
}

// ---------------------------------------------------------------- toolbar: search, filters, zoom, legend

function Toolbar({ view, set, types, tasks, collapse }) {
  const search = useRef();
  const menu = useRef();
  usePopover(menu);  // 视图
  useEffect(() => {
    const key = (ev) => { if (ev.key === '/' && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) { ev.preventDefault(); search.current?.focus(); } };
    addEventListener('keydown', key);
    return () => removeEventListener('keydown', key);
  }, []);
  const counts = {};
  for (const t of tasks) counts[t.status] = (counts[t.status] || 0) + 1;
  const toggle = (setName, key) => {
    const s = new Set(view[setName]);
    s.has(key) ? s.delete(key) : s.add(key);
    set({ [setName]: s });
  };
  const zoomTo = (z) => set({ zoom: Math.min(1.5, Math.max(0.4, Math.round(z * 100) / 100)) });
  // fit the whole width of the graph into the canvas
  const fit = () => {
    const canvas = document.querySelector('.canvas'), inner = document.querySelector('.canvas-inner');
    const width = parseFloat(inner?.style.width);
    if (canvas && width) zoomTo(Math.min(1, (canvas.clientWidth - 8) / width));
  };
  return html`<div class="toolbar" role="toolbar" aria-label=${tr('看板工具')}>
    <label class="search"><${Icon} name="search" /><span class="sr-only">${tr('搜索任务')}</span>
      <input ref=${search} type="search" placeholder=${tr('搜索标题、名称或下一步（/）')} value=${view.q} onInput=${(e) => set({ q: e.target.value })} /></label>
    <div class="chips" role="group" aria-label=${tr('按状态筛选（不选则显示全部）')}>
      ${STATUS_ORDER.filter((s) => counts[s]).map((s) => html`<button class=${`chip st-${s}`} aria-pressed=${view.statuses.has(s)}
        onClick=${() => toggle('statuses', s)}><${Icon} name=${STATUS[s].icon} />${STATUS[s].label} ${counts[s]}</button>`)}
    </div>
    <div class="chips" role="group" aria-label=${tr('显示的类型（泳道）')}>
      ${types.map((ty) => html`<button class="chip type" aria-pressed=${!view.hiddenTypes.has(ty.id)} onClick=${() => toggle('hiddenTypes', ty.id)}
        title=${tr('泳道 {lane}：{n} 个任务', { lane: ty.id, n: ty.count })}>${ty.id} ${ty.count}</button>`)}
    </div>
    <span class="spacer"></span>
    <div class="filters" role="group" aria-label=${tr('按日期筛选')}>
      <label class="select-inline"><span class="sr-only">${tr('日期')}</span>
        <select value=${view.window} onChange=${(e) => set({ window: e.target.value })} aria-label=${tr('按日期筛选')}
          title=${tr('任务在这段时间里有活动（从开始到完成；未完成的任务算到今天）')}>
          ${Object.keys(WINDOWS).map((k) => html`<option value=${k}>${windowLabel(k)}</option>`)}</select></label>
      ${view.window === 'custom' && html`<span class="date-range">
        <input type="date" value=${view.from} max=${view.to || undefined} onInput=${(e) => set({ from: e.target.value })} aria-label=${tr('开始日期')} />
        <span aria-hidden="true">–</span>
        <input type="date" value=${view.to} min=${view.from || undefined} onInput=${(e) => set({ to: e.target.value })} aria-label=${tr('结束日期')} /></span>`}
    </div>
    <details class="view-menu menu-wrap" ref=${menu}>
      <summary class="btn small">${tr('视图')}${collapse ? tr('：折叠已完成，') : tr('：')}${Math.round(view.zoom * 100)}%</summary>
      <div class="view-body">
        <label class="check"><input type="checkbox" checked=${collapse} onChange=${(e) => set({ collapse: e.target.checked })} />${tr('折叠已完成')}</label>
        <label class="check" title=${tr('A→B→C 已经表明 C 在 A 之后，A→C 这条线就不画了；依赖本身不变，任务详情里仍然列出')}>
          <input type="checkbox" checked=${!view.showAll} onChange=${(e) => set({ showAll: !e.target.checked })} />${tr('省略可由其他依赖推出的线')}</label>
        <div class="zoom" role="group" aria-label=${tr('缩放')}><span class="muted small">${tr('缩放')}</span>
          <button class="btn small icon-only" onClick=${() => zoomTo(view.zoom - 0.1)} aria-label=${tr('缩小')}>−</button>
          <button class="btn small ghost" onClick=${() => zoomTo(1)} title=${tr('恢复为 100%')}>${Math.round(view.zoom * 100)}%</button>
          <button class="btn small icon-only" onClick=${() => zoomTo(view.zoom + 0.1)} aria-label=${tr('放大')}>+</button>
          <button class="btn small ghost" onClick=${fit}>${tr('适应宽度')}</button>
        </div>
        <div class="legend" role="group" aria-label=${tr('显示的依赖类型')}>
          <span class="muted">${tr('显示依赖：')}</span>
          ${Object.entries(EDGE_KINDS).map(([k, v]) => html`<label title=${v.desc}><input type="checkbox" checked=${view.kinds.has(k)} onChange=${() => toggle('kinds', k)} />
            <svg width="26" height="10" aria-hidden="true"><line class=${`edge ${k}`} x1="1" y1="5" x2="25" y2="5" /></svg>${v.label}</label>`)}
          <span><span class="swatch-unmanaged" aria-hidden="true"></span>${tr('未托管（卡片左侧为点线）')}</span>
        </div>
      </div>
    </details>
  </div>`;
}

// The tasks the date filter keeps (the canvas, the table).
const kept = (view, now) => {
  const range = dateRange(view, now);
  return (t) => inDates(t, range, now);
};

// What the canvas shows after the filters: hidden tasks are left out of the layout; the date filter keeps the tasks
// active in those dates, and the direct upstreams (along the edges on display) of what it keeps, collapsed.
export function visibleTasks(tasks, edges, view, collapse, now) {
  let shown = tasks.filter((t) => !view.hiddenTypes.has(t.type) && (!view.statuses.size || view.statuses.has(t.status)));
  const keep = new Set(shown.filter(kept(view, now)).map((t) => t.id));
  const context = new Set();
  for (const e of edges) if (keep.has(e.to) && !keep.has(e.from) && view.kinds.has(e.kind)) context.add(e.from);
  shown = shown.filter((t) => keep.has(t.id) || context.has(t.id));
  const compact = new Set(shown.filter((t) => context.has(t.id) || (collapse && ['done', 'cancelled'].includes(t.status))).map((t) => t.id));
  return { shown, compact, hidden: tasks.length - shown.length };
}

function matches(t, q) {
  if (!q) return true;
  const s = q.toLowerCase();
  return [t.title, t.id, t.summary, t.progress?.next, t.last_message].some((x) => (x || '').toLowerCase().includes(s));
}

// ---------------------------------------------------------------- canvas

function Canvas({ pid, types, tasks, edges, view, set, selected, collapse, byId, onNewTask }) {
  const now = useNow(1000);
  const scroller = useRef();
  const [tip, setTip] = useState(null);
  // a lane order just saved, shown until the board comes back with it
  const [order, setOrder] = useState(null);
  useEffect(() => setOrder(null), [types.map((t) => t.id).join()]);
  const allLanes = order || types.map((t) => t.id);
  const laneIds = allLanes.filter((id) => !view.hiddenTypes.has(id));
  const countOf = Object.fromEntries(types.map((t) => [t.id, t.count]));
  const [drag, setDrag] = useState(null);   // {kind: 'card' | 'lane', id, dx, dy, to}: what is being dragged, by how much
  const dragged = useRef(false);            // a drag just ended: the click that follows it does not open the card
  const { shown, compact, hidden } = useMemo(() => visibleTasks(tasks, edges, view, collapse, Date.now()), [tasks, edges, view, collapse]);
  const shownIds = new Set(shown.map((t) => t.id));
  const visEdges = edges.filter((e) => view.kinds.has(e.kind) && shownIds.has(e.from) && shownIds.has(e.to));
  // edges another path already implies are left out unless 视图 says to draw every edge (rows stay the same)
  const implied = redundant(visEdges);
  const drawn = view.showAll ? visEdges : visEdges.filter((e) => !implied.has(e));
  const drawnKey = drawn.map((e) => `${e.from}>${e.to}:${e.kind}`).join();
  const L = useMemo(() => layout(shown, edges, laneIds, compact, drawn), [shown, edges, drawnKey, laneIds.join(), compact]);
  const lin = selected && byId.has(selected) ? lineage(selected, visEdges) : null;
  const related = (id) => !lin || id === selected || lin.up.has(id) || lin.down.has(id);
  const onPath = (e) => lin && (e.kind === 'review' ? e.from === selected || e.to === selected
    : ((e.to === selected || lin.up.has(e.to)) && lin.up.has(e.from)) || ((e.from === selected || lin.down.has(e.from)) && lin.down.has(e.to)));
  const hits = view.q ? shown.filter((t) => matches(t, view.q)) : [];
  const zoom = view.zoom;

  // keep the selected card in view
  useEffect(() => {
    const p = L.pos.get(selected), el = scroller.current;
    if (!p || !el) return;
    const x = p.x * zoom, y = p.y * zoom, w = NODE_W * zoom, h = p.h * zoom;
    const left = x < el.scrollLeft || x + w > el.scrollLeft + el.clientWidth ? Math.max(0, x - el.clientWidth / 2 + w / 2) : el.scrollLeft;
    const top = y < el.scrollTop || y + h > el.scrollTop + el.clientHeight ? Math.max(0, y - el.clientHeight / 3) : el.scrollTop;
    el.scrollTo({ left, top, behavior: 'smooth' });
  }, [selected]);

  const open = (id) => go(id === selected ? closeHref() : taskHref(pid, id));
  const edgeTip = (e) => {
    const from = byId.get(e.from)?.title || e.from, to = byId.get(e.to)?.title || e.to;
    return { title: `${from} → ${to}`, lines: [EDGE_KINDS[e.kind]?.label,
      e.children && tr('{task} 的子任务：{done}/{total} 已完成或取消', { task: from, ...e.children }),
      e.label, e.evidence && e.evidence !== e.label ? tr('prompt 原文：{evidence}', { evidence: e.evidence }) : '',
      e.source === 'prompt' ? tr('来源：prompt 中引用的任务路径') : e.kind === 'subtask' ? tr('来源：子任务 frontmatter 中的 parent')
        : tr('来源：下游任务 frontmatter 中的 depends_on')].filter(Boolean) };
  };

  // Dragging: a lane title to another place (the lane order, project.yaml lanes), a card to another lane (its type).
  // It starts once the pointer has moved a few pixels, so a click is still a click.
  const startDrag = (ev, kind, id, x0) => {
    if (ev.button !== 0) return;
    const sx = ev.clientX, sy = ev.clientY;
    let last = null;
    const move = (e) => {
      const dx = (e.clientX - sx) / zoom, dy = (e.clientY - sy) / zoom;
      if (!last && Math.hypot(e.clientX - sx, e.clientY - sy) < 6) return;
      last = { kind, id, dx, dy, to: laneIds[laneAt(x0 + dx + NODE_W / 2, laneIds.length)] };
      setDrag(last);
    };
    const end = (e) => {
      removeEventListener('pointermove', move);
      removeEventListener('pointerup', end);
      removeEventListener('pointercancel', end);
      setDrag(null);
      if (!last) return;
      dragged.current = true;
      setTimeout(() => { dragged.current = false; });
      if (e.type === 'pointerup') (kind === 'lane' ? dropLane : dropCard)(id, last.to);
    };
    addEventListener('pointermove', move);
    addEventListener('pointerup', end);
    addEventListener('pointercancel', end);
  };
  const dropLane = (id, to) => {
    if (!to || to === id) return;
    const next = moveLane(allLanes, laneIds, laneIds.indexOf(id), laneIds.indexOf(to));
    setOrder(next);
    api.put(`/api/projects/${pid}/lanes`, { order: next }).catch((e) => { setOrder(null); fail(e); });
  };
  const dropCard = async (id, to) => {
    const t = byId.get(id);
    if (!t || !to || to === t.type) return;
    if (t.status === 'running') return toast(tr('任务「{title}」正在运行，请先停止再移到其他泳道', { title: t.title }), { error: true });
    if (!await confirmMove(t, `${to}/${t.name}`)) return;
    api.post(`${paths.task(pid, id)}/rename`, { type: to, name: t.name }).then((d) => {
      toast(tr('已移到泳道 {lane}：{id}', { lane: to, id: d.id }));
      if (selected === id) go(taskHref(pid, d.id));
    }, fail);
  };
  // Alt+← / Alt+→ on a lane title moves the lane, for the keyboard
  const laneKey = (ev, id) => {
    const i = laneIds.indexOf(id), j = i + (ev.key === 'ArrowLeft' ? -1 : ev.key === 'ArrowRight' ? 1 : 0);
    if (!ev.altKey || j === i || j < 0 || j >= laneIds.length) return;
    ev.preventDefault();
    dropLane(id, laneIds[j]);
  };

  return html`<section class="canvas-wrap" aria-label=${tr('任务依赖图')}>
    ${(hidden > 0 || view.q || implied.size > 0) && html`<div class="filter-note muted small">
      ${hidden > 0 && html`<span>${tr('已按筛选条件隐藏 {n} 个任务。', { n: hidden })}</span>`}
      ${implied.size > 0 && html`<span title=${tr('A→B→C 已经表明 C 在 A 之后，A→C 这条线就不画了；依赖本身不变，任务详情里仍然列出')}>${view.showAll
        ? tr('显示全部 {n} 条依赖线。', { n: visEdges.length }) : tr('省略了 {n} 条可由其他依赖推出的线。', { n: implied.size })}
        <button class="btn ghost tiny" onClick=${() => set({ showAll: !view.showAll })}>${view.showAll ? tr('精简') : tr('全部显示')}</button></span>`}
      ${view.q && html`<span>${tr('「{q}」匹配 {n} 个任务', { q: view.q, n: hits.length })}${hits.length ? html`${tr('：')}${hits.slice(0, 6).map((t) => html`<a href=${taskHref(pid, t.id)}>${t.title}</a> `)}` : ''}</span>`}</div>`}
    <div class="canvas" ref=${scroller}>
      ${!tasks.length && html`<div class="canvas-empty"><p><b>${tr('暂无任务。')}</b>${tr('可先新建任务；如需提前规划后续步骤，可将其保存为草稿（草稿不会运行）。')}</p>
        ${onNewTask && html`<button class="btn primary" onClick=${onNewTask}><${Icon} name="plus" />${tr('新建任务')}</button>`}</div>`}
      <div class="canvas-size" style=${{ width: `${L.width * zoom}px`, height: `${L.height * zoom}px` }}>
        <div class=${`canvas-inner ${drag ? 'dragging' : ''}`} style=${{ width: `${L.width}px`, height: `${L.height}px`, transform: `scale(${zoom})` }}
          onClick=${(ev) => { if (!dragged.current && (ev.target === ev.currentTarget || ev.target.tagName === 'svg') && selected) go(closeHref()); }}>
          <svg width=${L.width} height=${L.height} class="graph">
            <defs>${Object.keys(EDGE_KINDS).map((k) => html`
              <marker id=${`arrow-${k}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">
                <path d="M0,1 L10,5 L0,9 z" class=${`arrow ${k}`} /></marker>`)}</defs>
            <text x="12" y="27" class="lane-title">${tr('开始日期')}</text>
            ${L.rows.map((r, i) => (i && r.day === L.rows[i - 1].day) ? null : html`
              ${i > 0 && !r.draft && html`<line x1="8" y1=${r.y - 5} x2=${L.width - 8} y2=${r.y - 5} class="day-rule" />`}
              ${!r.draft && html`<text x="12" y=${r.y + 19} class="day">${r.day}</text>`}`)}
            ${laneIds.map((id, i) => html`<rect x=${laneX(i) - 14} y="6" width=${NODE_W + 28} height=${L.height - 12} rx="8"
              class=${`lane-band ${drag && drag.to === id && (drag.kind === 'lane' ? drag.id !== id : byId.get(drag.id)?.type !== id) ? 'drop' : ''}`} />`)}
            ${L.draftTop != null && html`<rect x="6" y=${L.draftTop} width=${L.width - 12} height=${L.height - L.draftTop - 6} rx="8" class="draft-band" />
              <text x="12" y=${L.draftTop + 18} class="draft-title">${tr('草稿（未纳入托管，不会运行）')}</text>`}
            ${drawn.map((e, i) => {
              const cls = `edge ${e.kind} ${onPath(e) ? 'hl' : lin ? 'dim' : ''}`;
              return html`<g key=${`${e.from}>${e.to}`}>
                <path d=${L.paths[i]} class=${cls} marker-end=${`url(#arrow-${e.kind})`} />
                <path d=${L.paths[i]} class="edge-hit" onMouseMove=${(ev) => setTip({ ...edgeTip(e), x: ev.clientX, y: ev.clientY })}
                  onMouseLeave=${() => setTip(null)} onClick=${(ev) => { ev.stopPropagation(); setTip(null); open(e.to); }} /></g>`;
            })}
          </svg>
          ${laneIds.map((id, i) => {
            const d = drag?.kind === 'lane' && drag.id === id ? drag : null;
            return html`<button key=${id} class=${`lane-head ${d ? 'dragging' : ''}`} style=${{ left: `${laneX(i) - 14}px`, width: `${NODE_W + 28}px`,
              transform: d ? `translate(${d.dx}px, 0)` : undefined }} onPointerDown=${(ev) => startDrag(ev, 'lane', id, laneX(i))}
              onKeyDown=${(ev) => laneKey(ev, id)} title=${tr('泳道 {lane}：{n} 个任务。拖动标题调整泳道顺序（或 Alt+←/→）；把卡片拖到其他泳道可修改任务类型', { lane: id, n: countOf[id] ?? 0 })}>
              <span>${id}</span><${Icon} name="grip" /></button>`;
          })}
          ${shown.map((t) => {
            const p = L.pos.get(t.id);
            const d = drag?.kind === 'card' && drag.id === t.id ? drag : null;
            return html`<${Card} key=${t.id} t=${t} p=${p} titleOf=${(id) => byId.get(id)?.title || id} compact=${compact.has(t.id)} now=${now} selected=${t.id === selected}
              dim=${!related(t.id) || (view.q && !matches(t, view.q))} hit=${view.q && matches(t, view.q)} drag=${d}
              onPointerDown=${t.status === 'running' ? undefined : (ev) => startDrag(ev, 'card', t.id, p.x)}
              onClick=${() => { if (!dragged.current) open(t.id); }} />`;
          })}
        </div>
      </div>
    </div>
    ${tip && html`<div class="tooltip" style=${{ left: `${Math.min(tip.x + 14, innerWidth - 380)}px`, top: `${tip.y + 14}px` }}>
      <b>${tip.title}</b>${tip.lines.map((l) => html`<div>${l}</div>`)}</div>`}
  </section>`;
}

function Card({ t, p, titleOf, compact, now, selected, dim, hit, drag, onPointerDown, onClick }) {
  const line = statusLine(t, titleOf);
  const time = t.status === 'draft' ? '' : t.status === 'running' && t.managed ? tr('已运行 {time}', { time: since(t.since, now) }) : t.since ? (t.managed ? ago(t.since, now) : tr('更新于 {time}', { time: ago(t.since, now) })) : '';
  const cls = `card st-${t.status} ${t.managed ? '' : 'unmanaged'} ${compact ? 'compact' : ''} ${selected ? 'selected' : ''} ${dim ? 'dim' : ''} ${hit ? 'hit' : ''} ${drag ? 'dragging' : ''}`;
  const sep = tr('，');
  const label = `${t.title}${sep}${STATUS[t.status]?.label}${line ? `${sep}${line}` : ''}${t.managed ? '' : `${sep}${tr('未托管')}`}`;
  return html`<button class=${cls} style=${{ left: `${p.x}px`, top: `${p.y}px`, width: `${p.w}px`, height: `${p.h}px`,
      transform: drag ? `translate(${drag.dx}px, ${drag.dy}px)` : undefined }}
    onPointerDown=${onPointerDown} onClick=${(ev) => { ev.stopPropagation(); onClick(); }} aria-label=${label} aria-current=${selected ? 'true' : undefined}
    title=${`${t.title}\n${t.id}${t.summary ? `\n${t.summary}` : ''}${line ? `\n${line}` : ''}${t.managed ? '' : `\n${tr('未托管')}`}`}>
    <span class="c-title"><${StatusMark} t=${t} short />${t.unread ? html`<${Icon} name="unread" className="unread-dot" label=${tr('未查看')} />` : ''}<span class="ellipsis">${t.title}</span></span>
    ${!compact && html`
      <span class="c-meta"><span class="mono ellipsis">${t.name}</span><span class="nowrap">${time}</span></span>
      <span class="c-foot"><span class="c-status">${STATUS[t.status]?.label}${t.status === 'running' && t.detail?.kind === 'waiting' ? ` · ${tr('等待中')}` : ''}</span>
        <span class="ellipsis">${line}</span>
        ${t.review && !['done', 'in_review'].includes(t.status) && html`<span class="tag" title=${tr('完成后需审核')}>${tr('需审核')}</span>`}
        <${Badges} t=${t} />
        ${t.upstream_changed?.length > 0 && html`<span class="badge warn-text" title=${t.upstream_changed.map((c) => c.text).join('\n')}><${Icon} name="updated" label=${tr('上游已更新')} /></span>`}
        ${t.warnings?.length > 0 && html`<span class="warn" title=${t.warnings.join('\n')}><${Icon} name="warning" />${t.warnings.length}</span>`}</span>`}
  </button>`;
}

// What a card says besides its status: the machines it holds or waits for, its acceptance checks, its results.
function Badges({ t }) {
  const c = t.checks, r = t.results, m = t.machines || [];
  return html`${m.length > 0 && html`<span class="badge" title=${tr('机器：{machines}', { machines: m.join(', ') })}><${Icon} name="machine" />${m.length}</span>`}
    ${c && html`<span class=${`badge ${c.ok ? 'ok-text' : 'warn-text'}`} title=${c.ok ? tr('验收检查通过 {passed}/{total}', c) : tr('验收检查未通过：{failed}', { failed: c.failed.join(', ') })}>
      ${c.ok ? '✓' : '✗'}${c.passed}/${c.total}</span>`}
    ${r && html`<span class=${`badge ${r.errors ? 'warn-text' : ''}`} title=${tr('结果 {n} 个；最好的相对基线 {delta}', { n: r.count, delta: deltaText(r.best_delta_pct, r) })}>
      <${Icon} name="chart" />${deltaText(r.best_delta_pct, r)}</span>`}`;
}

// The project's machines: who holds each (one chip per holder: two only if the rule was broken), for how long, and
// whether the holder is waiting for you meanwhile.
function MachineStrip({ machines, pid }) {
  const now = useNow(30000);
  return html`<section class="machines" aria-label=${tr('机器')}>
    <span class="muted small"><${Icon} name="machine" />${tr('机器')}</span>
    ${machines.map((m) => {
      const label = `${m.name}${m.gpus ? ` · ${m.gpus} GPU` : ''}`;
      if (!m.holders.length) return html`<span class="machine free" title=${m.note || label}><span class="nowrap">${label}</span><span class="muted">${tr('空闲')}</span></span>`;
      return m.holders.map((h) => html`<a class=${`machine st-${h.status} ${h.idle_minutes > 0 ? 'idle' : ''}`} href=${h.project === pid ? taskHref(pid, h.task) : `#/p/${h.project}?t=${encodeURIComponent(`${h.project}/${h.task}`)}`}
        title=${`${m.note ? `${m.note}\n` : ''}${tr('被 {task} 占用', { task: `${h.project}/${h.task}` })}`}>
        <span class="nowrap">${label}</span><${Holder} h=${h} now=${now} /></a>`);
    })}</section>`;
}

// ---------------------------------------------------------------- task table

const SORTS = {
  status: (a, b) => STATUS_ORDER.indexOf(a.status) - STATUS_ORDER.indexOf(b.status),
  updated: (a, b) => lastActive(b).localeCompare(lastActive(a)),
  cost: (a, b) => (b.cost_usd || 0) - (a.cost_usd || 0),
  type: (a, b) => a.id.localeCompare(b.id),
  // the best result first, by its gain (the delta the right way up: the server's)
  result: (a, b) => (b.results?.best_gain ?? -Infinity) - (a.results?.best_gain ?? -Infinity),
};

function TaskTable({ pid, tasks, edges, view, titleOf, selected, tagUnmanaged }) {
  const [sort, setSort] = useState('status');
  const inView = kept(view, Date.now());
  const rows = tasks.filter((t) => !view.hiddenTypes.has(t.type) && (!view.statuses.size || view.statuses.has(t.status)) && inView(t) && matches(t, view.q))
    .sort((a, b) => SORTS[sort](a, b) || lastActive(b).localeCompare(lastActive(a)));
  const th = (key, label) => html`<th aria-sort=${sort === key ? 'descending' : undefined}>
    <button class="th-sort" onClick=${() => setSort(key)}>${label}${sort === key ? ' ↓' : ''}</button></th>`;
  const names = (ids) => ids.map((id) => html`<div><a href=${taskHref(pid, id)}>${titleOf(id)}</a></div>`);
  const shownEdges = edges.filter((e) => view.kinds.has(e.kind));
  return html`<section class="table-wrap" aria-label=${tr('任务列表')}>
    <h2>${tr('任务列表')} <span class="muted small">${tr('{shown} / {total}，任务较多时用于查找和排序', { shown: rows.length, total: tasks.length })}</span></h2>
    <table class="tasks">
      <thead><tr>${th('status', tr('状态'))}<th>${tr('任务')}</th>${th('type', tr('类型'))}<th>${tr('机器')}</th>${th('result', tr('结果'))}<th>${tr('状态说明')}</th>${th('cost', tr('花费'))}${th('updated', tr('更新时间'))}<th>${tr('上游')}</th><th>${tr('下游')}</th></tr></thead>
      <tbody>${!rows.length && html`<tr><td colspan="10" class="muted">${tasks.length ? tr('无符合筛选条件的任务') : tr('暂无任务')}</td></tr>`}
        ${rows.map((t) => html`<tr class=${t.id === selected ? 'selected' : ''}>
        <td class="nowrap"><${StatusMark} t=${t} /></td>
        <td><a href=${taskHref(pid, t.id)}><b>${t.title}</b></a>${t.unread ? html` <${Icon} name="unread" className="unread-dot" label=${tr('未查看')} />` : ''}
          <div class="mono small muted">${t.id}</div>${tagUnmanaged && !t.managed && html`<span class="tag">${tr('未托管')}</span>`}</td>
        <td>${t.type}</td>
        <td class="mono small">${(t.machines || []).join(', ')}</td>
        <td class="num">${t.results ? deltaText(t.results.best_delta_pct, t.results) : ''}</td>
        <td class="next">${statusLine(t, titleOf) || t.progress?.next || ''}${(t.warnings || []).map((w) => html`<div class="warn small"><${Icon} name="warning" />${w}</div>`)}</td>
        <td class="num">${t.managed ? money(t.cost_usd) : ''}</td>
        <td class="num nowrap">${clock(lastActive(t))}</td>
        <td class="small">${names(shownEdges.filter((e) => e.to === t.id).map((e) => e.from))}</td>
        <td class="small">${names(shownEdges.filter((e) => e.from === t.id).map((e) => e.to))}</td></tr>`)}</tbody>
    </table></section>`;
}
