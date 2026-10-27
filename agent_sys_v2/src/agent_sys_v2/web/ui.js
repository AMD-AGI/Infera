// Shared pieces: htm binding, status vocabulary, formatting, markdown, icons, small widgets and hooks.
import { h } from './vendor/preact.module.js';
import { useState, useEffect, useRef, useCallback, useMemo } from './vendor/hooks.module.js';
import htm from './vendor/htm.module.js';
import { marked } from './vendor/marked.esm.js';
import DOMPurify from './vendor/purify.es.mjs';
import { api } from './api.js';
import { t, list } from './i18n.js';

export const html = htm.bind(h);

// ---------------------------------------------------------------- status vocabulary

// Every status is shown as icon + word + color; the color never stands alone. Labels are looked up when read (the
// language is fixed in the browser, but tests switch it).
export const STATUS = {
  draft: { get label() { return t('草稿'); }, icon: 'draft' },
  queued: { get label() { return t('待开始'); }, icon: 'queued' },
  running: { get label() { return t('运行中'); }, icon: 'running' },
  needs_you: { get label() { return t('待确认'); }, icon: 'needs_you' },
  in_review: { get label() { return t('待审核'); }, icon: 'in_review' },
  done: { get label() { return t('已完成'); }, icon: 'done' },
  cancelled: { get label() { return t('已取消'); }, icon: 'cancelled' },
  unknown: { get label() { return t('其他'); }, icon: 'unknown' },
};
export const STATUS_ORDER = ['needs_you', 'in_review', 'running', 'queued', 'done', 'draft', 'cancelled', 'unknown'];

// 疑似卡住 (detail.stall, judged by the server from the run's events): which of the two, and for how long.
export function stallWhy(s, now = Date.now()) {
  const time = since(s.since, now);
  if (s.kind === 'timeout') return t('前台命令超时未返回（{tool} 已运行 {time}，超时 {timeout}）', { tool: s.tool, time, timeout: duration(s.timeout * 1000) });
  if (!s.tool) return t('模型调用无响应，已 {time}没有任何事件', { time });
  if (['Agent', 'Task'].includes(s.tool)) return t('前台子 agent 已 {time}没有任何事件', { time });
  return t('{tool} 已 {time}没有返回，也没有任何事件', { tool: s.tool, time });
}
export const stallText = (s, now = Date.now()) => t('疑似卡住：{why}', { why: stallWhy(s, now) });

// 正在执行：<command>, and from a minute on how long it has run and its own timeout (not stuck while under it).
function doing(d, now) {
  const what = t('正在执行：{what}', { what: d.current });
  const ran = d.current_since ? now - Date.parse(d.current_since) : 0;
  if (ran < 60000) return what;
  const time = duration(ran);
  return `${what} · ${d.current_timeout ? t('已运行 {time}（超时 {timeout}）', { time, timeout: duration(d.current_timeout * 1000) }) : t('已运行 {time}', { time })}`;
}

// The sub-state a card shows next to the status: what it is doing or waiting for.
export function statusLine(task, titleOf = (id) => id, now = Date.now()) {
  const d = task.detail || {};
  const next = task.progress?.next;
  switch (`${task.status}:${d.kind || ''}`) {
    case 'queued:upstream': return t('等待上游任务：{tasks}', { tasks: list((d.tasks || []).map(titleOf)) });
    case 'queued:children': return t('等待 {parents} 的子任务：{tasks}', { parents: list((d.parents || []).map(titleOf)), tasks: list((d.tasks || []).map(titleOf)) });
    case 'queued:concurrency': return d.scope === 'global' ? t('等待运行名额（全局上限 {n} 个）', { n: d.limit })
      : t('等待运行名额（项目上限 {n} 个）', { n: d.limit });
    case 'queued:machines': return t('等待机器：{machines}', { machines: list((d.machines || []).map((m) => t('{name}（被 {task} 占用）', { name: m.name, task: m.title || m.task }))) });
    case 'queued:template_review': return t('等待你审核模板：审核后在看板顶部点"开始运行"');
    case 'queued:manual': return t('已关闭自动开始，需手动开始');
    case 'queued:archived': return t('项目已归档，任务不会自动开始');
    case 'queued:invalid': return t('prompt 有错误，改正后才会开始：{error}', { error: d.error });
    case 'queued:': return t('上游已满足，等待调度');
    case 'running:waiting': return t('等待中：{what}', { what: next || d.waiting_for || t('后台命令') });
    case 'running:terminal': return d.current ? t('正在执行：{what}', { what: d.current }) : t('终端接管中');  // claude exited: the checks
    case 'running:unmanaged': return next || t('进行中');
    case 'running:working': return d.host ? t('在 {host} 上运行', { host: d.host }) : d.stall ? stallText(d.stall, now) : d.current ? doing(d, now) : t('运行中');
    case 'needs_you:asked': return t('待答复：{what}', { what: next || brief(task.last_message) });
    case 'needs_you:questions': return d.count ? t('{n} 个问题待回答（{path}）', { n: d.count, path: d.path || 'questions.md' }) : t('有问题待回答（{path}）', { path: d.path || 'questions.md' });
    case 'needs_you:stopped': return t('已手动停止');
    case 'needs_you:interrupted': return d.error ? t('运行出错：{error}', { error: d.error }) : t('运行中断（runner 进程已退出）');
    case 'needs_you:wait_interrupted': return t('后台等待中断：{what}', { what: d.waiting_for || '' });
    case 'needs_you:budget': return t('预算已用尽：{cost} / {budget}', { cost: money(task.cost_usd), budget: money(task.budget_usd) });
    case 'needs_you:checks_failed': return d.failed?.length ? t('验收未通过：{failed}', { failed: list(d.failed) }) : t('验收检查没有运行完');
    case 'needs_you:joined': return t('已纳入托管，请先确认终端中的会话已结束');
    case 'needs_you:unmanaged': return t('阻塞：{what}', { what: next || '' });
    case 'in_review:review': return t('报告：{file}', { file: task.report || 'report.md' });
    case 'done:': return next && !/^(无|none\b)/i.test(next) ? t('下一步：{next}', { next }) : '';  // i18n: reads the agent's progress.md
    case 'cancelled:': return '';
    case 'unknown:': return t('progress 状态：{raw}', { raw: d.raw || '' });
    default: return '';
  }
}

// The first line of a text (the agent's last message is markdown and can be long), at most n characters.
export function brief(text, n = 120) {
  const line = (text || '').split('\n').map((x) => x.trim()).find(Boolean) || '';
  return line.length > n ? `${line.slice(0, n)}…` : line;
}

// Icon name for a task: the status icon, refined by the running sub-state.
function iconOf(t) {
  if (t.status === 'running' && t.detail?.stall) return 'warning';
  if (t.status === 'running') return { waiting: 'waiting', terminal: 'terminal' }[t.detail?.kind] || 'running';
  return STATUS[t.status]?.icon || 'unknown';
}

export const EDGE_KINDS = {
  input: { get label() { return t('产出依赖'); }, get desc() { return t('使用上游的结果、报告或数据；阻塞下游'); } },
  code: { get label() { return t('代码基线'); }, get desc() { return t('基于上游的分支、commit 或镜像；阻塞下游'); } },
  children: { get label() { return t('等子任务'); }, get desc() { return t('上游完成、且它的子任务全部完成或取消后才开始（不含本任务和它的下游）；阻塞下游'); } },
  review: { get label() { return t('评审覆盖'); }, get desc() { return t('评审范围包含上游的改动；不阻塞下游，不计入上下游追溯'); } },
  subtask: { get label() { return t('父子任务'); }, get desc() { return t('由父任务拆分出的子任务；不阻塞下游'); } },
};

// ---------------------------------------------------------------- formatting

export const money = (v) => (v == null ? '—' : `$${Number(v).toFixed(2)}`);
const pad = (n) => String(n).padStart(2, '0');
export function clock(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}
export function duration(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return t('{n} 秒', { n: s });
  const m = Math.floor(s / 60);
  if (m < 60) return t('{n} 分钟', { n: m });
  const hrs = Math.floor(m / 60);
  if (hrs < 48) return m % 60 ? t('{h} 小时 {m} 分', { h: hrs, m: m % 60 }) : t('{h} 小时', { h: hrs });
  return t('{d} 天 {h} 小时', { d: Math.floor(hrs / 24), h: hrs % 24 });
}
// Elapsed time as a running clock: 1:02:11 / 4:05.
export function stopwatch(ms) {
  const s = Math.max(0, Math.floor(ms / 1000));
  const hh = Math.floor(s / 3600), mm = Math.floor((s % 3600) / 60), ss = s % 60;
  return hh ? `${hh}:${pad(mm)}:${pad(ss)}` : `${mm}:${pad(ss)}`;
}
export const ago = (iso, now = Date.now()) => (iso ? t('{time}前', { time: duration(now - Date.parse(iso)) }) : '');
export const since = (iso, now = Date.now()) => (iso ? duration(now - Date.parse(iso)) : '');
export const bytes = (n) => (n == null ? '' : n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`);

// ---------------------------------------------------------------- markdown

marked.setOptions({ gfm: true, breaks: false });
// What the agent writes is shown, not trusted: no scripts (DOMPurify's default), and no forms, controls or styles
// either, which could restyle agent_sys's page, cover it, or send a request to agent_sys's own API when clicked.
const SANITIZE = {
  FORBID_TAGS: ['style', 'form', 'input', 'button', 'select', 'option', 'textarea', 'dialog'],
  FORBID_ATTR: ['style'],
};
// Rendered markdown, sanitized after parsing (marked does not sanitize). Relative image links are
// resolved inside the task's workspace through api.rawUrl. Parsed again only when the text changes: a long run
// output re-renders often.
export function Markdown({ text, base, className = '', onLink }) {
  const ref = useRef();
  const out = useMemo(() => withImages(DOMPurify.sanitize(marked.parse(text || ''), SANITIZE), base),
    [text, base?.project, base?.task, base?.dir]);
  useEffect(() => {
    for (const a of ref.current?.querySelectorAll('a[href]') || []) {
      const href = a.getAttribute('href');
      if (/^https?:/.test(href)) { a.target = '_blank'; a.rel = 'noreferrer'; continue; }
      if (onLink) a.onclick = (ev) => { ev.preventDefault(); onLink(joinPath(base?.dir || '', href)); };
    }
  });
  return html`<div ref=${ref} class=${`md ${className}`} dangerouslySetInnerHTML=${{ __html: out }} />`;
}

// Point relative image links at the task's workspace. Done in an inert <template> so the browser
// never requests the unresolved path.
function withImages(htmlText, base) {
  if (!base || !htmlText.includes('<img')) return htmlText;
  const tpl = document.createElement('template');
  tpl.innerHTML = htmlText;
  for (const img of tpl.content.querySelectorAll('img')) {
    const src = img.getAttribute('src') || '';
    if (src && !/^(https?:|data:|\/)/.test(src)) img.setAttribute('src', api.rawUrl(base.project, base.task, joinPath(base.dir || '', src)));
  }
  return tpl.innerHTML;
}

function joinPath(dir, rel) {
  const parts = (dir ? `${dir}/${rel}` : rel).split('/');
  const out = [];
  for (const p of parts) {
    if (p === '..') out.pop();
    else if (p && p !== '.') out.push(p);
  }
  return out.join('/');
}

// ---------------------------------------------------------------- icons

// 16×16 line icons; each status has its own shape so state never depends on color alone.
const PATHS = {
  queued: '<circle cx="8" cy="8" r="5.5"/>',
  // an agent at work: a core with an arc that orbits it (style.css turns the arc while it works)
  running: '<circle cx="8" cy="8" r="5.6" class="track"/><path class="orbit" d="M8 2.4a5.6 5.6 0 0 1 5.6 5.6"/><circle cx="8" cy="8" r="1.7" class="fill"/>',
  // the logo: a tile with a line of tasks climbing, step by step, to a better point (also favicon.svg and index.html)
  logo: '<rect x="1" y="1" width="14" height="14" rx="3.8" class="fill"/><path class="knock" d="M4.2 11.6 7.2 8.2 9.2 9.8 11.8 4.6"/>'
    + '<circle cx="4.2" cy="11.6" r="1.1" class="knock-fill"/><circle cx="7.2" cy="8.2" r="1.1" class="knock-fill"/><circle cx="9.2" cy="9.8" r="1.1" class="knock-fill"/><circle cx="11.8" cy="4.6" r="1.25" class="knock-fill"/>',
  waiting: '<path d="M4.5 2.5h7M4.5 13.5h7M5.5 2.5c0 3 5 3.2 5 5.5s-5 2.5-5 5.5M10.5 2.5c0 3-5 3.2-5 5.5"/>',
  terminal: '<rect x="1.8" y="3" width="12.4" height="10" rx="1.6"/><path d="M4.6 6.4 6.8 8.3 4.6 10.2M8.3 10.3h3"/>',
  needs_you: '<circle cx="8" cy="8" r="6" class="fill"/><path d="M8 4.6v4.2" class="knock"/><circle cx="8" cy="11.1" r=".25" class="knock"/>',
  in_review: '<path d="M8 1.8 14.2 8 8 14.2 1.8 8z" class="fill"/><path d="M5.6 8.1 7.3 9.7 10.5 6.4" class="knock"/>',
  done: '<circle cx="8" cy="8" r="6"/><path d="M5.2 8.2 7.2 10.1 10.9 6.1"/>',
  cancelled: '<circle cx="8" cy="8" r="5.6"/><path d="M4.1 11.9 11.9 4.1"/>',
  draft: '<circle cx="8" cy="8" r="5.6" stroke-dasharray="1.6 2.1"/>',
  unknown: '<circle cx="8" cy="8" r="5.6"/><path d="M6.3 6.4c0-1 .8-1.7 1.8-1.7s1.7.7 1.7 1.6c0 1.3-1.7 1.4-1.7 2.6"/><circle cx="8.1" cy="11.2" r=".2"/>',
  warning: '<path d="M8 2.2 14.3 13.3H1.7z"/><path d="M8 6.4v3.3M8 11.5v.1"/>',
  unread: '<circle cx="8" cy="8" r="3.4" class="fill"/>',
  x: '<path d="M3.5 3.5 12.5 12.5M12.5 3.5 3.5 12.5"/>',
  plus: '<path d="M8 3v10M3 8h10"/>',
  search: '<circle cx="7" cy="7" r="4.2"/><path d="M10.2 10.2 13.5 13.5"/>',
  copy: '<rect x="5.5" y="5.5" width="7.5" height="7.5" rx="1.2"/><path d="M3 10.3V4.2C3 3.5 3.5 3 4.2 3h6"/>',
  chevron: '<path d="M6 3.8 10.2 8 6 12.2"/>',
  stop: '<rect x="4" y="4" width="8" height="8" rx="1.2" class="fill"/>',
  play: '<path d="M5.2 3.4v9.2L12.6 8z" class="fill"/>',
  sun: '<circle cx="8" cy="8" r="2.8"/><path d="M8 1.5v1.6M8 12.9v1.6M1.5 8h1.6M12.9 8h1.6M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M3.4 12.6l1.1-1.1M11.5 4.5l1.1-1.1"/>',
  moon: '<path d="M12.8 10.4A5.6 5.6 0 0 1 5.6 3.2a5.6 5.6 0 1 0 7.2 7.2z"/>',
  auto: '<circle cx="8" cy="8" r="5.6"/><path d="M8 2.4v11.2A5.6 5.6 0 0 0 8 2.4z" class="fill"/>',
  dot: '<circle cx="8" cy="8" r="2.6" class="fill"/>',
  tool: '<path d="M9.8 2.6a3.2 3.2 0 0 0-3.9 4.2L2.3 10.4a1.3 1.3 0 0 0 1.8 1.8l3.6-3.6a3.2 3.2 0 0 0 4.2-3.9L10 6.6 8.7 6.2 8.3 4.9z"/>',
  bg: '<path d="M2.5 8h2l1.5-3.5 2.2 7 1.7-5 1 1.5h2.6"/>',
  grip: '<circle cx="6" cy="4" r=".8" class="fill"/><circle cx="10" cy="4" r=".8" class="fill"/><circle cx="6" cy="8" r=".8" class="fill"/><circle cx="10" cy="8" r=".8" class="fill"/><circle cx="6" cy="12" r=".8" class="fill"/><circle cx="10" cy="12" r=".8" class="fill"/>',
  machine: '<rect x="2.2" y="2.5" width="11.6" height="4.4" rx="1"/><rect x="2.2" y="9.1" width="11.6" height="4.4" rx="1"/><path d="M4.6 4.7h.1M4.6 11.3h.1M7.4 4.7h4M7.4 11.3h4"/>',
  chart: '<path d="M2.2 13.6h11.6M3.4 11 6.3 7.4l2.6 2.1 4.2-5.3"/><circle cx="13.1" cy="4.2" r=".5" class="fill"/>',
  updated: '<path d="M13.2 8.6A5.3 5.3 0 1 1 11.6 4"/><path d="M12.4 1.6v2.9H9.5"/>',
  archive: '<rect x="2" y="3" width="12" height="3.2" rx=".8"/><path d="M3.2 6.2v6.3c0 .6.4 1 1 1h7.6c.6 0 1-.4 1-1V6.2M6.5 9h3"/>',
};
export function Icon({ name, size = 16, label, className = '' }) {
  return html`<svg class=${`icon icon-${name} ${className}`} width=${size} height=${size} viewBox="0 0 16 16"
    aria-hidden=${label ? undefined : 'true'} role=${label ? 'img' : undefined} aria-label=${label}
    dangerouslySetInnerHTML=${{ __html: PATHS[name] || '' }} />`;
}

// Status mark: icon + label in the status color.
export function StatusMark({ t: task, short = false }) {
  const s = STATUS[task.status] || STATUS.unknown;
  const stall = task.status === 'running' && task.detail?.stall;
  const sub = stall ? t('疑似卡住') : task.status === 'running' ? { waiting: t('等待中'), terminal: t('终端接管') }[task.detail?.kind] : '';
  return html`<span class=${`status st-${task.status} ${sub ? `sub-${stall ? 'stall' : task.detail.kind}` : ''}`}>
    <${Icon} name=${iconOf(task)} />${short ? '' : html`<span>${s.label}${sub ? ` · ${sub}` : ''}</span>`}</span>`;
}

// ---------------------------------------------------------------- hooks

// Data from the API that refreshes when the server reports a change in the given scope.
// scope: (change) => boolean. Changes that arrive together (one write touches several tasks) cause one reload.
export function useApi(path, scope = () => true) {
  const [state, set] = useState({ data: null, error: null, loading: true });
  const seq = useRef(0);
  const timer = useRef(0);
  const load = useCallback(() => {
    if (!path) return Promise.resolve();
    const n = ++seq.current;
    return api.get(path).then((data) => n === seq.current && set({ data, error: null, loading: false }),
      (error) => n === seq.current && set((s) => ({ data: s.data, error, loading: false })));
  }, [path]);
  useEffect(() => {
    set({ data: null, error: null, loading: !!path });
    load();
    const off = api.onChange((c) => {
      if (c.type !== 'reconnected' && (c.type !== 'changed' || !scope(c))) return;
      clearTimeout(timer.current);
      timer.current = setTimeout(load, 50);
    });
    return () => { off(); clearTimeout(timer.current); };
  }, [path]);
  return { ...state, reload: load };
}

// Re-render every `ms` (live durations).
export function useNow(ms = 1000) {
  const [now, set] = useState(Date.now());
  useEffect(() => { const id = setInterval(() => set(Date.now()), ms); return () => clearInterval(id); }, [ms]);
  return now;
}

// Esc closes (a drawer), unless a dialog is open over it.
export function useEscape(close) {
  useEffect(() => {
    const k = (ev) => { if (ev.key === 'Escape' && !document.querySelector('dialog[open]')) close(); };
    addEventListener('keydown', k);
    return () => removeEventListener('keydown', k);
  }, []);
}

// A <details> popover (ref) closes when you click elsewhere or press Esc (which then closes nothing else).
export function usePopover(ref) {
  useEffect(() => {
    const click = (ev) => { if (ref.current?.open && !ref.current.contains(ev.target)) ref.current.open = false; };
    const key = (ev) => { if (ev.key === 'Escape' && ref.current?.open) { ev.stopPropagation(); ref.current.open = false; } };
    addEventListener('click', click);
    addEventListener('keydown', key, true);
    return () => { removeEventListener('click', click); removeEventListener('keydown', key, true); };
  }, []);
}

// An id for a label's `for`, unique on the page (the same widget may show twice: a card and the drawer).
export const useId = (prefix) => useMemo(() => `${prefix}-${Math.random().toString(36).slice(2, 8)}`, []);

// A form filled from what the server sent: {value, base} (base: the data it was filled from, stale: the data changed
// since). A reload refills it while you have not changed it (or once it matches what you saved); otherwise your edits
// stay, and stale says the file changed meanwhile.
export function refill(state, next) {
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  if (!state || same(state.value, state.base) || same(state.value, next)) return { value: next, base: next, stale: false };
  return { ...state, stale: !same(state.base, next) };
}

// ---------------------------------------------------------------- small widgets

export function Copy({ text, label = t('复制') }) {
  const [done, set] = useState(false);
  const copy = () => {
    const ok = () => { set(true); setTimeout(() => set(false), 1400); };
    if (navigator.clipboard) navigator.clipboard.writeText(text).then(ok, () => fallbackCopy(text) && ok());
    else if (fallbackCopy(text)) ok();
  };
  return html`<button type="button" class="btn ghost small" onClick=${copy} title=${t('复制：{text}', { text })}>
    <${Icon} name="copy" />${done ? t('已复制') : label}</button>`;
}
function fallbackCopy(text) {
  const ta = document.createElement('textarea');
  ta.value = text;
  document.body.appendChild(ta);
  ta.select();
  const ok = document.execCommand('copy');
  ta.remove();
  return ok;
}

// Error box with a retry button: what failed, the server's message, and how to go on.
export function ErrorBox({ what, error, retry }) {
  return html`<div class="error-box" role="alert">
    <${Icon} name="warning" /><div><b>${t('{what}失败', { what })}</b><div class="muted">${String(error?.message || error)}</div></div>
    ${retry && html`<button class="btn small" onClick=${retry}>${t('重试')}</button>`}</div>`;
}

// Toasts: short confirmations of what just happened, with an optional link.
const toastListeners = new Set();
export function toast(text, opts = {}) { for (const fn of toastListeners) fn({ id: Math.random(), text, ...opts }); }
// A request that failed: its message as an error toast; a start refused by the check before a start lists its
// problems in a dialog (the reply's problems).
export const fail = (e) => (e.data?.problems
  ? problemsDialog(e.data.project, e.message.split('\n')[0], e.data.problems)
  : toast(e.message, { error: true }));

// What the check before a start found (preflight): errors, then warnings, each with its task and how to fix it.
export function Problems({ pid, problems }) {
  return html`<ul class="problems small">${problems.map((p) => html`<li class=${p.level}>
    <${Icon} name="warning" /><div><b class="p-level">${p.level === 'error' ? t('错误') : t('提醒')}</b>
      ${p.task ? html`<a class="mono" href=${taskHref(pid, p.task, { tab: 'task' })}>${p.task}</a>` : html`<a class="mono" href=${`#/p/${pid}/settings`}>project.yaml</a>`}
      <span>${p.message}</span>${p.fix && html`<div class="p-fix muted">→ ${p.fix}</div>`}</div></li>`)}</ul>`;
}
export const problemsDialog = (pid, title, problems) => confirmDialog({ title, cancel: t('关闭'),
  body: html`<${Problems} pid=${pid} problems=${problems} />` });
export function Toasts() {
  const [shown, set] = useState([]);
  useEffect(() => {
    const fn = (x) => {
      set((l) => [...l.slice(-3), x]);
      setTimeout(() => set((l) => l.filter((y) => y.id !== x.id)), x.error ? 9000 : 5000);
    };
    toastListeners.add(fn);
    return () => toastListeners.delete(fn);
  }, []);
  return html`<div class="toasts" role="status" aria-live="polite">${shown.map((x) => html`
    <div class=${`toast ${x.error ? 'error' : ''}`} key=${x.id}>${x.error && html`<${Icon} name="warning" />`}<span>${x.text}</span>
      ${x.href && html`<a href=${x.href}>${x.link || t('查看')}</a>`}</div>`)}</div>`;
}

// A page while its data loads. The skip link focuses <main id="main">: tabIndex -1 lets it take the focus.
export const Loading = ({ text = t('正在加载…') }) => html`<main id="main" tabIndex="-1" class="page"><div class="loading">${text}</div></main>`;

// Who holds a machine (a task summary) and, while it waits for you holding it, for how long.
export const Holder = ({ h, now }) => html`<${StatusMark} t=${h} short /><span class="ellipsis">${h.title}</span>
  ${h.idle_minutes > 0 && html`<span class="warn-text small nowrap">${t('闲置占用 {time}', { time: since(h.since, now) })}</span>`}`;

// Modal dialog on <dialog>: focus is trapped by the browser, Esc closes.
export function Dialog({ title, children, onClose }) {
  const ref = useRef();
  useEffect(() => {
    const d = ref.current;
    d.showModal();
    const close = (ev) => { ev.preventDefault(); onClose(); };
    d.addEventListener('cancel', close);
    return () => d.removeEventListener('cancel', close);
  }, []);
  return html`<dialog ref=${ref} class="dialog" aria-labelledby="dialog-title">
    <div class="dialog-head"><h2 id="dialog-title">${title}</h2>
      <button class="btn icon-only ghost" onClick=${onClose} aria-label=${t('关闭')}><${Icon} name="x" /></button></div>
    <div class="dialog-body">${children}</div></dialog>`;
}

// Confirm dialog driven by a promise: `await confirm({...})` → true / false. Without ok it only tells (cancel closes it);
// a link in it to a page of the app closes it too.
let confirmSetter = null;
export function confirmDialog(opts) { return new Promise((resolve) => confirmSetter?.({ ...opts, resolve })); }
export function ConfirmHost() {
  const [c, set] = useState(null);
  useEffect(() => { confirmSetter = set; return () => { confirmSetter = null; }; }, []);
  if (!c) return null;
  const done = (v) => { set(null); c.resolve(v); };
  return html`<${Dialog} title=${c.title} onClose=${() => done(false)}>
    <div class="confirm-body" onClick=${(ev) => ev.target.closest('a[href^="#"]') && done(false)}>${c.body}</div>
    <div class="dialog-actions">
      <button class="btn ghost" onClick=${() => done(false)}>${c.cancel || t('取消')}</button>
      ${c.ok && html`<button class=${`btn ${c.danger ? 'danger' : 'primary'}`} onClick=${() => done(true)} autofocus>${c.ok}</button>`}
    </div></${Dialog}>`;
}

// The models the server offers (/api/models: [{id, label, resolves}]), fetched once.
let models = null;
function useModels() {
  const [m, set] = useState(null);
  useEffect(() => { (models ||= api.get('/api/models').catch((e) => { models = null; throw e; })).then(set, () => {}); }, []);
  return m;
}

// A model: one of the offered ones (an alias says what it stands for here), or any other name typed in.
export function ModelSelect({ value, onChange }) {
  const m = useModels();
  const [custom, setCustom] = useState(false);
  const offered = m?.models || [];
  const typing = custom || (!!m && !offered.some((x) => x.id === value));
  const option = (x) => html`<option value=${x.id}>${x.id.includes('-') ? t('{label}（{id}）', x)
    : t('{id}（别名 → {model}）', { id: x.id, model: x.resolves || t('Claude Code 的默认') })}</option>`;
  const pick = (e) => { const v = e.target.value; setCustom(!v); if (v) onChange(v); };
  return html`<span class="model-select">
    <select value=${typing ? '' : value} onChange=${pick} aria-label="model">
      ${!m && html`<option value=${value}>${value}</option>`}
      ${m && html`<optgroup label=${t('模型 ID')}>${offered.filter((x) => x.id.includes('-')).map(option)}</optgroup>
        <optgroup label=${t('别名')}>${offered.filter((x) => !x.id.includes('-')).map(option)}</optgroup>`}
      <option value="">${t('其他模型…')}</option></select>
    ${typing && html`<input class="mono" value=${value} onInput=${(e) => onChange(e.target.value.trim())}
      placeholder=${t('模型名，例如 claude-opus-5-5')} aria-label=${t('模型名')} />`}</span>`;
}

// effort (claude --effort) and ultracode, Claude Code's multi-agent workflows (on with any effort), in the forms that
// set a task's agent: onChange gets {effort} or {ultracode}.
const EFFORTS = ['low', 'medium', 'high', 'xhigh', 'max'];
export function EffortSelect({ value, onChange }) {
  return html`<label class="field">effort<select value=${value} onChange=${(e) => onChange({ effort: e.target.value })}
    title=${t('xhigh 介于 high 和 max 之间；模型不支持时按 high 运行')}>${EFFORTS.map((x) => html`<option>${x}</option>`)}</select></label>`;
}
export function UltracodeCheck({ value, onChange }) {
  return html`<label class="check"><input type="checkbox" checked=${!!value} onChange=${(e) => onChange({ ultracode: e.target.checked })} />
    <span>ultracode <span class="muted small">${t('agent 用 Claude Code 的多 agent 工作流做每个实质性的步骤：更彻底，花费也高得多（预算上限照常有效）')}</span></span></label>`;
}

// Text area that grows with its content; Ctrl/⌘+Enter submits. onKeyDown runs first and may take the key
// (preventDefault); textRef gets the textarea.
export function AutoText({ value, onInput, onSubmit, onKeyDown, textRef, rows = 2, ...rest }) {
  const own = useRef();
  const ref = textRef || own;
  useEffect(() => {
    const el = ref.current;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight + 2, 320)}px`;
  }, [value]);
  const key = (ev) => {
    onKeyDown?.(ev);
    if (!ev.defaultPrevented && ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey) && onSubmit) { ev.preventDefault(); onSubmit(); }
  };
  return html`<textarea ref=${ref} rows=${rows} value=${value} onInput=${(e) => onInput(e.target.value)} onKeyDown=${key} ...${rest} />`;
}

// ---------------------------------------------------------------- @ mentions of tasks in a prompt

// The @ being typed before the caret: {start, query} (start = the index of the @), or null. An @ right after a
// letter or digit (an e-mail address, git@host) is not one.
export function mentionAt(text, caret) {
  const m = /(?:^|[^\w\-@])@([\w\-/]*)$/.exec(text.slice(0, caret));
  return m ? { start: caret - m[1].length - 1, query: m[1] } : null;
}

// The tasks an @query offers, [{id, title}]: id or title containing it, those whose name starts with it first.
export function mentionOptions(tasks, query, n = 8) {
  const q = query.toLowerCase();
  const hit = tasks.filter((x) => x.id.toLowerCase().includes(q) || (x.title || '').toLowerCase().includes(q));
  const first = (x) => (x.id.toLowerCase().startsWith(q) || x.id.split('/')[1]?.toLowerCase().startsWith(q) ? 0 : 1);
  return hit.sort((a, b) => first(a) - first(b)).slice(0, n);
}

// The text with the @query at start replaced by @<id> and a space: {text, caret} (the caret after the space).
export function insertMention(text, mention, id) {
  const end = mention.start + 1 + mention.query.length;
  const head = `${text.slice(0, mention.start)}@${id} `;
  return { text: head + text.slice(end).replace(/^ /, ''), caret: head.length };
}

// Where the character at pos of a textarea is drawn, relative to the textarea's box: a hidden copy of the
// textarea's text up to pos, with the same font, padding and width, measured.
function caretXY(el, pos) {
  const cs = getComputedStyle(el);
  const div = document.createElement('div');
  for (const k of ['fontFamily', 'fontSize', 'fontWeight', 'lineHeight', 'letterSpacing', 'tabSize', 'paddingTop',
    'paddingRight', 'paddingBottom', 'paddingLeft', 'borderTopWidth', 'borderRightWidth', 'borderBottomWidth',
    'borderLeftWidth', 'borderStyle', 'boxSizing', 'width']) div.style[k] = cs[k];
  Object.assign(div.style, { position: 'absolute', visibility: 'hidden', top: '0', left: '0', whiteSpace: 'pre-wrap', overflowWrap: 'break-word' });
  div.textContent = el.value.slice(0, pos);
  const mark = document.createElement('span');
  mark.textContent = '@';
  div.appendChild(mark);
  document.body.appendChild(div);
  const xy = { top: mark.offsetTop + mark.offsetHeight - el.scrollTop, left: Math.min(mark.offsetLeft, el.clientWidth - 260) };
  div.remove();
  return xy;
}

// A prompt editor (AutoText) where typing @ lists the project's tasks; picking one writes @<type>/<name>, which the
// server reads as a dependency, as it reads a task's path in the prompt. tasks: [{id, title}].
export function MentionText({ tasks = [], value, onInput, ...rest }) {
  const ref = useRef();
  const [menu, setMenu] = useState(null);  // {start, query, sel, top, left}
  const look = () => {
    const el = ref.current;
    const m = el.selectionStart === el.selectionEnd ? mentionAt(el.value, el.selectionStart) : null;
    setMenu((x) => m && { ...m, sel: x?.start === m.start ? x.sel : 0, ...caretXY(el, m.start) });
  };
  const options = menu ? mentionOptions(tasks, menu.query) : [];
  const open = options.length > 0;
  const pick = (x) => {
    const r = insertMention(value, menu, x.id), el = ref.current;
    setMenu(null);
    // written into the textarea here, so the caret is in place before the next key (Preact leaves an equal value)
    el.value = r.text;
    el.setSelectionRange(r.caret, r.caret);
    el.focus();
    onInput(r.text);
  };
  const key = (ev) => {
    if (!open) return;
    const move = { ArrowDown: 1, ArrowUp: -1 }[ev.key];
    if (move) setMenu({ ...menu, sel: (menu.sel + move + options.length) % options.length });
    else if (ev.key === 'Enter' || ev.key === 'Tab') pick(options[Math.min(menu.sel, options.length - 1)]);
    else if (ev.key === 'Escape') setMenu(null);
    else return;
    ev.preventDefault();
    ev.stopPropagation();  // Escape closes the list, not the drawer
  };
  const id = useId('mention');
  return html`<div class="mention-box">
    <${AutoText} ...${rest} value=${value} textRef=${ref} onKeyDown=${key}
      onInput=${(v) => { onInput(v); look(); }} onClick=${look} onBlur=${() => setMenu(null)}
      onKeyUp=${(ev) => /^(Arrow\w+|Home|End|PageUp|PageDown)$/.test(ev.key) && look()}
      aria-autocomplete="list" aria-expanded=${open} aria-controls=${open ? id : undefined}
      aria-activedescendant=${open ? `${id}-${Math.min(menu.sel, options.length - 1)}` : undefined} />
    ${open && html`<ul class="mention-menu" id=${id} role="listbox" aria-label=${t('引用任务')} style=${{ top: `${menu.top}px`, left: `${Math.max(0, menu.left)}px` }}>
      ${options.map((x, i) => html`<li id=${`${id}-${i}`} role="option" aria-selected=${i === Math.min(menu.sel, options.length - 1)}
        onMouseDown=${(ev) => { ev.preventDefault(); pick(x); }}><span class="mono">@${x.id}</span><span class="muted ellipsis">${x.title}</span></li>`)}</ul>`}
  </div>`;
}

// project.yaml machines as the settings form edits them: one line per machine, "name [gpus] [note]".
export const machinesText = (list) => (list || []).map((m) => (typeof m === 'string' ? m : [m.name, m.gpus, m.note].filter((x) => x !== undefined && x !== null && x !== '').join(' '))).join('\n');
export function parseMachines(text) {
  return (text || '').split('\n').map((l) => l.trim().split(/\s+/)).filter((w) => w[0]).map(([name, ...rest]) => {
    const gpus = /^\d+$/.test(rest[0] || '') ? Number(rest.shift()) : undefined;
    return { name, ...(gpus !== undefined ? { gpus } : {}), ...(rest.length ? { note: rest.join(' ') } : {}) };
  });
}

// Hash route helpers: #/p/<project>?t=<project>/<type>/<name>&tab=docs
export function parseHash() {
  const [path, query = ''] = location.hash.slice(1).split('?');
  return { path: path || '/', params: Object.fromEntries(new URLSearchParams(query)) };
}
export function hrefWith(params, path) {
  const cur = parseHash();
  const merged = { ...cur.params, ...params };
  for (const k of Object.keys(merged)) if (merged[k] == null || merged[k] === '') delete merged[k];
  const q = new URLSearchParams(merged).toString();
  return `#${path ?? cur.path}${q ? `?${q}` : ''}`;
}
export const go = (href) => { if (location.hash !== href) location.hash = href; };
export const taskHref = (project, id, extra = {}) => hrefWith({ t: `${project}/${id}`, tab: extra.tab, file: extra.file, new: null });
// The drawer closed: the page's own hash.
export const closeHref = () => hrefWith({ t: null, tab: null, file: null });
