// What needs you, one line per task: the home page's queue and the board's 待处理 panel use these.
// A 待确认 card gives one main button by why the task stopped; 回复 opens the input box under it. A running task that
// may be stuck (疑似卡住) is listed too, still running: 发消息, 停止 or 在终端中接管.
import { useState, useRef, useEffect } from './vendor/hooks.module.js';
import { api, paths } from './api.js';
import { html, Icon, StatusMark, statusLine, money, ago, since, clock, AutoText, confirmDialog, toast, fail, taskHref, brief } from './ui.js';
import { join, CHECKS } from './task.js';
import { BudgetButton, sendMessage, stopRun, AttachCommand } from './run.js';
import { t } from './i18n.js';

// The main button of a card: {kind, label}. kind: reply | questions | checks | resume | budget | join | review | stall.
export function needAction(task) {
  const k = task.detail?.kind;
  if (task.status === 'running') return { kind: 'stall', label: t('发消息') };  // 疑似卡住
  if (task.status === 'in_review') return { kind: 'review', label: t('查看报告并审核') };
  if (!task.managed) return { kind: 'join', label: t('纳入托管') };
  // count 0: questions.md has no table of 你的意见 to fill in, only text to read
  if (k === 'questions') return { kind: 'questions', label: task.detail.count ? t('回答 {n} 个问题', { n: task.detail.count }) : t('回答问题') };
  if (k === 'interrupted' || k === 'wait_interrupted') return { kind: 'resume', label: t('继续') };
  if (k === 'budget') return { kind: 'budget', label: t('追加预算并继续') };
  if (k === 'checks_failed') return { kind: 'checks', label: t('查看检查') };
  return { kind: 'reply', label: t('回复') };  // asked, stopped, joined
}

// What the input box of an opened card says, by why the task stopped.
export function replyPlaceholder(task) {
  const k = task.detail?.kind;
  if (task.status === 'running') return t('发消息（agent 在当前的调用或命令返回后读到；一直没有反应就停止或接管）');
  if (k === 'questions') return t('直接回复（不填表格）');
  if (k === 'interrupted' || k === 'wait_interrupted') return t('补充说明后继续');
  if (k === 'checks_failed') return t('告诉 agent 怎么处理（例如"继续修"或"这条检查可以放宽"）');
  if (k === 'joined') return t('回复（将与"请先阅读 progress.md"一起作为新会话的首条消息）');
  return t('回复');
}

// 已等待多久 / progress 更新于多久前 (an unmanaged task has no state.json); a running one: how long its run has gone
const waited = (task, now) => (task.status === 'running' ? t('已运行 {time}', { time: since(task.since, now) })
  : task.managed ? t('已等待 {time}', { time: since(task.since, now) }) : t('progress 更新于 {ago}', { ago: ago(task.since, now) }));

const send = (task, text) => sendMessage(task.project, task.id, text)
  .then(() => toast(task.status === 'running' ? t('已发消息给「{title}」', { title: task.title }) : t('已回复「{title}」，任务将继续运行', { title: task.title }),
    { href: taskHref(task.project, task.id, { tab: 'run' }), link: t('查看输出') }),
  (e) => { fail(e); throw e; });

export function NeedsCard({ t: task, now, here = false }) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const box = useRef();
  useEffect(() => { if (open) box.current?.querySelector('textarea')?.focus(); }, [open]);
  const a = needAction(task);
  // 继续 without words: the server words it in the agent's language
  const reply = () => {
    if ((!text.trim() && a.kind !== 'resume') || busy) return;
    setBusy(true);
    send(task, text).then(() => { setText(''); setOpen(false); }, () => {}).finally(() => setBusy(false));
  };
  const main = {
    reply: html`<button class="btn primary small" aria-expanded=${open} onClick=${() => setOpen(!open)}>${a.label}</button>`,
    questions: html`<a class="btn primary small" href=${taskHref(task.project, task.id, { tab: 'docs', file: task.detail?.path })}>${a.label}</a>`,
    checks: html`<a class="btn primary small" href=${taskHref(task.project, task.id, { tab: 'docs', file: CHECKS })}>${a.label}</a>`,
    resume: html`<button class="btn primary small" disabled=${busy} onClick=${reply}>${a.label}</button>`,
    budget: html`<${BudgetButton} pid=${task.project} t=${task} />`,
    join: html`<button class="btn primary small" onClick=${() => join(task.project, task)}>${a.label}</button>`,
    stall: html`<button class="btn primary small" aria-expanded=${open} onClick=${() => setOpen(!open)}>${a.label}</button>
      <button class="btn danger-ghost small" onClick=${() => stopRun(task.project, task)}>${t('停止')}</button>
      <button class="btn small" onClick=${() => takeover(task)}><${Icon} name="terminal" />${t('在终端中接管')}</button>`,
  }[a.kind];
  // besides the main button: reply in words (not when the budget is used up: a reply could not go on either)
  const more = task.managed && ['questions', 'checks', 'resume'].includes(a.kind);
  const id = `r-${task.project}-${task.id}`.replace(/\W/g, '-');
  const why = statusLine(task, undefined, now);
  return html`<article class=${`need-card kind-${a.kind === 'stall' ? 'stall' : task.detail?.kind || ''} ${open ? 'open' : ''}`} aria-label=${t('{title}：{status}', { title: task.title, status: why })}>
    <div class="nc-line">
      <${StatusMark} t=${task} short />
      <a class="nc-title" href=${taskHref(task.project, task.id)} title=${`${task.project} / ${task.id}`}>${task.title}</a>
      ${!here && html`<span class="nc-project mono small muted">${task.project}</span>`}
      ${!task.managed && html`<span class="tag">${t('未托管')}</span>`}
      <span class="nc-why" title=${why}>${why}</span>
      <span class="muted small nowrap" title=${clock(task.since)}>${waited(task, now)}</span>
      <span class="nc-actions">${main}
        ${more && html`<button class="btn ghost small" aria-expanded=${open} aria-controls=${id} onClick=${() => setOpen(!open)}>${t('回复')}</button>`}</span>
    </div>
    ${open && html`<div class="nc-reply" id=${id} ref=${box}>
      ${task.last_message && html`<blockquote class="nc-last"><span class="muted small">${t('agent 最近消息')}</span>${brief(task.last_message, 300)}</blockquote>`}
      <div class="reply"><label class="sr-only" for=${`${id}-text`}>${t('回复 {title}', { title: task.title })}</label>
        <${AutoText} id=${`${id}-text`} value=${text} onInput=${setText} onSubmit=${reply} rows=${2} placeholder=${replyPlaceholder(task)} />
        <button class="btn primary" disabled=${busy || (!text.trim() && a.kind !== 'resume')}
          onClick=${reply}>${a.kind === 'resume' && !text.trim() ? t('继续') : t('发送')}</button>
        <button class="btn ghost" onClick=${() => setOpen(false)}>${t('收起')}</button></div>
      <p class="muted small">${a.kind === 'stall' ? t('Ctrl+Enter 发送。消息进入正在运行的会话，不会重新开始。') : t('Ctrl+Enter 发送。回复后任务在原会话中继续运行（resume）。')}</p></div>`}
  </article>`;
}

// 在终端中接管 from a card of a running task: through the page's dialog, which stays when the card goes (the task is
// no longer stuck once taken over), so the command to attach is not lost with it.
async function takeover(task) {
  if (!await confirmDialog({ title: t('在终端中接管'), ok: t('接管'),
    body: html`<p>${t('agent_sys 将停止正在运行的 runner、清理该会话的残留进程，并在 tmux 中以同一会话启动交互式 claude（权限和配置与 runner 一致）。')}</p>
      <ul class="small"><li>${t('接管期间任务显示为"运行中 · 终端接管"，网页输入框暂停使用。')}</li>
        <li>${t('在终端中退出 claude 后，agent_sys 将根据 progress.md 自动判断任务状态，无需手动标记。')}</li></ul>` })) return;
  api.post(`${paths.task(task.project, task.id)}/actions/takeover`).then((r) => confirmDialog({ title: t('在终端中接管'), cancel: t('确定'),
    body: html`<p>${t('已在 tmux 窗口')} <code>${r.takeover.window}</code>${t(' 中启动。请在终端中运行：')}</p><${AttachCommand} attach=${r.takeover.attach} />` }), fail);
}

// 报告：report.md · 本轮花费 $0.12
export function reviewLine(task) {
  return [statusLine(task), task.managed && task.round_cost_usd != null ? t('本轮花费 {cost}', { cost: money(task.round_cost_usd) }) : ''].filter(Boolean).join(' · ');
}

export function ReviewCard({ t: task, now, here = false }) {
  return html`<article class="need-card kind-review" aria-label=${t('{title}：{status}', { title: task.title, status: reviewLine(task) })}>
    <div class="nc-line">
      <${StatusMark} t=${task} short />
      <a class="nc-title" href=${taskHref(task.project, task.id, { tab: 'docs', file: task.report })} title=${`${task.project} / ${task.id}`}>${task.title}</a>
      ${!here && html`<span class="nc-project mono small muted">${task.project}</span>`}
      ${!task.managed && html`<span class="tag">${t('未托管')}</span>`}
      <span class="nc-why">${reviewLine(task)}</span>
      <span class="muted small nowrap" title=${clock(task.since)}>${waited(task, now)}</span>
      <span class="nc-actions"><a class="btn primary small" href=${taskHref(task.project, task.id, { tab: 'docs', file: task.report })}>${t('查看报告并审核')}</a></span>
    </div>
  </article>`;
}

// 继续全部中断任务: the confirmation lists what goes on.
export async function resumeAll(tasks) {
  if (!await confirmDialog({ title: t('继续全部 {n} 个中断的任务？', { n: tasks.length }), ok: t('全部继续'),
    body: html`<p>${t('每个任务都将在原会话中继续运行（resume），发送的消息为"继续"。原本在后台等待的任务，agent 会先检查等待条件，再重新开始等待。')}</p>
      <ul class="small">${tasks.map((task) => html`<li>${task.title} <span class="muted">${task.project} · ${statusLine(task)}</span></li>`)}</ul>` })) return;
  api.post('/api/resume', { tasks: tasks.map((task) => ({ project: task.project, task: task.id })) }).then((r) => {
    toast(t('已继续 {n} 个任务', { n: r.resumed.length }));
    for (const e of r.errors || []) toast(t('{task}：{error}', { task: e.task, error: e.error }), { error: true });
  }, fail);
}

// 全部标为已查看
export function MarkAllRead({ tasks }) {
  const [busy, setBusy] = useState(false);
  if (!tasks.length) return null;
  const run = () => {
    setBusy(true);
    api.post('/api/mark_read', { tasks: tasks.map((task) => ({ project: task.project, task: task.id })) })
      .then((r) => toast(t('已将 {n} 个任务标为已查看', { n: r.marked.length })), fail).finally(() => setBusy(false));
  };
  return html`<button class="btn ghost small" disabled=${busy} onClick=${run}>${t('全部标为已查看')}</button>`;
}

// 待确认 on the home page: the running tasks that may be stuck (they need you as much), then the tasks that wait for you.
export const needsOf = (home) => [...(home.stalled || []), ...home.needs_you];
// 待确认 in a project's counts (project.counts): the tasks that wait for you and the running ones that may be stuck.
export const needsCount = (c) => c.needs_you + (c.stalled || 0);
// The same as needsOf among a project's tasks, the longest quiet first.
export const stalledOf = (tasks) => tasks.filter((task) => task.status === 'running' && task.detail?.stall)
  .sort((a, b) => a.detail.stall.since.localeCompare(b.detail.stall.since));

export const interruptedOf = (tasks) => tasks.filter((task) => task.managed && task.status === 'needs_you' && /interrupted/.test(task.detail?.kind || ''));
