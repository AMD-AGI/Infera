// What needs you, one line per task: the home page's queue and the board's 待处理 panel use these.
// A 待确认 card gives one main button by why the task stopped; 回复 opens the input box under it.
import { useState, useRef, useEffect } from './vendor/hooks.module.js';
import { api } from './api.js';
import { html, StatusMark, statusLine, money, ago, since, clock, AutoText, confirmDialog, toast, fail, taskHref, brief } from './ui.js';
import { join, CHECKS } from './task.js';
import { BudgetButton, sendMessage } from './run.js';
import { t } from './i18n.js';

// The main button of a card: {kind, label}. kind: reply | questions | checks | resume | budget | join | review.
export function needAction(task) {
  const k = task.detail?.kind;
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
  if (k === 'questions') return t('直接回复（不填表格）');
  if (k === 'interrupted' || k === 'wait_interrupted') return t('补充说明后继续');
  if (k === 'checks_failed') return t('告诉 agent 怎么处理（例如"继续修"或"这条检查可以放宽"）');
  if (k === 'joined') return t('回复（将与"请先阅读 progress.md"一起作为新会话的首条消息）');
  return t('回复');
}

// 已等待多久 / progress 更新于多久前 (an unmanaged task has no state.json)
const waited = (task, now) => (task.managed ? t('已等待 {time}', { time: since(task.since, now) }) : t('progress 更新于 {ago}', { ago: ago(task.since, now) }));

const send = (task, text) => sendMessage(task.project, task.id, text)
  .then(() => toast(t('已回复「{title}」，任务将继续运行', { title: task.title }), { href: taskHref(task.project, task.id, { tab: 'run' }), link: t('查看输出') }),
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
  }[a.kind];
  // besides the main button: reply in words (not when the budget is used up: a reply could not go on either)
  const more = task.managed && ['questions', 'checks', 'resume'].includes(a.kind);
  const id = `r-${task.project}-${task.id}`.replace(/\W/g, '-');
  return html`<article class=${`need-card kind-${task.detail?.kind || ''} ${open ? 'open' : ''}`} aria-label=${t('{title}：{status}', { title: task.title, status: statusLine(task) })}>
    <div class="nc-line">
      <${StatusMark} t=${task} short />
      <a class="nc-title" href=${taskHref(task.project, task.id)} title=${`${task.project} / ${task.id}`}>${task.title}</a>
      ${!here && html`<span class="nc-project mono small muted">${task.project}</span>`}
      ${!task.managed && html`<span class="tag">${t('未托管')}</span>`}
      <span class="nc-why" title=${statusLine(task)}>${statusLine(task)}</span>
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
      <p class="muted small">${t('Ctrl+Enter 发送。回复后任务在原会话中继续运行（resume）。')}</p></div>`}
  </article>`;
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

export const interruptedOf = (tasks) => tasks.filter((task) => task.managed && task.status === 'needs_you' && /interrupted/.test(task.detail?.kind || ''));
