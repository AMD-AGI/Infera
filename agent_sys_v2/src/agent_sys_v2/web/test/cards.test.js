// The home page's cards (cards.js): which main button a task gets, and what its card says.
//   node --test src/agent_sys_v2/web/test/*.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { needAction, replyPlaceholder, reviewLine, interruptedOf, needsOf, stalledOf, needsCount } from '../cards.js';
import { statusLine } from '../ui.js';
import { setLang } from '../i18n.js';

setLang('zh');  // these expectations are the Chinese texts; i18n.test.js covers the English

const needs = (kind, extra = {}) => ({ status: 'needs_you', managed: true, detail: { kind }, ...extra });

test('needAction: one main button by why the task stopped', () => {
  const label = (t) => `${needAction(t).kind}:${needAction(t).label}`;
  assert.equal(label(needs('asked')), 'reply:回复');
  assert.equal(label(needs('stopped')), 'reply:回复');
  assert.equal(label(needs('joined')), 'reply:回复');
  assert.equal(label({ ...needs('questions'), detail: { kind: 'questions', count: 5, path: 'questions.md' } }), 'questions:回答 5 个问题');
  // no table to fill in: no count to show
  assert.equal(label({ ...needs('questions'), detail: { kind: 'questions', count: 0, path: 'questions.md' } }), 'questions:回答问题');
  assert.equal(label(needs('interrupted')), 'resume:继续');
  assert.equal(label(needs('interrupted', { detail: { kind: 'interrupted', error: '自动开始失败：…' } })), 'resume:继续');
  assert.equal(label(needs('wait_interrupted')), 'resume:继续');
  assert.equal(label(needs('budget')), 'budget:追加预算并继续');
  assert.equal(label({ status: 'needs_you', managed: false, detail: { kind: 'unmanaged' } }), 'join:纳入托管');
  assert.equal(label({ status: 'in_review', managed: true, detail: { kind: 'review' } }), 'review:查看报告并审核');
  assert.equal(label({ status: 'in_review', managed: false, detail: { kind: 'review' } }), 'review:查看报告并审核');
});

test('the card says why, and what the reply box is for', () => {
  assert.equal(statusLine(needs('asked', { progress: { next: '选 A 还是 B？' } })), '待答复：选 A 还是 B？');
  assert.equal(statusLine(needs('wait_interrupted', { detail: { kind: 'wait_interrupted', waiting_for: '等作业 123' } })), '后台等待中断：等作业 123');
  assert.equal(statusLine(needs('interrupted', { detail: { kind: 'interrupted', error: '自动开始失败：tmux' } })), '运行出错：自动开始失败：tmux');
  assert.equal(statusLine({ status: 'queued', detail: { kind: 'concurrency', scope: 'project', limit: 3 } }), '等待运行名额（项目上限 3 个）');
  assert.equal(statusLine({ status: 'queued', detail: { kind: 'concurrency', scope: 'global', limit: 4 } }), '等待运行名额（全局上限 4 个）');
  assert.equal(replyPlaceholder(needs('interrupted')), '补充说明后继续');
  assert.equal(replyPlaceholder(needs('questions')), '直接回复（不填表格）');
  assert.equal(replyPlaceholder(needs('asked')), '回复');
  const review = { status: 'in_review', managed: true, detail: { kind: 'review' }, report: 'report.md', round_cost_usd: 0.123 };
  assert.equal(reviewLine(review), '报告：report.md · 本轮花费 $0.12');
  assert.equal(reviewLine({ ...review, managed: false, round_cost_usd: undefined }), '报告：report.md');
});

test('继续全部中断任务 takes only managed tasks whose run was interrupted', () => {
  const list = [needs('interrupted', { id: 'a' }), needs('wait_interrupted', { id: 'b' }), needs('asked', { id: 'c' }),
    { status: 'needs_you', managed: false, detail: { kind: 'unmanaged' }, id: 'd' }];
  assert.deepEqual(interruptedOf(list).map((t) => t.id), ['a', 'b']);
});

// 疑似卡住: a running task listed with what needs you; it says which of the two and for how long
const NOW = Date.parse('2026-10-27T12:00:00Z');
const at = (min) => new Date(NOW - min * 60000).toISOString().replace(/\.\d+Z$/, 'Z');
const stalled = (stall, extra = {}) => ({ status: 'running', managed: true, since: at(60), detail: { kind: 'working', stall }, ...extra });

test('a task that may be stuck: 发消息 first, and what the card says', () => {
  const silent = stalled({ kind: 'silent', tool: null, since: at(16) });
  assert.deepEqual(needAction(silent), { kind: 'stall', label: '发消息' });
  assert.match(replyPlaceholder(silent), /^发消息/);
  assert.equal(statusLine(silent, undefined, NOW), '疑似卡住：模型调用无响应，已 16 分钟没有任何事件');
  assert.equal(statusLine(stalled({ kind: 'silent', tool: 'Agent', since: at(20) }), undefined, NOW), '疑似卡住：前台子 agent 已 20 分钟没有任何事件');
  assert.equal(statusLine(stalled({ kind: 'silent', tool: 'mcp__db__query', since: at(15) }), undefined, NOW),
    '疑似卡住：mcp__db__query 已 15 分钟没有返回，也没有任何事件');
  assert.equal(statusLine(stalled({ kind: 'timeout', tool: 'Bash', since: at(16), timeout: 600 }), undefined, NOW),
    '疑似卡住：前台命令超时未返回（Bash 已运行 16 分钟，超时 10 分钟）');
});

test('a long command under its timeout says how long it has run, and is not stuck', () => {
  const run = (current_since, current_timeout) => ({ status: 'running', managed: true, detail: { kind: 'working', current: 'Bash make -j', current_since, current_timeout } });
  assert.equal(statusLine(run(at(0.5), 600), undefined, NOW), '正在执行：Bash make -j');  // under a minute: as before
  assert.equal(statusLine(run(at(12), 1800), undefined, NOW), '正在执行：Bash make -j · 已运行 12 分钟（超时 30 分钟）');
  assert.equal(statusLine(run(at(25)), undefined, NOW), '正在执行：Bash make -j · 已运行 25 分钟');  // a subagent: no timeout
});

test('the list of what needs you: the tasks that may be stuck first, the longest quiet first', () => {
  const a = stalled({ kind: 'silent', tool: null, since: at(16) }, { id: 'a' });
  const b = stalled({ kind: 'timeout', tool: 'Bash', since: at(30), timeout: 120 }, { id: 'b' });
  const tasks = [a, { status: 'running', id: 'c', detail: { kind: 'working' } }, needs('asked', { id: 'd' }), b];
  assert.deepEqual(stalledOf(tasks).map((t) => t.id), ['b', 'a']);
  assert.deepEqual(needsOf({ stalled: [b, a], needs_you: [tasks[2]] }).map((t) => t.id), ['b', 'a', 'd']);
  assert.deepEqual(needsOf({ needs_you: [tasks[2]] }).map((t) => t.id), ['d']);  // a server from before
  assert.deepEqual(interruptedOf(needsOf({ stalled: [a], needs_you: [] })), []);  // 继续全部 is not for them
  assert.equal(needsCount({ needs_you: 2, stalled: 1 }), 3);  // a project's 待确认 counts them too
  assert.equal(needsCount({ needs_you: 2 }), 2);
});
