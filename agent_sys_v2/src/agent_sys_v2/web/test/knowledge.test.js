// The pure parts of the rules / knowledge pages and of the task form (checks).  node --test web/test/*.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { KTYPES, RULE_FILE, ruleFileOf, ruleText, replaces, replaceNote, indexNote, hasIndex } from '../knowledge.js';
import { agentFrontmatter, agentError, frontmatterChanges, checksText, checksEditable, checkList } from '../task.js';
import { statusLine } from '../ui.js';
import { setLang } from '../i18n.js';
import { EN } from '../i18n_en.js';

setLang('zh');  // these expectations are the Chinese texts; i18n.test.js covers the English

test('entry types use the formal words', () => {
  assert.deepEqual(Object.values(KTYPES), ['注意事项', '环境信息', '方法', '已验证无效', '规则']);
});

test('改写为规则: default file and text', () => {
  assert.equal(ruleFileOf('global'), 'task_rules.md');
  assert.equal(ruleFileOf('project'), 'project.md');
  assert.equal(ruleText({ name: 'tee', description: '以后长命令都 tee 到 log/', body: '' }), '- 以后长命令都 tee 到 log/');
  assert.equal(ruleText({ name: 'tee', description: 'tee 到 log/', body: '你说：以后都这样。\n第二行' }), '- tee 到 log/');
  assert.equal(ruleText({ name: 'tee', description: '', body: '\n以后都 tee 到 log/。\n原因：…' }), '- 以后都 tee 到 log/。');
  assert.ok(RULE_FILE.test('task_rules.md') && RULE_FILE.test('a-b.c.md'));
  assert.ok(!RULE_FILE.test('../x.md') && !RULE_FILE.test('.hidden.md') && !RULE_FILE.test('x.txt'));
});

test('adopting over an entry of the same name is said before it happens', () => {
  const c = { name: 'ssh-hang', conflict: ['global'] };
  assert.ok(replaces(c, 'global') && !replaces(c, 'project'));
  assert.ok(!replaces(c, 'global', 'ssh-hang-2'));  // renamed before adopting: the server's 409 tells
  assert.equal(replaceNote('global', 'ssh-hang'), '全局经验库中已有同名经验「ssh-hang」，采纳后将覆盖该条目（原条目移到 .trash/）。');
});

test('index note warns past 60 lines', () => {
  assert.deepEqual(indexNote({ lines: 2, bytes: 412, tokens: 129, limit: 60 }),
    { over: false, text: '索引共 2 行、412 B（约 129 tokens），每次运行都会注入。' });
  assert.ok(indexNote({ lines: 61, bytes: 9000, tokens: 2812, limit: 60 }).over);
});

test('the injected parts are told apart by kind, not by their (translated) titles', () => {
  const part = (kind, title) => ({ kind, title, path: 'x', bytes: 1 });
  assert.ok(!hasIndex([part('protocol', 'agent_sys protocol'), part('rules', 'Global rules'), part('skills', 'Skills index')]));
  assert.ok(hasIndex([part('protocol', 'agent_sys 约定'), part('knowledge', 'Global knowledge index')]));
  assert.ok(hasIndex([part('project_knowledge', '项目经验索引')]));
});

test('saving agent settings keeps what the form does not show (replay)', () => {
  const defaults = { model: 'opus', effort: 'high', budget_usd: 200, extra_args: [] };
  const form = { model: 'opus', effort: 'low', budget_usd: 200, extra_args: [] };
  assert.deepEqual(agentFrontmatter(form, defaults, { effort: 'max', replay: { events: 'x.jsonl' } }),
    { replay: { events: 'x.jsonl' }, effort: 'low' });
  assert.deepEqual(agentFrontmatter(form, defaults, undefined), { effort: 'low' });
});

test('saving the task form writes only what you changed; a budget of $0 is not saved', () => {
  const defaults = { model: 'opus', effort: 'high', ultracode: false, budget_usd: 200, extra_args: [] };
  const saved = { title: 'A', summary: '', parent: '', machines: [], checks: '', review: false, deps: [],
    model: 'opus', effort: 'high', ultracode: false, budget: 200, extra: '' };
  assert.deepEqual(frontmatterChanges(saved, saved, defaults, {}), {});
  // only effort changed: the other agent fields stay the project's (not written into the task file)
  assert.deepEqual(frontmatterChanges({ ...saved, effort: 'low' }, saved, defaults, { replay: { events: 'x' } }),
    { agent: { replay: { events: 'x' }, effort: 'low' } });
  assert.deepEqual(frontmatterChanges({ ...saved, title: ' ', machines: ['n1'], checks: 'a\n b', deps: [{ task: 'x/y', kind: 'input', label: '' }] }, saved, defaults, {}),
    { title: null, machines: ['n1'], checks: ['a', 'b'], depends_on: ['x/y'] });
  assert.equal(agentError(saved), '');
  assert.notEqual(agentError({ ...saved, budget: '' }), '');
  assert.notEqual(agentError({ ...saved, budget: '0' }), '');
});

test('acceptance checks as the task form edits them, and the lines of a task waiting for machines or checks', () => {
  assert.equal(checksText(['test -s report.md', { name: 'results', run: 'python -m agent_sys_v2.ledger check results' }]),
    'test -s report.md\npython -m agent_sys_v2.ledger check results');
  assert.ok(checksEditable(['a', 'b']));
  assert.ok(!checksEditable(['a', { run: 'b', timeout: 5 }]));
  assert.deepEqual(checkList(' a \n\n b'), ['a', 'b']);
  assert.equal(checkList('  \n'), null);
  assert.equal(statusLine({ status: 'queued', detail: { kind: 'machines', machines: [{ name: 'n1', task: 'experiment/e1', title: 'E1' }] } }),
    '等待机器：n1（被 E1 占用）');
  assert.equal(statusLine({ status: 'needs_you', detail: { kind: 'checks_failed', failed: ['results', 'report'] } }), '验收未通过：results、report');
  assert.equal(statusLine({ status: 'needs_you', detail: { kind: 'checks_failed', failed: [] } }), '验收检查没有运行完');
});

test('the same lines in English', () => {
  setLang('en');
  try {
    assert.equal(statusLine({ status: 'needs_you', detail: { kind: 'checks_failed', failed: ['results'] } }), EN['验收未通过：{failed}'].replace('{failed}', 'results'));
    assert.doesNotMatch(statusLine({ status: 'queued', detail: { kind: 'machines', machines: [{ name: 'n1', task: 'x/y', title: 'Y' }] } }), /[一-鿿]/);
    assert.doesNotMatch(statusLine({ status: 'queued', detail: { kind: 'upstream', tasks: ['develop/a', 'develop/b'] } }), /[一-鿿、]/);
    assert.match(statusLine({ status: 'queued', detail: { kind: 'upstream', tasks: ['develop/a', 'develop/b'] } }), /develop\/a, develop\/b/);
  } finally {
    setLang('zh');
  }
});
