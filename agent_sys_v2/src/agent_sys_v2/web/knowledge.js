// The rules page and the knowledge page: what every run gets injected, the rules
// files, the lesson candidates the agents wrote and the two knowledge libraries.
import { useState } from './vendor/hooks.module.js';
import { api } from './api.js';
import { html, Icon, StatusMark, useApi, clock, bytes, Markdown, AutoText, ErrorBox, Loading, confirmDialog, toast, fail,
  taskHref } from './ui.js';
import { t, list, LANGS } from './i18n.js';

// ---------------------------------------------------------------- pure helpers (web/test/knowledge.test.js)

// Labels are looked up when read, not when the module loads (the language can be set after, in tests).
const labels = (o) => Object.defineProperties({}, Object.fromEntries(Object.entries(o)
  .map(([k, v]) => [k, { get: () => t(v), enumerable: true }])));
export const KTYPES = labels({ pitfall: '注意事项', 'env-fact': '环境信息', method: '方法', 'negative-result': '已验证无效', rule: '规则' });  // i18n: translated where shown
const LIB_NAME = labels({ global: '全局经验库', project: '项目经验库' });  // i18n: translated where shown
export const RULE_FILE = /^[A-Za-z0-9_][A-Za-z0-9_.-]*\.md$/;

// Where 改写为规则 writes by default: the global task_rules.md, or the project's project.md.
export const ruleFileOf = (scope) => (scope === 'global' ? 'task_rules.md' : 'project.md');

// The rule text a candidate starts from: its description as one list item (a rule is injected into every run, so it
// stays one line; the candidate's text is context, there to edit from).
export function ruleText(c) {
  const line = (c.description || '').trim() || (c.body || '').trim().split('\n')[0] || c.name;
  return `- ${line}`;
}

// Whether the library has an entry of this name, as the page knows it (a renamed candidate: the server's 409 says).
export const replaces = (c, scope, name = c.name) => name === c.name && (c.conflict || []).includes(scope);
// The confirm dialog's sentence before adopting over an entry.
export const replaceNote = (scope, name) => t('{lib}中已有同名经验「{name}」，采纳后将覆盖该条目（原条目移到 .trash/）。', { lib: LIB_NAME[scope], name });

// Whether system.md gets a knowledge index (a part of kind knowledge or project_knowledge: a library with entries).
export const hasIndex = (parts) => parts.some((p) => p.kind === 'knowledge' || p.kind === 'project_knowledge');

// The line about an index: how big it is, and whether it has grown past the size it should keep to.
export function indexNote(index) {
  const over = index.lines > index.limit;
  return { over, text: t('索引共 {lines} 行、{size}（约 {tokens} tokens），每次运行都会注入。',
    { lines: index.lines, size: bytes(index.bytes), tokens: index.tokens })
    + (over ? t('已超过约 {limit} 行：索引常驻上下文，请合并或删除过时的条目。', { limit: index.limit }) : '') };
}

// ---------------------------------------------------------------- rules

export function RulesPage({ pid }) {
  const { data, error, reload } = useApi(`/api/rules${pid ? `?project=${pid}` : ''}`, (c) => !c.task);
  if (error && !data) return html`<main id="main" tabIndex="-1" class="page"><${ErrorBox} what=${t('加载规则')} error=${error} retry=${reload} /></main>`;
  if (!data) return html`<${Loading} />`;
  const inj = data.injected;
  return html`<main id="main" tabIndex="-1" class="page rules-page">
    <header class="page-head"><div><h1>${pid ? t('规则') : t('全局规则')}</h1>
      <p class="muted">${pid ? t('每次运行时，按下表的顺序拼成 system.md，追加在 Claude Code 自带的 system prompt 之后。')
        : t('每次运行时，按下表的顺序拼成 system.md，追加在 Claude Code 自带的 system prompt 之后（下表只含所有项目共有的部分；项目页还会加上项目规则、skills 索引和项目经验索引）。')}
        ${t('各段大小与运行记录 launch.md 中列出的一致。')}</p>
      <p class="muted small">${t('修改规则后：新任务直接使用新规则；已经开始的任务保留会话创建时的 system prompt，下次继续时，agent_sys 会在消息开头提示 agent"规则有更新，请先读新的 system.md"。')}</p></div></header>
    <section class="side-card injected"><h2>${t('每次运行注入的内容')} <span class="muted small">${t('合计 {size}（约 {tokens} tokens）', { size: bytes(inj.bytes), tokens: inj.tokens })}</span></h2>
      <table class="inject-table"><thead><tr><th>#</th><th>${t('内容')}</th><th>${t('来源')}</th><th>${t('大小')}</th></tr></thead>
        <tbody>${inj.parts.map((p, i) => html`<tr><td>${i + 1}</td><td>${p.title}</td><td class="mono small">${p.path}</td><td class="nowrap">${bytes(p.bytes)}</td></tr>`)}</tbody></table>
      ${!hasIndex(inj.parts) && html`<p class="muted small">${t('经验库还没有条目，所以没有注入经验索引。')}</p>`}</section>
    <div class=${pid ? 'two-col' : ''}>
      <${RuleScope} title=${t('全局规则')} scope="global" lib=${data.global} />
      ${pid && html`<${RuleScope} title=${t('项目规则')} scope="project" lib=${data.project} pid=${pid} />`}
    </div>
    ${data.protocol && html`<section class="rule-scope protocol"><h2>${t('agent_sys 约定')} <span class="tag">${t('只读')}</span> <span class="tag">${LANGS[data.protocol.language]}</span> <span class="mono small muted">${data.protocol.path}</span></h2>
      <p class="muted small">${t('progress / report / questions.md / lessons 等写法的约定，随 agent_sys 的代码维护，每次运行从代码注入，不复制进项目。要修改请改代码仓库中的这个文件。')}
        ${t('约定有中文和英文两个版本，默认注入中文，在全局设置中切换；agent 的工作语言跟任务 prompt 的语言一致。')}
        <a href="#/settings">${t('全局设置')}</a></p>
      <div class="rule-view scroll"><${Markdown} text=${data.protocol.content} className="doc" /></div></section>`}
  </main>`;
}

function RuleScope({ title, scope, lib, pid }) {
  const [sel, setSel] = useState(lib.files[0]?.name);
  const [edit, setEdit] = useState(null);  // {name, text, base}: base = the file's mtime when editing began; null base = a new file
  const f = lib.files.find((x) => x.name === sel) || lib.files[0];
  const nameErr = edit && edit.base == null && !RULE_FILE.test(edit.name) ? t('文件名应以 .md 结尾，只能包含字母、数字、_、- 和 .') : '';
  const save = () => api.put('/api/rules', { scope, project: pid, name: edit.name, content: edit.text, base_mtime: edit.base || '' })
    .then(() => { setSel(edit.name); setEdit(null); toast(t('已保存 {path}；之后开始的运行将使用新规则', { path: `${lib.dir}/${edit.name}` })); }, fail);
  return html`<section class="rule-scope"><h2>${title} <span class="mono small muted">${lib.dir}/</span></h2>
    <div class="file-tabs" role="tablist">${lib.files.map((x) => html`<button role="tab" aria-selected=${!edit?.isNew && x.name === f?.name}
        onClick=${() => { setSel(x.name); setEdit(null); }}>${x.name}<span class="muted small">${bytes(x.size)}</span></button>`)}
      <button class="ghost" onClick=${() => setEdit({ name: 'new_rules.md', text: '', base: null, isNew: true })}><${Icon} name="plus" />${t('新建文件')}</button></div>
    ${edit ? html`<div class="editor form-grid">
        ${edit.isNew && html`<label class="field">${t('文件名')}<input class="mono" value=${edit.name} aria-invalid=${!!nameErr}
          onInput=${(e) => setEdit({ ...edit, name: e.target.value.trim() })} />${nameErr && html`<span class="error-text small">${nameErr}</span>`}</label>`}
        <${AutoText} value=${edit.text} onInput=${(v) => setEdit({ ...edit, text: v })} rows=${edit.isNew ? 6 : 14} class="mono"
          aria-label=${t('编辑 {name}', { name: edit.name })} />
        <div class="row-actions"><button class="btn primary" disabled=${!!nameErr} onClick=${save}>${t('保存')}</button>
          <button class="btn ghost" onClick=${() => setEdit(null)}>${edit.isNew ? t('取消') : t('放弃修改')}</button></div></div>`
      : f ? html`<div class="rule-view"><div class="row-actions"><span class="muted small">${t('更新于 {time}', { time: clock(f.mtime) })} · ${bytes(f.size)}</span>
          <button class="btn small" onClick=${() => setEdit({ name: f.name, text: f.content, base: f.mtime })}>${t('编辑')}</button></div>
          <${Markdown} text=${f.content} className="doc" /></div>`
        : html`<p class="empty-line">${t('还没有{title}文件。点"新建文件"添加，例如', { title })} <code>${scope === 'global' ? 'task_rules.md' : 'project.md'}</code>${t('。')}</p>`}
  </section>`;
}

// ---------------------------------------------------------------- knowledge

export function KnowledgePage({ pid }) {
  // candidates are files in the tasks' workspaces: any change of the project may add or remove one
  const { data, error, reload } = useApi(`/api/knowledge${pid ? `?project=${pid}` : ''}`, (c) => !pid || !c.project || c.project === pid);
  const [tab, setTab] = useState('candidates');
  if (error && !data) return html`<main id="main" tabIndex="-1" class="page"><${ErrorBox} what=${t('加载经验')} error=${error} retry=${reload} /></main>`;
  if (!data) return html`<${Loading} />`;
  const total = data.libraries.reduce((n, l) => n + l.entries.length, 0);
  return html`<main id="main" tabIndex="-1" class="page knowledge-page">
    <header class="page-head"><div><h1>${pid ? t('经验') : t('全局经验')}</h1>
      <p class="muted">${t('agent 认为值得留给以后任务的经验写在任务工作区的')} <code>lessons/</code> ${t('里，在这里作为候选，经你批准后才进入经验库。')}
        ${t('每次运行只注入经验库的索引（每条一行），正文由 agent 在需要时读取。')}${pid ? '' : t('这里列出所有未归档项目的候选。')}</p></div></header>
    <div class="tabs page-tabs" role="tablist">
      <button role="tab" aria-selected=${tab === 'candidates'} onClick=${() => setTab('candidates')}>${t('候选')}<span class="badge">${data.candidates.length}</span></button>
      <button role="tab" aria-selected=${tab === 'library'} onClick=${() => setTab('library')}>${t('经验库')}<span class="badge">${total}</span></button></div>
    ${tab === 'candidates' ? html`<section>${data.candidates.length
      ? data.candidates.map((c) => html`<${Candidate} key=${c.id} c=${c} global=${!pid} />`)
      : html`<p class="empty-line">${t('暂无待处理的候选。任务写入工作区')} <code>lessons/</code> ${t('的经验会出现在这里（不必等任务完成）；处理过的候选移到')} <code>lessons/.done/</code>${t('，移回原处即恢复为候选。')}</p>`}</section>`
      : html`<div class=${data.libraries.length > 1 ? 'two-col' : ''}>${data.libraries.map((lib) => html`<${Library} key=${lib.dir} lib=${lib} types=${data.types} />`)}</div>`}
  </main>`;
}

function Candidate({ c, global }) {
  const [f, setF] = useState({ name: c.name, description: c.description, type: c.type, body: c.body });
  const [editing, setEditing] = useState(c.problems.length > 0);
  const [rule, setRule] = useState(null);
  const set = (k) => (v) => setF((x) => ({ ...x, [k]: v?.target ? v.target.value : v }));
  const post = (action, body, msg) => api.post(`/api/knowledge/candidates/${action}`, { id: c.id, ...body }).then(() => toast(msg), fail);
  // asks first when the library has the name; a 409 (a name taken that the page did not know of) asks then
  const adopt = async (scope, replace = replaces(c, scope, f.name)) => {
    if (replace && !await confirmDialog({ title: t('覆盖{lib}中的「{name}」？', { lib: LIB_NAME[scope], name: f.name }), body: html`<p>${replaceNote(scope, f.name)}</p><p class="muted small">${t('如果两条说的不是一回事，请先在"编辑"中改名。')}</p>`, ok: t('覆盖并采纳'), danger: true })) return;
    api.post('/api/knowledge/candidates/adopt', { id: c.id, scope, ...f, replace }).then(
      () => toast(t('已采纳到{lib}：{name}。之后开始的运行会注入它的索引行', { lib: LIB_NAME[scope], name: f.name })),
      (e) => (e.status === 409 && !replace ? adopt(scope, true) : fail(e)));
  };
  const toRule = (scope) => setRule({ scope, file: ruleFileOf(scope), text: ruleText(f) });
  const isRule = f.type === 'rule';
  const clash = (c.conflict || []).map((s) => LIB_NAME[s]);
  return html`<article class="candidate">
    <div class="cand-head"><span class=${`tag ktype-${f.type}`}>${KTYPES[f.type] || f.type || t('未标类型')}</span><b class="mono">${f.name}</b>
      <span class="spacer"></span>
      <span class="muted small cand-src">${t('来自')}${global ? html` <b>${c.project}</b>${t(' 的')}` : ''} <a href=${taskHref(c.project, c.task, { tab: 'docs', file: `lessons/${c.file}` })}>${c.task_title}</a>
        <${StatusMark} t=${{ status: c.task_status }} short /> · ${clock(c.created)}</span></div>
    <div class="mono small muted">${c.path}</div>
    ${c.problems.length > 0 && html`<p class="warn-text small"><${Icon} name="warning" />${t('这条候选的格式有问题：{problems}。请在下面补全后再采纳。', { problems: c.problems.join(t('；')) })}</p>`}
    ${clash.length > 0 && html`<p class="warn-text small"><${Icon} name="warning" />${t('{libs}中已有同名经验「{name}」：采纳到该库会覆盖它，或先改名。', { libs: list(clash), name: c.name })}</p>`}
    ${editing ? html`<div class="form-grid">
        <div class="field-row"><label class="field">${t('名称（文件名，英文短名）')}<input class="mono" value=${f.name} onInput=${set('name')} /></label>
          <label class="field">${t('类型')}<select value=${f.type} onChange=${set('type')}>${!KTYPES[f.type] && html`<option value=${f.type}>${t('（请选择）')}</option>`}
            ${Object.entries(KTYPES).map(([k, v]) => html`<option value=${k}>${t('{label}（{type}）', { label: v, type: k })}</option>`)}</select></label></div>
        <label class="field">${t('描述（索引里只有这一行，写清楚什么时候用得上）')}<input value=${f.description} onInput=${set('description')} /></label>
        <label class="field">${t('正文')}<${AutoText} value=${f.body} onInput=${set('body')} rows=${5} /></label></div>`
      : html`<p class="cand-desc">${f.description || html`<span class="muted">${t('（没有描述）')}</span>`}</p>
        <details><summary>${t('正文')}</summary><${Markdown} text=${f.body} className="doc small" /></details>`}
    <div class="row-actions">
      ${isRule && html`<button class="btn primary" onClick=${() => toRule('global')}>${t('改写为规则')}</button>`}
      <button class=${`btn ${isRule ? '' : 'primary'}`} onClick=${() => adopt('global')}>${t('采纳到全局经验库')}</button>
      <button class="btn" onClick=${() => adopt('project')}>${t('采纳到项目经验库')}</button>
      ${!isRule && html`<button class="btn" onClick=${() => toRule('project')}>${t('改写为规则')}</button>`}
      <button class="btn ghost" onClick=${() => setEditing(!editing)}>${editing ? t('收起编辑') : t('编辑')}</button>
      <button class="btn ghost danger-text" onClick=${() => post('discard', {}, t('已丢弃 {name}（移到 lessons/.done/，移回原处即恢复）', { name: f.name }))}>${t('丢弃')}</button>
    </div>
    ${rule && html`<div class="to-rule form-grid"><h3>${t('改写为规则')}</h3>
      <div class="field-row"><label class="field">${t('写入')}<select value=${rule.scope} onChange=${(e) => setRule({ ...rule, scope: e.target.value, file: ruleFileOf(e.target.value) })}>
          <option value="global">${t('全局规则')}</option><option value="project">${t('项目规则（{project}）', { project: c.project })}</option></select></label>
        <label class="field">${t('文件')}<input class="mono" value=${rule.file} onInput=${(e) => setRule({ ...rule, file: e.target.value.trim() })} aria-invalid=${!RULE_FILE.test(rule.file)} /></label></div>
      <label class="field">${t('追加到文件末尾的内容')}<${AutoText} value=${rule.text} onInput=${(v) => setRule({ ...rule, text: v })} rows=${3} /></label>
      <p class="muted small">${t('规则每次运行都全文注入：请写成简短、明确的要求。写入后这条候选移到 lessons/.done/。')}</p>
      <div class="row-actions"><button class="btn primary" disabled=${!RULE_FILE.test(rule.file) || !rule.text.trim()}
          onClick=${() => post('to_rule', rule, t('已写入{where} {file}；之后开始的运行将使用它', { where: rule.scope === 'global' ? t('全局规则') : t('项目规则'), file: rule.file }))}>${t('写入规则')}</button>
        <button class="btn ghost" onClick=${() => setRule(null)}>${t('取消')}</button></div></div>`}
  </article>`;
}

function Library({ lib, types }) {
  const note = indexNote(lib.index);
  return html`<section class="library"><h2>${LIB_NAME[lib.scope]} <span class="mono small muted">${lib.dir}/</span></h2>
    <p class=${`small ${note.over ? 'warn-text' : 'muted'}`}>${note.over && html`<${Icon} name="warning" />`}${note.text}</p>
    ${lib.entries.length ? html`<ul class="entries">${lib.entries.map((e) => html`<${Entry} key=${e.name} e=${e} lib=${lib} types=${types} />`)}</ul>`
      : html`<p class="empty-line">${t('经验库为空。在"候选"中采纳的经验会写到')} <code>${lib.dir}/entries/</code>${t('。')}</p>`}
    <details><summary>${t('INDEX.md（每次运行注入的内容，由程序根据 entries/ 生成）')}</summary><pre class="log">${lib.index.text || t('（空）')}</pre></details>
  </section>`;
}

function Entry({ e, lib, types }) {
  const [edit, setEdit] = useState(null);
  const where = { scope: lib.scope, project: lib.project || undefined, name: e.name };
  const save = () => api.put('/api/knowledge/entries', { ...where, ...edit, base_mtime: e.mtime })
    .then(() => { setEdit(null); toast(t('已保存 {name}，INDEX.md 已重新生成', { name: e.name })); }, fail);
  const remove = async () => {
    if (!await confirmDialog({ title: t('删除经验「{name}」？', { name: e.name }), body: html`<p>${t('条目移到')} <code>${lib.dir}/.trash/</code>${t('（移回 entries/ 即恢复），INDEX.md 重新生成，之后的运行不再注入它的索引行。')}</p>`, ok: t('删除'), danger: true })) return;
    api.post('/api/knowledge/entries/delete', { ...where, base_mtime: e.mtime }).then(() => toast(t('已删除 {name}', { name: e.name })), fail);
  };
  return html`<li>
    <div class="entry-head"><b class="mono">${e.name}</b><span class=${`tag ktype-${e.type}`}>${KTYPES[e.type] || e.type}</span><span class="spacer"></span>
      ${!edit && html`<button class="btn ghost small" onClick=${() => setEdit({ description: e.description, type: e.type, body: e.body })}>${t('编辑')}</button>
        <button class="btn ghost small danger-text" onClick=${remove}>${t('删除')}</button>`}</div>
    ${edit ? html`<div class="form-grid">
        <label class="field">${t('描述')}<input value=${edit.description} onInput=${(ev) => setEdit({ ...edit, description: ev.target.value })} /></label>
        <label class="field">${t('类型')}<select value=${edit.type} onChange=${(ev) => setEdit({ ...edit, type: ev.target.value })}>
          ${(types || Object.keys(KTYPES)).map((k) => html`<option value=${k}>${t('{label}（{type}）', { label: KTYPES[k] || k, type: k })}</option>`)}</select></label>
        <label class="field">${t('正文')}<${AutoText} value=${edit.body} onInput=${(v) => setEdit({ ...edit, body: v })} rows=${4} /></label>
        <div class="row-actions"><button class="btn primary small" onClick=${save}>${t('保存')}</button><button class="btn ghost small" onClick=${() => setEdit(null)}>${t('取消')}</button></div></div>`
      : html`<p class="entry-desc">${e.description}</p>
        ${e.problems?.length > 0 && html`<p class="warn-text small">${e.problems.join(t('；'))}</p>`}
        <div class="entry-meta"><span>${t('来源')} <span class="mono">${e.source || '—'}</span></span><span>${t('批准于 {date}', { date: e.approved || '—' })}</span>
          <span class="mono">${e.path}</span></div>
        <details><summary>${t('正文')}</summary><${Markdown} text=${e.body} className="doc small" /></details>`}
  </li>`;
}
